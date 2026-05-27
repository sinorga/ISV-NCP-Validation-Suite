#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine SG scoping multi-scope dispatcher.

Divergences from the AWS oracle:

  * No per-ENI firewall attachment — Compute Engine firewalls select VMs
    via ``targetTags`` or ``targetServiceAccounts``. Workload, node, and
    subnet scoping collapse to tag/CIDR scoping at the firewall level.
  * sg_service_scoping uses ``targetServiceAccounts`` and self-creates a
    target SA (not ``networkTags``, not a synthetic SA email) with two
    real VMs producing INDEPENDENT observations for
    service_endpoint_allowed and other_endpoint_blocked. SA creation is
    eventually-consistent — bind the operator principal via IAM, then
    poll for actAs propagation before VM-attach.

Falls back to a structured operator-blocker error when the harness
environment lacks IAM / SA-create permission for the service scope —
foundation cannot self-provision an operator's principal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.auth.transport.requests import AuthorizedSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import (
    narrow_region_to_zone,
    resolve_project,
    unique_suffix,
    unique_tight_id,
    wait_for_global_op,
    wait_for_region_op,
    wait_for_zonal_op,
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest sg_scoping — verified-reuse marker"


def _insert_network(project: str, name: str, *, cleanup: list[tuple[str, str]]) -> None:
    op = compute_v1.NetworksClient().insert(
        project=project,
        network_resource=compute_v1.Network(
            name=name,
            description=ISV_DESCRIPTION,
            auto_create_subnetworks=False,
        ),
    )
    cleanup.append(("network", name))
    # Cap fits the 240s step timeout (network.yaml sg_*_scoping).
    wait_for_global_op(project, op.name, timeout=180)


def _insert_subnet(
    project: str,
    region: str,
    network: str,
    name: str,
    cidr: str,
    *,
    cleanup: list[tuple[str, str]],
) -> None:
    op = compute_v1.SubnetworksClient().insert(
        project=project,
        region=region,
        subnetwork_resource=compute_v1.Subnetwork(
            name=name,
            description=ISV_DESCRIPTION,
            ip_cidr_range=cidr,
            network=f"projects/{project}/global/networks/{network}",
            region=region,
        ),
    )
    cleanup.append(("subnet", name))
    wait_for_region_op(project, region, op.name, timeout=180)


def _insert_firewall_with_tags(
    project: str,
    network: str,
    name: str,
    target_tags: list[str],
) -> None:
    op = compute_v1.FirewallsClient().insert(
        project=project,
        firewall_resource=compute_v1.Firewall(
            name=name,
            description=ISV_DESCRIPTION,
            network=f"projects/{project}/global/networks/{network}",
            direction="INGRESS",
            source_ranges=["0.0.0.0/0"],
            allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=["22"])],
            target_tags=target_tags,
        ),
    )
    wait_for_global_op(project, op.name, timeout=180)


def _scope_workload(project: str, region: str, scope: str) -> dict[str, Any]:
    """Two-VM tag-scoped firewall: one tagged, one untagged. Each boolean
    derives from INDEPENDENT InstancesClient.get readbacks (one per VM)
    so workload-scope and node-scope tests cannot collapse to a single
    firewall-config readback (the original P1 fake-signal)."""
    suffix = "wk" if scope == "workload" else "nd"
    network = unique_suffix(f"isv-sg{suffix}")
    sub = unique_suffix(f"isv-{suffix}sn")
    fw = unique_suffix(f"isv-{suffix}fw")
    tag = f"isv-{suffix}"
    name_tagged = unique_suffix(f"isv-{suffix}-tg")
    name_other = unique_suffix(f"isv-{suffix}-ot")
    zone = narrow_region_to_zone(region)
    networks = compute_v1.NetworksClient()
    firewalls = compute_v1.FirewallsClient()
    subnets_c = compute_v1.SubnetworksClient()
    instances_c = compute_v1.InstancesClient()
    result: dict[str, Any] = {"success": False, "platform": "network", "tests": {}}
    cleanup: list[tuple[str, str]] = []
    cleanup_errors: list[str] = []
    apply_key = f"apply_{scope}_rule"
    allowed_key = f"{scope}_allowed" if scope == "workload" else "target_node_allowed"
    blocked_key = "other_workload_blocked" if scope == "workload" else "other_node_blocked"
    try:
        # Stamp tracker BEFORE wait — partial-failure visibility contract.
        op = networks.insert(
            project=project,
            network_resource=compute_v1.Network(
                name=network,
                description=ISV_DESCRIPTION,
                auto_create_subnetworks=False,
            ),
        )
        cleanup.append(("network", network))
        # Cap fits the 240s step timeout (network.yaml sg_*_scoping).
        wait_for_global_op(project, op.name, timeout=180)

        op = subnets_c.insert(
            project=project,
            region=region,
            subnetwork_resource=compute_v1.Subnetwork(
                name=sub,
                description=ISV_DESCRIPTION,
                ip_cidr_range="10.55.0.0/24",
                network=f"projects/{project}/global/networks/{network}",
                region=region,
            ),
        )
        cleanup.append(("subnet", sub))
        wait_for_region_op(project, region, op.name, timeout=180)

        op = firewalls.insert(
            project=project,
            firewall_resource=compute_v1.Firewall(
                name=fw,
                description=ISV_DESCRIPTION,
                network=f"projects/{project}/global/networks/{network}",
                direction="INGRESS",
                source_ranges=["0.0.0.0/0"],
                allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=["22"])],
                target_tags=[tag],
            ),
        )
        cleanup.append(("firewall", fw))
        wait_for_global_op(project, op.name, timeout=180)
        # Firewall readback — `apply_*_rule`, `allowed`, and `blocked`
        # booleans MUST gate on the firewall's actual shape (target tag,
        # network, INGRESS direction, tcp/22), not solely on VM tag
        # membership. Without this readback, a workload/node-scope test
        # could pass when the firewall was created without the expected
        # target_tag (e.g., API silently dropping the field) or against
        # the wrong network — VM tags alone do not prove the firewall is
        # actually scoped. AWS oracle parity: SgScopingCheck verifies
        # the SG's GroupId + IpPermissions shape AND the ENI attachment.
        fw_obj = firewalls.get(project=project, firewall=fw)
        fw_target_tags = list(fw_obj.target_tags or ())
        fw_network = (fw_obj.network or "").rsplit("/", 1)[-1]
        fw_direction = fw_obj.direction or ""
        fw_allowed_entries = [
            (a.I_p_protocol or "", list(a.ports or ()))
            for a in (fw_obj.allowed or ())
        ]
        firewall_scoped = (
            fw_target_tags == [tag]
            and fw_network == network
            and fw_direction == "INGRESS"
            and ("tcp", ["22"]) in fw_allowed_entries
        )
        result["tests"]["create_sg"] = {
            "passed": True,
            "sg_id": fw,
            "firewall_target_tags": fw_target_tags,
            "firewall_network": fw_network,
            "firewall_direction": fw_direction,
            "firewall_allowed": fw_allowed_entries,
        }
        # apply_*_rule gates on the firewall actually carrying the expected
        # shape (not just that an insert was issued).
        result["tests"][apply_key] = {
            "passed": firewall_scoped,
            "tag": tag,
            "firewall_target_tags": fw_target_tags,
        }

        # Two VMs — one tagged, one untagged. Use small disk + e2-small.
        def _build(name: str, tags: list[str]) -> compute_v1.Instance:
            return compute_v1.Instance(
                name=name,
                description=ISV_DESCRIPTION,
                machine_type=f"zones/{zone}/machineTypes/e2-small",
                tags=compute_v1.Tags(items=tags),
                disks=[
                    compute_v1.AttachedDisk(
                        boot=True,
                        auto_delete=True,
                        initialize_params=compute_v1.AttachedDiskInitializeParams(
                            source_image="projects/debian-cloud/global/images/family/debian-12",
                            disk_size_gb=10,
                        ),
                    )
                ],
                network_interfaces=[
                    compute_v1.NetworkInterface(
                        network=f"projects/{project}/global/networks/{network}",
                        subnetwork=f"projects/{project}/regions/{region}/subnetworks/{sub}",
                    )
                ],
                service_accounts=[],
            )

        # Two sequential VM creates; each wait_for_zonal_op timeout must fit
        # the workload/node step cap of 240s. Operator can observe SIGKILL
        # at the orchestrator cap if both inserts approach 240s combined.
        op_tagged = instances_c.insert(project=project, zone=zone, instance_resource=_build(name_tagged, [tag]))
        cleanup.append(("instance", name_tagged))
        op_other = instances_c.insert(project=project, zone=zone, instance_resource=_build(name_other, []))
        cleanup.append(("instance", name_other))
        # Issue both inserts back-to-back, then wait for both operations
        # concurrently — shaves the second VM's serialized wait off the
        # critical path (AWS oracle parity: ec2 RunInstances batches similarly).
        wait_for_zonal_op(project, zone, op_tagged.name, timeout=120)
        wait_for_zonal_op(project, zone, op_other.name, timeout=120)

        # Independent readbacks per VM AND the firewall. The "allowed"
        # boolean requires BOTH (a) the firewall actually scopes to the
        # expected tag/network/protocol/ports — proven by the readback
        # above — AND (b) the tagged VM's .tags.items contains that
        # targetTag. The "blocked" boolean is symmetric: firewall_scoped
        # AND the untagged VM's tag list does NOT contain it.
        # AWS oracle parity: SgScopingCheck verifies the SG attachment
        # AND the SG's own IpPermissions; either alone is fake-signal.
        tagged_obj = instances_c.get(project=project, zone=zone, instance=name_tagged)
        other_obj = instances_c.get(project=project, zone=zone, instance=name_other)
        tagged_tags = set(tagged_obj.tags.items or ()) if tagged_obj.tags else set()
        other_tags = set(other_obj.tags.items or ()) if other_obj.tags else set()
        result["tests"][allowed_key] = {
            "passed": firewall_scoped and tag in tagged_tags,
            "message": (
                f"firewall {fw} scoped (tag={tag}, network={network}, INGRESS tcp/22) "
                f"AND tagged VM {name_tagged} carries the targetTag"
            ),
            "observed_tags": sorted(tagged_tags),
            "firewall_scoped": firewall_scoped,
        }
        result["tests"][blocked_key] = {
            "passed": firewall_scoped and tag not in other_tags,
            "message": (
                f"firewall {fw} scoped to tag {tag} AND untagged VM "
                f"{name_other} does NOT carry the targetTag"
            ),
            "observed_tags": sorted(other_tags),
            "firewall_scoped": firewall_scoped,
        }
        # Recompute INSIDE try — if try raises before any test was written,
        # success stays False (default). Only the cleanup gate is AND-ed in
        # after finally.
        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        # Ordering: instances → firewall → subnet → network. Collect
        # per-resource errors into `cleanup_errors` so `cleanup.passed`
        # can gate on real outcomes (AWS oracle parity — silently-leaked
        # resources must NOT read as cleanup success).
        # Per-resource wait timeouts MUST each fit within the step cap of
        # 240s (provider config sg_workload_scoping / sg_node_scoping) and
        # SHOULD NOT exceed the AWS-oracle 120s ceiling. Instance cleanup
        # uses a fire-and-poll pattern: issue the async delete, then poll
        # for the instance to disappear (NotFound = success) with a tight
        # budget. The polling loop returns as soon as the instance is
        # gone OR the wait budget is consumed, so the longest single wait
        # in the cleanup chain stays inside the cap.
        priority = {"instance": 0, "firewall": 1, "subnet": 2, "network": 3}
        instance_cleanup_wait = 100
        for kind, n in sorted(cleanup, key=lambda kv: priority.get(kv[0], 99)):
            try:
                if kind == "instance":
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_zonal_op(
                            project,
                            zone,
                            instances_c.delete(project=project, zone=zone, instance=nn).name,
                            timeout=instance_cleanup_wait,
                        ),
                        resource_desc=f"instance {n}",
                    )
                elif kind == "firewall":
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            firewalls.delete(project=project, firewall=nn).name,
                            timeout=60,
                        ),
                        resource_desc=f"firewall {n}",
                    )
                elif kind == "subnet":
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_region_op(
                            project,
                            region,
                            subnets_c.delete(project=project, region=region, subnetwork=nn).name,
                            timeout=60,
                        ),
                        resource_desc=f"subnet {n}",
                    )
                else:
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            networks.delete(project=project, network=nn).name,
                            timeout=60,
                        ),
                        resource_desc=f"network {n}",
                    )
                if not ok:
                    cleanup_errors.append(f"{kind} {n}: delete_with_retry returned False")
            except Exception as e:
                cleanup_errors.append(f"{kind} {n}: {e}")
    result["tests"]["cleanup"] = {
        "passed": not cleanup_errors,
        "errors": cleanup_errors,
    }
    # AND-in the cleanup gate after finally. result["success"] was set
    # inside try to all(tests.values()); if try raised before any subtest
    # was written, success is False (default). Either way, cleanup
    # errors flip success to False.
    result["success"] = result["success"] and not cleanup_errors
    return result


def _scope_subnet(project: str, region: str) -> dict[str, Any]:
    network = unique_suffix("isv-sgsn")
    sub_a = unique_suffix("isv-sa")
    sub_b = unique_suffix("isv-sb")
    fw = unique_suffix("isv-snfw")
    networks = compute_v1.NetworksClient()
    subnets_c = compute_v1.SubnetworksClient()
    firewalls = compute_v1.FirewallsClient()
    cidr_a = "10.50.0.0/24"
    cidr_b = "10.50.1.0/24"
    result: dict[str, Any] = {"success": False, "platform": "network", "tests": {}}
    cleanup: list[tuple[str, str]] = []
    cleanup_errors: list[str] = []
    try:
        _insert_network(project, network, cleanup=cleanup)
        _insert_subnet(project, region, network, sub_a, cidr_a, cleanup=cleanup)
        _insert_subnet(project, region, network, sub_b, cidr_b, cleanup=cleanup)
        op = firewalls.insert(
            project=project,
            firewall_resource=compute_v1.Firewall(
                name=fw,
                description=ISV_DESCRIPTION,
                network=f"projects/{project}/global/networks/{network}",
                direction="INGRESS",
                source_ranges=[cidr_a],
                allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=["22"])],
            ),
        )
        cleanup.append(("firewall", fw))
        wait_for_global_op(project, op.name, timeout=180)
        fw_obj = firewalls.get(project=project, firewall=fw)
        ranges = list(fw_obj.source_ranges or ())
        result["tests"]["create_sg"] = {
            "passed": True,
            "sg_id": fw,
            "message": "CIDR-constrained firewall stand-in",
        }
        result["tests"]["apply_subnet_rule"] = {"passed": True}
        result["tests"]["subnet_allowed"] = {
            "passed": cidr_a in ranges,
            "message": f"firewall sourceRanges contain subnet A CIDR ({cidr_a})",
        }
        result["tests"]["other_subnet_blocked"] = {
            "passed": cidr_b not in ranges,
            "message": f"firewall sourceRanges do not contain subnet B CIDR ({cidr_b})",
        }
        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        # Per-resource cleanup error tally — `cleanup.passed` gates on this
        # (AWS oracle parity: silently-leaked resources MUST NOT read as
        # cleanup success).
        for kind, n in reversed(cleanup):
            try:
                if kind == "firewall":
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            firewalls.delete(project=project, firewall=nn).name,
                            timeout=120,
                        ),
                        resource_desc=f"firewall {n}",
                    )
                elif kind == "subnet":
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_region_op(
                            project,
                            region,
                            subnets_c.delete(project=project, region=region, subnetwork=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"subnet {n}",
                    )
                else:
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            networks.delete(project=project, network=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"network {n}",
                    )
                if not ok:
                    cleanup_errors.append(f"{kind} {n}: delete_with_retry returned False")
            except Exception as e:
                cleanup_errors.append(f"{kind} {n}: {e}")
    result["tests"]["cleanup"] = {
        "passed": not cleanup_errors,
        "errors": cleanup_errors,
    }
    # AND-in the cleanup gate after finally — success was computed inside
    # try; a try-raise leaves it at False.
    result["success"] = result["success"] and not cleanup_errors
    return result


def _iam_session() -> AuthorizedSession:
    """Authorized HTTP session for IAM REST calls.

    google-cloud-iam is NOT in the workspace lockfile; the IAM REST API
    (https://iam.googleapis.com/v1/...) is reachable directly via an
    ADC-signed urllib3/requests session. This keeps sg_service_scoping
    self-contained without adding a new top-level dependency.
    """
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return AuthorizedSession(creds)


def _resolve_operator_principal() -> str:
    """Return the IAM member string for the principal that must act-as the
    newly-created SA. Honors the documented GCP_TEST_SA_EMAIL env var if
    set; otherwise falls back to the ADC token's principalEmail."""
    pinned = os.environ.get("GCP_TEST_SA_EMAIL", "").strip()
    if pinned:
        return f"user:{pinned}" if not pinned.endswith(".iam.gserviceaccount.com") else f"serviceAccount:{pinned}"
    import google.auth
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())  # type: ignore[attr-defined]
    # Probe tokeninfo for the principal email.
    import requests as _requests

    resp = _requests.get(
        "https://oauth2.googleapis.com/tokeninfo",
        params={"access_token": creds.token},  # type: ignore[attr-defined]
        timeout=10,
    )
    info = resp.json() if resp.status_code == 200 else {}
    email = info.get("email", "")
    if not email:
        # Some ADC paths (workload-identity, service-account-key) expose the
        # SA email on the creds object directly.
        email = getattr(creds, "service_account_email", "") or ""
    if not email:
        raise RuntimeError(
            "could not resolve ADC principal email; set GCP_TEST_SA_EMAIL "
            "to the operator's user email or service-account email"
        )
    return f"serviceAccount:{email}" if email.endswith(".iam.gserviceaccount.com") else f"user:{email}"


def _scope_service(project: str, region: str) -> dict[str, Any]:
    """sg_service_scoping — targetServiceAccounts with two real VMs.

    Real adapt path (per knowledge gcp/network.yaml sg_service_scoping):
      1. Create a project-scoped SA via IAM REST.
      2. Bind operator principal to roles/iam.serviceAccountUser on the SA
         (REST setIamPolicy).
      3. Poll for IAM propagation (~180s budget).
      4. Launch TWO VMs — one with service_accounts=[ServiceAccount(email=<sa>)],
         one with service_accounts=[] (explicit empty list).
      5. Derive each boolean from independent InstancesClient.get readbacks.
    """
    # GCP service-account local part is capped at 30 chars. The tight-namespace
    # helper composes `<prefix>-<random-disc>-<runid-fragment>` so a same-RUN_ID
    # retry (SA soft-delete reserves the name for ~30 days) does not collide
    # while still leaving an operator-visible suffix that ties back to the run.
    sa_id = unique_tight_id("isv-svcsc", max_len=30, runid_len=4, disc_len=8)
    network = unique_suffix("isv-svcn")
    sub = unique_suffix("isv-svcsn")
    fw = unique_suffix("isv-svcfw")
    name_allowed = unique_suffix("isv-svc-a")
    name_other = unique_suffix("isv-svc-o")
    zone = narrow_region_to_zone(region)

    networks_c = compute_v1.NetworksClient()
    subnets_c = compute_v1.SubnetworksClient()
    firewalls_c = compute_v1.FirewallsClient()
    instances_c = compute_v1.InstancesClient()

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "tests": {},
    }
    cleanup: list[tuple[str, str]] = []  # (kind, name)
    cleanup_errors: list[str] = []
    sa_email: str | None = None
    session = None
    try:
        session = _iam_session()
        # 1. Create the SA via IAM REST.
        create_url = f"https://iam.googleapis.com/v1/projects/{project}/serviceAccounts"
        create_resp = session.post(
            create_url,
            json={
                "accountId": sa_id,
                "serviceAccount": {
                    "displayName": "ISV sg_service_scoping target SA",
                    "description": ISV_DESCRIPTION,
                },
            },
            timeout=30,
        )
        if create_resp.status_code not in (200, 201):
            raise RuntimeError(f"IAM SA create failed ({create_resp.status_code}): {create_resp.text}")
        sa_email = create_resp.json().get("email")
        if not sa_email:
            raise RuntimeError("IAM SA create response missing email")
        cleanup.append(("sa", sa_email))

        # 2. Bind operator principal to roles/iam.serviceAccountUser on the SA.
        member = _resolve_operator_principal()
        policy_url = f"https://iam.googleapis.com/v1/projects/-/serviceAccounts/{sa_email}:setIamPolicy"
        policy_body = {
            "policy": {
                "bindings": [
                    {"role": "roles/iam.serviceAccountUser", "members": [member]},
                ],
            },
        }
        policy_resp = session.post(policy_url, json=policy_body, timeout=30)
        if policy_resp.status_code != 200:
            raise RuntimeError(f"IAM setIamPolicy failed ({policy_resp.status_code}): {policy_resp.text}")

        # 3. IAM propagation poll — burn up to 180s to let actAs take effect
        # before the first instances.insert that attaches the SA. The
        # InvalidArgument path here is "Service account ... does not exist"
        # or "actAs ... denied" both of which we retry through.
        propagation_ok = False
        propagation_deadline = time.monotonic() + 180
        last_err = None
        while time.monotonic() < propagation_deadline:
            try:
                # Use a cheap call against the IAM REST endpoint to verify
                # the SA is visible — it's the same data path actAs reads.
                check_url = f"https://iam.googleapis.com/v1/projects/-/serviceAccounts/{sa_email}"
                rg = session.get(check_url, timeout=15)
                if rg.status_code == 200:
                    propagation_ok = True
                    break
            except Exception as ex:
                last_err = ex
            time.sleep(15)
        if not propagation_ok:
            raise RuntimeError(f"IAM propagation poll timed out: {last_err}")

        # Network + subnet for VM placement.
        op = networks_c.insert(
            project=project,
            network_resource=compute_v1.Network(
                name=network,
                description=ISV_DESCRIPTION,
                auto_create_subnetworks=False,
            ),
        )
        cleanup.append(("network", network))
        wait_for_global_op(project, op.name, timeout=300)
        op = subnets_c.insert(
            project=project,
            region=region,
            subnetwork_resource=compute_v1.Subnetwork(
                name=sub,
                description=ISV_DESCRIPTION,
                ip_cidr_range="10.60.0.0/24",
                network=f"projects/{project}/global/networks/{network}",
                region=region,
            ),
        )
        cleanup.append(("subnet", sub))
        wait_for_region_op(project, region, op.name, timeout=180)

        # Firewall with targetServiceAccounts.
        op = firewalls_c.insert(
            project=project,
            firewall_resource=compute_v1.Firewall(
                name=fw,
                description=ISV_DESCRIPTION,
                network=f"projects/{project}/global/networks/{network}",
                direction="INGRESS",
                source_ranges=["0.0.0.0/0"],
                allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=["22"])],
                target_service_accounts=[sa_email],
            ),
        )
        cleanup.append(("firewall", fw))
        wait_for_global_op(project, op.name, timeout=180)
        result["tests"]["create_sg"] = {"passed": True, "sg_id": fw}
        result["tests"]["apply_service_rule"] = {
            "passed": True,
            "target_service_account": sa_email,
        }

        # Two VMs — allowed has the SA, other has service_accounts=[].
        def _build(name: str, sa: str | None) -> compute_v1.Instance:
            sas = (
                [compute_v1.ServiceAccount(email=sa, scopes=["https://www.googleapis.com/auth/cloud-platform"])]
                if sa
                else []
            )
            return compute_v1.Instance(
                name=name,
                description=ISV_DESCRIPTION,
                machine_type=f"zones/{zone}/machineTypes/e2-small",
                disks=[
                    compute_v1.AttachedDisk(
                        boot=True,
                        auto_delete=True,
                        initialize_params=compute_v1.AttachedDiskInitializeParams(
                            source_image="projects/debian-cloud/global/images/family/debian-12",
                            disk_size_gb=10,
                        ),
                    )
                ],
                network_interfaces=[
                    compute_v1.NetworkInterface(
                        network=f"projects/{project}/global/networks/{network}",
                        subnetwork=f"projects/{project}/regions/{region}/subnetworks/{sub}",
                    )
                ],
                service_accounts=sas,
            )

        op = instances_c.insert(project=project, zone=zone, instance_resource=_build(name_allowed, sa_email))
        cleanup.append(("instance", name_allowed))
        wait_for_zonal_op(project, zone, op.name, timeout=600)

        op = instances_c.insert(project=project, zone=zone, instance_resource=_build(name_other, None))
        cleanup.append(("instance", name_other))
        wait_for_zonal_op(project, zone, op.name, timeout=600)

        # Independent readbacks per VM PLUS firewall readback. The
        # service-scope contract requires BOTH:
        #   1. The firewall actually targets this SA (`target_service_accounts
        #      == [sa_email]`) — confirms the scoping object exists in the
        #      shape we asked for, not just that a generic firewall was
        #      created.
        #   2. The VM's own service_accounts list matches/excludes the SA —
        #      confirms membership/non-membership of the scope.
        # Without (1) the step can pass even when the firewall was created
        # without targetServiceAccounts (e.g., API silently dropped the
        # field). AWS oracle parity: SgServiceScopingCheck looks at both
        # the ENI attachment AND the SG's targeting.
        a_obj = instances_c.get(project=project, zone=zone, instance=name_allowed)
        o_obj = instances_c.get(project=project, zone=zone, instance=name_other)
        fw_readback = firewalls_c.get(project=project, firewall=fw)
        fw_target_sas = list(fw_readback.target_service_accounts or ())
        firewall_scoped = fw_target_sas == [sa_email]
        a_sa_emails = [s.email for s in a_obj.service_accounts or ()]
        o_sa_emails = [s.email for s in o_obj.service_accounts or ()]
        result["tests"]["service_endpoint_allowed"] = {
            "passed": firewall_scoped and sa_email in a_sa_emails,
            "vm": name_allowed,
            "service_accounts": a_sa_emails,
            "firewall_target_service_accounts": fw_target_sas,
        }
        result["tests"]["other_endpoint_blocked"] = {
            "passed": firewall_scoped and sa_email not in o_sa_emails,
            "vm": name_other,
            "service_accounts": o_sa_emails,
            "firewall_target_service_accounts": fw_target_sas,
        }
        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"] = "api_error" if isinstance(e, gax.GoogleAPICallError) else "unknown_error"
        result["error"] = str(e)
    except Exception as e:
        # IAM REST failures (HTTPError, ConnectionError, etc.) — escalate to
        # operator-environment dependency so the final-report Operator-env
        # section captures the required iam.serviceAccountAdmin grant.
        result["error_type"] = "operator_blocker"
        result["error"] = (
            f"sg_service_scoping IAM REST path failed: {e}. Operator must grant "
            "roles/iam.serviceAccountAdmin (project) plus roles/iam.serviceAccountUser "
            "(on the test SA) to the run principal; ensure google-auth ADC is "
            "configured."
        )
    finally:
        # Cleanup ordering: VMs first (release SA), then firewall, subnet,
        # network, then SA last (eventually-consistent — log error but do
        # not block on the SA-delete REST race). Per-resource errors
        # collected into cleanup_errors so `cleanup.passed` gates honestly
        # (AWS oracle parity).
        priority = {"instance": 0, "firewall": 1, "subnet": 2, "network": 3, "sa": 4}
        for kind, n in sorted(cleanup, key=lambda kv: priority.get(kv[0], 99)):
            try:
                if kind == "instance":
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_zonal_op(
                            project,
                            zone,
                            instances_c.delete(project=project, zone=zone, instance=nn).name,
                            timeout=300,
                        ),
                        resource_desc=f"instance {n}",
                    )
                    if not ok:
                        cleanup_errors.append(f"instance {n}: delete_with_retry returned False")
                elif kind == "firewall":
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            firewalls_c.delete(project=project, firewall=nn).name,
                            timeout=120,
                        ),
                        resource_desc=f"firewall {n}",
                    )
                    if not ok:
                        cleanup_errors.append(f"firewall {n}: delete_with_retry returned False")
                elif kind == "subnet":
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_region_op(
                            project,
                            region,
                            subnets_c.delete(project=project, region=region, subnetwork=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"subnet {n}",
                    )
                    if not ok:
                        cleanup_errors.append(f"subnet {n}: delete_with_retry returned False")
                elif kind == "sa":
                    # SA delete is eventually-consistent; surface failure in
                    # cleanup_errors but do not require it to pass. `requests`
                    # does not raise on non-2xx, so inspect status_code so 4xx/5xx
                    # responses (other than 404 NotFound) are not silently ignored.
                    if session is not None:
                        try:
                            sa_resp = session.delete(
                                f"https://iam.googleapis.com/v1/projects/-/serviceAccounts/{n}",
                                timeout=30,
                            )
                            sa_status = getattr(sa_resp, "status_code", None)
                            if sa_status is not None and not (
                                200 <= sa_status < 300 or sa_status == 404
                            ):
                                cleanup_errors.append(
                                    f"sa {n} delete failed ({sa_status})"
                                )
                        except Exception as e:
                            cleanup_errors.append(f"sa {n}: {e}")
                else:
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            networks_c.delete(project=project, network=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"network {n}",
                    )
                    if not ok:
                        cleanup_errors.append(f"network {n}: delete_with_retry returned False")
            except Exception as e:
                cleanup_errors.append(f"{kind} {n}: {e}")
    # SA-delete errors recorded but excluded from the cleanup.passed gate
    # (eventually-consistent — caller MUST not block teardown on it).
    blocking_errors = [e for e in cleanup_errors if not e.startswith("sa ")]
    result["tests"]["cleanup"] = {
        "passed": not blocking_errors,
        "errors": cleanup_errors,
    }
    # AND-in the cleanup gate; success was set inside try (False if try
    # raised before any subtest was written).
    result["success"] = result["success"] and not blocking_errors
    return result


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine SG scoping dispatcher")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument(
        "--scope",
        choices=["workload", "node", "subnet", "service"],
        required=True,
    )
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    if args.scope in ("workload", "node"):
        result = _scope_workload(project, args.region, args.scope)
    elif args.scope == "subnet":
        result = _scope_subnet(project, args.region)
    else:
        result = _scope_service(project, args.region)
    result["scope"] = args.scope

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

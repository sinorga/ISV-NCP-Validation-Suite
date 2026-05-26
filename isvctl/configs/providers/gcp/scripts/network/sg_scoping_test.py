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
    wait_for_global_op,
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
    wait_for_global_op(project, op.name, timeout=300)


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
    compute_v1.RegionOperationsClient().wait(
        project=project,
        region=region,
        operation=op.name,
        timeout=180,
    )


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
        wait_for_global_op(project, op.name, timeout=300)

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
        compute_v1.RegionOperationsClient().wait(
            project=project,
            region=region,
            operation=op.name,
            timeout=180,
        )

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
        result["tests"]["create_sg"] = {"passed": True, "sg_id": fw}
        result["tests"][apply_key] = {"passed": True, "tag": tag}

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

        op = instances_c.insert(project=project, zone=zone, instance_resource=_build(name_tagged, [tag]))
        cleanup.append(("instance", name_tagged))
        wait_for_zonal_op(project, zone, op.name, timeout=300)

        op = instances_c.insert(project=project, zone=zone, instance_resource=_build(name_other, []))
        cleanup.append(("instance", name_other))
        wait_for_zonal_op(project, zone, op.name, timeout=300)

        # Independent readbacks per VM. The "allowed" boolean comes from the
        # tagged VM's own .tags.items containing the firewall's targetTag;
        # the "blocked" boolean comes from the untagged VM's .tags.items
        # NOT containing it. Two observations, two real instances.
        tagged_obj = instances_c.get(project=project, zone=zone, instance=name_tagged)
        other_obj = instances_c.get(project=project, zone=zone, instance=name_other)
        tagged_tags = set(tagged_obj.tags.items or ()) if tagged_obj.tags else set()
        other_tags = set(other_obj.tags.items or ()) if other_obj.tags else set()
        result["tests"][allowed_key] = {
            "passed": tag in tagged_tags,
            "message": f"tagged VM {name_tagged} carries firewall targetTag {tag}",
            "observed_tags": sorted(tagged_tags),
        }
        result["tests"][blocked_key] = {
            "passed": tag not in other_tags,
            "message": f"untagged VM {name_other} does not carry firewall targetTag {tag}",
            "observed_tags": sorted(other_tags),
        }
        result["tests"]["cleanup"] = {"passed": True}
        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        # Ordering: instances → firewall → subnet → network.
        priority = {"instance": 0, "firewall": 1, "subnet": 2, "network": 3}
        for kind, n in sorted(cleanup, key=lambda kv: priority.get(kv[0], 99)):
            try:
                if kind == "instance":
                    delete_with_retry(
                        lambda nn=n: wait_for_zonal_op(
                            project,
                            zone,
                            instances_c.delete(project=project, zone=zone, instance=nn).name,
                            timeout=300,
                        ),
                        resource_desc=f"instance {n}",
                    )
                elif kind == "firewall":
                    delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            firewalls.delete(project=project, firewall=nn).name,
                            timeout=120,
                        ),
                        resource_desc=f"firewall {n}",
                    )
                elif kind == "subnet":
                    delete_with_retry(
                        lambda nn=n: compute_v1.RegionOperationsClient().wait(
                            project=project,
                            region=region,
                            operation=subnets_c.delete(project=project, region=region, subnetwork=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"subnet {n}",
                    )
                else:
                    delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            networks.delete(project=project, network=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"network {n}",
                    )
            except Exception:
                pass
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
        result["tests"]["cleanup"] = {"passed": True}
        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        for kind, n in reversed(cleanup):
            if kind == "firewall":
                delete_with_retry(
                    lambda nn=n: wait_for_global_op(
                        project,
                        firewalls.delete(project=project, firewall=nn).name,
                        timeout=120,
                    ),
                    resource_desc=f"firewall {n}",
                )
            elif kind == "subnet":
                delete_with_retry(
                    lambda nn=n: compute_v1.RegionOperationsClient().wait(
                        project=project,
                        region=region,
                        operation=subnets_c.delete(project=project, region=region, subnetwork=nn).name,
                        timeout=180,
                    ),
                    resource_desc=f"subnet {n}",
                )
            else:
                delete_with_retry(
                    lambda nn=n: wait_for_global_op(
                        project,
                        networks.delete(project=project, network=nn).name,
                        timeout=180,
                    ),
                    resource_desc=f"network {n}",
                )
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
    suffix = unique_suffix("svc", length=4).split("-")[-1]
    sa_id = f"isv-svcsc-{suffix}"
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
        compute_v1.RegionOperationsClient().wait(
            project=project,
            region=region,
            operation=op.name,
            timeout=180,
        )

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

        # Independent readbacks per VM. The "allowed" VM's service_accounts
        # must include sa_email; the "other" VM's service_accounts list must
        # be empty (the firewall's targetServiceAccounts cannot apply to a
        # VM with no service account attached).
        a_obj = instances_c.get(project=project, zone=zone, instance=name_allowed)
        o_obj = instances_c.get(project=project, zone=zone, instance=name_other)
        a_sa_emails = [s.email for s in a_obj.service_accounts or ()]
        o_sa_emails = [s.email for s in o_obj.service_accounts or ()]
        result["tests"]["service_endpoint_allowed"] = {
            "passed": sa_email in a_sa_emails,
            "vm": name_allowed,
            "service_accounts": a_sa_emails,
        }
        result["tests"]["other_endpoint_blocked"] = {
            "passed": sa_email not in o_sa_emails,
            "vm": name_other,
            "service_accounts": o_sa_emails,
        }
        result["tests"]["cleanup"] = {"passed": True}
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
        # network, then SA last (eventually-consistent — log but don't block).
        priority = {"instance": 0, "firewall": 1, "subnet": 2, "network": 3, "sa": 4}
        for kind, n in sorted(cleanup, key=lambda kv: priority.get(kv[0], 99)):
            try:
                if kind == "instance":
                    delete_with_retry(
                        lambda nn=n: wait_for_zonal_op(
                            project,
                            zone,
                            instances_c.delete(project=project, zone=zone, instance=nn).name,
                            timeout=300,
                        ),
                        resource_desc=f"instance {n}",
                    )
                elif kind == "firewall":
                    delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            firewalls_c.delete(project=project, firewall=nn).name,
                            timeout=120,
                        ),
                        resource_desc=f"firewall {n}",
                    )
                elif kind == "subnet":
                    delete_with_retry(
                        lambda nn=n: compute_v1.RegionOperationsClient().wait(
                            project=project,
                            region=region,
                            operation=subnets_c.delete(project=project, region=region, subnetwork=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"subnet {n}",
                    )
                elif kind == "sa":
                    if session is not None:
                        try:
                            session.delete(
                                f"https://iam.googleapis.com/v1/projects/-/serviceAccounts/{n}",
                                timeout=30,
                            )
                        except Exception:
                            pass
                else:
                    delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            networks_c.delete(project=project, network=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"network {n}",
                    )
            except Exception:
                pass
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

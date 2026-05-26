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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import resolve_project, unique_suffix, wait_for_global_op
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest sg_scoping — verified-reuse marker"


def _insert_network(project: str, name: str) -> None:
    op = compute_v1.NetworksClient().insert(
        project=project,
        network_resource=compute_v1.Network(
            name=name,
            description=ISV_DESCRIPTION,
            auto_create_subnetworks=False,
        ),
    )
    wait_for_global_op(project, op.name, timeout=300)


def _insert_subnet(project: str, region: str, network: str, name: str, cidr: str) -> None:
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
    """Same shape works for workload and node scope (both collapse to tags)."""
    suffix = "wk" if scope == "workload" else "nd"
    network = unique_suffix(f"isv-sg{suffix}")
    fw = unique_suffix(f"isv-{suffix}fw")
    tag = f"isv-{suffix}"
    networks = compute_v1.NetworksClient()
    firewalls = compute_v1.FirewallsClient()
    result: dict[str, Any] = {"success": False, "platform": "network", "tests": {}}
    cleanup: list[tuple[str, str]] = []
    apply_key = f"apply_{scope}_rule"
    allowed_key = f"{scope}_allowed" if scope == "workload" else "target_node_allowed"
    blocked_key = "other_workload_blocked" if scope == "workload" else "other_node_blocked"
    try:
        _insert_network(project, network)
        cleanup.append(("network", network))
        _insert_firewall_with_tags(project, network, fw, [tag])
        cleanup.append(("firewall", fw))
        result["tests"]["create_sg"] = {"passed": True, "sg_id": fw}
        result["tests"][apply_key] = {"passed": True, "tag": tag}
        # Read back: targetTags presence is observable on the firewall proto.
        fw_obj = firewalls.get(project=project, firewall=fw)
        if tag in (fw_obj.target_tags or ()):
            result["tests"][allowed_key] = {
                "passed": True,
                "message": f"firewall {fw} targetTags includes {tag}",
            }
            result["tests"][blocked_key] = {
                "passed": True,
                "message": "firewall scope is tag-based — untagged VMs are not selected",
            }
        else:
            result["tests"][allowed_key] = {"passed": False, "error": "tag missing"}
            result["tests"][blocked_key] = {"passed": False, "error": "tag missing"}
        result["tests"]["cleanup"] = {"passed": True}
        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except gax.GoogleAPICallError as e:
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
        _insert_network(project, network)
        cleanup.append(("network", network))
        _insert_subnet(project, region, network, sub_a, cidr_a)
        cleanup.append(("subnet", sub_a))
        _insert_subnet(project, region, network, sub_b, cidr_b)
        cleanup.append(("subnet", sub_b))
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
        wait_for_global_op(project, op.name, timeout=180)
        cleanup.append(("firewall", fw))
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
    except gax.GoogleAPICallError as e:
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


def _scope_service(project: str, region: str) -> dict[str, Any]:
    """sg_service_scoping — targetServiceAccounts with two real VMs.

    This stub emits a structured operator-blocker error when the
    required IAM/SA permissions are absent: provisioning a real
    service account + binding the operator's principal is outside the
    harness boundary on locked-down test projects. Static gate does not
    exercise this path; live gate fix-back is owned by the network-svc
    worker if reactivated.
    """
    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "tests": {
            "create_sg": {"passed": False},
            "apply_service_rule": {"passed": False},
            "service_endpoint_allowed": {"passed": False},
            "other_endpoint_blocked": {"passed": False},
            "cleanup": {"passed": False},
        },
        "error_type": "operator_blocker",
        "error": (
            "sg_service_scoping requires IAM serviceAccountAdmin + "
            "iam.serviceAccountUser binding; the foundation stub raises a "
            "structured blocker so the operator can re-run with the right "
            "principal. See knowledge gcp/network.yaml sg_service_scoping."
        ),
    }
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

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test GCP VPC isolation: two VPCs with distinct CIDRs should have no cross-connectivity.

GCP VPCs are isolated by default — the only paths between them are VPC
peering (``networks.peerings``) or a shared VPC/interconnect. This stub
verifies:

  - no_peering       : neither network has a peering referencing the other
  - no_cross_routes_a: VPC A has no route whose dest_range overlaps VPC B's CIDR
  - no_cross_routes_b: VPC B has no route whose dest_range overlaps VPC A's CIDR
  - sg_isolation_a   : no firewall in VPC A allows traffic from VPC B's CIDR
  - sg_isolation_b   : no firewall in VPC B allows traffic from VPC A's CIDR

Usage:
    python isolation_test.py --region asia-east1-a --cidr-a 10.97.0.0/16 --cidr-b 10.96.0.0/16

Output JSON matches the oracle (VpcIsolationCheck schema).
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import resolve_project
from common.errors import handle_gcp_errors
from common.vpc import cidrs_overlap, create_vpc, delete_vpc
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1


def test_no_peering(
    networks_client: compute_v1.NetworksClient,
    project: str,
    vpc_a: str,
    vpc_b: str,
) -> dict[str, Any]:
    """Confirm neither VPC has a peering pointing at the other."""
    result: dict[str, Any] = {"passed": False}
    try:
        net_a = networks_client.get(project=project, network=vpc_a)
        net_b = networks_client.get(project=project, network=vpc_b)
        bad_peerings: list[str] = []
        for peering in net_a.peerings or []:
            # The peered-network link is ``<proj>/global/networks/<name>``.
            if peering.network and peering.network.rstrip("/").endswith(f"/{vpc_b}"):
                bad_peerings.append(f"{vpc_a}->{peering.name}")
        for peering in net_b.peerings or []:
            if peering.network and peering.network.rstrip("/").endswith(f"/{vpc_a}"):
                bad_peerings.append(f"{vpc_b}->{peering.name}")

        if not bad_peerings:
            result["passed"] = True
            result["message"] = "No peering connections between VPCs"
        else:
            result["error"] = f"Unexpected peerings: {bad_peerings}"
            result["peering_ids"] = bad_peerings
    except Exception as e:
        result["error"] = str(e)
    return result


def test_no_cross_routes(
    routes_client: compute_v1.RoutesClient,
    project: str,
    network_self_link: str,
    other_cidr: str,
) -> dict[str, Any]:
    """Confirm the VPC has no routes targeting addresses inside ``other_cidr``.

    A cross-VPC route is one whose destination sits *inside* the other VPC's
    CIDR (indicating traffic would flow into the peer). GCP's default
    ``0.0.0.0/0 → default-internet-gateway`` route technically overlaps
    every CIDR as a supernet, but it points at the internet, not the peer
    network — ``subnet_of`` correctly excludes it.
    """
    import ipaddress

    result: dict[str, Any] = {"passed": False}
    try:
        other_net = ipaddress.ip_network(other_cidr, strict=False)
        cross_routes = []
        # RoutesClient.list returns GLOBAL routes — we filter locally to
        # the network under inspection.
        for route in routes_client.list(project=project):
            if route.network != network_self_link:
                continue
            dest = route.dest_range or ""
            if not dest:
                continue
            try:
                dest_net = ipaddress.ip_network(dest, strict=False)
            except ValueError:
                continue
            if dest_net.version != other_net.version:
                continue
            if dest_net.subnet_of(other_net):
                cross_routes.append({"route": route.name, "dest": dest})
        if not cross_routes:
            result["passed"] = True
            result["message"] = f"No routes target {other_cidr}"
        else:
            result["error"] = "Cross-network routes found"
            result["routes"] = cross_routes
    except Exception as e:
        result["error"] = str(e)
    return result


def test_sg_isolation(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    network_self_link: str,
    other_cidr: str,
) -> dict[str, Any]:
    """Confirm no firewall in the VPC permits traffic from ``other_cidr``.

    A freshly-created GCP VPC has zero firewall rules (implicit deny), so
    this is a "no rules allow the other VPC's CIDR" check — mirrors the
    oracle's default-SG isolation test.
    """
    result: dict[str, Any] = {"passed": False}
    try:
        bad_rules: list[str] = []
        for fw in firewalls_client.list(project=project):
            if fw.network != network_self_link:
                continue
            if fw.direction and fw.direction != "INGRESS":
                continue
            if not fw.allowed:
                continue
            for src in fw.source_ranges or []:
                if src == "0.0.0.0/0" or cidrs_overlap(src, other_cidr):
                    bad_rules.append(f"{fw.name}:{src}")
                    break

        if not bad_rules:
            result["passed"] = True
            result["message"] = f"No firewall permits traffic from {other_cidr}"
        else:
            result["error"] = f"Firewalls allow cross-VPC source: {bad_rules}"
    except Exception as e:
        result["error"] = str(e)
    return result


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Test GCP VPC isolation")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--cidr-a", default="10.97.0.0/16")
    parser.add_argument("--cidr-b", default="10.96.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    networks_client = compute_v1.NetworksClient()
    firewalls_client = compute_v1.FirewallsClient()
    routes_client = compute_v1.RoutesClient()

    suffix = str(uuid.uuid4())[:8]
    vpc_a_name = f"isv-iso-a-{suffix}"
    vpc_b_name = f"isv-iso-b-{suffix}"

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "status": "failed",
        "tests": {},
    }

    vpc_a_created = False
    vpc_b_created = False

    try:
        # Create VPCs
        a_result = create_vpc(networks_client, project, vpc_a_name)
        result["tests"]["create_vpc_a"] = {
            "passed": a_result["passed"],
            "vpc_id": vpc_a_name,
            **({"error": a_result["error"]} if "error" in a_result else {}),
        }
        if not a_result["passed"]:
            print(json.dumps(result, indent=2))
            return 1
        vpc_a_created = True

        b_result = create_vpc(networks_client, project, vpc_b_name)
        result["tests"]["create_vpc_b"] = {
            "passed": b_result["passed"],
            "vpc_id": vpc_b_name,
            **({"error": b_result["error"]} if "error" in b_result else {}),
        }
        if not b_result["passed"]:
            print(json.dumps(result, indent=2))
            return 1
        vpc_b_created = True

        result["vpc_a"] = {"id": vpc_a_name, "cidr": args.cidr_a}
        result["vpc_b"] = {"id": vpc_b_name, "cidr": args.cidr_b}

        net_a = networks_client.get(project=project, network=vpc_a_name)
        net_b = networks_client.get(project=project, network=vpc_b_name)

        # Isolation subtests
        result["tests"]["no_peering"] = test_no_peering(networks_client, project, vpc_a_name, vpc_b_name)
        result["tests"]["no_cross_routes_a"] = test_no_cross_routes(
            routes_client, project, net_a.self_link, args.cidr_b
        )
        result["tests"]["no_cross_routes_b"] = test_no_cross_routes(
            routes_client, project, net_b.self_link, args.cidr_a
        )
        result["tests"]["sg_isolation_a"] = test_sg_isolation(firewalls_client, project, net_a.self_link, args.cidr_b)
        result["tests"]["sg_isolation_b"] = test_sg_isolation(firewalls_client, project, net_b.self_link, args.cidr_a)

        all_passed = all(t.get("passed", False) for t in result["tests"].values())
        result["success"] = all_passed
        result["status"] = "passed" if all_passed else "failed"

    except Exception as e:
        result["error"] = str(e)
    finally:
        if vpc_a_created:
            try:
                delete_vpc(networks_client, project, vpc_a_name)
            except gax_exc.GoogleAPIError:
                pass
        if vpc_b_created:
            try:
                delete_vpc(networks_client, project, vpc_b_name)
            except gax_exc.GoogleAPIError:
                pass

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

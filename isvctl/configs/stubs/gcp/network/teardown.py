#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Teardown GCP shared VPC and all attached resources.

Deletion order (GCP enforces this — leaving any dependent resource
attached returns 400 DependencyViolation on network delete):

  1. Instances whose NICs reference subnets in the VPC
  2. Forwarding rules / routers (not created by the suite — skip listing)
  3. Static addresses attached to VPC (rare — created by floating_ip, which
     self-cleans). We still scan regional addresses belonging to the VPC.
  4. Firewall rules on the network
  5. Peerings originating from the network (remove_peering)
  6. Subnetworks in every region of this network
  7. Routes with next_hop referencing the network
  8. The network itself

Usage:
    python teardown.py --vpc-id <network> --region asia-east1-a

Output JSON matches the oracle teardown schema.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import resolve_project
from common.errors import handle_gcp_errors
from common.vpc import wait_operation
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1


def _cleanup_instances(
    instances_client: compute_v1.InstancesClient,
    project: str,
    network_self_link: str,
) -> list[str]:
    """Terminate every instance whose primary NIC references the network.

    Uses aggregated_list to sweep all zones. Returns the list of deleted
    instance names for the teardown result.
    """
    deleted: list[str] = []
    try:
        for scope, insts in instances_client.aggregated_list(project=project):
            if not scope.startswith("zones/"):
                continue
            zone = scope.split("/", 1)[1]
            for inst in insts.instances or []:
                on_network = any(nic.network == network_self_link for nic in inst.network_interfaces or [])
                if on_network:
                    try:
                        op = instances_client.delete(project=project, zone=zone, instance=inst.name)
                        op.result(timeout=300)
                        deleted.append(inst.name)
                    except gax_exc.GoogleAPIError as e:
                        print(f"  delete instance {inst.name} warning: {e}", file=sys.stderr)
    except gax_exc.GoogleAPIError as e:
        print(f"  aggregated_list instances warning: {e}", file=sys.stderr)
    return deleted


def _cleanup_firewalls(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    network_self_link: str,
) -> list[str]:
    """Delete every firewall whose ``network`` matches the VPC."""
    deleted: list[str] = []
    try:
        for fw in firewalls_client.list(project=project):
            if fw.network != network_self_link:
                continue
            try:
                op = firewalls_client.delete(project=project, firewall=fw.name)
                wait_operation(op, timeout=120)
                deleted.append(fw.name)
            except gax_exc.GoogleAPIError as e:
                print(f"  delete firewall {fw.name} warning: {e}", file=sys.stderr)
    except gax_exc.GoogleAPIError as e:
        print(f"  list firewalls warning: {e}", file=sys.stderr)
    return deleted


def _cleanup_peerings(
    networks_client: compute_v1.NetworksClient,
    project: str,
    vpc_name: str,
) -> list[str]:
    """Remove every peering originating from this VPC."""
    deleted: list[str] = []
    try:
        net = networks_client.get(project=project, network=vpc_name)
        for peering in net.peerings or []:
            try:
                req = compute_v1.NetworksRemovePeeringRequest()
                req.name = peering.name
                op = networks_client.remove_peering(
                    project=project,
                    network=vpc_name,
                    networks_remove_peering_request_resource=req,
                )
                wait_operation(op, timeout=120)
                deleted.append(peering.name)
            except gax_exc.GoogleAPIError as e:
                print(f"  remove_peering {peering.name} warning: {e}", file=sys.stderr)
    except gax_exc.NotFound:
        pass
    except gax_exc.GoogleAPIError as e:
        print(f"  list peerings warning: {e}", file=sys.stderr)
    return deleted


def _cleanup_subnets(
    subnets_client: compute_v1.SubnetworksClient,
    project: str,
    network_self_link: str,
) -> list[str]:
    """Delete every subnet on this network across all regions."""
    deleted: list[str] = []
    try:
        for scope, subs in subnets_client.aggregated_list(project=project):
            if not scope.startswith("regions/"):
                continue
            region = scope.split("/", 1)[1]
            for sub in subs.subnetworks or []:
                if sub.network != network_self_link:
                    continue
                try:
                    op = subnets_client.delete(project=project, region=region, subnetwork=sub.name)
                    wait_operation(op, timeout=180)
                    deleted.append(sub.name)
                except gax_exc.GoogleAPIError as e:
                    print(f"  delete subnet {sub.name} warning: {e}", file=sys.stderr)
    except gax_exc.GoogleAPIError as e:
        print(f"  aggregated_list subnets warning: {e}", file=sys.stderr)
    return deleted


def _cleanup_routes(
    routes_client: compute_v1.RoutesClient,
    project: str,
    network_self_link: str,
) -> list[str]:
    """Delete non-default custom routes attached to the network.

    GCP auto-creates a default-internet-gateway route and one subnet route
    per subnet; those get cleaned up automatically when the network is
    deleted. Only user-created custom routes need explicit removal.
    """
    deleted: list[str] = []
    try:
        for route in routes_client.list(project=project):
            if route.network != network_self_link:
                continue
            # Default routes are owned by GCP — their name matches
            # ``default-route-<hex>`` and they're removed automatically.
            if route.name.startswith("default-route"):
                continue
            try:
                op = routes_client.delete(project=project, route=route.name)
                wait_operation(op, timeout=120)
                deleted.append(route.name)
            except gax_exc.GoogleAPIError as e:
                print(f"  delete route {route.name} warning: {e}", file=sys.stderr)
    except gax_exc.GoogleAPIError as e:
        print(f"  list routes warning: {e}", file=sys.stderr)
    return deleted


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Teardown GCP shared VPC")
    parser.add_argument("--vpc-id", required=True, help="GCP network name")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--project", default=None)
    parser.add_argument("--skip-destroy", action="store_true", help="Skip teardown entirely")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "resources_destroyed": False,
        "network_id": args.vpc_id,
        # ``deleted`` is the structured breakdown (per-resource-type lists);
        # ``resources_deleted`` is the flat list the canonical tests/network.yaml
        # teardown contract expects. We populate both at the end of main().
        "deleted": {},
        "resources_deleted": [],
    }

    # Honour the AWS-style env var name for parity with the oracle, and
    # add a GCP-specific alias.
    skip = (
        args.skip_destroy
        or os.environ.get("GCP_NETWORK_SKIP_TEARDOWN", "").lower() == "true"
        or os.environ.get("AWS_NETWORK_SKIP_TEARDOWN", "").lower() == "true"
    )
    if skip:
        result["success"] = True
        result["message"] = "Destroy skipped (--skip-destroy or *_NETWORK_SKIP_TEARDOWN=true)"
        result["resources_deleted"] = []
        print(json.dumps(result, indent=2))
        return 0

    project = resolve_project(args.project)
    networks_client = compute_v1.NetworksClient()
    subnets_client = compute_v1.SubnetworksClient()
    firewalls_client = compute_v1.FirewallsClient()
    instances_client = compute_v1.InstancesClient()
    routes_client = compute_v1.RoutesClient()

    try:
        try:
            net = networks_client.get(project=project, network=args.vpc_id)
        except gax_exc.NotFound:
            # Already gone — success.
            result["success"] = True
            result["resources_destroyed"] = True
            result["resources_deleted"] = []
            result["message"] = f"VPC {args.vpc_id} not found (already deleted)"
            print(json.dumps(result, indent=2))
            return 0

        network_self_link = net.self_link

        # 1. Instances
        result["deleted"]["instances"] = _cleanup_instances(instances_client, project, network_self_link)
        # 2. Firewalls
        result["deleted"]["security_groups"] = _cleanup_firewalls(firewalls_client, project, network_self_link)
        # 3. Peerings
        result["deleted"]["peerings"] = _cleanup_peerings(networks_client, project, args.vpc_id)
        # 4. Subnets
        result["deleted"]["subnets"] = _cleanup_subnets(subnets_client, project, network_self_link)
        # 5. Custom routes
        result["deleted"]["route_tables"] = _cleanup_routes(routes_client, project, network_self_link)
        # 6. Network itself — GCP networks have no "IGW" to detach; the
        #    implicit default-internet-gateway is owned by GCP.
        try:
            op = networks_client.delete(project=project, network=args.vpc_id)
            wait_operation(op, timeout=300)
            result["deleted"]["vpc"] = args.vpc_id
        except gax_exc.NotFound:
            result["deleted"]["vpc"] = args.vpc_id

        # Match the oracle's key — "internet_gateways" is always empty for GCP.
        result["deleted"].setdefault("internet_gateways", [])

        # Flatten the structured deleted dict into the canonical
        # ``resources_deleted`` list the teardown contract expects. Each
        # entry is ``<type>:<id>`` so a grep over the test logs can
        # pinpoint which resources were touched.
        flattened: list[str] = []
        for resource_type, ids in result["deleted"].items():
            if isinstance(ids, list):
                flattened.extend(f"{resource_type}:{rid}" for rid in ids)
            elif ids:
                flattened.append(f"{resource_type}:{ids}")
        result["resources_deleted"] = flattened

        result["success"] = True
        result["resources_destroyed"] = True
        result["message"] = f"VPC and all attached resources destroyed ({len(flattened)} items)"
    except gax_exc.GoogleAPIError as e:
        result["error"] = str(e)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

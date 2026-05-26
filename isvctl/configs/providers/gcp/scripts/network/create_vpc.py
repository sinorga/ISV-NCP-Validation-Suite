#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Create a Compute Engine custom-mode VPC + subnetworks + firewall.

Compute Engine divergences from the AWS oracle (per reviewed knowledge):

  * Networks have NO CIDR — CIDRs are per-subnetwork. The emitted ``cidr``
    is the create-time aggregate (the --cidr arg) used by VpcIpConfigCheck
    for subnet_of containment math.
  * Subnetworks are regional, not zonal — populate the contract's ``az``
    field from real zones in the configured region (RegionsClient.get).
  * No internet_gateway resource — implicit default-internet-gateway is
    pre-installed; omit ``internet_gateway_id`` and ``route_table_id``.
  * No DHCP options API — emit a synthesised dhcp_options object pointing
    at the metadata-server resolver at 169.254.169.254.
  * No per-VPC default SG — create a project-scoped Firewall bound to
    the test network with sourceRanges + at least one Allowed entry
    with I_p_protocol set. Emit Firewall.name as ``security_group_id``.
  * Firewall ``allowed[]`` is REJECTED with HTTP 400 unless I_p_protocol
    is set on every entry (factory override).
  * Network / Subnetwork / Firewall protos have NO ``labels`` field —
    use only ``description`` for provenance.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import resolve_project, unique_suffix, wait_for_global_op
from common.errors import handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest network domain — verified-reuse marker"
PROVENANCE_TAG = "isvtest"
# Compute Engine reserves 4 IPs at the bottom of every subnet (network,
# default gateway, second-to-last, broadcast). available_ips = size - 4.
_GCE_RESERVED_PER_SUBNET = 4


def _list_region_zones(project: str, region: str) -> list[str]:
    client = compute_v1.RegionsClient()
    region_obj = client.get(project=project, region=region)
    return [url.rsplit("/", 1)[-1] for url in region_obj.zones or ()]


def _wait_region_op(project: str, region: str, op_name: str, *, timeout: int = 300) -> None:
    client = compute_v1.RegionOperationsClient()
    client.wait(project=project, region=region, operation=op_name, timeout=timeout)


def _carve_subnet_cidrs(aggregate: str, count: int) -> list[str]:
    net = ipaddress.ip_network(aggregate, strict=False)
    # /24 subnets carved from the /16 aggregate.
    new_prefix = max(net.prefixlen + 8, 24)
    subs = list(net.subnets(new_prefix=new_prefix))
    return [str(s) for s in subs[:count]]


@handle_gcp_errors
def main() -> int:
    """Create-network is flattened into main() so the result dict is the
    single source of truth — every mutation (`network_created=True`,
    `subnets.append(...)`, `security_group_id=...`) happens on the outer
    `result` dict BEFORE the corresponding async wait. If a wait raises,
    the partial-state JSON still surfaces what was created so teardown's
    forwarded `--network-created` / `--vpc-id` args can drive cleanup.
    """
    parser = argparse.ArgumentParser(description="Create Compute Engine VPC + subnets + firewall")
    parser.add_argument("--name", default="isv-shared-vpc", help="Network name prefix")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr", default="10.0.0.0/16", help="Aggregate from which subnet CIDRs are carved")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    network_name = unique_suffix(args.name)
    firewall_name = unique_suffix(f"{args.name}-fw")

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "network_id": network_name,
        "cidr": args.cidr,
        "subnets": [],
        "security_group_id": None,
        "dhcp_options": None,
        "network_created": False,
        "region": args.region,
        "name": args.name,
    }

    networks = compute_v1.NetworksClient()
    subnets_client = compute_v1.SubnetworksClient()
    firewalls = compute_v1.FirewallsClient()

    try:
        # --- Custom-mode network --------------------------------------------
        # Stamp network_created BEFORE the wait — partial-state-recovery
        # contract: teardown reads `network_created` and `network_id` from
        # whatever JSON THIS run emits (including a partial-failure JSON).
        op = networks.insert(
            project=project,
            network_resource=compute_v1.Network(
                name=network_name,
                description=ISV_DESCRIPTION,
                auto_create_subnetworks=False,
                routing_config=compute_v1.NetworkRoutingConfig(routing_mode="REGIONAL"),
            ),
        )
        result["network_created"] = True
        wait_for_global_op(project, op.name, timeout=300)

        # --- Subnetworks (two minimum) --------------------------------------
        zones = _list_region_zones(project, args.region)
        if len(zones) < 2:
            raise RuntimeError(f"region {args.region} reports fewer than 2 zones: {zones}")

        for i, sub_cidr in enumerate(_carve_subnet_cidrs(args.cidr, 2)):
            sub_name = f"{network_name}-sub{i}"
            op = subnets_client.insert(
                project=project,
                region=args.region,
                subnetwork_resource=compute_v1.Subnetwork(
                    name=sub_name,
                    description=ISV_DESCRIPTION,
                    ip_cidr_range=sub_cidr,
                    network=f"projects/{project}/global/networks/{network_name}",
                    region=args.region,
                ),
            )
            # Append BEFORE the wait — same partial-state-recovery argument.
            net = ipaddress.ip_network(sub_cidr)
            result["subnets"].append(
                {
                    "subnet_id": sub_name,
                    "cidr": sub_cidr,
                    "az": zones[i % len(zones)],
                    "auto_assign_public_ip": False,
                    "available_ips": max(0, net.num_addresses - _GCE_RESERVED_PER_SUBNET),
                }
            )
            _wait_region_op(project, args.region, op.name, timeout=300)

        # --- Project-scoped firewall (SG analog) ----------------------------
        op = firewalls.insert(
            project=project,
            firewall_resource=compute_v1.Firewall(
                name=firewall_name,
                description=ISV_DESCRIPTION,
                network=f"projects/{project}/global/networks/{network_name}",
                direction="INGRESS",
                source_ranges=["0.0.0.0/0"],
                allowed=[
                    compute_v1.Allowed(I_p_protocol="tcp", ports=["22"]),
                    compute_v1.Allowed(I_p_protocol="icmp"),
                ],
                target_tags=[PROVENANCE_TAG],
            ),
        )
        # Stamp firewall id BEFORE the wait so teardown's enumerate-by-network
        # filter has the explicit name to lookup if the wait raises.
        result["security_group_id"] = firewall_name
        wait_for_global_op(project, op.name, timeout=300)

        # --- DHCP options (synthesised — no API) ----------------------------
        result["dhcp_options"] = {
            "dhcp_options_id": network_name,
            "domain_name": None,
            "domain_name_servers": ["169.254.169.254"],
            "ntp_servers": [],
        }
        result["success"] = True
    except gax.AlreadyExists as e:
        result["error_type"] = "api_error"
        result["error"] = f"resource already exists (verified-reuse not implemented for create_network): {e}"
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        et = "api_error" if isinstance(e, gax.GoogleAPICallError) else "unknown_error"
        result["error_type"] = et
        result["error"] = str(e)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

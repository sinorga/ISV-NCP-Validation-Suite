#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Create a GCP VPC (custom-mode network) with subnets for the shared test fixture.

Usage:
    python create_vpc.py --name isv-shared-vpc --region asia-east1-a --cidr 10.0.0.0/16

The --region flag is a zone (the test config passes {{region}} which for GCP
is a zone; see providers/gcp/vm.yaml). The subnet goes into the parent region.

Output JSON — mirrors the AWS oracle shape so downstream steps resolve
{{steps.create_network.*}} the same way:
{
    "success": true,
    "platform": "network",
    "network_id": "isv-shared-vpc",
    "cidr": "10.0.0.0/16",
    "subnets": [
        {"subnet_id": "<name>", "cidr": "10.0.1.0/24", "az": "asia-east1",
         "auto_assign_public_ip": false, "available_ips": 251}
    ],
    "internet_gateway_id": "default-internet-gateway",  # GCP has one per network
    "route_table_id": null,                             # GCP routes are VPC-level
    "security_group_id": "<firewall-name>",
    "dhcp_options": {
        "dhcp_options_id": "default",
        "domain_name": "<region>.c.<project>.internal",
        "domain_name_servers": ["169.254.169.254"],
        "ntp_servers": ["metadata.google.internal"]
    }
}
"""

import argparse
import ipaddress
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import resolve_project, zone_to_region
from common.errors import classify_gcp_error, handle_gcp_errors
from common.vpc import (
    build_firewall,
    create_subnet,
    create_vpc,
    default_dhcp_options,
    wait_operation,
)
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1

# GCP subnetworks get the first 4 addresses reserved (network, gateway, 2x GCP-reserved)
# plus the broadcast. So a /24 yields 256 - 4 = 252 usable. Match that tally here so
# VpcIpConfigCheck's available_ips threshold behaves the same as on AWS.
_GCP_RESERVED_PER_SUBNET = 4


def _derive_subnet_cidr(vpc_cidr: str, index: int) -> str:
    """Carve a /24 out of ``vpc_cidr`` using the third octet as ``index``.

    ``10.0.0.0/16`` + index 1 → ``10.0.1.0/24``. Matches the oracle's
    naming so BYOIP and traffic tests see the same layout.
    """
    net = ipaddress.ip_network(vpc_cidr, strict=False)
    base_octets = str(net.network_address).split(".")
    return f"{base_octets[0]}.{base_octets[1]}.{index}.0/24"


def _available_ips(subnet_cidr: str) -> int:
    net = ipaddress.ip_network(subnet_cidr, strict=False)
    return max(0, net.num_addresses - _GCP_RESERVED_PER_SUBNET - 1)


@handle_gcp_errors
def main() -> int:
    """Provision the shared VPC used by downstream network tests."""
    parser = argparse.ArgumentParser(description="Create VPC for GCP testing")
    parser.add_argument("--name", default="isv-shared-vpc", help="VPC name (must be DNS-1123)")
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_ZONE", "asia-east1-a"),
        help="GCP zone; parent region is used for the subnet",
    )
    parser.add_argument("--cidr", default="10.0.0.0/16", help="VPC CIDR — used to carve subnet ranges")
    parser.add_argument("--project", default=None, help="GCP project ID")
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = args.region
    region = zone_to_region(zone)

    networks_client = compute_v1.NetworksClient()
    subnets_client = compute_v1.SubnetworksClient()
    firewalls_client = compute_v1.FirewallsClient()

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "network_id": args.name,
        "name": args.name,
        "cidr": args.cidr,
        "region": zone,
        "gcp_region": region,
        "project": project,
        "subnets": [],
        # GCP has one implicit default-internet-gateway route per VPC; expose a
        # stable identifier so provider tooling has something to reference.
        "internet_gateway_id": "default-internet-gateway",
        # GCP has no per-subnet route tables — routes are VPC-level. We surface
        # null rather than a fake ID so downstream checks don't pattern-match on
        # an AWS-style rtb-* value.
        "route_table_id": None,
        "security_group_id": None,
        "dhcp_options": default_dhcp_options(project, region),
    }

    firewall_name = f"{args.name}-fw"

    try:
        # ── VPC ──────────────────────────────────────────────────────────
        vpc_result = create_vpc(networks_client, project, args.name)
        if not vpc_result["passed"]:
            result["error"] = vpc_result.get("error", "VPC creation failed")
            print(json.dumps(result, indent=2))
            return 1
        net = networks_client.get(project=project, network=args.name)
        network_self_link = net.self_link
        result["network_self_link"] = network_self_link

        # ── Subnets ──────────────────────────────────────────────────────
        # AWS creates two subnets in different AZs. GCP subnets are regional,
        # so "multi-AZ" is satisfied by creating two subnets in the same
        # region with distinct CIDRs — or by spanning two regions. For the
        # shared fixture we stay in one region to keep cross-subnet latency
        # predictable; NetworkProvisionedCheck only requires >= min_subnets.
        subnet_specs = [
            (f"{args.name}-sn-1", _derive_subnet_cidr(args.cidr, 1)),
            (f"{args.name}-sn-2", _derive_subnet_cidr(args.cidr, 2)),
        ]
        for subnet_name, subnet_cidr in subnet_specs:
            sn = create_subnet(
                subnets_client,
                project,
                region,
                subnet_name,
                network_self_link,
                subnet_cidr,
            )
            if not sn["passed"]:
                raise RuntimeError(f"Subnet {subnet_name} creation failed: {sn.get('error')}")
            result["subnets"].append(
                {
                    "subnet_id": subnet_name,
                    "cidr": subnet_cidr,
                    # az holds the region (GCP subnets are regional, see vpc.py).
                    "az": region,
                    # GCP assigns external IPs per-instance via accessConfig,
                    # not at the subnet level. Honestly reporting False keeps
                    # VpcIpConfigCheck's subnet mode consistent; the provider
                    # config overrides auto_assign_ip_mode to "instance".
                    "auto_assign_public_ip": False,
                    "available_ips": _available_ips(subnet_cidr),
                }
            )

        # ── Firewall (GCP equivalent of AWS security group) ──────────────
        # SSH + ICMP from anywhere (for SSH-based DhcpIpManagementCheck) and
        # an intra-VPC allow covering everything so the shared VPC can host
        # the DHCP IP test and connectivity_test instances without needing
        # per-test firewall edits.
        fw = build_firewall(
            name=firewall_name,
            network_self_link=network_self_link,
            allowed=[("tcp", ["22"]), ("icmp", None)],
            source_ranges=["0.0.0.0/0"],
            description="ISV shared VPC SSH + ICMP ingress",
        )
        try:
            op = firewalls_client.insert(project=project, firewall_resource=fw)
            wait_operation(op, timeout=120)
        except gax_exc.Conflict:
            print(f"  Firewall {firewall_name} already exists — reusing", file=sys.stderr)
        result["security_group_id"] = firewall_name

        # Intra-VPC allow-all so downstream tests can ping between instances
        # in the shared VPC without opening a second rule each time.
        intra_name = f"{args.name}-fw-intra"
        intra_fw = build_firewall(
            name=intra_name,
            network_self_link=network_self_link,
            allowed=[("tcp", None), ("udp", None), ("icmp", None)],
            source_ranges=[args.cidr],
            description="ISV shared VPC intra-VPC allow",
        )
        try:
            op = firewalls_client.insert(project=project, firewall_resource=intra_fw)
            wait_operation(op, timeout=120)
        except gax_exc.Conflict:
            print(f"  Firewall {intra_name} already exists — reusing", file=sys.stderr)

        result["success"] = True

    except gax_exc.GoogleAPIError as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    except Exception as e:
        result["error"] = str(e)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

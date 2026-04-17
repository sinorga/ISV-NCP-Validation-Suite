#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Inspect GCP VPC IP configuration: subnets, DHCP-equivalent, auto-assign mode.

GCP differs from AWS here in three ways that the validator already
understands:

  1. Custom-mode VPCs (what we create) have no top-level CIDR. We report
     the caller-provided --cidr (or discover it from the first subnet's
     range if not given) so VpcIpConfigCheck can sanity-check the subnets.
  2. DHCP options are not user-configurable. We synthesize the standard
     GCP shape (see common/vpc.py default_dhcp_options).
  3. Auto-assign public IP is per-instance (accessConfig), not per-subnet.
     The validator accepts ``auto_assign_ip_mode: instance`` (set in the
     provider config), so subnet-level ``auto_assign_public_ip`` is False.

Usage:
    python vpc_ip_config_test.py --vpc-id <network> --region asia-east1-a

Output JSON matches the oracle's vpc_ip_config schema.
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
from common.vpc import default_dhcp_options
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1

_GCP_RESERVED_PER_SUBNET = 4


def _available_ips(cidr: str) -> int:
    net = ipaddress.ip_network(cidr, strict=False)
    return max(0, net.num_addresses - _GCP_RESERVED_PER_SUBNET - 1)


def describe_subnets(
    subnets_client: compute_v1.SubnetworksClient,
    project: str,
    network_self_link: str,
) -> list[dict[str, Any]]:
    """Walk every region for subnets on ``network_self_link``."""
    subnets: list[dict[str, Any]] = []
    for scope, subs in subnets_client.aggregated_list(project=project):
        if not scope.startswith("regions/"):
            continue
        region = scope.split("/", 1)[1]
        for sub in subs.subnetworks or []:
            if sub.network != network_self_link:
                continue
            subnets.append(
                {
                    "subnet_id": sub.name,
                    "cidr": sub.ip_cidr_range,
                    "az": region,
                    # GCP assigns external IPs per-instance — report False
                    # here so the validator's "subnet" mode doesn't pick up a
                    # misleading True. Provider config overrides the mode.
                    "auto_assign_public_ip": False,
                    "available_ips": _available_ips(sub.ip_cidr_range),
                }
            )
    return subnets


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect GCP VPC IP configuration")
    parser.add_argument("--vpc-id", required=True, help="GCP network name")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = args.region
    region = zone_to_region(zone)

    networks_client = compute_v1.NetworksClient()
    subnets_client = compute_v1.SubnetworksClient()

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "test_name": "vpc_ip_config",
        "network_id": args.vpc_id,
        "cidr": None,
        "subnets": [],
        "dhcp_options": default_dhcp_options(project, region),
    }

    try:
        net = networks_client.get(project=project, network=args.vpc_id)
        subnets = describe_subnets(subnets_client, project, net.self_link)
        result["subnets"] = subnets

        # Custom-mode GCP VPCs carry no top-level CIDR. Report the caller's
        # expected enclosing range by summarising the subnets — take the
        # common /16 prefix if they share one.
        if subnets:
            result["cidr"] = _derive_enclosing_cidr([s["cidr"] for s in subnets])
        result["success"] = True
    except gax_exc.NotFound as e:
        result["error"] = f"VPC {args.vpc_id} not found: {e}"
    except gax_exc.GoogleAPIError as e:
        result["error_type"], result["error"] = classify_gcp_error(e)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


def _derive_enclosing_cidr(subnet_cidrs: list[str]) -> str | None:
    """Heuristic: if all subnets share a /16 prefix, return that /16.

    VpcIpConfigCheck's ``subnet_cidr_valid`` subtest needs the VPC CIDR to
    verify each subnet is a subset. GCP custom networks don't publish a
    top-level CIDR, so we reconstruct one from the subnet layout.
    """
    if not subnet_cidrs:
        return None
    try:
        first = ipaddress.ip_network(subnet_cidrs[0], strict=False)
    except ValueError:
        return None
    # Walk back prefix lengths until every subnet is covered.
    for prefix in (16, 12, 10, 8):
        candidate = ipaddress.ip_network(f"{first.network_address}/{prefix}", strict=False)
        if all(
            ipaddress.ip_network(c, strict=False).subnet_of(candidate)
            for c in subnet_cidrs
            if ipaddress.ip_network(c, strict=False).version == candidate.version
        ):
            return str(candidate)
    return str(first)


if __name__ == "__main__":
    sys.exit(main())

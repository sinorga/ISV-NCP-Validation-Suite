#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test GCP BYOIP (Bring-Your-Own-IP) via custom subnet CIDR ranges.

GCP custom-mode networks don't carry a top-level CIDR; the CIDR lives on
the subnet. So "BYOIP" here means creating a subnet with a non-standard
CIDR (e.g. ``100.64.0.0/16`` — RFC 6598 carrier-grade NAT range) and
confirming it works alongside a standard-CIDR subnet in a second VPC.

Subtests (match oracle schema):
  - custom_cidr_create   : VPC + subnet in the custom CIDR
  - custom_cidr_verify   : re-read the subnet and confirm ip_cidr_range matches
  - standard_cidr_create : second VPC + subnet in a standard CIDR
  - no_conflict          : both CIDRs differ and both subnets are READY
  - custom_cidr_subnet   : the subnet in the custom CIDR exists + is READY

Usage:
    python byoip_test.py --region asia-east1-a \\
        --custom-cidr 100.64.0.0/16 --standard-cidr 10.90.0.0/16

Output JSON matches the oracle byoip schema.
"""

import argparse
import ipaddress
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import resolve_project, zone_to_region
from common.errors import handle_gcp_errors
from common.vpc import (
    create_subnet,
    create_vpc,
    delete_subnet,
    delete_vpc,
)
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1


def _first_slash24(cidr: str) -> str:
    """Carve the first /24 out of ``cidr`` (works for /16 or larger)."""
    net = ipaddress.ip_network(cidr, strict=False)
    base = str(net.network_address).split(".")
    return f"{base[0]}.{base[1]}.1.0/24"


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Test GCP BYOIP")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--custom-cidr", default="100.64.0.0/16")
    parser.add_argument("--standard-cidr", default="10.90.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    region = zone_to_region(args.region)

    networks_client = compute_v1.NetworksClient()
    subnets_client = compute_v1.SubnetworksClient()

    suffix = str(uuid.uuid4())[:8]
    custom_vpc_name = f"isv-byoip-custom-{suffix}"
    standard_vpc_name = f"isv-byoip-standard-{suffix}"
    custom_sub_name = f"isv-byoip-custom-sn-{suffix}"
    standard_sub_name = f"isv-byoip-standard-sn-{suffix}"

    custom_subnet_cidr = _first_slash24(args.custom_cidr)
    standard_subnet_cidr = _first_slash24(args.standard_cidr)

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "status": "failed",
        "tests": {},
    }

    created = {
        "custom_vpc": False,
        "standard_vpc": False,
        "custom_sub": False,
        "standard_sub": False,
    }

    try:
        # ── custom_cidr_create: VPC + subnet in the custom CIDR ────────
        vpc_c = create_vpc(networks_client, project, custom_vpc_name)
        if not vpc_c["passed"]:
            result["tests"]["custom_cidr_create"] = {
                "passed": False,
                "error": vpc_c.get("error", "VPC create failed"),
            }
            print(json.dumps(result, indent=2))
            return 1
        created["custom_vpc"] = True
        net_c = networks_client.get(project=project, network=custom_vpc_name)

        sub_c = create_subnet(
            subnets_client,
            project,
            region,
            custom_sub_name,
            net_c.self_link,
            custom_subnet_cidr,
        )
        if sub_c["passed"]:
            created["custom_sub"] = True
        result["tests"]["custom_cidr_create"] = {
            "passed": sub_c["passed"],
            "vpc_id": custom_vpc_name,
            "cidr": custom_subnet_cidr,
            "subnet_id": custom_sub_name,
            **({"error": sub_c["error"]} if "error" in sub_c else {}),
        }

        # ── custom_cidr_verify: re-read and match ──────────────────────
        verify: dict[str, Any] = {"passed": False}
        try:
            sn = subnets_client.get(project=project, region=region, subnetwork=custom_sub_name)
            if sn.ip_cidr_range == custom_subnet_cidr:
                verify["passed"] = True
                verify["cidr"] = sn.ip_cidr_range
                verify["state"] = sn.state or "READY"
                verify["message"] = f"Subnet {custom_sub_name} has CIDR {sn.ip_cidr_range}"
            else:
                verify["error"] = f"Expected {custom_subnet_cidr}, got {sn.ip_cidr_range}"
        except gax_exc.NotFound as e:
            verify["error"] = f"Subnet {custom_sub_name} not found: {e}"
        result["tests"]["custom_cidr_verify"] = verify

        # ── standard_cidr_create: second VPC + subnet ──────────────────
        vpc_s = create_vpc(networks_client, project, standard_vpc_name)
        if vpc_s["passed"]:
            created["standard_vpc"] = True
            net_s = networks_client.get(project=project, network=standard_vpc_name)
            sub_s = create_subnet(
                subnets_client,
                project,
                region,
                standard_sub_name,
                net_s.self_link,
                standard_subnet_cidr,
            )
            if sub_s["passed"]:
                created["standard_sub"] = True
            result["tests"]["standard_cidr_create"] = {
                "passed": sub_s["passed"],
                "vpc_id": standard_vpc_name,
                "cidr": standard_subnet_cidr,
                "subnet_id": standard_sub_name,
                **({"error": sub_s["error"]} if "error" in sub_s else {}),
            }
        else:
            result["tests"]["standard_cidr_create"] = {
                "passed": False,
                "error": vpc_s.get("error", "standard VPC create failed"),
            }

        # ── no_conflict: CIDRs differ ──────────────────────────────────
        no_conflict: dict[str, Any] = {"passed": False}
        if custom_subnet_cidr != standard_subnet_cidr:
            no_conflict["passed"] = True
            no_conflict["cidr_a"] = custom_subnet_cidr
            no_conflict["cidr_b"] = standard_subnet_cidr
            no_conflict["message"] = f"No conflict: {custom_subnet_cidr} and {standard_subnet_cidr}"
        else:
            no_conflict["error"] = f"Both CIDRs are {custom_subnet_cidr}"
        result["tests"]["no_conflict"] = no_conflict

        # ── custom_cidr_subnet: the custom subnet is READY ─────────────
        cust_sub: dict[str, Any] = {"passed": False}
        try:
            sn = subnets_client.get(project=project, region=region, subnetwork=custom_sub_name)
            state = sn.state or "READY"
            if state == "READY":
                cust_sub["passed"] = True
                cust_sub["subnet_id"] = custom_sub_name
                cust_sub["subnet_cidr"] = sn.ip_cidr_range
                cust_sub["message"] = f"Subnet {custom_sub_name} READY in {region}"
            else:
                cust_sub["error"] = f"Subnet state={state}"
        except gax_exc.NotFound as e:
            cust_sub["error"] = f"Subnet missing: {e}"
        result["tests"]["custom_cidr_subnet"] = cust_sub

        all_passed = all(t.get("passed", False) for t in result["tests"].values())
        result["success"] = all_passed
        result["status"] = "passed" if all_passed else "failed"

    except Exception as e:
        result["error"] = str(e)
    finally:
        # Cleanup in dependency order.
        if created["custom_sub"]:
            delete_subnet(subnets_client, project, region, custom_sub_name)
        if created["standard_sub"]:
            delete_subnet(subnets_client, project, region, standard_sub_name)
        if created["custom_vpc"]:
            delete_vpc(networks_client, project, custom_vpc_name)
        if created["standard_vpc"]:
            delete_vpc(networks_client, project, standard_vpc_name)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

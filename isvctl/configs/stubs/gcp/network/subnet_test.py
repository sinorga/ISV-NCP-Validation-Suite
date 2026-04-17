#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test GCP subnet configuration across multiple regions (GCP's "AZ" equivalent).

GCP subnetworks are REGIONAL — a single subnet spans every zone in its
region, so the AWS idea of "distribute across availability zones" maps
to "distribute across regions". We create N subnets split between two
regions from docs/gcp.yaml (asia-east1 and us-central1) and report each
region as the subnet's ``az`` so the SubnetConfigCheck multi-AZ test
passes with az_count >= 2.

Per docs/gcp.yaml: do NOT include a ``route_table_exists`` subtest —
routes in GCP are VPC-level, not per-subnet. Omitting the key makes
SubnetConfigCheck pass it by default.

Usage:
    python subnet_test.py --region asia-east1-a --cidr 10.98.0.0/16 --subnet-count 4

Output JSON (matches oracle schema; ``route_table_exists`` intentionally absent):
{
    "success": true,
    "platform": "network",
    "tests": {
        "create_vpc": {"passed": true},
        "create_subnets": {"passed": true, "count": 4},
        "az_distribution": {"passed": true, "azs": ["asia-east1", "us-central1"]},
        "subnets_available": {"passed": true}
    },
    "subnets": [...]
}
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

# Regions used for multi-AZ coverage. asia-east1 matches the VM test's
# preferred zone; us-central1 is a geographically distinct GCP region
# with high capacity, giving real multi-region spread.
_MULTI_REGION_POOL = ["asia-east1", "us-central1"]


def _derive_subnet_cidr(vpc_cidr: str, index: int) -> str:
    net = ipaddress.ip_network(vpc_cidr, strict=False)
    base_octets = str(net.network_address).split(".")
    return f"{base_octets[0]}.{base_octets[1]}.{index + 1}.0/24"


def test_create_subnets(
    subnets_client: compute_v1.SubnetworksClient,
    project: str,
    vpc_cidr: str,
    network_self_link: str,
    regions: list[str],
    count: int,
    suffix: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Create N subnetworks spanning ``regions`` round-robin."""
    result: dict[str, Any] = {"passed": False, "count": 0}
    subnets: list[dict[str, str]] = []

    try:
        for i in range(count):
            region = regions[i % len(regions)]
            subnet_cidr = _derive_subnet_cidr(vpc_cidr, i)
            name = f"isv-subnet-{suffix}-{i}"
            sn = create_subnet(
                subnets_client,
                project,
                region,
                name,
                network_self_link,
                subnet_cidr,
            )
            if not sn["passed"]:
                raise RuntimeError(f"Subnet {name} failed: {sn.get('error')}")
            subnets.append(
                {
                    "subnet_id": name,
                    "cidr": subnet_cidr,
                    "az": region,
                    "region": region,
                }
            )

        result["count"] = len(subnets)
        result["passed"] = result["count"] == count
        result["message"] = f"Created {result['count']} subnets across {len(set(s['region'] for s in subnets))} regions"
    except Exception as e:
        result["error"] = str(e)

    return result, subnets


def test_az_distribution(subnets: list[dict[str, str]], min_azs: int = 2) -> dict[str, Any]:
    """Confirm subnets span at least min_azs regions (AZ-equivalent)."""
    result: dict[str, Any] = {"passed": False}
    azs = sorted({s["az"] for s in subnets})
    result["azs"] = azs
    result["az_count"] = len(azs)
    if len(azs) >= min_azs:
        result["passed"] = True
        result["message"] = f"Subnets span {len(azs)} regions: {', '.join(azs)}"
    else:
        result["error"] = f"Only {len(azs)} regions used, need {min_azs}"
    return result


def test_subnets_available(
    subnets_client: compute_v1.SubnetworksClient,
    project: str,
    subnets: list[dict[str, str]],
) -> dict[str, Any]:
    """Re-read each subnet to confirm it's READY (GCP's 'available')."""
    result: dict[str, Any] = {"passed": False, "states": {}}
    try:
        all_ready = True
        for sn in subnets:
            obj = subnets_client.get(project=project, region=sn["region"], subnetwork=sn["subnet_id"])
            # GCP Subnetwork.state is "READY" for usable subnets (or empty
            # on immediate describe — treat empty as ready since insert
            # already awaited the op).
            state = obj.state or "READY"
            result["states"][sn["subnet_id"]] = state
            if state not in ("READY",):
                all_ready = False
        result["passed"] = all_ready
        result["message"] = f"All {len(subnets)} subnets READY" if all_ready else "One or more subnets not READY"
    except Exception as e:
        result["error"] = str(e)
    return result


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Test GCP subnet configuration")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--cidr", default="10.98.0.0/16")
    parser.add_argument("--subnet-count", type=int, default=4)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    # Use the caller's region as the primary and pick a second region from
    # the pool to guarantee multi-region coverage. If the caller's region
    # isn't in the pool we still span two regions.
    primary_region = zone_to_region(args.region)
    secondary = next((r for r in _MULTI_REGION_POOL if r != primary_region), _MULTI_REGION_POOL[1])
    regions = [primary_region, secondary]

    networks_client = compute_v1.NetworksClient()
    subnets_client = compute_v1.SubnetworksClient()

    suffix = str(uuid.uuid4())[:8]
    vpc_name = f"isv-subnet-test-{suffix}"

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "status": "failed",
        "tests": {},
        "subnets": [],
    }

    vpc_created = False
    subnets_made: list[dict[str, str]] = []

    try:
        # Test 1: Create VPC
        vpc_result = create_vpc(networks_client, project, vpc_name)
        result["tests"]["create_vpc"] = {
            "passed": vpc_result["passed"],
            "vpc_id": vpc_name,
            "cidr": args.cidr,
            **({"error": vpc_result["error"]} if "error" in vpc_result else {}),
        }
        if not vpc_result["passed"]:
            print(json.dumps(result, indent=2))
            return 1
        vpc_created = True
        result["network_id"] = vpc_name

        net = networks_client.get(project=project, network=vpc_name)

        # Test 2: Create subnets
        subnets_result, subnets_made = test_create_subnets(
            subnets_client,
            project,
            args.cidr,
            net.self_link,
            regions,
            args.subnet_count,
            suffix,
        )
        result["tests"]["create_subnets"] = subnets_result
        result["subnets"] = subnets_made

        if subnets_result["passed"]:
            # Test 3: AZ distribution
            result["tests"]["az_distribution"] = test_az_distribution(subnets_made)

            # Test 4: Subnets available
            result["tests"]["subnets_available"] = test_subnets_available(subnets_client, project, subnets_made)

        all_passed = all(t.get("passed", False) for t in result["tests"].values())
        result["success"] = all_passed
        result["status"] = "passed" if all_passed else "failed"

    except Exception as e:
        result["error"] = str(e)
    finally:
        # Cleanup: subnets first, then VPC.
        for sn in subnets_made:
            try:
                delete_subnet(subnets_client, project, sn["region"], sn["subnet_id"])
            except gax_exc.GoogleAPIError:
                pass
        if vpc_created:
            delete_vpc(networks_client, project, vpc_name)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

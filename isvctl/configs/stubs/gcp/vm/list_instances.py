#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""List GCP Compute Engine instances attached to a VPC network.

GCP's Compute API is zone-scoped. The canonical test config passes
``--region`` with a zone value (per the provider config), so the list is
primarily a zone scan filtered to the caller's VPC. For instances in
other zones on the same network, we fall back to the aggregated list.

Usage:
    python list_instances.py --vpc-id default --region asia-east1-a \
        --instance-id isv-test-gpu

Output JSON (per docs/existing-patterns/vm.md):
{
    "success": true,
    "platform": "vm",
    "instances": [{"instance_id": "...", "state": "running", ...}],
    "total_count": 1,
    "found_target": true,
    "target_instance": "isv-test-gpu"
}
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import (
    canonical_state,
    get_instance_external_ip,
    get_instance_internal_ip,
    resolve_project,
)
from common.errors import handle_gcp_errors
from google.cloud import compute_v1


def _summarise(instance: Any) -> dict[str, Any]:
    network = ""
    if instance.network_interfaces:
        network = instance.network_interfaces[0].network.rsplit("/", 1)[-1]
    return {
        "instance_id": instance.name,
        "instance_type": instance.machine_type.rsplit("/", 1)[-1],
        "state": canonical_state(instance.status),
        "gcp_status": instance.status,
        "public_ip": get_instance_external_ip(instance),
        "private_ip": get_instance_internal_ip(instance),
        "vpc_id": network,
        "zone": instance.zone.rsplit("/", 1)[-1] if instance.zone else "",
    }


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="List Compute Engine instances on a VPC")
    parser.add_argument("--vpc-id", required=True, help="GCP network name (e.g. 'default')")
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_ZONE", "asia-east1-a"),
        help="Primary zone to list (region setting is a zone per the provider config)",
    )
    parser.add_argument("--instance-id", help="Target instance name to verify exists in list")
    parser.add_argument("--project", help="GCP project ID (default: $GOOGLE_CLOUD_PROJECT)")
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = args.region

    client = compute_v1.InstancesClient()

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "instances": [],
    }

    try:
        seen: set[str] = set()

        # ===== Primary: zone list =====
        # GCP removes deleted VMs immediately (no lingering "terminated"
        # entry the way EC2 does), so filtering is by VPC only.
        for inst in client.list(project=project, zone=zone):
            summary = _summarise(inst)
            if summary["vpc_id"] != args.vpc_id:
                continue
            if inst.name in seen:
                continue
            seen.add(inst.name)
            result["instances"].append(summary)

        # ===== Fallback: aggregated list =====
        # If the target instance isn't in our zone, scan every zone.
        # This also covers multi-zone deployments on the same VPC.
        if args.instance_id and not any(i["instance_id"] == args.instance_id for i in result["instances"]):
            agg_request = compute_v1.AggregatedListInstancesRequest(project=project)
            for _scope, scoped_list in client.aggregated_list(request=agg_request):
                for inst in scoped_list.instances or []:
                    summary = _summarise(inst)
                    if summary["vpc_id"] != args.vpc_id:
                        continue
                    if inst.name in seen:
                        continue
                    seen.add(inst.name)
                    result["instances"].append(summary)

        result["count"] = len(result["instances"])
        # The test config inspects ``total_count``; keep ``count`` for parity with the oracle.
        result["total_count"] = result["count"]

        if args.instance_id:
            result["target_instance"] = args.instance_id
            result["found_target"] = any(i["instance_id"] == args.instance_id for i in result["instances"])

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

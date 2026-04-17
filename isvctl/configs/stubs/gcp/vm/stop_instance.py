#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Stop a Compute Engine instance and wait for it to reach TERMINATED.

GCP's analogue of AWS's ``stop_instances`` is ``InstancesClient.stop``.
A stopped GCP VM transitions to status ``TERMINATED`` (not ``STOPPED``),
which maps back to the canonical ``stopped`` state via
``canonical_state()``. See docs/gcp.yaml.

Usage:
    python stop_instance.py --instance-id isv-test-gpu --region asia-east1-a

Output JSON (per docs/existing-patterns/vm.md):
{
    "success": true,
    "platform": "vm",
    "instance_id": "...",
    "state": "stopped",
    "stop_initiated": true
}
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import canonical_state, describe_instance, resolve_project
from common.errors import handle_gcp_errors
from google.cloud import compute_v1


@handle_gcp_errors
def main() -> int:
    """Stop a Compute Engine instance and wait for TERMINATED."""
    parser = argparse.ArgumentParser(description="Stop Compute Engine instance")
    parser.add_argument("--instance-id", required=True, help="Instance name")
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_ZONE", "asia-east1-a"),
        help="GCP zone (provider config 'region' is a zone)",
    )
    parser.add_argument("--project", help="GCP project ID (default: $GOOGLE_CLOUD_PROJECT)")
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = args.region

    client = compute_v1.InstancesClient()

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "instance_id": args.instance_id,
        "region": zone,
        "stop_initiated": False,
    }

    try:
        # ============================================================
        # Step 1: Verify current state
        # ============================================================
        print("Checking instance state before stop...", file=sys.stderr)
        instance = describe_instance(client, project, zone, args.instance_id)
        gcp_status = instance.status
        current_state = canonical_state(gcp_status)

        if gcp_status == "TERMINATED":
            # Already stopped — idempotent no-op (matches oracle behaviour)
            result["state"] = current_state
            result["gcp_status"] = gcp_status
            result["stop_initiated"] = True
            result["success"] = True
            print(f"  Instance {args.instance_id} already TERMINATED (no-op)", file=sys.stderr)
            print(json.dumps(result, indent=2))
            return 0

        if current_state != "running":
            result["error"] = f"Instance is {gcp_status}, expected RUNNING"
            result["state"] = current_state
            result["gcp_status"] = gcp_status
            print(json.dumps(result, indent=2))
            return 1

        # ============================================================
        # Step 2: Initiate stop
        # ============================================================
        print(f"Stopping instance {args.instance_id}...", file=sys.stderr)
        op = client.stop(project=project, zone=zone, instance=args.instance_id)
        result["stop_initiated"] = True
        print("  Stop API call succeeded", file=sys.stderr)

        # ============================================================
        # Step 3: Wait for TERMINATED
        # ============================================================
        # ``op.result`` blocks on the zone operation; generous timeout
        # per docs/gcp.yaml (stop typically 30-60 s, sometimes longer).
        op.result(timeout=540)

        # ============================================================
        # Step 4: Refresh state
        # ============================================================
        instance = describe_instance(client, project, zone, args.instance_id)
        result["gcp_status"] = instance.status
        result["state"] = canonical_state(instance.status)
        result["success"] = result["state"] == "stopped"
        if not result["success"]:
            result["error"] = f"Instance did not reach TERMINATED (status={instance.status})"
        else:
            print("Stop completed successfully!", file=sys.stderr)

    except Exception as e:
        result["error"] = str(e)
        print(f"ERROR: {e}", file=sys.stderr)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

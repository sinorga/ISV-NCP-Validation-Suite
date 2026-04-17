#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Start a stopped (TERMINATED) Compute Engine instance.

GCP re-allocates an ephemeral external IP on every start, so ``public_ip``
in the output reflects the post-start address (often different from the
launch-time IP). Downstream steps that reference
``steps.start_instance.public_ip`` pick up the new address automatically.

Usage:
    python start_instance.py --instance-id isv-test-gpu --region asia-east1-a \\
        --key-file /tmp/isv-test-key.pem --public-ip <launch-time-ip>

Output JSON (per docs/existing-patterns/vm.md):
{
    "success": true,
    "platform": "vm",
    "instance_id": "...",
    "state": "running",
    "public_ip": "...",
    "key_file": "...",
    "start_initiated": true,
    "ssh_ready": true
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
    describe_instance,
    get_instance_external_ip,
    get_instance_internal_ip,
    resolve_project,
    wait_for_ssh,
)
from common.errors import handle_gcp_errors
from google.cloud import compute_v1


@handle_gcp_errors
def main() -> int:
    """Start a TERMINATED Compute Engine instance and wait for SSH."""
    parser = argparse.ArgumentParser(description="Start stopped Compute Engine instance")
    parser.add_argument("--instance-id", required=True, help="Instance name")
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_ZONE", "asia-east1-a"),
        help="GCP zone (provider config 'region' is a zone)",
    )
    parser.add_argument("--key-file", required=True, help="Path to SSH private key")
    parser.add_argument("--public-ip", required=True, help="Pre-stop external IP (may change on start)")
    parser.add_argument("--ssh-user", default="ubuntu", help="SSH username")
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
        "key_file": args.key_file,
        "ssh_user": args.ssh_user,
        "start_initiated": False,
        "ssh_ready": False,
    }

    try:
        # ============================================================
        # Step 1: Verify instance is TERMINATED
        # ============================================================
        print("Verifying instance is TERMINATED before start...", file=sys.stderr)
        instance = describe_instance(client, project, zone, args.instance_id)
        gcp_status = instance.status
        current_state = canonical_state(gcp_status)

        if gcp_status != "TERMINATED":
            result["error"] = f"Instance is {gcp_status}, expected TERMINATED"
            result["state"] = current_state
            result["gcp_status"] = gcp_status
            print(json.dumps(result, indent=2))
            return 1

        # ============================================================
        # Step 2: Initiate start
        # ============================================================
        print(f"Starting instance {args.instance_id}...", file=sys.stderr)
        op = client.start(project=project, zone=zone, instance=args.instance_id)
        result["start_initiated"] = True
        print("  Start API call succeeded", file=sys.stderr)

        # ============================================================
        # Step 3: Wait for the zone operation to finish
        # ============================================================
        op.result(timeout=540)

        # ============================================================
        # Step 4: Describe and extract fresh IPs
        # ============================================================
        instance = describe_instance(client, project, zone, args.instance_id)
        result["gcp_status"] = instance.status
        result["state"] = canonical_state(instance.status)
        new_ip = get_instance_external_ip(instance) or args.public_ip
        result["public_ip"] = new_ip
        result["private_ip"] = get_instance_internal_ip(instance)

        if not new_ip:
            result["error"] = "Instance has no external IP after start"
            print(json.dumps(result, indent=2))
            return 1

        # ============================================================
        # Step 5: Wait for SSH
        # ============================================================
        print("Waiting for SSH to be ready...", file=sys.stderr)
        ssh_ready = wait_for_ssh(
            new_ip,
            args.ssh_user,
            args.key_file,
            max_attempts=30,
            interval=10,
        )
        result["ssh_ready"] = ssh_ready

        if not ssh_ready:
            result["error"] = "SSH not ready after start"
            print("WARNING: SSH did not become ready after start", file=sys.stderr)
            print(json.dumps(result, indent=2))
            return 1

        result["success"] = result["state"] == "running"
        if result["success"]:
            print("Start completed successfully!", file=sys.stderr)
        else:
            result["error"] = f"Instance state is {instance.status}, expected RUNNING"

    except Exception as e:
        result["error"] = str(e)
        print(f"ERROR: {e}", file=sys.stderr)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

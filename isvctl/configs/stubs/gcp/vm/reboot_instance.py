#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Reboot a Compute Engine instance and validate it comes back healthy.

GCP's equivalent of ``reboot_instances`` is ``InstancesClient.reset`` —
it hard-resets the VM (keeping the instance record) and returns a zone
operation. After the operation completes we wait for SSH and capture
uptime to confirm the reboot actually happened.

Usage:
    python reboot_instance.py --instance-id isv-test-gpu --region asia-east1-a \\
        --key-file /tmp/isv-test-key.pem --public-ip 34.x.x.x

Output JSON (per docs/existing-patterns/vm.md):
{
    "success": true,
    "platform": "vm",
    "instance_id": "...",
    "instance_state": "running",
    "public_ip": "...",
    "key_file": "...",
    "uptime_seconds": 45,
    "ssh_connectivity": true
}
"""

import argparse
import json
import os
import subprocess
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


def _get_uptime_via_ssh(host: str, user: str, key_file: str) -> float | None:
    """Return ``/proc/uptime`` seconds via SSH, or None on any failure."""
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "PasswordAuthentication=no",
                "-o",
                "ConnectTimeout=10",
                "-i",
                key_file,
                f"{user}@{host}",
                "cat /proc/uptime | cut -d' ' -f1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return None


@handle_gcp_errors
def main() -> int:
    """Reset a Compute Engine VM and verify recovery."""
    parser = argparse.ArgumentParser(description="Reboot (reset) Compute Engine instance")
    parser.add_argument("--instance-id", required=True, help="Instance name")
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_ZONE", "asia-east1-a"),
        help="GCP zone (provider config 'region' is a zone)",
    )
    parser.add_argument("--key-file", required=True, help="Path to SSH private key")
    parser.add_argument("--public-ip", required=True, help="Pre-reboot external IP")
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
        "reboot_initiated": False,
        "ssh_ready": False,
        "ssh_connectivity": False,
    }

    try:
        # ============================================================
        # Step 1: Verify instance is RUNNING, record pre-reboot uptime
        # ============================================================
        print("Verifying instance is RUNNING before reboot...", file=sys.stderr)
        instance = describe_instance(client, project, zone, args.instance_id)
        gcp_status = instance.status
        current_state = canonical_state(gcp_status)

        if current_state != "running":
            result["error"] = f"Instance is {gcp_status}, expected RUNNING"
            result["state"] = current_state
            result["gcp_status"] = gcp_status
            print(json.dumps(result, indent=2))
            return 1

        pre_uptime = _get_uptime_via_ssh(args.public_ip, args.ssh_user, args.key_file)
        if pre_uptime is not None:
            result["pre_reboot_uptime"] = round(pre_uptime, 1)
            print(f"  Pre-reboot uptime: {pre_uptime:.0f}s", file=sys.stderr)

        # ============================================================
        # Step 2: Issue reset
        # ============================================================
        print(f"Resetting instance {args.instance_id}...", file=sys.stderr)
        op = client.reset(project=project, zone=zone, instance=args.instance_id)
        result["reboot_initiated"] = True
        op.result(timeout=300)
        print("  Reset API call completed", file=sys.stderr)

        # ============================================================
        # Step 3: Re-describe and pick up any IP changes
        # ============================================================
        instance = describe_instance(client, project, zone, args.instance_id)
        result["gcp_status"] = instance.status
        result["state"] = canonical_state(instance.status)
        result["instance_state"] = result["state"]
        new_ip = get_instance_external_ip(instance) or args.public_ip
        result["public_ip"] = new_ip
        result["private_ip"] = get_instance_internal_ip(instance)

        # ============================================================
        # Step 4: Wait for SSH
        # ============================================================
        print("Waiting for SSH after reboot...", file=sys.stderr)
        ssh_ready = wait_for_ssh(
            new_ip,
            args.ssh_user,
            args.key_file,
            max_attempts=30,
            interval=10,
        )
        result["ssh_ready"] = ssh_ready
        result["ssh_connectivity"] = ssh_ready

        if not ssh_ready:
            result["error"] = "SSH not ready after reboot"
            print("WARNING: SSH did not become ready after reboot", file=sys.stderr)
            print(json.dumps(result, indent=2))
            return 1

        # ============================================================
        # Step 5: Capture post-reboot uptime and confirm reset
        # ============================================================
        post_uptime = _get_uptime_via_ssh(new_ip, args.ssh_user, args.key_file)
        if post_uptime is not None:
            result["uptime_seconds"] = round(post_uptime, 1)
            print(f"  Post-reboot uptime: {post_uptime:.0f}s", file=sys.stderr)

            if pre_uptime is not None and post_uptime < pre_uptime:
                result["reboot_confirmed"] = True
                print("  Reboot confirmed (uptime reset)", file=sys.stderr)
            elif pre_uptime is not None:
                result["reboot_confirmed"] = False
                print(
                    f"  WARNING: Uptime did not decrease (pre={pre_uptime:.0f}s, post={post_uptime:.0f}s)",
                    file=sys.stderr,
                )
            else:
                # No pre-reboot uptime to compare — treat <10min as freshly booted
                result["reboot_confirmed"] = post_uptime < 600
                print(
                    f"  Reboot likely confirmed (uptime={post_uptime:.0f}s)",
                    file=sys.stderr,
                )

        result["success"] = result["state"] == "running"
        if result["success"]:
            print("Reboot completed successfully!", file=sys.stderr)
        else:
            result["error"] = f"Instance state is {instance.status}, expected RUNNING"

    except Exception as e:
        result["error"] = str(e)
        print(f"ERROR: {e}", file=sys.stderr)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

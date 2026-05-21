#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

"""Hard-reset a Compute Engine instance and affirm the reboot occurred.

Compute Engine exposes only ``instances.reset`` (hard reset); the call
returns before the guest restarts. Flow:
  1. Pre-gate on ``cloud-init status --wait`` so the reset doesn't fire
     mid-cloud-init.
  2. Issue reset.
  3. Wait for SSH to DROP (confirms the pre-reboot sshd is gone) before
     sampling post-reboot uptime — otherwise the probe may hit the
     lingering pre-reboot sshd and falsely confirm.
  4. Wait for SSH to come back + cloud-init readiness.
  5. Sample uptime via SSH; reboot is confirmed only if boot timestamp
     is after the reset request OR uptime decreased relative to pre-reboot.

Usage:
    python3 reboot_instance.py --instance-id <name> --region <zone> \\
        --key-file <pem> --public-ip <ip>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import (
    canonical_state,
    is_zone_unavailable,
    resolve_project,
    select_zones,
    wait_for_public_ip,
    wait_for_zonal_op,
)
from common.ssh_utils import ssh_run, wait_for_cloud_init, wait_for_ssh, wait_for_ssh_drop


def _uptime_seconds(host: str, user: str, key_file: str) -> float | None:
    """Read /proc/uptime over SSH; returns seconds or None on failure."""
    rc, stdout, _stderr = ssh_run(
        host, user, key_file, "cat /proc/uptime | cut -d' ' -f1", timeout=30, connect_timeout=10
    )
    if rc == 0:
        try:
            return float(stdout.strip())
        except ValueError:
            return None
    return None


def main() -> int:
    """Issue the reset, wait for SSH drop, then confirm post-reboot health."""
    parser = argparse.ArgumentParser(description="Hard-reset Compute Engine instance")
    parser.add_argument("--instance-id", required=True, help="Instance name")
    parser.add_argument("--region", required=True, help="Effective zone")
    parser.add_argument("--key-file", required=True, help="SSH key path")
    parser.add_argument("--public-ip", required=True, help="Pre-reset IP (re-validated post-reset)")
    parser.add_argument("--ssh-user", default="ubuntu", help="SSH username")
    parser.add_argument("--project", default="", help="GCP project (default: ADC)")
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = select_zones(args.region)[0]

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "instance_id": args.instance_id,
        "region": args.region,
        "zone": zone,
        "key_file": args.key_file,
        "ssh_user": args.ssh_user,
        "reboot_initiated": False,
        "reboot_confirmed": False,
        "ssh_ready": False,
        "state": "",
    }

    try:
        from google.cloud import compute_v1

        client = compute_v1.InstancesClient()
        inst = client.get(project=project, zone=zone, instance=args.instance_id)
        if canonical_state(getattr(inst, "status", None)) != "running":
            result["error"] = "instance is not running"
            print(json.dumps(result, indent=2, default=str))
            return 1

        # Pre-gate cloud-init so the reset doesn't fire mid-cloud-init.
        # Honor the helper's return — issuing reset while cloud-init is
        # still applying metadata/user-data leaves the guest in a dirty
        # state that only surfaces on the next boot.
        if not wait_for_cloud_init(args.public_ip, args.ssh_user, args.key_file):
            result["error"] = "pre-reset cloud-init did not finish cleanly; refusing to reset mid-cloud-init"
            print(json.dumps(result, indent=2, default=str))
            return 1
        pre_uptime = _uptime_seconds(args.public_ip, args.ssh_user, args.key_file)
        if pre_uptime is not None:
            result["pre_reboot_uptime"] = round(pre_uptime, 1)

        reset_requested_at = time.time()
        last_error: Exception | None = None
        for attempt, backoff in enumerate([0, 60, 120], start=1):
            if backoff:
                print(f"  retrying reset after {backoff}s backoff (attempt {attempt})", file=sys.stderr)
                time.sleep(backoff)
            try:
                op = client.reset(project=project, zone=zone, instance=args.instance_id)
                result["reboot_initiated"] = True
                wait_for_zonal_op(client, project, zone, op, timeout=300)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if is_zone_unavailable(exc):
                    continue
                raise
        if last_error is not None:
            result["error"] = f"reset failed after retries: {last_error}"
            print(json.dumps(result, indent=2, default=str))
            return 1

        # Require the pre-reset sshd to actually drop before probing
        # post-reset state — otherwise the probe may hit the lingering
        # pre-reset sshd and falsely affirm. instances.reset acks before
        # the VM actually power-cycles; the 12-attempt default (~60s)
        # is too tight for some L4 zones where reset propagation is
        # slow. Allow up to ~180s of probing for the drop.
        if not wait_for_ssh_drop(args.public_ip, args.ssh_user, args.key_file, max_attempts=36, interval=5):
            result["error"] = "pre-reset sshd did not drop within wait window; cannot affirm reboot"
            print(json.dumps(result, indent=2, default=str))
            return 1

        # Never silently fall back to the pre-reset IP — that would mask
        # a stale IP if the external NAT got reassigned. Fail honestly
        # when the post-reset poll returns no IP.
        public_ip = wait_for_public_ip(client, project, zone, args.instance_id, timeout=180, interval=5)
        if not public_ip:
            result["error"] = "no external IP observed after reset"
            print(json.dumps(result, indent=2, default=str))
            return 1
        result["public_ip"] = public_ip
        inst = client.get(project=project, zone=zone, instance=args.instance_id)
        result["state"] = canonical_state(getattr(inst, "status", None))
        nic = (inst.network_interfaces or [None])[0]
        result["private_ip"] = getattr(nic, "network_i_p", "") if nic else ""

        # Match the post-stop/start SSH wait budget (60 attempts ~660s)
        # to cover the documented 5-7min sshd recovery window. 36 attempts
        # (~400s) trips at the edge of the recovery window on capacity-
        # heavy days.
        if not wait_for_ssh(public_ip, args.ssh_user, args.key_file, max_attempts=60, interval=10):
            result["error"] = "SSH not ready after reset"
            print(json.dumps(result, indent=2, default=str))
            return 1

        # cloud-init wait MUST succeed before we report ssh_ready; rc 1
        # means semantic cloud-init failure on the post-reset boot, rc 255
        # / 124 means the helper exhausted retries without a real signal.
        if not wait_for_cloud_init(public_ip, args.ssh_user, args.key_file):
            result["error"] = "cloud-init did not finish cleanly after reset"
            print(json.dumps(result, indent=2, default=str))
            return 1

        # Consecutive-success SSH stability gate — first SSH success after
        # a reset may be a transient sshd that immediately drops while
        # the guest re-applies metadata keys.
        for stability_attempt in range(3):
            rc, _stdout, _stderr = ssh_run(
                public_ip, args.ssh_user, args.key_file, "exit 0", timeout=10, connect_timeout=5
            )
            if rc != 0:
                result["error"] = f"post-reset SSH stability probe failed at attempt {stability_attempt + 1} (rc={rc})"
                print(json.dumps(result, indent=2, default=str))
                return 1
            if stability_attempt < 2:
                time.sleep(10)
        result["ssh_ready"] = True

        post_uptime = _uptime_seconds(public_ip, args.ssh_user, args.key_file)
        if post_uptime is None:
            result["error"] = "post-reset uptime sample failed; reboot not affirmed"
            print(json.dumps(result, indent=2, default=str))
            return 1
        result["uptime_seconds"] = round(post_uptime, 1)

        boot_started_at = time.time() - post_uptime
        if boot_started_at >= reset_requested_at:
            result["reboot_confirmed"] = True
        elif pre_uptime is not None and post_uptime < pre_uptime:
            result["reboot_confirmed"] = True

        result["success"] = result["reboot_confirmed"]
        if not result["success"]:
            result["error"] = "uptime did not decrease and boot timestamp not after reset request"
    except Exception as exc:
        result["error"] = str(exc)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

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

"""Start a stopped Compute Engine instance and confirm post-start health.

Re-reads the external IP after start (ephemeral IPs are released on stop)
and gates ``ssh_ready=true`` on N consecutive SSH probes plus cloud-init
readiness so a transient sshd rebind doesn't ship as success.

Usage:
    python3 start_instance.py --instance-id <name> --region <zone> --key-file <pem>
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
from common.ssh_utils import ssh_run, wait_for_cloud_init, wait_for_ssh

SSH_STABILITY_PROBES = 3
SSH_STABILITY_INTERVAL = 10


def _ssh_stability_probe(host: str, user: str, key_file: str) -> bool:
    """Require ``SSH_STABILITY_PROBES`` consecutive successful SSH probes."""
    for attempt in range(SSH_STABILITY_PROBES):
        rc, _stdout, _stderr = ssh_run(host, user, key_file, "exit 0", timeout=10, connect_timeout=5)
        if rc != 0:
            return False
        if attempt < SSH_STABILITY_PROBES - 1:
            time.sleep(SSH_STABILITY_INTERVAL)
    return True


def main() -> int:
    """Drive the start operation and post-start stability gates."""
    parser = argparse.ArgumentParser(description="Start Compute Engine instance")
    parser.add_argument("--instance-id", required=True, help="Instance name")
    parser.add_argument("--region", required=True, help="Effective zone")
    parser.add_argument("--key-file", required=True, help="SSH key path for stability probes")
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
        "start_initiated": False,
        "ssh_ready": False,
        "state": "",
    }

    try:
        from google.cloud import compute_v1

        client = compute_v1.InstancesClient()
        inst = client.get(project=project, zone=zone, instance=args.instance_id)
        current = canonical_state(getattr(inst, "status", None))
        if current != "stopped":
            result["state"] = current
            result["error"] = f"instance is {current}; expected stopped"
            print(json.dumps(result, indent=2, default=str))
            return 1

        last_error: Exception | None = None
        for attempt, backoff in enumerate([0, 60, 120], start=1):
            if backoff:
                print(f"  retrying start after {backoff}s backoff (attempt {attempt})", file=sys.stderr)
                time.sleep(backoff)
            try:
                op = client.start(project=project, zone=zone, instance=args.instance_id)
                result["start_initiated"] = True
                # The orchestrator caps the step at 600s; the start op is
                # ack-only and almost always completes inside 2 minutes
                # — bound the internal wait at ~3 minutes so the SSH /
                # cloud-init / stability gates downstream still have a
                # comfortable budget.
                wait_for_zonal_op(client, project, zone, op, timeout=180)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if is_zone_unavailable(exc):
                    continue
                raise
        if last_error is not None:
            result["error"] = f"start failed after retries: {last_error}"
            print(json.dumps(result, indent=2, default=str))
            return 1

        # Poll state + fresh public IP (the pre-stop IP was released).
        public_ip = wait_for_public_ip(client, project, zone, args.instance_id, timeout=120, interval=5)
        if not public_ip:
            result["error"] = "no external IP observed after start"
            print(json.dumps(result, indent=2, default=str))
            return 1
        result["public_ip"] = public_ip

        inst = client.get(project=project, zone=zone, instance=args.instance_id)
        result["state"] = canonical_state(getattr(inst, "status", None))
        nic = (inst.network_interfaces or [None])[0]
        result["private_ip"] = getattr(nic, "network_i_p", "") if nic else ""

        # Compute Engine post-stop/start sshd recovery on g2-standard-* +
        # L4 GPUs is documented at ~5-7 minutes; 30 attempts at ~11s/cycle
        # (5s connect timeout + 10s sleep when sshd is not yet bound) is
        # ~330s which trips right at the edge of the recovery window.
        # 60 attempts (~660s) leaves comfortable margin without exceeding
        # the 1200s step budget.
        if not wait_for_ssh(public_ip, args.ssh_user, args.key_file, max_attempts=60, interval=10):
            result["error"] = "SSH not ready after start"
            print(json.dumps(result, indent=2, default=str))
            return 1

        cloud_init_ok = wait_for_cloud_init(public_ip, args.ssh_user, args.key_file)
        stable_ok = _ssh_stability_probe(public_ip, args.ssh_user, args.key_file)
        result["ssh_ready"] = cloud_init_ok and stable_ok
        result["cloud_init_ready"] = cloud_init_ok
        if not result["ssh_ready"]:
            result["error"] = "post-start stability gate failed (cloud-init or stable SSH probes)"
            print(json.dumps(result, indent=2, default=str))
            return 1
        result["success"] = True
    except Exception as exc:
        result["error"] = str(exc)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

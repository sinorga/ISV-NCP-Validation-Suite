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

"""Stop a Compute Engine instance and confirm it reaches TERMINATED.

Compute Engine's TERMINATED is the canonical "stopped" state. The stub
pre-gates the stop on cloud-init readiness over SSH so the API call
doesn't fire mid-cloud-init.

Usage:
    python3 stop_instance.py --instance-id <name> --region <zone> \\
        [--key-file <pem>] [--public-ip <ip>]
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
    is_sentinel,
    is_zone_unavailable,
    resolve_project,
    select_zones,
    wait_for_zonal_op,
)
from common.ssh_utils import wait_for_cloud_init


def main() -> int:
    """Stop an instance and wait for the canonical stopped state."""
    parser = argparse.ArgumentParser(description="Stop Compute Engine instance")
    parser.add_argument("--instance-id", required=True, help="Instance name")
    parser.add_argument("--region", required=True, help="Effective zone")
    parser.add_argument("--key-file", default="", help="SSH key path for cloud-init pre-gate")
    parser.add_argument("--public-ip", default="", help="SSH host for cloud-init pre-gate")
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
        "stop_initiated": False,
        "state": "",
    }

    try:
        from google.cloud import compute_v1

        client = compute_v1.InstancesClient()
        inst = client.get(project=project, zone=zone, instance=args.instance_id)
        current = canonical_state(getattr(inst, "status", None))
        if current == "stopped":
            # Idempotent: contract treats end-state satisfied as initiated=True.
            result["state"] = current
            result["stop_initiated"] = True
            result["already_stopped"] = True
            result["success"] = True
            print(json.dumps(result, indent=2, default=str))
            return 0
        if current != "running":
            result["state"] = current
            result["error"] = f"instance is {current}; expected running"
            print(json.dumps(result, indent=2, default=str))
            return 1

        if not (is_sentinel(args.public_ip) or is_sentinel(args.key_file)):
            # Honor the cloud-init helper's return — issuing stop while
            # cloud-init is still applying user-data leaves the guest
            # half-configured and surfaces only on the next boot.
            if not wait_for_cloud_init(args.public_ip, args.ssh_user, args.key_file):
                result["error"] = "pre-stop cloud-init did not finish cleanly; refusing to stop mid-cloud-init"
                print(json.dumps(result, indent=2, default=str))
                return 1

        # Retry-in-place on capacity-class errors (lifecycle ops are zone-bound).
        last_error: Exception | None = None
        for attempt, backoff in enumerate([0, 60, 120], start=1):
            if backoff:
                print(f"  retrying stop after {backoff}s backoff (attempt {attempt})", file=sys.stderr)
                time.sleep(backoff)
            try:
                op = client.stop(project=project, zone=zone, instance=args.instance_id)
                result["stop_initiated"] = True
                wait_for_zonal_op(client, project, zone, op, timeout=600)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if is_zone_unavailable(exc):
                    continue
                raise
        if last_error is not None:
            result["error"] = f"stop failed after retries: {last_error}"
            print(json.dumps(result, indent=2, default=str))
            return 1

        inst = client.get(project=project, zone=zone, instance=args.instance_id)
        result["state"] = canonical_state(getattr(inst, "status", None))
        result["success"] = result["state"] == "stopped"
        if not result["success"]:
            result["error"] = f"post-stop state is {result['state']}; expected stopped"
    except Exception as exc:
        result["error"] = str(exc)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

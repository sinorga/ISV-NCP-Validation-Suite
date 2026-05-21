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

"""Describe a Compute Engine VM and forward SSH info for downstream checks.

Runs in the test phase after reboot so host-level validations (Connectivity,
GPU, Driver, etc.) anchor to the post-reboot state.

Usage:
    python3 describe_instance.py --instance-id <name> --region <zone> --key-file <pem>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import canonical_state, resolve_project, select_zones


def main() -> int:
    """Emit the canonical describe-instance JSON."""
    parser = argparse.ArgumentParser(description="Describe Compute Engine instance")
    parser.add_argument("--instance-id", required=True, help="Instance name")
    parser.add_argument("--region", required=True, help="Effective zone")
    parser.add_argument("--key-file", required=True, help="SSH key path")
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
        "availability_zone": zone,
        "key_file": args.key_file,
        "ssh_user": args.ssh_user,
    }

    try:
        from google.cloud import compute_v1

        client = compute_v1.InstancesClient()
        inst = client.get(project=project, zone=zone, instance=args.instance_id)
        nic = (inst.network_interfaces or [None])[0]
        access = (nic.access_configs or [None])[0] if nic else None
        result["instance_type"] = (inst.machine_type or "").rsplit("/", 1)[-1]
        result["state"] = canonical_state(getattr(inst, "status", None))
        result["public_ip"] = getattr(access, "nat_i_p", "") if access else ""
        result["private_ip"] = getattr(nic, "network_i_p", "") if nic else ""
        if nic:
            if nic.network:
                result["vpc_id"] = nic.network.rsplit("/", 1)[-1]
            if nic.subnetwork:
                result["subnet_id"] = nic.subnetwork.rsplit("/", 1)[-1]
        result["success"] = bool(result["state"])
        if not result["public_ip"]:
            result["error"] = "instance has no external IP"
            result["success"] = False
    except Exception as exc:
        result["error"] = str(exc)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

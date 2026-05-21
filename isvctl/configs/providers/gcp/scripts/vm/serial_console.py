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

"""Probe Compute Engine serial console output for the target instance.

Compute Engine has no account-level serial-console enable toggle; access is
gated entirely by IAM. ``serial_access_enabled`` is therefore derived from
whether ``getSerialPortOutput`` succeeded under the active credentials.

Usage:
    python3 serial_console.py --instance-id <name> --region <zone>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import resolve_project, select_zones


def main() -> int:
    """Emit the canonical serial-console JSON with derived booleans."""
    parser = argparse.ArgumentParser(description="Probe Compute Engine serial console output")
    parser.add_argument("--instance-id", required=True, help="Instance name")
    parser.add_argument("--region", required=True, help="Effective zone")
    parser.add_argument("--project", default="", help="GCP project (default: ADC)")
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = select_zones(args.region)[0]

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "instance_id": args.instance_id,
        "console_available": False,
        "serial_access_enabled": False,
        "output_length": 0,
    }

    try:
        from google.cloud import compute_v1

        client = compute_v1.InstancesClient()
        # Default port=1 — the flattened (project, zone, instance) form does
        # NOT accept `port=` as a kwarg; build an explicit Request when a
        # non-default port is needed.
        response = client.get_serial_port_output(project=project, zone=zone, instance=args.instance_id)
        result["serial_access_enabled"] = True
        contents = getattr(response, "contents", "") or ""
        result["output_length"] = len(contents)
        result["console_available"] = bool(contents)
        if contents:
            result["output_snippet"] = contents[-500:] if len(contents) > 500 else contents
        result["success"] = result["console_available"] or result["serial_access_enabled"]
    except Exception as exc:
        # Permission or auth failure — drive both booleans from the real
        # probe outcome instead of hardcoding True. The check still fails
        # if both probes failed.
        message = str(exc)
        if "403" in message or "permission" in message.lower() or "forbidden" in message.lower():
            result["serial_access_enabled"] = False
        result["error"] = message

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

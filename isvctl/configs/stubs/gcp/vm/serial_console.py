#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Retrieve serial port output from a Compute Engine instance (read-only).

GCP's equivalent of EC2's ``get_console_output`` is
``InstancesClient.get_serial_port_output`` (port 1 = default kernel log).

Unlike AWS, GCP does not gate console output behind an
account-wide "serial console access" toggle — the API is always
available to anyone with the ``compute.instances.getSerialPortOutput``
permission. We report ``serial_access_enabled: true`` when the API call
succeeds for this instance, and let ``SerialConsoleCheck`` cross-check
the output length.

Usage:
    python serial_console.py --instance-id isv-test-gpu --region asia-east1-a

Output JSON (per docs/existing-patterns/vm.md):
{
    "success": true,
    "platform": "vm",
    "instance_id": "...",
    "console_available": true,
    "serial_access_enabled": true,
    "output_length": 4096
}
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import resolve_project
from common.errors import handle_gcp_errors
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1


def _fetch_serial_output(
    client: compute_v1.InstancesClient,
    project: str,
    zone: str,
    name: str,
    retries: int = 3,
) -> dict[str, Any]:
    """Retry ``get_serial_port_output`` up to ``retries`` times.

    The buffer is briefly empty on fresh instances (mirrors the oracle's
    behaviour with ``Latest=True`` on Nitro/bare-metal).
    """
    out: dict[str, Any] = {"available": False, "output_length": 0}
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            response = client.get_serial_port_output(
                request=compute_v1.GetSerialPortOutputInstanceRequest(
                    project=project,
                    zone=zone,
                    instance=name,
                    port=1,
                )
            )
            contents = response.contents or ""
            if contents:
                out["available"] = True
                out["output_length"] = len(contents)
                out["output_snippet"] = contents[-500:] if len(contents) > 500 else contents
                return out
        except gax_exc.GoogleAPIError as e:
            last_error = e

        if attempt < retries - 1:
            time.sleep(10)

    if last_error:
        out["error"] = str(last_error)
    else:
        out["message"] = "Console output empty after retries"
    return out


@handle_gcp_errors
def main() -> int:
    """Retrieve serial port 1 output for a Compute Engine instance."""
    parser = argparse.ArgumentParser(description="Get Compute Engine VM serial console output")
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
        "console_available": False,
        "serial_access_enabled": False,
        "output_length": 0,
    }

    try:
        console = _fetch_serial_output(client, project, zone, args.instance_id)
        result["console_available"] = console.get("available", False)
        result["output_length"] = console.get("output_length", 0)

        # GCP exposes serial port output as a standard API call — if we
        # got a valid response back, console access is effectively enabled
        # for this caller on this instance.
        result["serial_access_enabled"] = result["console_available"] or "error" not in console

        if console.get("output_snippet"):
            result["output_snippet"] = console["output_snippet"]
        if console.get("error"):
            result["error"] = console["error"]

        result["success"] = result["console_available"] or result["serial_access_enabled"]

    except Exception as e:
        result["error"] = str(e)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

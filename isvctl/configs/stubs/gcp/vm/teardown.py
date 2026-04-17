#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Teardown a Compute Engine instance and associated resources.

Mirrors the oracle's teardown flow:
  1. Describe the instance to capture attached resource names
  2. Delete the instance and wait for the zone op to finish
  3. Delete the SSH ingress firewall rule (our "security group" equivalent)
  4. Remove the local SSH key files (our "key pair" equivalent)

``--skip-destroy`` and the flag-list signature match the oracle so the
canonical test config can pass ``--delete-key-pair``/``--delete-security-group``
without modification.

Usage:
    python teardown.py --instance-id isv-test-gpu --region asia-east1-a

Output JSON (per docs/existing-patterns/vm.md):
{
    "success": true,
    "platform": "vm",
    "resources_deleted": [...],
    "message": "..."
}
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import resolve_project
from common.errors import handle_gcp_errors
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Teardown Compute Engine instance")
    parser.add_argument("--instance-id", required=True, help="Instance name")
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_ZONE", "asia-east1-a"),
        help="GCP zone (provider config 'region' is a zone)",
    )
    parser.add_argument("--project", help="GCP project ID (default: $GOOGLE_CLOUD_PROJECT)")
    parser.add_argument("--delete-key-pair", action="store_true", help="Also delete SSH key files")
    parser.add_argument(
        "--delete-security-group",
        action="store_true",
        help="Also delete SSH ingress firewall rule",
    )
    parser.add_argument("--skip-destroy", action="store_true", help="Skip actual destroy")
    parser.add_argument("--key-name", default="isv-test-key", help="SSH key basename")
    parser.add_argument(
        "--firewall-name",
        default=None,
        help="Firewall rule name (default: <instance>-ssh)",
    )
    args = parser.parse_args()

    firewall_name = args.firewall_name or f"{args.instance_id}-ssh"

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "resources_destroyed": False,
        "resources_deleted": [],
        "deleted": {
            "instances": [],
            "firewalls": [],
            "ssh_keys": [],
        },
    }

    if args.skip_destroy:
        result["success"] = True
        result["message"] = "Destroy skipped (--skip-destroy flag)"
        print(json.dumps(result, indent=2))
        return 0

    project = resolve_project(args.project)
    zone = args.region

    instances_client = compute_v1.InstancesClient()
    firewalls_client = compute_v1.FirewallsClient()

    try:
        # ============================================================
        # Step 1: Delete the instance (tolerate already-gone)
        # ============================================================
        try:
            op = instances_client.delete(project=project, zone=zone, instance=args.instance_id)
            op.result(timeout=540)
            result["deleted"]["instances"].append(args.instance_id)
            result["resources_deleted"].append(f"instance:{args.instance_id}")
            print(f"  Deleted instance {args.instance_id}", file=sys.stderr)
        except gax_exc.NotFound:
            print(f"  Instance {args.instance_id} already gone (ok)", file=sys.stderr)

        # ============================================================
        # Step 2: Delete the SSH firewall rule
        # ============================================================
        if args.delete_security_group:
            try:
                op = firewalls_client.delete(project=project, firewall=firewall_name)
                op.result(timeout=120)
                result["deleted"]["firewalls"].append(firewall_name)
                result["resources_deleted"].append(f"firewall:{firewall_name}")
                print(f"  Deleted firewall {firewall_name}", file=sys.stderr)
            except gax_exc.NotFound:
                print(f"  Firewall {firewall_name} already gone (ok)", file=sys.stderr)
            except Exception as fw_err:
                result.setdefault("warnings", []).append(f"Could not delete firewall {firewall_name}: {fw_err}")

        # ============================================================
        # Step 3: Delete local SSH key files
        # ============================================================
        if args.delete_key_pair:
            key_dir = Path("/tmp")
            for suffix in (".pem", ".pub"):
                key_file = key_dir / f"{args.key_name}{suffix}"
                if key_file.exists():
                    try:
                        key_file.chmod(0o600)
                        key_file.unlink()
                        result["deleted"]["ssh_keys"].append(str(key_file))
                        result["resources_deleted"].append(f"ssh_key:{key_file}")
                    except OSError as key_err:
                        result.setdefault("warnings", []).append(f"Could not delete {key_file}: {key_err}")

        result["success"] = True
        result["resources_destroyed"] = True
        result["message"] = f"Teardown complete ({len(result['resources_deleted'])} resources)"

    except Exception as e:
        result["error"] = str(e)
        print(f"ERROR: {e}", file=sys.stderr)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

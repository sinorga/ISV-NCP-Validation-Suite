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

"""Tear down the Compute Engine VM, owned firewall, and local SSH artifacts.

Reads verified-reuse flags forwarded from ``launch_instance`` and only
deletes resources this run created. Also sweeps any zones in
``--leaked-zones`` for phantom instance records left by the zone-walk
during launch.

Usage:
    python3 teardown.py --instance-id <name> --region <zone> \\
        --delete-key-pair --delete-security-group \\
        --firewall-name <name> --firewall-created <bool> \\
        --key-file <pem> --key-created <bool> \\
        --instance-created <bool> --leaked-zones <comma-list> \\
        [--skip-destroy]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import (
    delete_failed_zonal_instance,
    is_sentinel,
    resolve_project,
    select_zones,
    ssh_public_key_path,
    wait_for_global_op,
    wait_for_zonal_op,
)


def _as_bool(value: str) -> bool:
    """Parse a bool-shaped CLI value, treating sentinels as False."""
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in ("", "none", "null", "false", "0", "no"):
        return False
    return text in ("true", "1", "yes")


def _split_zones(value: str) -> list[str]:
    """Parse the comma-joined leaked-zones list (sentinel-aware)."""
    if is_sentinel(value):
        return []
    return [z.strip() for z in value.split(",") if z.strip()]


def main() -> int:
    """Delete owned resources, sweep leaked zones, drop local PEM artifacts."""
    parser = argparse.ArgumentParser(description="Teardown Compute Engine VM and owned resources")
    parser.add_argument("--instance-id", required=True, help="Instance name")
    parser.add_argument("--region", required=True, help="Effective zone")
    parser.add_argument("--delete-key-pair", action="store_true", help="Enable local PEM cleanup")
    parser.add_argument("--delete-security-group", action="store_true", help="Enable firewall cleanup")
    parser.add_argument("--firewall-name", default="", help="Firewall rule to delete")
    parser.add_argument("--firewall-created", default="false", help="True iff this run created the firewall")
    parser.add_argument("--key-file", default="", help="Local PEM path")
    parser.add_argument("--key-created", default="false", help="True iff this run generated the keypair")
    parser.add_argument(
        "--instance-created",
        default="false",
        help="True iff this run created the primary VM (gates primary-delete and leaked-zone sweep)",
    )
    parser.add_argument("--leaked-zones", default="", help="Comma-joined list of zones to sweep")
    parser.add_argument("--skip-destroy", action="store_true", help="Skip all destructive work")
    parser.add_argument("--project", default="", help="GCP project (default: ADC)")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "resources_destroyed": False,
        "deleted": {"instances": [], "firewalls": [], "key_files": []},
    }

    if args.skip_destroy:
        result["success"] = True
        result["instance_id"] = args.instance_id
        result["message"] = f"Instance {args.instance_id} preserved (--skip-destroy); terminate manually when done"
        print(json.dumps(result, indent=2, default=str))
        return 0

    project = resolve_project(args.project)
    primary_zone = select_zones(args.region)[0] if not is_sentinel(args.region) else ""
    instance_created = _as_bool(args.instance_created)
    firewall_created = _as_bool(args.firewall_created)
    key_created = _as_bool(args.key_created)
    leaked = _split_zones(args.leaked_zones)

    cleanup_success = True
    cleanup_errors: list[str] = []

    try:
        from google.cloud import compute_v1
    except ImportError as exc:
        result["error"] = f"google-cloud-compute missing: {exc}"
        print(json.dumps(result, indent=2, default=str))
        return 1

    instances_client = compute_v1.InstancesClient()
    firewalls_client = compute_v1.FirewallsClient()

    # 1. Primary instance delete (only when this run created it; otherwise we
    # leave the operator's pre-existing VM alone).
    if instance_created and primary_zone and not is_sentinel(args.instance_id):
        try:
            op = instances_client.delete(project=project, zone=primary_zone, instance=args.instance_id)
            wait_for_zonal_op(instances_client, project, primary_zone, op, timeout=300)
            result["deleted"]["instances"].append(args.instance_id)
        except Exception as exc:
            message = str(exc)
            if "404" in message or "notFound" in message.lower():
                result["deleted"]["instances"].append(args.instance_id)
            else:
                cleanup_success = False
                cleanup_errors.append(f"instance delete: {message}")
    else:
        result.setdefault("notes", []).append("primary instance not deleted (instance_created=false)")

    # 2. Leaked-zone phantom sweep (best-effort).
    if instance_created and not is_sentinel(args.instance_id):
        for zone in leaked:
            if zone == primary_zone:
                continue
            ok = delete_failed_zonal_instance(instances_client, project, zone, args.instance_id)
            if not ok:
                cleanup_success = False
                cleanup_errors.append(f"phantom delete in {zone}")

    # 3. Firewall cleanup (gated on firewall_created).
    if args.delete_security_group and firewall_created and not is_sentinel(args.firewall_name):
        try:
            op = firewalls_client.delete(project=project, firewall=args.firewall_name)
            wait_for_global_op(firewalls_client, project, op, timeout=180)
            result["deleted"]["firewalls"].append(args.firewall_name)
        except Exception as exc:
            message = str(exc)
            if "404" in message or "notFound" in message.lower():
                result["deleted"]["firewalls"].append(args.firewall_name)
            else:
                cleanup_success = False
                cleanup_errors.append(f"firewall delete: {message}")

    # 4. Local PEM + .pub cleanup (gated on key_created). The local-key
    # cleanup is NOT short-circuited by the cloud NotFound earlier — the
    # PEM may exist after an aborted run even if the cloud resource is
    # gone.
    if args.delete_key_pair and key_created and not is_sentinel(args.key_file):
        for path in (Path(args.key_file), ssh_public_key_path(args.key_file)):
            try:
                if path.exists():
                    path.chmod(0o600)
                    path.unlink()
                    result["deleted"]["key_files"].append(str(path))
            except OSError as exc:
                cleanup_success = False
                cleanup_errors.append(f"local key delete {path}: {exc}")

    if cleanup_errors:
        result["cleanup_errors"] = cleanup_errors
    result["success"] = cleanup_success
    result["resources_destroyed"] = cleanup_success and bool(
        result["deleted"]["instances"] or result["deleted"]["firewalls"] or result["deleted"]["key_files"]
    )
    if not result["success"]:
        result["error"] = "; ".join(cleanup_errors)
    result["message"] = "Teardown completed" if cleanup_success else "Teardown failed"

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

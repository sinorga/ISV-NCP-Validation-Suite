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

"""List Compute Engine instances on a given VPC network.

Compute Engine ``instances.list`` is zone-scoped; cross-zone listing requires
``instances.aggregatedList``. Network is filtered by URL-match against
``networkInterfaces[].network``.

Usage:
    python3 list_instances.py --vpc-id default --instance-id <name> --region <zone>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import canonical_state, is_sentinel, resolve_project


def main() -> int:
    """Emit the canonical instance-list JSON."""
    parser = argparse.ArgumentParser(description="List Compute Engine instances")
    parser.add_argument("--vpc-id", required=True, help="Network short name")
    parser.add_argument("--region", default="us-central1", help="Zone or region; informational")
    parser.add_argument("--instance-id", default="", help="Target instance name to verify")
    parser.add_argument("--project", default="", help="GCP project (default: ADC)")
    args = parser.parse_args()

    project = resolve_project(args.project)
    network = args.vpc_id

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "instances": [],
    }

    try:
        from google.cloud import compute_v1

        client = compute_v1.InstancesClient()
        agg = client.aggregated_list(project=project)
        seen: list[dict[str, Any]] = []
        for _zone_key, scoped in agg:
            instances = getattr(scoped, "instances", None) or []
            for inst in instances:
                nic = (inst.network_interfaces or [None])[0]
                network_url = getattr(nic, "network", "") if nic else ""
                if not network_url:
                    continue
                # Exact-match the network short name parsed off the self-link.
                if network_url.rsplit("/", 1)[-1] != network:
                    continue
                access = (nic.access_configs or [None])[0] if nic else None
                seen.append(
                    {
                        "instance_id": inst.name,
                        "instance_type": (inst.machine_type or "").rsplit("/", 1)[-1],
                        "state": canonical_state(getattr(inst, "status", None)),
                        "public_ip": getattr(access, "nat_i_p", None) if access else None,
                        "private_ip": getattr(nic, "network_i_p", None) if nic else None,
                        "vpc_id": network_url.rsplit("/", 1)[-1],
                    }
                )
        result["instances"] = seen
        result["count"] = len(seen)
        if not is_sentinel(args.instance_id):
            result["target_instance"] = args.instance_id
            result["found_target"] = any(i["instance_id"] == args.instance_id for i in seen)
        result["success"] = True
    except Exception as exc:
        result["error"] = str(exc)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

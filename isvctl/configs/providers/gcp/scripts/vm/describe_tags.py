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

"""Read Compute Engine instance labels and emit canonical-cased suite tags.

GCE label keys must be lowercase ``[a-z]([-a-z0-9_]*)`` so the stub
projects the lowercase API labels back to the canonical ``Name`` /
``CreatedBy`` suite keys for ``InstanceTagCheck``.

Usage:
    python3 describe_tags.py --instance-id <name> --region <zone>
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
    """Emit the canonical-cased suite tag dict from real GCE labels."""
    parser = argparse.ArgumentParser(description="Describe Compute Engine instance labels")
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
        "tags": {},
        "tag_count": 0,
    }

    try:
        from google.cloud import compute_v1

        client = compute_v1.InstancesClient()
        inst = client.get(project=project, zone=zone, instance=args.instance_id)
        labels = dict(inst.labels or {})

        # Project lowercase GCE labels back to canonical suite keys. The
        # underlying values are real (the labels exist on the instance);
        # only the canonical-casing keys are reconstructed for the suite
        # InstanceTagCheck.
        canonical: dict[str, str] = {}
        if "name" in labels:
            canonical["Name"] = labels["name"]
        if "createdby" in labels:
            canonical["CreatedBy"] = labels["createdby"]
        # Pass any additional labels through with original (lowercase) keys
        # so the JSON still carries them as evidence.
        for key, value in labels.items():
            if key not in ("name", "createdby"):
                canonical.setdefault(key, value)

        result["tags"] = canonical
        result["tag_count"] = len(canonical)
        result["raw_labels"] = labels
        result["success"] = True
    except Exception as exc:
        result["error"] = str(exc)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

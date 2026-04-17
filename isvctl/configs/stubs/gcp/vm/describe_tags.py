#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Retrieve labels on a Compute Engine instance.

GCP's closest analogue to AWS user-defined tags is ``labels`` — key/value
pairs attached at instance create and mutable with ``set_labels``. Keys
are restricted to lowercase letters, digits, ``_`` and ``-``, which is
why the provider config overrides ``InstanceTagCheck.required_keys`` to
``[name, created-by]`` instead of the oracle's ``[Name, CreatedBy]``.

Usage:
    python describe_tags.py --instance-id isv-test-gpu --region asia-east1-a

Output JSON (per docs/existing-patterns/vm.md):
{
    "success": true,
    "platform": "vm",
    "instance_id": "...",
    "tags": {"name": "...", "created-by": "..."},
    "tag_count": 2
}
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import describe_instance, resolve_project
from common.errors import handle_gcp_errors
from google.cloud import compute_v1


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Describe Compute Engine instance labels")
    parser.add_argument("--instance-id", required=True, help="Instance name")
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_ZONE", "asia-east1-a"),
        help="GCP zone (the provider config's 'region' is a zone)",
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
        "tags": {},
        "tag_count": 0,
    }

    try:
        instance = describe_instance(client, project, zone, args.instance_id)
        # compute_v1.Instance.labels is a mapping-like proto; materialise to a plain dict.
        labels = dict(instance.labels or {})
        result["tags"] = labels
        result["tag_count"] = len(labels)
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

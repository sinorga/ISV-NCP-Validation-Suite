#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine NVLink domain metadata probe (NvlinkDomainCheck).

Divergences from the AWS oracle:

  * AWS does not ship an nvlink_domain step. The validator was added
    after the AWS-shaped suite; non-AWS NCPs must adapt.
  * Compute Engine does NOT expose an NVLink-domain identifier via
    public APIs. The honest portable probe is:
      1. Resolve the node by --node-id via InstancesClient
         aggregated-list across the project's zones (the operator's
         --node-id is a Compute Engine instance name, which is regional
         in our domain config but globally unique via the zone scope).
         An unresolved node-id is a hard failure: the validator cannot
         skip on absence-of-hardware because absence-of-VM is a separate
         outcome from non-NVLink-shape. Emit node_resolved.passed=False.
      2. Read `guestAccelerators` on the resolved instance.
      3. Emit `nvlink_supported=false` ONLY when the resolved instance
         has no NVLink-capable accelerator type attached. The validator
         then surfaces an explicit pytest.skip for the non-NVLink shape.
      4. When NVLink IS attached, a `nvidia-smi topo` guest probe is
         required to populate `nvlink_domain_id`. This stub does NOT
         SSH into the guest, so the NVLink-capable path fails
         nvlink_support_detected with a clear "no provider-side
         topology probe; guest probe required" message. Do NOT force
         nvlink_supported=false to silently route through pytest.skip
         — knowledge: "do not invent an ID from machine type or zone"
         AND "Emit nvlink_supported=false ONLY from real evidence that
         the resolved node is non-NVLink."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import resolve_project
from common.errors import classify_gcp_error, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

# NVLink-capable Compute Engine accelerator types (interconnect topology
# exposes peer NVLink bridges within an HGX system). Source:
# https://cloud.google.com/compute/docs/gpus — A100/H100/H200/B200 ship
# with NVLink fabric on HGX baseboards.
NVLINK_ACCELERATOR_FAMILIES = (
    "nvidia-h100",
    "nvidia-h200",
    "nvidia-a100",
    "nvidia-b200",
)


def _accelerator_type_short(selflink: str) -> str:
    return selflink.rsplit("/", 1)[-1] if selflink else ""


def _is_nvlink_capable(accel_type: str) -> bool:
    return any(accel_type.startswith(family) for family in NVLINK_ACCELERATOR_FAMILIES)


def _resolve_instance(project: str, node_id: str) -> compute_v1.Instance | None:
    """Find a Compute Engine instance named ``node_id`` across all zones.

    Returns the first match found via aggregated-list, or None when no
    instance with that name exists in the project. The aggregated-list
    call is read-only and cheap; it is the canonical pattern when the
    caller knows only the instance name, not the zone.
    """
    client = compute_v1.InstancesClient()
    try:
        for _zone, scoped in client.aggregated_list(project=project):
            for inst in getattr(scoped, "instances", ()) or ():
                if inst.name == node_id:
                    return inst
    except gax.GoogleAPICallError:
        return None
    return None


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine NVLink domain metadata probe")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "test_name": "nvlink_domain",
        "region": args.region,
        "node_id": args.node_id,
        "nvlink_supported": False,
        "tests": {
            "node_resolved": {"passed": False, "node_id": args.node_id},
            "nvlink_support_detected": {"passed": False},
            "nvlink_domain_id_present": {"passed": False},
        },
    }

    try:
        instance = _resolve_instance(project, args.node_id)

        if instance is None:
            # Unresolved node-id is a hard failure: absence-of-VM is a
            # different outcome from non-NVLink-shape, and the validator's
            # pytest.skip on nvlink_supported=False is reserved for the
            # latter. Force node_resolved.passed=False so the operator sees
            # the missing-node signal explicitly, not a silent skip.
            result["tests"]["node_resolved"] = {
                "passed": False,
                "node_id": args.node_id,
                "found": False,
                "message": (
                    f"no Compute Engine instance named {args.node_id!r} found in project; "
                    "set --node-id to a live instance name"
                ),
            }
            result["tests"]["nvlink_support_detected"] = {
                "passed": False,
                "message": "node_resolved failed; NVLink support cannot be detected without a resolved instance",
            }
        else:
            result["tests"]["node_resolved"] = {
                "passed": True,
                "node_id": args.node_id,
                "found": True,
            }
            accelerator_types = [
                _accelerator_type_short(a.accelerator_type)
                for a in (instance.guest_accelerators or ())
                if a.accelerator_type
            ]
            nvlink_capable = any(_is_nvlink_capable(t) for t in accelerator_types)
            if nvlink_capable:
                # NVLink hardware attached — the validator requires a real
                # `nvlink_domain_id`. The public Compute Engine API does not
                # expose one; populating it requires an in-guest probe
                # (e.g., SSH + `nvidia-smi topo -m`) that this metadata-only
                # stub does NOT perform. Per knowledge: do not invent an ID
                # from machine type or zone, and do not emit
                # nvlink_supported=false to silently route through
                # pytest.skip on hardware that IS NVLink-capable. Fail the
                # detection subtest with a clear message so the gap is
                # explicit, not masked.
                result["tests"]["nvlink_support_detected"] = {
                    "passed": False,
                    "accelerators": accelerator_types,
                    "nvlink_capable_family_present": True,
                    "message": (
                        "NVLink-capable accelerators detected but no provider-side "
                        "NVLink-domain API and no guest topology probe is implemented; "
                        "an `nvidia-smi topo` SSH probe is required to emit a real "
                        "nvlink_domain_id (do not invent from machine type or zone)"
                    ),
                }
            else:
                # No NVLink-capable accelerator on a resolved live instance —
                # the canonical non-NVLink skip path. Validator emits
                # pytest.skip on nvlink_supported=False.
                result["tests"]["nvlink_support_detected"] = {
                    "passed": True,
                    "accelerators": accelerator_types,
                    "nvlink_capable_family_present": False,
                }
                result["nvlink_supported"] = False

        # tests.nvlink_domain_id_present stays False (default) — only meaningful
        # when nvlink_supported=True with a verified guest probe.
        result["success"] = (
            result["tests"]["node_resolved"]["passed"] and result["tests"]["nvlink_support_detected"]["passed"]
        )
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

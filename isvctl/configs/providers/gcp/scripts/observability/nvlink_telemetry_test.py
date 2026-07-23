#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GCP NVLink telemetry availability probes (observability test phase).

The GCP port of the AWS oracle ``nvlink_telemetry_test.py``. Compute Engine
exposes guest GPU metrics (Ops Agent / NVML / DCGM) for supported accelerator
VMs, but there is no general customer API for the provider-owned NVLink switch
fabric plane, so both aspects use the validator's all-subtests PROVIDER-HIDDEN
path after a real ``ProjectsClient.get_project`` identity probe:

  * ``gpu_nvlink_telemetry``    — link_metrics / links_checked
  * ``switch_nvlink_telemetry`` — port_metrics / ports_checked

Guest DCGM/NVML data is never relabeled as provider NVLink-fabric telemetry. Both
validators are enabled: the provider-hidden pass is gated on the real project
identity probe, and every required subtest is emitted passed=true +
provider_hidden=true (guest DCGM/NVML for VMs that actually expose NVLink is the
concrete alternative when the selected GPU topology has NVLink).

AWS reference implementation:
    ../../aws/scripts/observability/nvlink_telemetry_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

from common.compute import resolve_project
from common.errors import classify_gcp_error, handle_gcp_errors
from common.telemetry import probe_project_identity

ASPECT_TESTS: dict[str, list[str]] = {
    "gpu_nvlink_telemetry": ["telemetry_endpoint_reachable", "link_metrics_present", "samples_recent"],
    "switch_nvlink_telemetry": ["telemetry_endpoint_reachable", "port_metrics_present", "samples_recent"],
}

# The count-probe field each aspect's validator expects in provider-hidden evidence.
HIDDEN_ASPECT_PROBE_FIELDS: dict[str, str] = {
    "gpu_nvlink_telemetry": "links_checked",
    "switch_nvlink_telemetry": "ports_checked",
}

GCP_NO_CUSTOMER_NVLINK_MESSAGE = (
    "Compute Engine exposes guest GPU metrics for accelerator VMs but no customer API for the "
    "provider-owned NVLink switch fabric plane"
)


def _base_result(aspect: str) -> dict[str, Any]:
    """Build the common observability result envelope."""
    return {
        "success": False,
        "platform": "observability",
        "test_name": aspect,
        "tests": {name: {"passed": False} for name in ASPECT_TESTS[aspect]},
    }


def _failed(error: str, probes: dict[str, Any]) -> dict[str, Any]:
    """Build a failing subtest result."""
    return {"passed": False, "error": error, "probes": probes}


def _provider_hidden(test_name: str, *, probe_field: str, project: str) -> dict[str, Any]:
    """Build a passing provider-hidden subtest result."""
    return {
        "passed": True,
        "provider_hidden": True,
        "probes": {probe_field: 0, "telemetry_source": "", "metric_names": []},
        "message": f"{test_name}: {GCP_NO_CUSTOMER_NVLINK_MESSAGE} (project {project} reachable).",
    }


def check_hidden_nvlink_telemetry(project: str, *, aspect: str) -> dict[str, Any]:
    """Emit provider-hidden NVLink evidence after a real project identity probe."""
    result = _base_result(aspect)
    probe_field = HIDDEN_ASPECT_PROBE_FIELDS[aspect]
    try:
        project_id = probe_project_identity(project)
    except Exception as e:
        error_type, error_msg = classify_gcp_error(e)
        result["error_type"] = error_type
        result["error"] = error_msg
        probes = {probe_field: 0, "telemetry_source": "", "metric_names": []}
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(f"GCP project identity probe failed: {error_msg}", probes)
        return result

    result["tests"] = {
        name: _provider_hidden(name, probe_field=probe_field, project=project_id) for name in ASPECT_TESTS[aspect]
    }
    result["success"] = True
    return result


@handle_gcp_errors
def main() -> int:
    """Run the selected GCP NVLink telemetry probe and emit structured JSON."""
    parser = argparse.ArgumentParser(description="GCP NVLink telemetry availability test")
    parser.add_argument("--region", default="us-central1", help="GCP region (contextual)")
    parser.add_argument("--aspect", required=True, choices=sorted(ASPECT_TESTS))
    parser.add_argument("--project", default=None, help="GCP project ID (ADC fallback)")
    args = parser.parse_args()

    result = check_hidden_nvlink_telemetry(resolve_project(args.project), aspect=args.aspect)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

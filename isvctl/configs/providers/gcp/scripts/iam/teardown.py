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

"""GCP IAM reclamation of run-owned service accounts (optional ISV-facing scaffold).

Counterpart to the AWS reference `teardown.sh`, for the leftovers an interrupted
run recorded but could not clean up itself. This script is NOT invoked by
suites/iam.yaml — the wired teardown-phase step is `delete_user`, which deletes
the specific service account `create_user` made plus the same candidate handoff
this helper consumes.

Cleanup scope is EXPLICIT, never inferred: the only accounts this helper can
touch are the packed candidates `create_user` emitted in
``unreconciled_resources``, and each one is deleted only after its recorded
per-invocation marker is re-verified on that exact account. A name prefix is not
ownership — a sweep over "everything that looks like ours" deletes a concurrent
run's identity (or an operator's same-named account) with no way to tell the
difference — so no name-shaped sweep exists here.

A marker mismatch is another run's account and is never deleted; a lookup that
stays denied or failing deletes nothing either and fails honestly so the operator
can replay the reclamation. When the project / Application Default Credentials do
not resolve, the recorded candidates CANNOT be verified, so the run exits nonzero
rather than reporting a clean cleanup it never performed. ``--skip-destroy``
returns success without deleting anything.

Usage:
    python3 teardown.py --project=my-project \\
      --unreconciled-resources "$(jq -r '.unreconciled_resources | join(",")' create_user.json)"

Output JSON:
{
    "success": true,
    "platform": "iam",
    "resources_deleted": ["isv-test-user-...@my-project.iam.gserviceaccount.com"],
    "message": "Reclaimed 1 of 1 recorded candidate(s)"
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

from common.compute import resolve_project
from common.errors import classify_gcp_error, handle_gcp_errors
from common.ownership import parse_unreconciled_records
from common.service_account import reclaim_unreconciled_service_accounts


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="GCP IAM candidate reclamation (optional scaffold)")
    parser.add_argument("--project", default="", help="GCP project (falls back to env/ADC when blank)")
    parser.add_argument(
        "--unreconciled-resources",
        default="",
        help=(
            "Comma-joined packed candidates from create_user's unreconciled_resources "
            "(kind|name|project|zone|invocation); only marker-verified accounts are deleted"
        ),
    )
    parser.add_argument("--skip-destroy", action="store_true", help="Skip actual destroy")
    args = parser.parse_args()

    result: dict = {
        "success": True,
        "platform": "iam",
        "resources_deleted": [],
        "message": "",
    }

    candidates = parse_unreconciled_records(args.unreconciled_resources)

    if args.skip_destroy:
        result["message"] = "Cleanup skipped (--skip-destroy flag)"
        print(json.dumps(result, indent=2))
        return 0

    if not candidates:
        # Nothing was handed off, so there is nothing this helper is entitled to
        # delete. It does not go looking for more.
        result["message"] = "No recorded candidates supplied; nothing to reclaim"
        print(json.dumps(result, indent=2))
        return 0

    try:
        project = resolve_project(args.project or None)
    except Exception as e:
        # Recorded candidates exist and cannot be verified without a project /
        # credentials, so this is a BLOCKED cleanup, not a no-op: exit nonzero
        # so the operator retries instead of reading a green run as "clean".
        error_type, error_msg = classify_gcp_error(e)
        result["success"] = False
        result["error_type"] = error_type
        # The bucket token must live in the `error` STRING: only that field is
        # forwarded into the validation verdict.
        result["error"] = f"[bucket={error_type}] cannot reclaim {len(candidates)} recorded candidate(s): {error_msg}"
        result["message"] = "Cleanup blocked: project/credentials did not resolve"
        print(json.dumps(result, indent=2))
        return 1

    reclaimed, warnings, ok = reclaim_unreconciled_service_accounts(candidates, default_project=project)
    result["resources_deleted"] = list(reclaimed)
    if warnings:
        result["warnings"] = warnings

    result["success"] = ok
    result["message"] = f"Reclaimed {len(reclaimed)} of {len(candidates)} recorded candidate(s)"
    if not ok:
        result["error_type"] = "api_error"
        result["error"] = (
            "[bucket=api_error] recorded candidate(s) were not reclaimed "
            "(ownership unproven or delete failed after marker match)"
        )

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

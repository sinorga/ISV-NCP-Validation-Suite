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

"""Delete the test service account (IAM teardown).

In the keyless primary path there is no user-managed key to delete — the
``secret_access_key`` minted by create_user is a short-lived OAuth2 access
token that self-expires — so teardown only deletes the service account.

Deletion is idempotent: a NotFound service account is the desired terminal
state and reports success (the bounded-retry cleanup helper absorbs the
eventual-consistency window). ``--skip-destroy`` returns success without
deleting anything, mirroring the AWS reference.

The service-account email emitted as create_user.username is enough to build
``projects/-/serviceAccounts/<email>`` (the ``-`` project wildcard), so
``--project`` is accepted for provenance but not required for the delete.

``--unreconciled-resources`` carries create_user's ambiguous-create candidates:
accounts that may have been committed before the response was lost, whose
ownership its readback could not settle. They arrive WITHOUT a
``service_account_created`` bit because none could honestly be set, and they
arrive exactly when ``--username`` is the ``none`` sentinel (create_user failed,
so it emitted no identity) — so this is the only channel that keeps the handoff
alive. Each candidate's per-invocation marker is re-verified on the exact
account before anything is deleted: a marker mismatch is another run's account
and is NEVER deleted, and a lookup that stays denied or failing deletes nothing
either and fails the teardown honestly.

Usage:
    python3 delete_user.py --username sa@proj.iam.gserviceaccount.com --project=proj

Output JSON:
{
    "success": true,
    "platform": "iam",
    "resources_destroyed": true,
    "resources_deleted": ["service_account:sa@proj.iam.gserviceaccount.com"],
    "deleted": {"service_account": "sa@proj.iam.gserviceaccount.com"},
    "message": "Service account deleted (or already absent)"
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

from common.errors import handle_gcp_errors
from common.ownership import parse_unreconciled_records
from common.service_account import delete_service_account, reclaim_unreconciled_service_accounts

# Sentinels the config uses so an unresolved template keeps its argv pair intact
# instead of collapsing it and shifting the next flag into the value slot.
_FALSY_SENTINELS = frozenset({"", "none", "null", "false"})


def _truthy(arg: str | None) -> bool:
    """Per-arg sentinel check: '' / 'none' / 'null' / 'false' mean "not supplied"."""
    if arg is None:
        return False
    return arg.strip().lower() not in _FALSY_SENTINELS


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Delete the test service account (IAM teardown)")
    parser.add_argument(
        "--username",
        required=True,
        help="Service account email from create_user ('none' when create_user recorded no identity)",
    )
    parser.add_argument("--project", default="", help="GCP project (provenance only)")
    parser.add_argument(
        "--unreconciled-resources",
        default="",
        help="Comma-joined packed ambiguous-create candidates from create_user ('none' when there are none)",
    )
    parser.add_argument("--skip-destroy", action="store_true", help="Skip actual destroy")
    args = parser.parse_args()

    result: dict = {
        "success": False,
        "platform": "iam",
        "resources_destroyed": False,
        "resources_deleted": [],
        "deleted": {"service_account": None},
        # Keyless primary path: create_user minted a short-lived OAuth2 access
        # token, so there is no persistent credential for teardown to revoke.
        "credential_cleanup": "not_required_short_lived_token",
    }
    if args.project:
        result["project"] = args.project

    sa_email = args.username if _truthy(args.username) else ""
    candidates = parse_unreconciled_records(args.unreconciled_resources)

    if args.skip_destroy:
        result["success"] = True
        # Preservation suppresses the delete, never the bookkeeping: name the
        # exact identity retained so a later standalone cleanup can reclaim it.
        if sa_email:
            result["resources_preserved"] = [f"service_account:{sa_email}"]
        if candidates:
            # Unproven candidates are part of the fixture state the operator
            # asked to keep; a later marker-verified pass reclaims them.
            result["warnings"] = [
                "unreconciled candidates preserved (--skip-destroy): "
                + ", ".join(candidate.describe() for candidate in candidates)
            ]
        result["message"] = "Destroy skipped (--skip-destroy flag)"
        print(json.dumps(result, indent=2))
        return 0

    messages: list[str] = []
    identity_ok = True
    if sa_email:
        # delete_service_account returns True when the SA was deleted now OR is
        # already absent (NotFound), and False only on a persistent transient
        # failure past the retry budget — fold that bool into success so a genuine
        # leak surfaces rather than being silently swallowed.
        if delete_service_account(sa_email, project=args.project or None):
            result["deleted"]["service_account"] = sa_email
            result["resources_deleted"].append(f"service_account:{sa_email}")
            result["resources_destroyed"] = True
            messages.append("Service account deleted (or already absent)")
        else:
            identity_ok = False
            result["error_type"] = "api_error"
            # The bucket token must live in the `error` STRING: only that field is
            # forwarded into the validation verdict.
            result["error"] = (
                f"[bucket=api_error] failed to delete service account {sa_email} after bounded retry "
                "(delete denied and project inventory did not prove absence)"
            )
            messages.append("Service account deletion failed")
    else:
        # create_user emitted no identity (it failed before, or during, a create
        # that committed nothing it could name). Any account that may still have
        # been committed arrives as a marker-carrying candidate below.
        messages.append("No run-owned service account recorded; nothing to delete by name")

    # Ownership of an ambiguous-create candidate is decided HERE, by re-verifying
    # the recorded invocation marker on the exact account. Only a marker match is
    # deleted; a mismatch belongs to another run and is preserved.
    reclaimed, warnings, candidates_ok = reclaim_unreconciled_service_accounts(candidates, default_project=args.project)
    for email in reclaimed:
        result["resources_deleted"].append(f"service_account:{email}")
        result["resources_destroyed"] = True
    if reclaimed:
        messages.append(f"Reclaimed {len(reclaimed)} unreconciled candidate(s) after marker verification")
    if warnings:
        result["warnings"] = warnings

    result["success"] = identity_ok and candidates_ok
    if not candidates_ok and "error" not in result:
        result["error_type"] = "api_error"
        result["error"] = (
            "[bucket=api_error] unreconciled service-account candidate(s) could not be reclaimed "
            "(ownership unproven or delete failed after marker match)"
        )
        messages.append("Unreconciled candidate cleanup incomplete")
    result["message"] = "; ".join(messages)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

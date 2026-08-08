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

"""Create a GCP service account + short-lived impersonation token (IAM create_user).

Google Cloud has no human IAM users; the managed application principal is a
service account. This stub maps the suite's AWS-shaped create_user contract
onto a service account:

  1. Create a uniquely-named service account in the resolved project.
  2. Grant the ADC principal roles/iam.serviceAccountTokenCreator on the new
     service account's resource policy.
  3. Mint a short-lived (600s) OAuth2 access token for the service account via
     IAMCredentials.generateAccessToken, retrying while the tokenCreator
     binding propagates (eventually-consistent, observed up to ~180s on
     hardened orgs -> 12 x 15s budget).

The AWS-shaped output names are preserved for contract compatibility while
their GCP meaning is documented: ``access_key_id`` is the service account
unique_id (NON-SECRET; equals tokeninfo.azp) and ``secret_access_key`` is the
short-lived OAuth2 access token (sensitive, self-expiring — no key file).

Cleanup ownership is taken from the ACCEPTED create, not from a well-formed
response body: ``create_service_account_resource`` stamps a per-invocation
marker and transfers ownership either on acknowledgement or after reading the
exact candidate identity back with that marker. A create whose response is lost
in transit (or whose retry draws a 409 for this invocation's own candidate)
therefore still lands in the cleanup set instead of leaking a project-level
service account. When the reconciling readback itself is denied or keeps
failing, the candidate is NOT adopted for deletion — it is emitted as a packed
``unreconciled_resources`` record (kind|email|project|zone|invocation marker)
that the config forwards to the teardown step, which re-verifies that exact
marker before deleting anything. That is the fail-closed half of the same
contract: an email alone names a resource but proves nothing about who owns it,
so the marker travels with it.

If the tokenCreator binding or the token mint fails after the service account
exists, the identity this step created is rolled back (and its absence
confirmed) before the failure is returned. ``--skip-destroy`` (wired from the
same operator preservation flag the teardown step reads) suppresses that
compensating delete so a failed run can be debugged, and reports the exact
identity it retained — preservation never suppresses the ownership bookkeeping.

Usage:
    python3 create_user.py --username isv-test-user --create-access-key --project=my-project

Output JSON:
{
    "success": true,
    "platform": "iam",
    "username": "isv-test-user-ab12-cd34ef56@my-project.iam.gserviceaccount.com",
    "user_id": "1234567890",
    "service_account_name": "projects/my-project/serviceAccounts/...",
    "service_account_created": true,
    "access_key_id": "1234567890",
    "secret_access_key": "ya29...",
    "token_expiry": "2026-06-05T12:34:56+00:00",
    "project": "my-project",
    "unreconciled_resources": []
}
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

from common.compute import resolve_project, unique_suffix
from common.errors import (
    TRANSIENT_EXCEPTIONS,
    classify_gcp_error,
    handle_gcp_errors,
    modify_iam_policy_with_retry,
)
from common.ownership import CREATED_BY_DESCRIPTION, UnreconciledCandidate
from common.service_account import (
    create_service_account_resource,
    delete_service_account,
    resolve_principal_member,
    service_account_absent,
)
from google.api_core import exceptions as gax
from google.cloud import iam_admin_v1, iam_credentials_v1
from google.iam.v1 import iam_policy_pb2, policy_pb2
from google.protobuf import duration_pb2

# tokenCreator binding -> token mint propagation budget. A freshly-granted
# serviceAccountTokenCreator binding is not effective for generateAccessToken
# immediately; hardened orgs have required up to ~180s. 12 x 15s = 180s.
TOKEN_MINT_MAX_RETRIES = 12
TOKEN_MINT_RETRY_DELAY_SECONDS = 15
TOKEN_LIFETIME_SECONDS = 600

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_TOKEN_CREATOR_ROLE = "roles/iam.serviceAccountTokenCreator"

# GCP service-account ID local-part: 6-30 chars, must start with a lowercase
# letter, then lowercase letters / digits / hyphens.
_SA_ID_MAX_LEN = 30

# generateAccessToken propagation shapes worth retrying: the tokenCreator
# binding is not yet effective (PermissionDenied), the service account is not
# yet visible (NotFound), or a transient backend error.
_RETRYABLE_MINT: tuple[type[Exception], ...] = (
    gax.PermissionDenied,
    gax.NotFound,
    *TRANSIENT_EXCEPTIONS,
)


def _service_account_id(username: str) -> str:
    """Normalize ``username`` into a unique, valid service-account ID (<=30 chars).

    GCP service-account IDs are lowercase RFC1035-ish identifiers (6-30 chars,
    must start with a letter). The 30-char cap means a run-id suffix alone is
    not collision-safe under truncation (a wider prefix family collapses to the
    same wire identifier), so fold a per-invocation discriminator BEFORE the
    run-id suffix and truncate the base after reserving room for both.
    """
    base = re.sub(r"[^a-z0-9-]+", "-", username.lower()).strip("-")
    if not base or not base[0].isalpha():
        base = f"isv-{base}".strip("-")
    run_suffix = unique_suffix("x", length=8).split("-", 1)[-1]  # 8 hex chars
    disc = secrets.token_hex(2)  # 4 hex chars, fresh per invocation
    # Reserve "-<disc>-<run_suffix>" so the assembled ID fits in 30 chars.
    reserve = len(disc) + len(run_suffix) + 2
    base = base[: _SA_ID_MAX_LEN - reserve].rstrip("-") or "isv"
    return f"{base}-{disc}-{run_suffix}"[:_SA_ID_MAX_LEN]


def _predicted_sa_email(project: str, account_id: str) -> str:
    """Return the deterministic email of the SA this invocation attempts to create.

    GCP derives a new service account's email from its account id and project:
    ``<account_id>@<project>.iam.gserviceaccount.com``. When an ambiguous create
    loses its response before returning the ServiceAccount, this predicted email
    is the exact, invocation-specific coordinate cleanup and the reconciling
    readback address — the account id embeds a per-invocation random
    discriminator (see :func:`_service_account_id`), so no same-run foreign actor
    shares it.
    """
    return f"{account_id}@{project}.iam.gserviceaccount.com"


def _rollback_service_account(project: str, sa_email: str) -> list[str]:
    """Best-effort delete the SA this invocation created and confirm it is gone.

    Returns a list of rollback errors (empty only on a CONFIRMED-absent
    rollback). ``delete_service_account`` fails closed on an existence-hiding
    403 unless project inventory proves absence; this second project-scoped
    readback additionally confirms an acknowledged delete converged. Only True
    proves the SA is gone — False (still present) and None (inventory
    unreadable) keep the identity as a teardown handoff rather than silently
    orphaning a project-level account.
    """
    rollback_errors: list[str] = []
    if delete_service_account(sa_email, project=project):
        absent = service_account_absent(project, sa_email)
        if absent is False:
            rollback_errors.append(f"rollback service account {sa_email} still present after delete")
        elif absent is None:
            rollback_errors.append(f"rollback service account {sa_email} deletion unconfirmed (SA list unreadable)")
    else:
        rollback_errors.append(f"rollback delete service account {sa_email} failed")
    return rollback_errors


def _grant_token_creator(iam: iam_admin_v1.IAMClient, sa_email: str) -> None:
    """Grant the ADC principal tokenCreator on the new SA's resource policy.

    The member comes from ``resolve_principal_member``, which resolves the
    CANONICAL tokeninfo email of the active ADC principal and derives the
    ``user:`` / ``serviceAccount:`` prefix from that email's shape. Never take it
    from an SDK credential attribute: metadata-server ADC reports the alias
    ``default`` and federated credentials report nothing, so either would bind a
    member this account's policy does not actually name.

    Read-modify-write the service account's own IAM policy (carrying its etag)
    rather than overwriting it, so the binding is added without clobbering any
    policy the create call may have seeded. The read-modify-write is wrapped in
    a bounded retry that re-reads on each attempt (fresh etag), matching the
    sibling token-mint path's transient-shape resilience: a 5xx / 429 or a
    stale-etag 409 on the bind otherwise aborts the whole create + cleanup +
    re-provision cycle.
    """
    member = resolve_principal_member()
    resource = f"projects/-/serviceAccounts/{sa_email}"

    def _read() -> Any:
        return iam.get_iam_policy(request=iam_policy_pb2.GetIamPolicyRequest(resource=resource))

    def _mutate(policy: Any) -> None:
        policy.bindings.append(policy_pb2.Binding(role=_TOKEN_CREATOR_ROLE, members=[member]))

    def _write(policy: Any) -> None:
        iam.set_iam_policy(request=iam_policy_pb2.SetIamPolicyRequest(resource=resource, policy=policy))

    modify_iam_policy_with_retry(_read, _write, _mutate, resource_desc=f"service account {sa_email}")


def _mint_access_token(sa_email: str) -> tuple[str, str]:
    """Mint a short-lived impersonation token, retrying while the binding propagates.

    Returns ``(access_token, rfc3339_expiry)``. The retry envelope lives here
    (not in test_credentials) because the tokenCreator-binding propagation
    happens during create-time impersonation. Only the propagation /
    transient shapes are retried; any other error re-raises immediately so a
    malformed request fails fast.
    """
    creds_client = iam_credentials_v1.IAMCredentialsClient()
    name = f"projects/-/serviceAccounts/{sa_email}"
    lifetime = duration_pb2.Duration(seconds=TOKEN_LIFETIME_SECONDS)

    last_error: str | None = None
    last_exception: Exception | None = None
    for attempt in range(1, TOKEN_MINT_MAX_RETRIES + 1):
        try:
            response = creds_client.generate_access_token(
                name=name,
                scope=[_CLOUD_PLATFORM_SCOPE],
                lifetime=lifetime,
            )
        except _RETRYABLE_MINT as e:
            last_exception = e
            last_error = f"{type(e).__name__}: {e}"
            if attempt < TOKEN_MINT_MAX_RETRIES:
                print(
                    f"  generate_access_token attempt {attempt}/{TOKEN_MINT_MAX_RETRIES} "
                    f"({type(e).__name__}); tokenCreator binding propagating, "
                    f"retrying in {TOKEN_MINT_RETRY_DELAY_SECONDS}s",
                    file=sys.stderr,
                )
                time.sleep(TOKEN_MINT_RETRY_DELAY_SECONDS)
                continue
            break
        # proto-plus exposes the Timestamp field as a datetime at runtime;
        # widen to Any so the isoformat call type-checks against the proto stub.
        expiry: Any = response.expire_time
        rfc3339 = expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry)
        return response.access_token, rfc3339

    budget = TOKEN_MINT_MAX_RETRIES * TOKEN_MINT_RETRY_DELAY_SECONDS
    msg = f"generate_access_token did not converge within {budget}s: {last_error}"
    # Chain the last typed failure: `classify_gcp_error` walks the cause chain,
    # so the exhausted mint reports the disposition that actually dominated
    # (access_denied while the binding propagates, transient on a backend blip)
    # instead of collapsing to unknown_error.
    raise RuntimeError(msg) from last_exception


def _emit_failure(
    result: dict[str, Any],
    error: Exception,
    *,
    cleanup_errors: list[str] | None = None,
    preserved: str = "",
) -> int:
    """Print the structured failure payload and return the stub exit code.

    ``classify_gcp_error`` embeds the canonical ``[bucket=<name>]`` token in the
    message, which is the only channel that survives into the validation verdict
    (the validator reads ``error``, not ``error_type``).

    ``preserved`` names an identity this step created and deliberately RETAINED
    because the operator asked for preservation; ``cleanup_errors`` reports the
    outcome of a rollback that did run. Candidates whose ownership could not be
    proven are already in ``result["unreconciled_resources"]`` (stamped by the
    create path's handoff) — they are neither deleted nor dropped, and travel to
    teardown packed with the invocation marker that decides their fate.
    """
    bucket, message = classify_gcp_error(error)
    result["error_type"] = bucket
    result["error"] = message
    result["success"] = False
    if preserved:
        # Preservation suppresses the delete, never the ownership bookkeeping:
        # the exact identity stays in the payload so a later standalone cleanup
        # can reclaim it.
        result["cleanup"] = {"service_account_deleted": False, "preserved": True}
        result["resources_preserved"] = [f"service_account:{preserved}"]
    elif cleanup_errors is not None:
        result["cleanup"] = {"service_account_deleted": not cleanup_errors}
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
    print(json.dumps(result, indent=2))
    return 1


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Create a GCP service account + impersonation token")
    parser.add_argument("--username", default="isv-test-user", help="Service account ID base")
    parser.add_argument(
        "--create-access-key",
        action="store_true",
        default=False,
        help="Mint short-lived credential material for test_credentials",
    )
    parser.add_argument("--project", default="", help="GCP project (falls back to env/ADC when blank)")
    parser.add_argument(
        "--skip-destroy",
        action="store_true",
        help="Preserve a partially-created service account instead of rolling it back on failure",
    )
    args = parser.parse_args()

    project = resolve_project(args.project or None)
    account_id = _service_account_id(args.username)
    predicted_email = _predicted_sa_email(project, account_id)

    result: dict[str, Any] = {
        "success": False,
        "platform": "iam",
        "project": project,
        "service_account_created": False,
        # Ambiguous-create candidates whose ownership could NOT be decided: the
        # create may have committed, but the reconciling readback was denied or
        # kept failing, so neither "ours" nor "not ours" is proven. Nothing is
        # deleted on that unproven claim; each packed record carries
        # kind|email|project|zone|invocation so the teardown step can re-verify
        # the marker itself and delete ONLY what still echoes this invocation.
        # Always emitted (empty on the happy path) so the config's forwarding
        # template never dangles.
        "unreconciled_resources": [],
    }

    sa_owned = False

    def _record_sa_acceptance() -> None:
        nonlocal sa_owned
        sa_owned = True

    def _retain_unreconciled(candidate: UnreconciledCandidate, error: Exception, detail: str) -> None:
        """Persist one candidate whose ownership the create path could not prove."""
        record = candidate.pack()
        if record not in result["unreconciled_resources"]:
            result["unreconciled_resources"].append(record)
        print(
            f"  {candidate.describe()} ownership unproven after ambiguous create "
            f"({error.__class__.__name__}: {detail}); retaining cleanup handoff for teardown",
            file=sys.stderr,
        )

    # 1. Create the run-owned service account. The create is non-idempotent, so
    #    ownership comes from the shared accepted-create envelope (marker stamp
    #    + exact-identity readback), never from the presence of a response body.
    try:
        created = create_service_account_resource(
            project,
            account_id,
            display_name="ISV IAM lifecycle validation",
            description=f"IAM suite create_user identity ({CREATED_BY_DESCRIPTION}).",
            on_accepted=_record_sa_acceptance,
            on_unreconciled=_retain_unreconciled,
        )
    except Exception as e:
        if sa_owned:
            # Committed before the response was lost: this invocation owns the
            # identity. Preservation keeps it for debugging; otherwise reclaim
            # it, and an unconfirmed rollback keeps the identity in the payload
            # as the teardown handoff.
            result["service_account_created"] = True
            result["username"] = predicted_email
            result["service_account_name"] = f"projects/{project}/serviceAccounts/{predicted_email}"
            if args.skip_destroy:
                return _emit_failure(result, e, preserved=predicted_email)
            return _emit_failure(result, e, cleanup_errors=_rollback_service_account(project, predicted_email))
        # Ownership was decided inside the create envelope against the exact
        # invocation marker: a conclusive pre-commit refusal, a foreign account,
        # and a proven absence all leave nothing behind, while an unprovable
        # readback already handed the packed candidate to _retain_unreconciled.
        # A project-scoped inventory is deliberately NOT consulted here — it can
        # only answer "an account with this name exists", which is the question
        # the marker, not the name, has to settle.
        return _emit_failure(result, e)

    result["service_account_created"] = True
    result["username"] = created.email
    result["user_id"] = created.unique_id
    result["service_account_name"] = created.name
    # NON-SECRET: the service account unique_id. test_credentials compares this
    # to tokeninfo.azp to prove the token is the expected identity.
    result["access_key_id"] = created.unique_id

    # 2. Mint credential material. Any failure after the identity exists rolls
    #    the identity back so a partial failure does not leak a project-level
    #    service account — unless the operator asked to preserve the fixture,
    #    which suppresses the delete but never the ownership bookkeeping.
    if args.create_access_key:
        try:
            iam = iam_admin_v1.IAMClient()
            _grant_token_creator(iam, created.email)
            token, token_expiry = _mint_access_token(created.email)
        except Exception as e:
            if args.skip_destroy:
                return _emit_failure(result, e, preserved=created.email)
            return _emit_failure(result, e, cleanup_errors=_rollback_service_account(project, created.email))
        result["secret_access_key"] = token
        result["token_expiry"] = token_expiry

    result["success"] = True
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

"""Verify a GCP service-account impersonation token authenticates (IAM test_credentials).

The AWS reference builds a boto3 session from the created access key and proves
it authenticates via STS GetCallerIdentity. The GCP analog of "these
credentials work" for a short-lived impersonation token is:

  1. Call the OAuth2 tokeninfo endpoint for the token. Require
     ``tokeninfo.azp == --credential-id`` (the service account unique_id
     emitted as create_user.access_key_id) and ``expires_in > 0``. This proves
     the token is live AND belongs to the expected identity.
  2. Build google.oauth2.credentials.Credentials(token=...) and call
     IAMClient.get_service_account for the expected service account. Success is
     authenticated access; PermissionDenied still proves the token
     authenticated but lacks that specific permission (limited permissions are
     not a failure). Unauthenticated / token expiry / transport errors are
     credential failures.

``account_id`` is derived from the project segment of the service-account email
(or the --project argument), preserving the AWS-shaped field name the suite's
``credentials`` validation requires.

Both probes retry only the transient / propagation shapes, and both draw from a
single wall-clock budget stamped at entry. The provider config caps this step at
120s with ``subprocess.run(timeout=...)``, which SIGKILLs the child — no
``except`` runs and no JSON is printed, so a validation would see no signal at
all. Deriving every retry from what is LEFT of the budget keeps the structured
failure payload reachable on the slowest path.

The access token must never be printed to stderr or diagnostic logs.

Usage:
    python3 test_credentials.py --username sa@proj.iam.gserviceaccount.com \\
        --credential-id 1234567890 --credential-secret ya29... --project=proj

Output JSON:
{
    "success": true,
    "platform": "iam",
    "account_id": "my-project",
    "identity_id": "1234567890",
    "tests": {
        "identity": {"passed": true, "identity": "1234567890", "expires_in": 599},
        "access": {"passed": true, "note": "iam_get_self_ok"}
    }
}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

import google.oauth2.credentials
from common.errors import TRANSIENT_EXCEPTIONS, classify_gcp_error, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import iam_admin_v1

_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
_HTTP_TIMEOUT_S = 30

# Wall clock this step allows itself, stamped at entry. The provider config caps
# the step at 120s; reserving ~20s leaves room for the interpreter start, the
# final IAM call already in flight, and printing the payload before SIGKILL.
_STEP_BUDGET_SECONDS = 100

# tokeninfo / IAM-read retry budget. The tokenCreator binding already propagated
# during create_user (it minted the token), so these are short hedges against
# transient backend errors, not the full propagation budget. Every retry is
# additionally gated on the step budget being able to fund the NEXT attempt.
_TOKENINFO_MAX_RETRIES = 3
_TOKENINFO_RETRY_DELAY_SECONDS = 5
_READ_MAX_RETRIES = 4
_READ_RETRY_DELAY_SECONDS = 5
# Worst-case wall cost charged to the budget before starting one more attempt,
# and the shortest window an IAM call is still granted so a nearly-spent budget
# never turns a healthy sub-second read into a fabricated failure.
_IAM_CALL_COST_SECONDS = 20
_MIN_IAM_CALL_SECONDS = 5

# Leading `[bucket=<name>]` token stamped by ``classify_gcp_error``. Nested
# diagnostics keep it; the top-level emitter strips it before re-prefixing so a
# forwarded message can never carry the token twice.
_BUCKET_TOKEN_RE = re.compile(r"^\[bucket=[^\]]*\]\s*")

# tokeninfo HTTP statuses mapped to the shared google.api_core bucket taxonomy
# (common.yaml http_error_handling) so a REST probe and an SDK call reach the
# same disposition vocabulary.
_TOKENINFO_STATUS_BUCKETS: dict[int, str] = {
    400: "credentials_invalid",  # invalid_token
    401: "credentials_invalid",
    403: "access_denied",
    404: "not_found",
    429: "transient",
}

# IAM self-read shapes worth retrying: the service account not yet visible
# (NotFound) or a transient backend error. Authenticated-but-limited
# (PermissionDenied) and Unauthenticated are terminal and handled explicitly.
_RETRYABLE_READ: tuple[type[Exception], ...] = (gax.NotFound, *TRANSIENT_EXCEPTIONS)


def _project_from_email(email: str) -> str:
    """Derive the project ID from a ``<id>@<project>.iam.gserviceaccount.com`` email."""
    if "@" in email:
        host = email.split("@", 1)[1]
        if host.endswith(".iam.gserviceaccount.com"):
            return host.split(".", 1)[0]
    return email


def _sleep_within_budget(deadline: float, delay: float, next_attempt_cost: float) -> bool:
    """Sleep ``delay`` only while the budget can still fund the next attempt.

    Returns False (without sleeping) once the remaining budget cannot cover the
    backoff PLUS the worst case of the attempt it would enable, so the caller
    stops retrying and emits its structured failure instead of being SIGKILLed
    mid-wait with no output.
    """
    if time.monotonic() + delay + next_attempt_cost > deadline:
        return False
    time.sleep(delay)
    return True


def _strip_bucket_token(message: str) -> str:
    """Drop a leading ``[bucket=...]`` token from an already-classified message."""
    return _BUCKET_TOKEN_RE.sub("", message or "")


def _tokeninfo_bucket(error: Exception) -> str:
    """Map a tokeninfo transport/HTTP failure onto a canonical error bucket."""
    if isinstance(error, urllib.error.HTTPError):
        if error.code in _TOKENINFO_STATUS_BUCKETS:
            return _TOKENINFO_STATUS_BUCKETS[error.code]
        return "transient" if error.code >= 500 else "credentials_invalid"
    # URLError / exhausted-retry RuntimeError: the endpoint was unreachable, so
    # the token itself was never judged.
    return "transient"


def _tokeninfo(token: str) -> dict:
    """Fetch OAuth2 tokeninfo for ``token``. Raises on HTTP / transport error."""
    query = urllib.parse.urlencode({"access_token": token})
    request = urllib.request.Request(f"{_TOKENINFO_URL}?{query}", method="GET")
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def _tokeninfo_with_retry(token: str, deadline: float) -> dict:
    """Call tokeninfo, retrying only on transport / 5xx errors (4xx is terminal)."""
    last_error: str | None = None
    for attempt in range(1, _TOKENINFO_MAX_RETRIES + 1):
        try:
            return _tokeninfo(token)
        except urllib.error.HTTPError as e:
            # 4xx (e.g. 400 invalid_token) is a real credential failure; only
            # 5xx is transient and worth retrying.
            if e.code < 500 or attempt >= _TOKENINFO_MAX_RETRIES:
                raise
            last_error = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            if attempt >= _TOKENINFO_MAX_RETRIES:
                raise
            last_error = str(e)
        if not _sleep_within_budget(deadline, _TOKENINFO_RETRY_DELAY_SECONDS, _HTTP_TIMEOUT_S):
            break
    msg = f"tokeninfo did not succeed within the step budget: {last_error}"
    raise RuntimeError(msg)


def _iam_self_read(token: str, sa_email: str, deadline: float) -> tuple[dict[str, Any], str]:
    """Probe IAM get_service_account with the minted token.

    Returns ``(tests.access-shaped dict, error bucket)``. ``passed`` is True when
    the token authenticates (a successful read OR PermissionDenied, which still
    proves authentication with limited permissions) and False for
    Unauthenticated, transport errors, or exhausted transient retries. The
    bucket is empty on a passing probe and carries the canonical disposition
    otherwise, so the caller can embed it in the ``[bucket=<name>]`` token.
    """
    credentials = google.oauth2.credentials.Credentials(token=token)
    client = iam_admin_v1.IAMClient(credentials=credentials)
    name = f"projects/-/serviceAccounts/{sa_email}"

    last_error: str | None = None
    for attempt in range(1, _READ_MAX_RETRIES + 1):
        # Bound the CALL itself by what is left of the budget, not just the
        # backoff between calls: a slow-but-eventually-successful tokeninfo can
        # leave little budget, and an unbounded trailing RPC would then be
        # SIGKILLed at the step cap with no JSON at all.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                "passed": False,
                "note": "budget_exhausted",
                "error": "step budget exhausted before the IAM self-read completed",
            }, "transient"
        call_timeout = max(_MIN_IAM_CALL_SECONDS, min(float(_IAM_CALL_COST_SECONDS), remaining))
        try:
            client.get_service_account(name=name, timeout=call_timeout)
            return {"passed": True, "note": "iam_get_self_ok"}, ""
        except gax.PermissionDenied:
            # Authenticated but lacks iam.serviceAccounts.get — the token is
            # still valid; limited permissions are not a credential failure.
            return {"passed": True, "note": "permission_denied_expected"}, ""
        except gax.Unauthenticated as e:
            bucket, message = classify_gcp_error(e)
            return {"passed": False, "note": "unauthenticated", "error": message}, bucket
        except _RETRYABLE_READ as e:
            bucket, last_error = classify_gcp_error(e)
            if attempt < _READ_MAX_RETRIES and _sleep_within_budget(
                deadline, _READ_RETRY_DELAY_SECONDS, _IAM_CALL_COST_SECONDS
            ):
                continue
            return {"passed": False, "note": "transient_exhausted", "error": last_error}, bucket
        except Exception as e:
            bucket, message = classify_gcp_error(e)
            return {"passed": False, "note": "transport_error", "error": message}, bucket
    return {"passed": False, "note": "transient_exhausted", "error": last_error}, "transient"


def _emit(result: dict[str, Any], *, bucket: str, message: str) -> int:
    """Attach the bucketed failure to ``result``, print it, and return rc=1.

    The validation verdict only forwards the ``error`` STRING, so the canonical
    bucket has to travel inside it as a ``[bucket=<name>]`` token; ``error_type``
    alone never reaches the reader. Messages that were already classified
    upstream are stripped first so the emitted string carries exactly one token.
    """
    result["error_type"] = bucket
    result["error"] = f"[bucket={bucket}] {_strip_bucket_token(message)}"
    result["success"] = False
    print(json.dumps(result, indent=2))
    return 1


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a GCP service-account impersonation token")
    parser.add_argument("--username", required=True, help="Service account email from create_user")
    parser.add_argument("--credential-id", required=True, help="Service account unique_id (== tokeninfo.azp)")
    parser.add_argument("--credential-secret", required=True, help="Short-lived OAuth2 access token")
    parser.add_argument("--project", default="", help="GCP project; falls back to the SA email's project")
    args = parser.parse_args()

    deadline = time.monotonic() + _STEP_BUDGET_SECONDS
    account_id = args.project.strip() or _project_from_email(args.username)
    result: dict[str, Any] = {
        "success": False,
        "platform": "iam",
        "account_id": account_id,
        "tests": {},
    }

    # 1. Prove the token is live for the expected identity via tokeninfo.
    try:
        info = _tokeninfo_with_retry(args.credential_secret, deadline)
    except (urllib.error.URLError, RuntimeError) as e:
        result["tests"]["identity"] = {"passed": False}
        return _emit(result, bucket=_tokeninfo_bucket(e), message=f"tokeninfo failed: {e}")

    azp = str(info.get("azp") or info.get("sub") or "")
    expires_in = int(info.get("expires_in") or 0)
    if azp != args.credential_id:
        result["tests"]["identity"] = {"passed": False, "identity": azp}
        return _emit(
            result,
            bucket="credentials_invalid",
            message=f"tokeninfo azp {azp!r} != expected credential id {args.credential_id!r}",
        )
    if expires_in <= 0:
        result["tests"]["identity"] = {"passed": False, "expires_in": expires_in}
        return _emit(
            result,
            bucket="credentials_invalid",
            message="access token has expired (expires_in <= 0)",
        )
    result["identity_id"] = azp
    result["tests"]["identity"] = {"passed": True, "identity": azp, "expires_in": expires_in}

    # 2. Authenticated self-read with the minted token.
    access, access_bucket = _iam_self_read(args.credential_secret, args.username, deadline)
    result["tests"]["access"] = access
    if not access["passed"]:
        # The nested diagnostic keeps its own token; strip it here so the
        # top-level string is not double-prefixed by _emit.
        detail = _strip_bucket_token(str(access.get("error", "")))
        return _emit(
            result,
            bucket=access_bucket or "unknown_error",
            message=f"IAM self-read probe failed ({access['note']}): {detail}",
        )

    result["success"] = True
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

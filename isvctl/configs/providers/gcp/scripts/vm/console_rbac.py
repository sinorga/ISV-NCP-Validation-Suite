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

"""Console RBAC probe for Compute Engine serial console access.

The suite contract requires three subtests:
  1. denied_principal_cannot_access_console
  2. allowed_principal_can_access_console
  3. allowed_principal_is_resource_scoped

GCP has no AWS-style ``simulate_principal_policy`` equivalent for Compute
Engine serial console access (``instances.testIamPermissions`` and
``instances.getSerialPortOutput`` both evaluate the caller). Because the
APIs evaluate the caller rather than a simulated principal, RBAC evidence
must come from REAL probe principals:

  * a denied service account WITHOUT
    ``compute.instances.getSerialPortOutput`` on the target VM,
  * an allowed service account WITH that permission scoped to the target
    VM only,
  * a real second VM where the allowed SA must still be denied.

The default path is SELF-PROVISIONED: this stub creates two temporary
probe service accounts and a second probe VM, grants the caller
``roles/iam.serviceAccountTokenCreator`` on the probe SAs, grants the
allowed SA a minimal serial-output role scoped to the target VM only,
mints short-lived access tokens with
``iam_credentials_v1.IAMCredentialsClient.generate_access_token``, and probes
``instances.getSerialPortOutput`` as each principal. Cleanup deletes the
temporary SAs and probe VM and removes the IAM bindings with
read-modify-write retry on etag conflicts.

The pre-provisioned env-var path (``GCP_DENIED_PRINCIPAL_SA``,
``GCP_ALLOWED_PRINCIPAL_SA``, ``GCP_OTHER_INSTANCE_ID``) is a FALLBACK
for projects where the operator cannot allow IAM mutation. The fallback
is opt-in via the env vars themselves; otherwise the self-provisioned
path runs.

Compute Engine serial-console RBAC implementation notes:
  * Token minting goes through the declared IAMCredentials SDK client
    (``iam_credentials_v1.IAMCredentialsClient.generate_access_token``),
    which is what owns this API's request shape, typed errors, and
    version compatibility. Avoid
    ``google.auth.impersonated_credentials.Credentials`` with local
    authorized-user ADC — its refresh code can call a private
    ``_refresh_token`` member that is a string on authorized-user
    credentials, raising ``TypeError: 'str' object is not callable``.
    Calling the SDK client directly sidesteps that helper without
    hand-rolling the API.
  * Resolve the ADC caller from tokeninfo when local user ADC has no
    ``service_account_email`` and an empty ``account``.

HTTP 404 on the second VM probe is treated as a FAILURE, not as proof of
RBAC scoping (the resource being missing means IAM enforcement could not
be observed).

Fixture ownership (self-provisioned path):
  * Every probe fixture carries a per-invocation ownership marker in its
    ``description``. Ownership transfers to the cleanup tracker the moment a
    create is ACCEPTED — not when the response body happens to carry an
    identity field — and, on an ambiguous outcome (transport loss, 429/5xx,
    409 on the exact candidate), only when a marker-verified readback proves
    this invocation created the resource. Cleanup deletes exactly that
    accepted set; an unproven candidate is retained as a handoff, never
    deleted on a guess and never dropped.
  * ``--skip-destroy`` (the operator's preservation decision, shared with the
    terminal teardown step) suppresses that cleanup WHOLE and emits
    ``preserved_fixtures``; ``--reclaim-preserved <payload>`` replays the same
    ownership gates later to reclaim them.
  * ``preserved_fixtures`` is emitted by every run whose cleanup did not
    finish, not just by a preserved one: a failed delete, a revoke that could
    not be confirmed, or an unproven candidate all keep the record. Cleanup
    that cannot prove a fixture is gone must hand off the identities needed to
    finish the job, so ``--reclaim-preserved`` can always be replayed against a
    run that may have leaked. A delete or revoke hidden behind HTTP 403 is
    settled by a paginated inventory of the exact identity; only proven absence
    releases the handoff.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

from common.compute import (
    narrow_region_to_zone,
    resolve_project,
    unique_suffix,
    wait_for_zonal_op,
)
from common.errors import TRANSIENT_EXCEPTIONS, handle_gcp_errors
from common.ownership import (
    description_carries_invocation,
    description_with_invocation,
    new_invocation_id,
)
from google.api_core import exceptions as gax
from google.cloud import iam_credentials_v1
from google.protobuf import duration_pb2

# Serial console permission. The validator's ``restricted_actions`` field
# is populated with this exact permission name so downstream audits can
# correlate the IAM action with the API call.
_CONSOLE_PERMISSION = "compute.instances.getSerialPortOutput"
_RESTRICTED_ACTIONS = (_CONSOLE_PERMISSION,)

# Compute Engine REST endpoints. Token minting is NOT here: it goes through
# the google-cloud-iam SDK's IAMCredentialsClient (see _mint_access_token).
_COMPUTE_BASE = "https://compute.googleapis.com/compute/v1"
_IAM_BASE = "https://iam.googleapis.com/v1"
_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"

# Short-lived probe-token settings. The lifetime only has to outlast the three
# getSerialPortOutput probes that follow the mint.
_TOKEN_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_TOKEN_LIFETIME_S = 300

# tokenCreator binding -> generate_access_token propagation budget. A
# freshly-granted serviceAccountTokenCreator binding is not effective for token
# minting immediately; convergence has taken up to ~3 minutes, so 12 x 15s.
_TOKEN_MINT_ATTEMPTS = 12
_TOKEN_MINT_DELAY_S = 15.0

# Mint outcomes worth waiting out: the tokenCreator binding has not converged
# yet (PermissionDenied), the freshly-created probe SA is not visible yet
# (NotFound), or the backend returned a transient class. Anything else is a real
# defect and fails immediately rather than burning the whole budget.
_RETRYABLE_MINT: tuple[type[Exception], ...] = (
    gax.PermissionDenied,
    gax.NotFound,
    *TRANSIENT_EXCEPTIONS,
)

# Default per-call timeout for raw HTTP calls (seconds). The enclosing
# step timeout is the real deadline; this is just a safeguard against
# the urllib request hanging on a single API call.
_HTTP_TIMEOUT_S = 30

# Minimal predefined role granting serial-port-output read. ``roles/
# compute.viewer`` includes the permission and is broader than necessary,
# but is the acceptable predefined role for this probe (an equivalent
# minimal serial-output custom role would be tighter but requires extra
# provisioning the probe does not need).
_ALLOWED_TARGET_ROLE = "roles/compute.viewer"
_TOKEN_CREATOR_ROLE = "roles/iam.serviceAccountTokenCreator"

# Self-provisioning is the DEFAULT path. The pre-provisioned fallback is
# opt-in by exporting all three env vars.
_DENIED_SA_ENV = "GCP_DENIED_PRINCIPAL_SA"
_ALLOWED_SA_ENV = "GCP_ALLOWED_PRINCIPAL_SA"
_OTHER_INSTANCE_ENV = "GCP_OTHER_INSTANCE_ID"
_OTHER_INSTANCE_ZONE_ENV = "GCP_OTHER_INSTANCE_ZONE"
# Operators that cannot grant IAM mutations to the caller can set this
# to ``"0"`` to force-skip the self-provisioned path even when no fallback
# env vars are supplied; otherwise the stub will try self-provisioning
# and surface an honest failure if any step is denied.
_SELF_PROVISION_ENABLED_ENV = "GCP_SELF_PROVISION_RBAC"

# Description text the per-invocation ownership marker is appended to. The
# marker is what makes an ambiguous create reconcilable: a resource carrying
# THIS invocation's marker is provably ours to delete, and one carrying another
# marker (or none) belongs to a different run and must never be adopted.
_PROBE_SA_DESCRIPTION = "isvtest console-rbac probe principal"
_PROBE_VM_DESCRIPTION = "isvtest console-rbac probe instance"

# Reconciliation verdicts for an exact, project-scoped readback of a candidate
# resource after an ambiguous create outcome.
_OWNED = "owned"  # exists AND carries this invocation's marker -> ours to delete
_ABSENT = "absent"  # proven not to exist -> nothing was created
_FOREIGN = "foreign"  # exists with a different marker -> another run owns it
_INCONCLUSIVE = "inconclusive"  # lookup denied/failed -> retain the handoff

# Bounded readback envelope for reconciliation. A just-accepted create is
# eventually consistent, so a single 404 is not proof of absence.
_RECONCILE_ATTEMPTS = 3
_RECONCILE_BACKOFF_S = 2.0

# Page cap for the absence-proof inventory below. The probe fixtures live in
# the run's own project, so these inventories are small; the cap only stops an
# endlessly-paging response from consuming the step timeout. Reaching it is
# reported as an INCOMPLETE inventory (handoff retained), never as absence.
_INVENTORY_MAX_PAGES = 20

_RECLAIM_EPILOG = """\
Reclaiming preserved probe fixtures
-----------------------------------

    python3 console_rbac.py --reclaim-preserved console_rbac.json

Save this step's JSON output to a file first (or pipe it in with
`--reclaim-preserved -`). Use this instead of hand-written delete commands,
because that payload is the only record of what the run actually created:

  * every delete is gated on a readback that must echo the payload's
    per-invocation ownership marker, so a same-named resource belonging to
    another run is reported and left alone instead of destroyed;
  * both IAM bindings are removed by their recorded role + member -- the
    target-VM serial-output role granted to the allowed probe SA, and the
    caller's TokenCreator grant on each probe SA -- neither of which is
    visible from the resource names alone;
  * an already-absent resource is idempotent success, so re-running after a
    partial reclamation is safe, and an unproven (denied readback) candidate
    fails closed and keeps the handoff for a later retry -- an incomplete
    reclamation re-emits that handoff in its own output, so the payload you
    just ran can be fed straight back in.
"""


def _http_request(
    method: str,
    url: str,
    token: str,
    *,
    body: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Issue an HTTPS call with a bearer token; return ``(status, body)``.

    Body is the parsed JSON when the response is JSON, or ``{}`` when
    the response is empty / non-JSON. Errors raise ``urllib.error.HTTPError``
    which the caller catches to read the status code.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
            status = response.getcode() or 0
            raw = response.read()
    except urllib.error.HTTPError:
        raise
    parsed: dict[str, Any] = {}
    if raw:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            parsed = {"raw": raw.decode("utf-8", errors="replace")}
    return status, parsed


def _http_error_body(error: urllib.error.HTTPError) -> dict[str, Any]:
    """Best-effort extract JSON body from an HTTPError."""
    try:
        raw = error.read()
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {"raw": raw.decode("utf-8", errors="replace")}


def _adc_access_token() -> str:
    """Refresh ADC and return the access token string."""
    import google.auth
    import google.auth.transport.requests
    from google.auth.credentials import Credentials

    raw_creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds: Credentials = raw_creds  # type: ignore[assignment]
    creds.refresh(google.auth.transport.requests.Request())
    token = getattr(creds, "token", None)
    if not isinstance(token, str) or not token:
        msg = "ADC refresh produced no access token"
        raise RuntimeError(msg)
    return token


def _resolve_caller_member(access_token: str) -> str:
    """Resolve the calling principal to a ``user:`` / ``serviceAccount:`` member.

    Local user ADC (``gcloud auth application-default login``) commonly
    has no ``service_account_email`` and an empty ``account`` attribute;
    in that case the only reliable identifier is the tokeninfo endpoint,
    which returns the authenticated email for the refreshed access token.
    """
    import google.auth

    creds, _ = google.auth.default()
    sa_email = getattr(creds, "service_account_email", None)
    if isinstance(sa_email, str) and sa_email:
        return f"serviceAccount:{sa_email}"
    account = getattr(creds, "account", "")
    if isinstance(account, str) and account:
        # gcloud user ADC populates ``account`` with the user email.
        return f"user:{account}"

    # Last resort: probe the tokeninfo endpoint.
    url = f"{_TOKENINFO_URL}?access_token={urllib.parse.quote(access_token)}"
    with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_S) as response:
        info = json.loads(response.read().decode("utf-8"))
    email = info.get("email") or info.get("audience") or ""
    if not isinstance(email, str) or not email:
        msg = "could not resolve caller principal from ADC or tokeninfo"
        raise RuntimeError(msg)
    member_type = "serviceAccount" if email.endswith(".gserviceaccount.com") else "user"
    return f"{member_type}:{email}"


def _create_outcome_ambiguous(error: Exception) -> bool:
    """Return whether a failed REST create may still have committed server-side.

    An outcome is ambiguous whenever the client cannot prove the server did NOT
    create the resource:

      * a dropped connection / read timeout / DNS-or-socket failure — the
        request may have been committed and only the response lost;
      * HTTP 429 or 5xx — the rejection can be raised after the write landed;
      * HTTP 409 ALREADY_EXISTS on the exact per-invocation candidate identity —
        the classic "a previous attempt landed and its response was lost" shape.

    HTTP 400 / 401 / 403 / 404 are conclusive refusals evaluated before commit,
    so they are NOT ambiguous and never trigger a readback. ``HTTPError`` is a
    subclass of ``URLError`` (hence of ``OSError``), so it must be matched first.
    """
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (409, 429) or 500 <= error.code < 600
    return isinstance(error, urllib.error.URLError | TimeoutError | OSError)


def _describe_error(error: Exception) -> str:
    """Render a create/readback failure for structured evidence."""
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}: {_http_error_body(error)}"
    return f"{type(error).__name__}: {error}"


def _reconcile_marked_resource(
    *,
    url: str,
    token: str,
    invocation_id: str,
    identity: str,
    absence_proof: Callable[[], tuple[bool | None, str]] | None = None,
) -> tuple[str, str, str]:
    """Read back one exact resource and decide whether this invocation owns it.

    Returns ``(verdict, identity, detail)``. The verdict is derived ONLY from
    the per-invocation marker stamped into the resource ``description`` at
    create time — never from the name alone, so a same-named resource from
    another run is reported ``foreign`` and is never adopted for deletion. A
    denied or failing lookup is ``inconclusive``: the caller must retain the
    cleanup handoff rather than assume the resource is absent. ``404`` is
    retried within a bounded envelope first, because a just-accepted create is
    eventually consistent and one miss is not proof of absence.

    Absence is decided from the WHOLE attempt sequence, never from the last
    attempt alone, and the unobserved state is monotonic — the same rule the
    shared SDK reconciler (``common.ownership.reconcile_owned``) applies, so the
    two readers cannot drift. A transport failure followed by a ``404`` is a
    MIXED sequence: the failed attempt answered nothing, so the later 404 proves
    only that one read missed, which is precisely what a committed-but-still-
    propagating create looks like. Returning ``absent`` there would skip
    ``on_unreconciled`` and drop a real probe service account or probe VM out of
    both immediate cleanup and terminal reclamation.

    ``absence_proof`` settles the one denied shape that a direct readback can
    never settle by itself: HTTP 403 means the caller may not read THIS
    resource, which is neither presence nor absence. The paginated inventory of
    the exact identity can still prove absence — and only absence; it cannot
    read the marker, so a listed resource stays ``inconclusive`` (handoff
    retained) rather than becoming ``owned``. Without the proof the answer is
    ``inconclusive``, which never deletes anything and never claims cleanup.
    """
    last_detail = ""
    unobserved_detail = ""
    saw_unobserved = False
    not_found_attempts = 0
    for attempt in range(1, _RECONCILE_ATTEMPTS + 1):
        try:
            _, body = _http_request("GET", url, token)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                last_detail = "readback 404 (not found)"
                not_found_attempts += 1
                if attempt < _RECONCILE_ATTEMPTS:
                    time.sleep(_RECONCILE_BACKOFF_S * attempt)
                    continue
                break
            if e.code == 403 and absence_proof is not None:
                absent, proof_detail = absence_proof()
                verdict = _ABSENT if absent is True else _INCONCLUSIVE
                return verdict, identity, f"readback {_describe_error(e)}; {proof_detail}"
            return _INCONCLUSIVE, identity, f"readback {_describe_error(e)}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_detail = f"readback {_describe_error(e)}"
            unobserved_detail = last_detail
            saw_unobserved = True
            if attempt < _RECONCILE_ATTEMPTS:
                time.sleep(_RECONCILE_BACKOFF_S * attempt)
                continue
            break
        if description_carries_invocation(body.get("description"), invocation_id):
            return _OWNED, str(body.get("email") or body.get("name") or identity), "readback marker matched"
        return _FOREIGN, identity, "readback found a resource without this invocation's marker"

    # Exhausted (or broke out of) the envelope: every attempt was either a
    # conclusive 404 or an attempt that observed nothing. Only the all-404
    # sequence proves absence; the `> 0` guard also stops a zero-attempt
    # envelope from claiming a proof it never gathered.
    if not_found_attempts > 0 and not saw_unobserved:
        return _ABSENT, identity, last_detail
    return _INCONCLUSIVE, identity, unobserved_detail or last_detail or "readback exhausted"


def _reconcile_service_account(*, project: str, token: str, email: str, invocation_id: str) -> tuple[str, str, str]:
    """Reconcile ownership of one exact probe service account by marker."""
    return _reconcile_marked_resource(
        url=f"{_IAM_BASE}/projects/{project}/serviceAccounts/{email}",
        token=token,
        invocation_id=invocation_id,
        identity=email,
        absence_proof=lambda: _service_account_absent(project=project, token=token, email=email),
    )


def _reconcile_probe_vm(*, project: str, zone: str, token: str, name: str, invocation_id: str) -> tuple[str, str, str]:
    """Reconcile ownership of the exact probe instance by marker."""
    return _reconcile_marked_resource(
        url=f"{_COMPUTE_BASE}/projects/{project}/zones/{zone}/instances/{name}",
        token=token,
        invocation_id=invocation_id,
        identity=name,
        absence_proof=lambda: _probe_vm_absent(project=project, zone=zone, token=token, name=name),
    )


def _inventory_absence_proof(
    *,
    list_url: str,
    token: str,
    items_key: str,
    identity_fields: tuple[str, ...],
    identity: str,
) -> tuple[bool | None, str]:
    """Decide absence of one EXACT identity from a fully paginated inventory.

    A delete that answers HTTP 403 is a permission-hidden response: it proves
    only that the caller may not perform THAT operation on THAT resource, never
    that the resource is gone. Treating it as "already cleaned up" silently
    drops a live probe fixture, so absence is decided here instead — from a
    project/zone inventory walked to its last page, matched against the exact
    recorded identity.

    Returns ``(True, detail)`` only when every page was read and none carried
    the identity; ``(False, detail)`` when the identity is still listed; and
    ``(None, detail)`` when the inventory itself was denied, failed, or was
    truncated. An unreadable or incomplete inventory is never absence — the
    caller must fail closed and keep the ownership handoff for a later retry.
    """
    page_token = ""
    pages = 0
    while pages < _INVENTORY_MAX_PAGES:
        url = list_url
        if page_token:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}pageToken={urllib.parse.quote(page_token)}"
        try:
            _, body = _http_request("GET", url, token)
        except Exception as e:
            return None, f"inventory unreadable after {pages} page(s): {_describe_error(e)}"
        pages += 1
        for item in body.get(items_key) or []:
            if not isinstance(item, dict):
                continue
            for field in identity_fields:
                value = item.get(field)
                if isinstance(value, str) and identity in (value, value.rsplit("/", 1)[-1]):
                    return False, f"inventory still lists {identity}"
        page_token = str(body.get("nextPageToken") or "")
        if not page_token:
            return True, f"inventory ({pages} page(s)) does not list {identity}"
    return None, f"inventory truncated after {_INVENTORY_MAX_PAGES} pages; absence unproven"


def _service_account_absent(*, project: str, token: str, email: str) -> tuple[bool | None, str]:
    """Paginated project inventory proof that an exact probe SA is gone."""
    return _inventory_absence_proof(
        list_url=f"{_IAM_BASE}/projects/{project}/serviceAccounts?pageSize=100",
        token=token,
        items_key="accounts",
        identity_fields=("email", "name"),
        identity=email,
    )


def _probe_vm_absent(*, project: str, zone: str, token: str, name: str) -> tuple[bool | None, str]:
    """Paginated zone inventory proof that the exact probe VM is gone."""
    return _inventory_absence_proof(
        list_url=f"{_COMPUTE_BASE}/projects/{project}/zones/{zone}/instances?maxResults=500",
        token=token,
        items_key="items",
        identity_fields=("name", "selfLink"),
        identity=name,
    )


def _create_service_account(
    *,
    project: str,
    token: str,
    sa_id: str,
    display_name: str,
    invocation_id: str,
    on_owned: Callable[[str], None],
    on_unreconciled: Callable[[str, str], None],
) -> str:
    """Create a probe service account, transferring cleanup ownership on acceptance.

    Ownership must not be derived from "the response body contained an email":
    IAM can commit the account and then lose the response, and a committed
    response whose body lacks ``email`` describes an account that exists. Either
    shape would leave the caller's tracker empty and the ``finally`` block would
    skip the delete, leaking the account.

    So ownership transfers through ``on_owned`` the moment the create is
    ACCEPTED (any 2xx — the deterministic ``<accountId>@<project>.iam.
    gserviceaccount.com`` identity is known before the call), or, on an
    ambiguous failure, only when an exact marker-verified readback proves this
    invocation created it. When that readback is denied or inconclusive the
    candidate is handed to ``on_unreconciled`` so the handoff is retained for a
    later reclamation pass instead of being silently dropped. A conclusive
    refusal (400/401/403/404) transfers nothing.
    """
    candidate_email = f"{sa_id}@{project}.iam.gserviceaccount.com"
    url = f"{_IAM_BASE}/projects/{project}/serviceAccounts"
    body = {
        "accountId": sa_id,
        "serviceAccount": {
            "displayName": display_name,
            "description": description_with_invocation(_PROBE_SA_DESCRIPTION, invocation_id),
        },
    }
    try:
        _, response = _http_request("POST", url, token, body=body)
    except Exception as exc:
        if _create_outcome_ambiguous(exc):
            verdict, identity, detail = _reconcile_service_account(
                project=project,
                token=token,
                email=candidate_email,
                invocation_id=invocation_id,
            )
            if verdict == _OWNED:
                on_owned(identity)
            elif verdict == _INCONCLUSIVE:
                on_unreconciled(candidate_email, detail)
            msg = f"create_service_account({sa_id}) ambiguous: {_describe_error(exc)}; reconciled={verdict} ({detail})"
        else:
            msg = f"create_service_account({sa_id}) failed: {_describe_error(exc)}"
        raise RuntimeError(msg) from exc

    # Accepted. Transfer ownership BEFORE anything else can fail, using the
    # response identity when present and the deterministic candidate otherwise.
    email = response.get("email")
    owned_email = email if isinstance(email, str) and email else candidate_email
    on_owned(owned_email)
    return owned_email


def _delete_service_account(
    *,
    project: str,
    token: str,
    email: str,
    invocation_id: str,
    attempts: int = 5,
    backoff: float = 2.0,
) -> bool:
    """Delete a probe service account; only a PROVEN absence is success.

    Bounded retry/backoff on transient IAM cleanup failures (HTTP 429 / 5xx
    and socket-level errors) so a single flaky delete call doesn't orphan the
    probe SA into the run namespace — the same transient envelope the
    grant-side read-modify-write retries use. Non-404 4xx are non-retryable.

    HTTP 404 is NOT taken as proof of removal. This helper only ever runs
    against an identity whose create was ACCEPTED, and IAM is eventually
    consistent in exactly that window: a delete served by a replica that has
    not seen the create answers 404 while the account exists and goes on
    existing. Reporting success there is unrecoverable, because the caller
    retains ``preserved_fixtures`` — the only replayable ownership record this
    step ever writes — solely when cleanup reports a failure. So a 404 is
    settled by the same bounded, marker-verified readback the create path uses
    (``_reconcile_service_account``: exact-name reads, 404 retried inside its
    own envelope, inventory proof for a denied read):

      * ``absent``       — proven gone; cleanup succeeded;
      * ``owned``        — still live and still ours, so the 404 was a
        propagation miss: retry the delete inside this envelope;
      * ``foreign``      — the identity is held by a resource without this
        invocation's marker; never delete it, and report failure so the handoff
        survives (the verdict the reclamation path already applies);
      * ``inconclusive`` — nothing was proven, so fail closed.

    HTTP 403 is permission-hidden, not proof of removal: it is settled against
    the paginated project inventory of this exact email, and only a proven
    absence reports cleanup success. Present, denied, or truncated inventory
    fails closed so the caller keeps the ownership handoff.
    """
    url = f"{_IAM_BASE}/projects/{project}/serviceAccounts/{email}"
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            _http_request("DELETE", url, token)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                verdict, _, detail = _reconcile_service_account(
                    project=project,
                    token=token,
                    email=email,
                    invocation_id=invocation_id,
                )
                print(
                    f"  delete_service_account({email}) HTTP 404; reconciled={verdict} ({detail})",
                    file=sys.stderr,
                )
                if verdict == _ABSENT:
                    return True
                if verdict == _OWNED and attempt < attempts:
                    time.sleep(backoff * attempt)
                    continue
                return False
            last_error = f"HTTP {e.code}: {_http_error_body(e)}"
            if e.code == 403:
                absent, detail = _service_account_absent(project=project, token=token, email=email)
                print(f"  delete_service_account({email}) {last_error}; {detail}", file=sys.stderr)
                return absent is True
            if e.code == 429 or 500 <= e.code < 600:
                time.sleep(backoff * attempt)
                continue
            print(f"  delete_service_account({email}) {last_error}", file=sys.stderr)
            return False
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = f"network error ({type(e).__name__}): {e}"
            time.sleep(backoff * attempt)
            continue
    print(f"  delete_service_account({email}) exhausted: {last_error}", file=sys.stderr)
    return False


def _service_account_resource(project: str, sa_email: str) -> str:
    """REST resource path for an SA (used by SA-resource IAM policy calls)."""
    return f"projects/{project}/serviceAccounts/{sa_email}"


def _is_member_propagation_error(body: dict[str, Any]) -> bool:
    """True if a setIamPolicy 400 body means the member SA has not propagated.

    A freshly-created service account referenced as an IAM *member* is
    eventually consistent: the policy write is rejected with HTTP 400
    ('... does not exist') until the member converges (same window as the
    TokenCreator binding handled in ``_mint_access_token``). Malformed-policy
    400s (unknown member type, unsupported role) carry different text and are
    intentionally NOT matched, so they stay non-retryable. ``body`` is the
    parsed JSON from ``_http_error_body``; serialize it so the match works
    whether the text lands in ``error.message`` or the ``raw`` fallback.
    """
    return "does not exist" in json.dumps(body).lower()


def _modify_iam_policy(
    *,
    get_url: str,
    set_url: str,
    token: str,
    operation: str,
    role: str,
    member: str,
    get_method: str = "POST",
    attempts: int = 5,
    backoff: float = 1.0,
    member_propagation_delay: float = 0.0,
) -> bool:
    """Read-modify-write an IAM policy with etag retry.

    ``operation`` is ``"add"`` or ``"remove"``. Returns True on success.

    ``get_method`` selects the HTTP verb for ``getIamPolicy``. The IAM API
    (service-account resources at ``iam.googleapis.com``) uses
    ``POST {resource}:getIamPolicy`` with a JSON body, while the Compute
    Engine API (instance / disk / zonal resources at
    ``compute.googleapis.com``) uses
    ``GET {resource}/getIamPolicy?optionsRequestedPolicyVersion=3`` and
    rejects POST with HTTP 400. ``setIamPolicy`` is POST on both APIs.
    """
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            if get_method == "GET":
                _, policy = _http_request(
                    "GET",
                    f"{get_url}?optionsRequestedPolicyVersion=3",
                    token,
                )
            else:
                _, policy = _http_request(
                    "POST",
                    get_url,
                    token,
                    body={"options": {"requestedPolicyVersion": 3}},
                )
        except urllib.error.HTTPError as e:
            last_error = f"getIamPolicy HTTP {e.code}: {_http_error_body(e)}"
            if e.code in (404, 403):
                # No policy / no permission — caller decides whether this is fatal.
                return False
            if e.code == 429 or 500 <= e.code < 600:
                time.sleep(backoff * attempt)
                continue
            # Other 4xx (malformed request, etc.) — retry will not help.
            return False
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # Transient socket-level failure (connect/read timeout, connection
            # reset, DNS). On Python >= 3.10 a mid-read `socket.timeout` is
            # `TimeoutError` and may NOT be wrapped in `URLError`, so catch
            # both directly to avoid escaping into the outer try.
            last_error = f"getIamPolicy network error ({type(e).__name__}): {e}"
            time.sleep(backoff * attempt)
            continue

        # Edit a deep copy of the freshly read policy and send it back whole.
        # setIamPolicy REPLACES the resource policy, so a response built from a
        # fresh dict would silently drop every live top-level field this probe
        # does not model (auditConfigs today, anything a later API version
        # adds). Deep — not shallow — because the binding dicts are nested and
        # a shallow copy would alias them back into the parsed response.
        new_policy = copy.deepcopy(policy)
        bindings = list(new_policy.get("bindings", []) or [])
        # An IAM condition makes a binding a DISTINCT grant even when its role
        # matches, so a conditional binding is never the target of this
        # unconditional add/remove. Matching one would rewrite an
        # operator-owned conditional grant into an unconditional one.
        target_idx = next(
            (i for i, b in enumerate(bindings) if b.get("role") == role and not b.get("condition")),
            None,
        )
        if operation == "add":
            if target_idx is None:
                bindings.append({"role": role, "members": [member]})
            else:
                members = list(bindings[target_idx].get("members", []))
                if member in members:
                    return True  # desired state already holds; skip the write
                members.append(member)
                bindings[target_idx]["members"] = members
        elif operation == "remove":
            if target_idx is None:
                return True  # already absent
            current = list(bindings[target_idx].get("members", []))
            if member not in current:
                return True  # already absent from this binding; skip the write
            members = [m for m in current if m != member]
            if members:
                bindings[target_idx]["members"] = members
            else:
                bindings.pop(target_idx)
        else:
            msg = f"invalid operation: {operation!r}"
            raise ValueError(msg)

        new_policy["bindings"] = bindings
        new_policy.setdefault("etag", "")
        new_policy.setdefault("version", 1)
        try:
            _http_request("POST", set_url, token, body={"policy": new_policy})
            return True
        except urllib.error.HTTPError as e:
            body = _http_error_body(e)
            last_error = f"setIamPolicy HTTP {e.code}: {body}"
            # A brand-new SA referenced as a member is eventually consistent;
            # setIamPolicy rejects the binding with HTTP 400 ('... does not
            # exist') until it propagates (~3 min observed). Callers that add a
            # just-created SA opt in via member_propagation_delay and retry the
            # read-modify-write on a flat delay until the member converges.
            if e.code == 400 and member_propagation_delay > 0 and _is_member_propagation_error(body):
                print(
                    f"  iam grant: member {member} not yet propagated; retrying in {member_propagation_delay:.0f}s",
                    file=sys.stderr,
                )
                time.sleep(member_propagation_delay)
                continue
            # 409 stale etag (refresh GET on next iter), 429 rate-limit, and 5xx
            # transient server errors all warrant the read-modify-write retry.
            if e.code in (409, 429) or 500 <= e.code < 600:
                time.sleep(backoff * attempt)
                continue
            print(f"  setIamPolicy non-retryable: {last_error}", file=sys.stderr)
            return False
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # Same socket-level coverage as the getIamPolicy arm — mid-read
            # timeouts can escape URLError on Python >= 3.10.
            last_error = f"setIamPolicy network error ({type(e).__name__}): {e}"
            time.sleep(backoff * attempt)
            continue

    if last_error:
        print(f"  iam_policy_retry exhausted: {last_error}", file=sys.stderr)
    return False


def _grant_token_creator(*, project: str, token: str, sa_email: str, member: str) -> bool:
    """Grant the caller TokenCreator on a probe SA."""
    base = f"{_IAM_BASE}/projects/{project}/serviceAccounts/{sa_email}"
    return _modify_iam_policy(
        get_url=f"{base}:getIamPolicy",
        set_url=f"{base}:setIamPolicy",
        token=token,
        operation="add",
        role=_TOKEN_CREATOR_ROLE,
        member=member,
    )


def _revoke_token_creator(*, project: str, token: str, sa_email: str, member: str) -> bool:
    base = f"{_IAM_BASE}/projects/{project}/serviceAccounts/{sa_email}"
    return _modify_iam_policy(
        get_url=f"{base}:getIamPolicy",
        set_url=f"{base}:setIamPolicy",
        token=token,
        operation="remove",
        role=_TOKEN_CREATOR_ROLE,
        member=member,
    )


def _grant_target_role(
    *,
    project: str,
    zone: str,
    instance: str,
    token: str,
    sa_email: str,
) -> bool:
    """Grant the allowed SA the target-VM serial-output role.

    ``sa_email`` is a probe SA created seconds earlier in this same step.
    A brand-new SA referenced as an IAM member is eventually consistent, so
    the instance setIamPolicy is rejected with HTTP 400 ('... does not exist')
    until the member propagates. Opt into the member-propagation retry (flat
    15s, ~3 min budget — the convergence window already documented on
    ``_mint_access_token``) so the grant lands deterministically instead of
    nondeterministically failing when the SA was minted moments ago.
    """
    base = f"{_COMPUTE_BASE}/projects/{project}/zones/{zone}/instances/{instance}"
    return _modify_iam_policy(
        get_url=f"{base}/getIamPolicy",
        set_url=f"{base}/setIamPolicy",
        token=token,
        operation="add",
        role=_ALLOWED_TARGET_ROLE,
        member=f"serviceAccount:{sa_email}",
        get_method="GET",
        attempts=14,
        member_propagation_delay=15.0,
    )


def _revoke_target_role(
    *,
    project: str,
    zone: str,
    instance: str,
    token: str,
    sa_email: str,
) -> bool:
    base = f"{_COMPUTE_BASE}/projects/{project}/zones/{zone}/instances/{instance}"
    return _modify_iam_policy(
        get_url=f"{base}/getIamPolicy",
        set_url=f"{base}/setIamPolicy",
        token=token,
        operation="remove",
        role=_ALLOWED_TARGET_ROLE,
        member=f"serviceAccount:{sa_email}",
        get_method="GET",
    )


def _mint_access_token(
    *,
    sa_email: str,
    attempts: int = _TOKEN_MINT_ATTEMPTS,
    delay: float = _TOKEN_MINT_DELAY_S,
) -> str:
    """Mint a short-lived access token for ``sa_email`` via the IAMCredentials SDK.

    Uses ``iam_credentials_v1.IAMCredentialsClient.generate_access_token`` — the
    declared client for this API — rather than a hand-rolled POST to the
    ``generateAccessToken`` endpoint. Hand-building the request would put this
    probe's serialization, error mapping, and endpoint/version compatibility on
    a private copy of a contract the SDK already owns, and would report token
    failures as bare HTTP codes that share no vocabulary with the typed
    ``google.api_core`` errors every other call in this tree classifies.

    The client authenticates with the same ADC principal whose access token the
    surrounding REST calls carry — ``generate_access_token`` needs
    ``iam.serviceAccounts.getAccessToken`` on ``sa_email``, which is exactly the
    tokenCreator binding this stub grants the caller — so the minted token still
    represents the probe SA, not the caller.

    ``google.auth.impersonated_credentials.Credentials`` is deliberately NOT used
    here: under local authorized-user ADC its refresh path can call a private
    ``_refresh_token`` member that is a string on authorized-user credentials,
    raising ``TypeError: 'str' object is not callable``. Calling the SDK client
    directly avoids that helper while staying on the reviewed SDK.

    TokenCreator IAM bindings on freshly-created probe service accounts are
    eventually consistent; observed convergence in this suite has required up to
    ~3 minutes after the binding is granted. Retry the propagation shapes
    (``PermissionDenied`` for a binding that has not landed, ``NotFound`` for an
    SA that is not visible yet, and the transient classes) on a bounded
    12 x 15s budget so the probe does not nondeterministically fail against a
    binding that converges seconds after the call. Every other error is a real
    defect and is raised immediately with its typed class named, so the caller's
    structured failure reporting stays honest instead of waiting out the budget.
    """
    client = iam_credentials_v1.IAMCredentialsClient()
    name = f"projects/-/serviceAccounts/{sa_email}"
    lifetime = duration_pb2.Duration(seconds=_TOKEN_LIFETIME_S)

    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.generate_access_token(
                name=name,
                scope=[_TOKEN_SCOPE],
                lifetime=lifetime,
            )
        except _RETRYABLE_MINT as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < attempts:
                print(
                    f"  generate_access_token attempt {attempt}/{attempts} "
                    f"({type(e).__name__}); tokenCreator binding propagating, "
                    f"retrying in {delay:.0f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
                continue
            break
        minted = response.access_token
        if isinstance(minted, str) and minted:
            return minted
        last_error = "response carried no access_token"
        break

    budget = attempts * delay
    msg = f"generate_access_token for {sa_email} did not converge within {budget:.0f}s: {last_error}"
    raise RuntimeError(msg)


def _probe_serial_console(
    *,
    project: str,
    zone: str,
    instance: str,
    access_token: str,
) -> tuple[int, str]:
    """Probe ``getSerialPortOutput`` with ``access_token``.

    Returns ``(http_status, evidence_text)``. Honest signal: the HTTP
    status comes from a real probe (200 = allowed, 403 = denied, 404 =
    diagnostic gap). The evidence text records the response body / error
    detail for audit.
    """
    url = f"{_COMPUTE_BASE}/projects/{project}/zones/{zone}/instances/{instance}/serialPort?port=1"
    try:
        status, response = _http_request("GET", url, access_token)
    except urllib.error.HTTPError as e:
        body = _http_error_body(e)
        message = body.get("error", {}).get("message", "")
        return e.code, f"HTTP {e.code}: {message or body}"
    contents = response.get("contents") or ""
    return status, f"HTTP {status}: contents_length={len(contents)}"


def _submit_probe_vm_insert(
    *,
    project: str,
    zone: str,
    network: str,
    token: str,
    name: str,
    invocation_id: str,
    on_owned: Callable[[str], None],
    on_unreconciled: Callable[[str, str], None],
) -> tuple[bool, str, str]:
    """Submit the probe VM insert; return ``(ack_ok, op_name, evidence)``.

    The probe VM is e2-micro with the Debian image family — no GPU, no
    NIM, no persistent disk reuse. It exists solely so the allowed-SA
    probe can produce a real HTTP 403 against an instance the SA was
    NOT granted access to.

    Stamp-on-accept, before the wait: ownership transfers through
    ``on_owned`` as soon as the insert is ACCEPTED — not when the response
    happens to carry an operation name, and never after
    ``wait_for_zonal_op``. Compute Engine can commit the instance and then
    lose the response, and a committed response missing ``name`` still
    describes a running, billable VM; deriving ownership from that field
    would leave the caller's tracker empty and the ``finally`` block would
    skip the delete. On an ambiguous failure ownership transfers only when
    an exact marker-verified readback proves this invocation created the
    instance; an inconclusive readback goes to ``on_unreconciled`` so the
    handoff survives for a later reclamation pass.
    """
    url = f"{_COMPUTE_BASE}/projects/{project}/zones/{zone}/instances"
    body = {
        "name": name,
        # Per-invocation ownership marker. It lives in ``description``
        # rather than in ``labels`` for the same reason the launch step
        # keeps it there: labels are part of the user-visible tag surface.
        "description": description_with_invocation(_PROBE_VM_DESCRIPTION, invocation_id),
        "machineType": f"zones/{zone}/machineTypes/e2-micro",
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "initializeParams": {
                    "sourceImage": "projects/debian-cloud/global/images/family/debian-12",
                    "diskType": f"zones/{zone}/diskTypes/pd-balanced",
                    "diskSizeGb": "10",
                },
            }
        ],
        "networkInterfaces": [
            {
                "network": f"projects/{project}/global/networks/{network}",
            }
        ],
        "labels": {
            "createdby": "isvtest",
            "isv_role": "console-rbac-probe",
        },
    }
    try:
        _, op = _http_request("POST", url, token, body=body)
    except Exception as exc:
        if not _create_outcome_ambiguous(exc):
            return False, "", f"insert failed: {_describe_error(exc)}"
        verdict, identity, detail = _reconcile_probe_vm(
            project=project,
            zone=zone,
            token=token,
            name=name,
            invocation_id=invocation_id,
        )
        if verdict == _OWNED:
            on_owned(identity)
        elif verdict == _INCONCLUSIVE:
            on_unreconciled(name, detail)
        return False, "", f"insert ambiguous: {_describe_error(exc)}; reconciled={verdict} ({detail})"

    # Accepted — the server holds the create regardless of what the body
    # carries, so ownership transfers here and the op name only decides
    # whether the caller can WAIT on completion.
    on_owned(name)
    op_name = op.get("name", "")
    if not op_name:
        return False, "", f"insert accepted but response missing operation name: {op}"
    return True, op_name, f"probe VM {name} insert accepted in {zone}"


def _wait_probe_vm_insert(
    *,
    project: str,
    zone: str,
    op_name: str,
) -> tuple[bool, str]:
    """Block on the probe VM insert op; return ``(ok, evidence)``."""
    try:
        wait_for_zonal_op(project, zone, op_name, timeout=300)
    except Exception as e:
        return False, f"insert wait failed: {e}"
    return True, "probe VM insert wait done"


def _delete_probe_vm(
    *,
    project: str,
    zone: str,
    token: str,
    name: str,
    invocation_id: str,
    attempts: int = 5,
    backoff: float = 2.0,
) -> bool:
    """Delete the probe VM; only a PROVEN absence is success.

    Bounded retry/backoff on transient delete-submit failures (HTTP 429 / 5xx
    and socket-level errors) mirrors the SA-cleanup envelope so a flaky
    Compute Engine call doesn't orphan the probe VM into the run namespace.
    A wait-side failure is NOT retried here (the delete op is already in
    flight); it surfaces as a cleanup error for the next sweep.

    HTTP 404 is NOT taken as proof of removal, for the same reason the probe
    SA's delete does not take it: this helper runs against an instance whose
    insert was ACCEPTED, and a delete routed to a replica that has not yet seen
    that insert answers 404 for a VM that is running and billable. Claiming
    cleanup success there discards ``preserved_fixtures``, the one record a
    later reclamation can replay. The 404 is therefore settled by the bounded
    marker-verified readback (``_reconcile_probe_vm``): ``absent`` is success,
    ``owned`` means the 404 was a propagation miss and the delete is retried in
    this envelope, ``foreign`` refuses the delete and keeps the handoff, and
    ``inconclusive`` fails closed.

    HTTP 403 is permission-hidden, not proof of removal: it is settled against
    the paginated zone inventory of this exact instance name, and only a proven
    absence reports cleanup success. Present, denied, or truncated inventory
    fails closed so the caller keeps the ownership handoff.
    """
    url = f"{_COMPUTE_BASE}/projects/{project}/zones/{zone}/instances/{name}"
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            _, op = _http_request("DELETE", url, token)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                verdict, _, detail = _reconcile_probe_vm(
                    project=project,
                    zone=zone,
                    token=token,
                    name=name,
                    invocation_id=invocation_id,
                )
                print(
                    f"  delete_probe_vm({name}) HTTP 404; reconciled={verdict} ({detail})",
                    file=sys.stderr,
                )
                if verdict == _ABSENT:
                    return True
                if verdict == _OWNED and attempt < attempts:
                    time.sleep(backoff * attempt)
                    continue
                return False
            last_error = f"HTTP {e.code}: {_http_error_body(e)}"
            if e.code == 403:
                absent, detail = _probe_vm_absent(project=project, zone=zone, token=token, name=name)
                print(f"  delete_probe_vm({name}) {last_error}; {detail}", file=sys.stderr)
                return absent is True
            if e.code == 429 or 500 <= e.code < 600:
                time.sleep(backoff * attempt)
                continue
            print(f"  delete_probe_vm({name}) {last_error}", file=sys.stderr)
            return False
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = f"network error ({type(e).__name__}): {e}"
            time.sleep(backoff * attempt)
            continue
        op_name = op.get("name", "")
        if op_name:
            try:
                wait_for_zonal_op(project, zone, op_name, timeout=180)
            except Exception as e:
                print(f"  delete_probe_vm wait failed: {e}", file=sys.stderr)
                return False
        return True
    print(f"  delete_probe_vm({name}) exhausted: {last_error}", file=sys.stderr)
    return False


def _self_provisioned_probe(
    *,
    project: str,
    zone: str,
    instance: str,
    network: str,
    result: dict[str, Any],
    skip_destroy: bool = False,
) -> int:
    """Run the self-provisioned RBAC probe.

    Returns 0 ONLY when every subtest passed AND cleanup succeeded on
    every probe resource. Cleanup runs in the ``finally`` block so the
    SAs / VM / IAM bindings created by this run are removed even on
    partial failure; the ``cleanup_errors`` list is then AND-ed into
    ``result['success']`` and the return code, mirroring the AWS oracle
    (providers/aws/scripts/vm/console_rbac.py — cleanup failures flip
    success to False).

    ``skip_destroy`` (the operator's preservation decision, the same one the
    terminal teardown step receives) suppresses that whole ``finally`` cleanup:
    this step owns fixtures it would otherwise destroy long before terminal
    teardown is reached, so preservation has to reach it too. It preserves the
    fixture set WHOLE — probe SAs, probe VM, and both IAM bindings — and emits
    ``preserved_fixtures`` so a later ``--reclaim-preserved`` pass can reclaim
    exactly what this invocation created.

    ``preserved_fixtures`` is NOT limited to that preservation path. Any run
    that ends with an unconfirmed owned fixture — a delete or revoke that
    failed, or a candidate whose ownership could not be reconciled — emits the
    same record, because a cleanup that may have leaked is precisely the case
    where the replayable ownership evidence matters.
    """
    caller_token = _adc_access_token()
    caller_member = _resolve_caller_member(caller_token)
    result["caller"] = caller_member

    # The GCP service-account local-part (segment before ``@<project>.iam.``)
    # is hard-capped at 30 chars. A run-id-only suffix is NOT enough: the
    # run-id alone collapses two distinct logical SAs onto the same name if a
    # transient in-step cleanup of one fails and the next attempt inside the
    # same run hits 409 ALREADY_EXISTS. Fold a per-invocation discriminator
    # (4 hex chars) BETWEEN the static prefix and the run-id suffix so every
    # invocation gets a fresh name, and so the 30-char truncation can never
    # drop the discriminator or the trailing run-id token. The ``isv-`` prefix
    # + trailing run-id suffix still match the external sweep regex
    # ``^isv-.*-<run_id_suffix>$``. Shorter ``-d`` / ``-a`` prefixes (vs
    # ``denied`` / ``allowed``) keep clear headroom under the cap; the human
    # role stays legible via each SA's displayName.
    run_suffix = unique_suffix("rbac", length=8).split("-", 1)[-1]
    invocation_tag = secrets.token_hex(2)  # 4 hex chars, per-invocation
    denied_sa_id = f"isv-rbac-d-{invocation_tag}-{run_suffix}"[:30]
    allowed_sa_id = f"isv-rbac-a-{invocation_tag}-{run_suffix}"[:30]
    probe_vm_name = f"isv-rbac-probe-{invocation_tag}-{run_suffix}"[:62]

    # Per-invocation ownership marker stamped into every fixture this step
    # creates. It is what lets an ambiguous create outcome be reconciled by
    # exact readback without ever adopting another run's identically-shaped
    # resource for deletion.
    invocation_id = new_invocation_id()
    result["invocation_id"] = invocation_id

    # Cleanup ownership. The three identity slots hold an identifier only once
    # this invocation is entitled to delete it; the three boolean slots mean "a
    # binding write was issued against this principal, so a revoke is owed" —
    # NOT "the grant returned success", because a policy write can commit and
    # still report failure.
    created: dict[str, Any] = {
        "denied_sa": "",
        "allowed_sa": "",
        "probe_vm": "",
        "token_creator_denied": False,
        "token_creator_allowed": False,
        "target_role_allowed": False,
    }
    # Candidates whose ownership could NOT be decided (the reconciling readback
    # was denied or failed). They are never deleted here — deletion is reserved
    # for the accepted set — but the handoff is retained and reported so a later
    # reclamation pass can re-verify the marker and finish the job.
    unreconciled: list[dict[str, str]] = []
    cleanup_errors: list[str] = []
    subtests_passed = False
    early_failure: str | None = None

    def _own(slot: str) -> Callable[[str], None]:
        def _record(identity: str) -> None:
            created[slot] = identity

        return _record

    def _retain(kind: str) -> Callable[[str, str], None]:
        def _record(identity: str, detail: str) -> None:
            unreconciled.append(
                {
                    "kind": kind,
                    "identity": identity,
                    "zone": zone if kind == "probe_vm" else "",
                    "detail": detail,
                }
            )

        return _record

    try:
        # 1. Create probe service accounts. Ownership transfers inside the
        #    helper — on acceptance, or on a marker-verified readback of an
        #    ambiguous outcome — so a lost response can never strand a real
        #    account outside the cleanup set.
        print(f"Creating denied probe SA {denied_sa_id}...", file=sys.stderr)
        denied_email = _create_service_account(
            project=project,
            token=caller_token,
            sa_id=denied_sa_id,
            display_name="ISV RBAC denied probe",
            invocation_id=invocation_id,
            on_owned=_own("denied_sa"),
            on_unreconciled=_retain("service_account"),
        )

        print(f"Creating allowed probe SA {allowed_sa_id}...", file=sys.stderr)
        allowed_email = _create_service_account(
            project=project,
            token=caller_token,
            sa_id=allowed_sa_id,
            display_name="ISV RBAC allowed probe",
            invocation_id=invocation_id,
            on_owned=_own("allowed_sa"),
            on_unreconciled=_retain("service_account"),
        )

        # 2. Grant the caller TokenCreator on both probe SAs so we can
        #    mint access tokens for them.
        #
        #    The revoke trackers are armed BEFORE each grant, for the same
        #    reason the creates stamp ownership on acceptance: a policy write
        #    that commits and then loses its response returns False here, and
        #    deriving "must revoke" from that return value would leave the
        #    binding in place forever. Revoking a binding that never landed is
        #    a no-op (the read-modify-write finds the role absent and reports
        #    success), so arming early can only ever over-clean this run's own
        #    binding — never under-clean it.
        created["token_creator_denied"] = True
        denied_grant_ok = _grant_token_creator(
            project=project,
            token=caller_token,
            sa_email=denied_email,
            member=caller_member,
        )
        created["token_creator_allowed"] = True
        allowed_grant_ok = _grant_token_creator(
            project=project,
            token=caller_token,
            sa_email=allowed_email,
            member=caller_member,
        )
        if not (denied_grant_ok and allowed_grant_ok):
            early_failure = "could not grant TokenCreator on probe SAs"
            result["error"] = early_failure
            return 1

        # 3. Grant the allowed SA the serial-output role on the TARGET VM
        #    only. The denied SA gets nothing; the allowed SA's binding is
        #    scoped to this one instance so the resource-scope subtest
        #    against the second VM is a genuine deny.
        #
        #    Armed before the call as above — this is the one binding that
        #    lives on a resource the step does NOT own (the operator's target
        #    VM), so a silently-retained grant here is the most costly leak in
        #    the step.
        created["target_role_allowed"] = True
        if not _grant_target_role(
            project=project,
            zone=zone,
            instance=instance,
            token=caller_token,
            sa_email=allowed_email,
        ):
            early_failure = "could not grant target-VM role to allowed probe SA"
            result["error"] = early_failure
            return 1

        # 4. Submit the second probe VM. The allowed SA was NOT granted
        #    any role on this VM, so the resource-scope subtest is a real
        #    HTTP 403 (or honest failure if 404 — the instance is missing).
        #    Stamp-before-wait: the helper records the probe VM in the
        #    cleanup tracker on the insert ACCEPTANCE (or on a
        #    marker-verified readback of an ambiguous outcome), so neither a
        #    wait-side failure nor a lost insert response can leave a live
        #    probe VM outside the finally-block teardown.
        ack_ok, probe_op, ack_evidence = _submit_probe_vm_insert(
            project=project,
            zone=zone,
            network=network,
            token=caller_token,
            name=probe_vm_name,
            invocation_id=invocation_id,
            on_owned=_own("probe_vm"),
            on_unreconciled=_retain("probe_vm"),
        )
        if not ack_ok:
            early_failure = f"could not create probe VM: {ack_evidence}"
            result["error"] = early_failure
            return 1
        wait_ok, wait_evidence = _wait_probe_vm_insert(
            project=project,
            zone=zone,
            op_name=probe_op,
        )
        if not wait_ok:
            early_failure = f"probe VM insert wait failed: {wait_evidence}"
            result["error"] = early_failure
            return 1

        # 5. Mint short-lived access tokens (with TokenCreator-propagation
        #    retry budget — see _mint_access_token) and run the three
        #    subtests.
        denied_token = _mint_access_token(sa_email=denied_email)
        allowed_token = _mint_access_token(sa_email=allowed_email)

        denied_status, denied_evidence = _probe_serial_console(
            project=project,
            zone=zone,
            instance=instance,
            access_token=denied_token,
        )
        result["tests"]["denied_principal_cannot_access_console"] = {
            "passed": denied_status == 403,
            "principal": f"serviceAccount:{denied_email}",
            "evidence": denied_evidence,
        }

        allowed_status, allowed_evidence = _probe_serial_console(
            project=project,
            zone=zone,
            instance=instance,
            access_token=allowed_token,
        )
        result["tests"]["allowed_principal_can_access_console"] = {
            "passed": allowed_status == 200,
            "principal": f"serviceAccount:{allowed_email}",
            "evidence": allowed_evidence,
        }

        scope_status, scope_evidence = _probe_serial_console(
            project=project,
            zone=zone,
            instance=probe_vm_name,
            access_token=allowed_token,
        )
        # HTTP 404 is NOT proof of scoping — the resource is missing, so
        # IAM enforcement could not be observed. Only 403 counts.
        result["tests"]["allowed_principal_is_resource_scoped"] = {
            "passed": scope_status == 403,
            "principal": f"serviceAccount:{allowed_email}",
            "evidence": f"probe_vm={probe_vm_name}; {scope_evidence}",
        }

        result["access_restricted"] = (
            result["tests"]["denied_principal_cannot_access_console"]["passed"]
            and result["tests"]["allowed_principal_is_resource_scoped"]["passed"]
        )
        subtests_passed = all(t["passed"] for t in result["tests"].values())
        if not subtests_passed:
            result["error"] = "one or more console RBAC subtests failed; see tests.* evidence"
        # Defer the final success/rc computation to the finally block so
        # cleanup failures can flip success to False (matches AWS oracle).
        return 0  # placeholder — finally overrides with the cleanup-AND-ed rc

    except Exception as e:
        # Capture probe-setup errors (SA create/grant HTTP errors, token
        # mint failures, etc.) so the operator sees a structured root
        # cause rather than a generic three-False-subtest failure. The
        # finally block still runs to clean up partial probe resources.
        early_failure = f"{type(e).__name__}: {e}"
        result["error"] = early_failure
        return 1
    finally:
        # The fixture record this invocation owns. It is emitted whenever this
        # run cannot prove its fixtures are gone — preserved on purpose, left
        # unreconciled, or left behind by a cleanup call that failed — because
        # it is the provenance a later reclamation replays, and in every one of
        # those cases it is the only place the retained identities and binding
        # flags are written down.
        fixture_record = {
            "invocation_id": invocation_id,
            "project": project,
            "zone": zone,
            "target_instance": instance,
            "caller": caller_member,
            "denied_sa": created["denied_sa"],
            "allowed_sa": created["allowed_sa"],
            "probe_vm": created["probe_vm"],
            "token_creator_denied": created["token_creator_denied"],
            "token_creator_allowed": created["token_creator_allowed"],
            "target_role_allowed": created["target_role_allowed"],
            "target_role": _ALLOWED_TARGET_ROLE,
            "token_creator_role": _TOKEN_CREATOR_ROLE,
        }
        if unreconciled:
            result["unreconciled_fixtures"] = list(unreconciled)
        if skip_destroy:
            # Preservation mode: suppress the WHOLE compensating cleanup. This
            # step's `finally` runs long before the terminal teardown step
            # consults the same operator decision, so gating only there would
            # destroy the exact failed fixture state the operator asked to keep.
            # Never a partial clean — the probe VM, both probe SAs, and both IAM
            # bindings are retained together, because reproducing the failure
            # needs the whole set. Preservation suppresses the delete, never the
            # ownership bookkeeping: the identifiers below stay truthful and are
            # what `--reclaim-preserved` consumes later.
            result["skip_destroy"] = True
            result["preserved_fixtures"] = fixture_record
            print(
                "Skip-destroy set: preserving console RBAC probe fixtures "
                f"(denied_sa={created['denied_sa']!r}, allowed_sa={created['allowed_sa']!r}, "
                f"probe_vm={created['probe_vm']!r}@{zone!r}, target_role_allowed="
                f"{created['target_role_allowed']} on {instance!r}). Reclaim them later with: "
                "console_rbac.py --reclaim-preserved <this-step-output.json>",
                file=sys.stderr,
            )
        else:
            # Cleanup runs unconditionally so this stub never leaks probe
            # resources / IAM bindings even on partial failure. Each
            # cleanup helper returns bool and is AND-ed into success. Only the
            # ACCEPTED set is deleted; unreconciled candidates are reported
            # instead, never deleted on an unproven claim.
            if created["target_role_allowed"]:
                ok = _revoke_target_role(
                    project=project,
                    zone=zone,
                    instance=instance,
                    token=caller_token,
                    sa_email=created["allowed_sa"],
                )
                if not ok:
                    cleanup_errors.append(f"revoke target role on {instance}")
            if created["probe_vm"]:
                ok = _delete_probe_vm(
                    project=project,
                    zone=zone,
                    token=caller_token,
                    name=created["probe_vm"],
                    invocation_id=invocation_id,
                )
                if not ok:
                    cleanup_errors.append(f"delete probe VM {created['probe_vm']}")
            for sa_email, created_flag in (
                (created["denied_sa"], created["token_creator_denied"]),
                (created["allowed_sa"], created["token_creator_allowed"]),
            ):
                if not sa_email:
                    continue
                if created_flag:
                    if not _revoke_token_creator(
                        project=project,
                        token=caller_token,
                        sa_email=sa_email,
                        member=caller_member,
                    ):
                        cleanup_errors.append(f"revoke tokenCreator on {sa_email}")
                if not _delete_service_account(
                    project=project,
                    token=caller_token,
                    email=sa_email,
                    invocation_id=invocation_id,
                ):
                    cleanup_errors.append(f"delete service account {sa_email}")
            for pending in unreconciled:
                # Fail closed: a candidate whose ownership could not be proven
                # may exist, so cleanup is NOT complete. Retain it as a handoff
                # rather than deleting a resource this run cannot prove it owns.
                cleanup_errors.append(
                    f"unreconciled {pending['kind']} {pending['identity']} "
                    f"(ownership unproven: {pending['detail']}); "
                    "retry with --reclaim-preserved on this step output"
                )
            if cleanup_errors or unreconciled:
                # UNCONFIRMED cleanup keeps the handoff, exactly like the
                # preservation and unreconciled paths above. A helper returning
                # False means "this fixture was not proven gone" — a denied or
                # transient delete, a delete whose 404 could not be reconciled
                # into a proven absence, a revoke whose policy write never
                # landed, or an inventory that could not prove absence — so the
                # probe VM, a probe SA, the TokenCreator grants, or the viewer
                # binding on the operator's own target VM may still be live.
                # Emitting the record only for skip-destroy / unreconciled would
                # discard the one machine-replayable ownership record for exactly
                # the runs that need it, and `--reclaim-preserved` rejects a
                # payload without it. The full record is retained (not pruned to
                # the failures): re-verifying a fixture that is genuinely gone is
                # idempotent success in reclamation, while dropping one that is
                # not gone is an unrecoverable leak. Which fixtures are
                # outstanding is in `cleanup_errors`.
                result["preserved_fixtures"] = fixture_record
                print(
                    "Console RBAC cleanup left fixtures unconfirmed "
                    f"({'; '.join(cleanup_errors)}). Ownership handoff retained in "
                    "preserved_fixtures; reclaim with: console_rbac.py "
                    "--reclaim-preserved <this-step-output.json>",
                    file=sys.stderr,
                )
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
        # Final success/rc: subtests AND cleanup AND no early failure.
        final_success = subtests_passed and not cleanup_errors and early_failure is None
        result["success"] = final_success
        if cleanup_errors and not result.get("error"):
            result["error"] = "console RBAC cleanup failed: " + "; ".join(cleanup_errors)
        elif cleanup_errors and result.get("error"):
            result["error"] = f"{result['error']}; cleanup failed: {'; '.join(cleanup_errors)}"
        return 0 if final_success else 1


def _reclaim_preserved(*, source: str, result: dict[str, Any]) -> int:
    """Reclaim console-RBAC probe fixtures a preserved or aborted run left behind.

    This is the later-cleanup handoff for ``--skip-destroy``, for candidates
    whose ownership a run could not reconcile in-step, and for fixtures an
    in-step delete or IAM revoke failed to remove. It replays the SAME
    ownership rules the in-step cleanup applies, from the step's own emitted
    payload — the only record of what this run created:

      * every delete is marker-verified against the payload's
        ``invocation_id``, so a resource that belongs to another run (or to a
        later run that reused the name) is reported and left alone rather than
        destroyed;
      * a proven-absent resource is idempotent success, so re-running after a
        partial reclamation is safe;
      * a denied or failing readback fails closed and keeps the handoff, so the
        operator can retry instead of believing cleanup finished.

    Reads ``-`` from stdin. Hand-written ``gcloud iam service-accounts delete``
    / ``compute instances delete`` cannot do any of that: they see no marker,
    and they miss the two IAM bindings (the target-VM role and the caller's
    TokenCreator grants) entirely.
    """
    raw = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        msg = f"console_rbac output must be a JSON object, got {type(payload).__name__}"
        raise ValueError(msg)
    record = payload.get("preserved_fixtures") or {}
    if not isinstance(record, dict) or not record.get("invocation_id"):
        result["error_type"] = "configuration_error"
        result["error"] = (
            "payload has no preserved_fixtures.invocation_id; only a console_rbac "
            "output that preserved its fixtures, or could not confirm their cleanup, "
            "can be reclaimed"
        )
        return 1

    project = str(record.get("project") or "")
    zone = str(record.get("zone") or "")
    invocation_id = str(record["invocation_id"])
    caller_token = _adc_access_token()
    caller_member = str(record.get("caller") or "") or _resolve_caller_member(caller_token)

    result.update(
        {
            "mode": "reclaim_preserved",
            "project": project,
            "zone": zone,
            "invocation_id": invocation_id,
            "caller": caller_member,
        }
    )
    reclaimed: list[str] = []
    cleanup_errors: list[str] = []

    # 1. Remove the target-VM binding first: it is the only fixture that lives
    #    on a resource this step does NOT own, so it is the one an operator
    #    most wants gone. It is identified by the recorded role + member, never
    #    by scanning the policy for bindings that "look like ours".
    target_instance = str(record.get("target_instance") or "")
    allowed_sa = str(record.get("allowed_sa") or "")
    if record.get("target_role_allowed") and target_instance and allowed_sa:
        if _revoke_target_role(
            project=project,
            zone=zone,
            instance=target_instance,
            token=caller_token,
            sa_email=allowed_sa,
        ):
            reclaimed.append(f"revoked {_ALLOWED_TARGET_ROLE} for {allowed_sa} on {target_instance}")
        else:
            cleanup_errors.append(f"revoke target role on {target_instance}")

    # 2. Probe VM and probe service accounts, each gated on a marker-verified
    #    readback of the exact recorded identity.
    candidates: list[dict[str, str]] = []
    if record.get("probe_vm"):
        candidates.append({"kind": "probe_vm", "identity": str(record["probe_vm"]), "zone": zone})
    for slot in ("denied_sa", "allowed_sa"):
        if record.get(slot):
            candidates.append({"kind": "service_account", "identity": str(record[slot]), "zone": ""})
    pending_fixtures = [p for p in payload.get("unreconciled_fixtures") or [] if isinstance(p, dict)]
    for pending in pending_fixtures:
        if pending.get("identity"):
            candidate = {
                "kind": str(pending.get("kind") or "service_account"),
                "identity": str(pending["identity"]),
                "zone": str(pending.get("zone") or zone),
            }
            if candidate not in candidates:
                candidates.append(candidate)

    token_creator_granted = {
        str(record.get("denied_sa") or ""): bool(record.get("token_creator_denied")),
        str(record.get("allowed_sa") or ""): bool(record.get("token_creator_allowed")),
    }
    for candidate in candidates:
        identity = candidate["identity"]
        if candidate["kind"] == "probe_vm":
            verdict, _, detail = _reconcile_probe_vm(
                project=project,
                zone=candidate["zone"] or zone,
                token=caller_token,
                name=identity,
                invocation_id=invocation_id,
            )
        else:
            verdict, _, detail = _reconcile_service_account(
                project=project,
                token=caller_token,
                email=identity,
                invocation_id=invocation_id,
            )
        if verdict == _ABSENT:
            reclaimed.append(f"{candidate['kind']} {identity} already absent")
            continue
        if verdict == _FOREIGN:
            cleanup_errors.append(f"refused to delete {candidate['kind']} {identity}: {detail}")
            continue
        if verdict == _INCONCLUSIVE:
            cleanup_errors.append(f"ownership of {candidate['kind']} {identity} unproven: {detail}")
            continue
        if candidate["kind"] == "probe_vm":
            if _delete_probe_vm(
                project=project,
                zone=candidate["zone"] or zone,
                token=caller_token,
                name=identity,
                invocation_id=invocation_id,
            ):
                reclaimed.append(f"deleted probe VM {identity}")
            else:
                cleanup_errors.append(f"delete probe VM {identity}")
            continue
        if token_creator_granted.get(identity) and not _revoke_token_creator(
            project=project,
            token=caller_token,
            sa_email=identity,
            member=caller_member,
        ):
            cleanup_errors.append(f"revoke tokenCreator on {identity}")
        if _delete_service_account(
            project=project,
            token=caller_token,
            email=identity,
            invocation_id=invocation_id,
        ):
            reclaimed.append(f"deleted service account {identity}")
        else:
            cleanup_errors.append(f"delete service account {identity}")

    result["reclaimed"] = reclaimed
    if cleanup_errors:
        result["cleanup_errors"] = cleanup_errors
        result["error"] = "console RBAC reclamation incomplete: " + "; ".join(cleanup_errors)
        # An incomplete reclamation carries the SAME handoff forward, so this
        # output is itself replayable: an operator who kept only the reclaim
        # payload (the natural thing to keep after running it) would otherwise
        # hold a record of a leak with no way to retry it, because this function
        # rejects any payload without preserved_fixtures.invocation_id. The
        # ownership record and the still-unreconciled candidates are re-emitted
        # unpruned for the same reason the in-step handoff is: a fixture is
        # released from the record only when it is proven absent, and a
        # re-verified absence costs one readback while a dropped identity costs
        # a permanent leak.
        result["preserved_fixtures"] = record
        if pending_fixtures:
            result["unreconciled_fixtures"] = pending_fixtures
    result["success"] = not cleanup_errors
    return 0 if result["success"] else 1


def _preprovisioned_probe(
    *,
    project: str,
    zone: str,
    instance: str,
    denied_sa: str,
    allowed_sa: str,
    other_instance: str,
    other_zone: str,
    result: dict[str, Any],
) -> int:
    """Run the pre-provisioned RBAC probe with operator-supplied principals.

    The fallback path for projects where IAM mutation is not allowed.
    Operators must pre-create denied / allowed SAs, grant the caller
    TokenCreator on both, scope the allowed SA's serial-output role to
    the target VM, and create a real ``GCP_OTHER_INSTANCE_ID`` that the
    allowed SA has NOT been granted access to.

    Workflow exceptions (ADC failure, token-mint propagation timeout,
    HTTP errors from getSerialPortOutput) are caught here so the
    contract-shaped result populated by ``main()`` survives — mirrors
    the AWS oracle's try/except around its console RBAC workflow.
    Escaping to ``handle_gcp_errors`` would drop ``platform``,
    ``test_name``, ``rbac_model``, ``access_restricted``, and
    ``tests.*`` from the printed JSON.
    """
    try:
        caller_token = _adc_access_token()
        result["caller"] = _resolve_caller_member(caller_token)
        denied_token = _mint_access_token(sa_email=denied_sa)
        allowed_token = _mint_access_token(sa_email=allowed_sa)

        denied_status, denied_evidence = _probe_serial_console(
            project=project,
            zone=zone,
            instance=instance,
            access_token=denied_token,
        )
        result["tests"]["denied_principal_cannot_access_console"] = {
            "passed": denied_status == 403,
            "principal": f"serviceAccount:{denied_sa}",
            "evidence": denied_evidence,
        }

        allowed_status, allowed_evidence = _probe_serial_console(
            project=project,
            zone=zone,
            instance=instance,
            access_token=allowed_token,
        )
        result["tests"]["allowed_principal_can_access_console"] = {
            "passed": allowed_status == 200,
            "principal": f"serviceAccount:{allowed_sa}",
            "evidence": allowed_evidence,
        }

        scope_status, scope_evidence = _probe_serial_console(
            project=project,
            zone=other_zone,
            instance=other_instance,
            access_token=allowed_token,
        )
        result["tests"]["allowed_principal_is_resource_scoped"] = {
            "passed": scope_status == 403,
            "principal": f"serviceAccount:{allowed_sa}",
            "evidence": f"other_instance={other_instance}; {scope_evidence}",
        }

        result["access_restricted"] = (
            result["tests"]["denied_principal_cannot_access_console"]["passed"]
            and result["tests"]["allowed_principal_is_resource_scoped"]["passed"]
        )
        all_passed = all(t["passed"] for t in result["tests"].values())
        result["success"] = all_passed
        if not all_passed:
            result["error"] = "one or more console RBAC subtests failed; see tests.* evidence"
        return 0 if all_passed else 1
    except Exception as e:
        result["success"] = False
        result["access_restricted"] = False
        result["error"] = str(e)
        return 1


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Console RBAC probe (Compute Engine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_RECLAIM_EPILOG,
    )
    parser.add_argument("--instance-id", default=None, help="Target instance name")
    parser.add_argument("--region", default=None, help="GCP region or zone")
    parser.add_argument("--zone", default=None, help="GCP zone (overrides region)")
    parser.add_argument("--project", default=None, help="GCP project ID (ADC fallback)")
    parser.add_argument(
        "--network",
        default="default",
        help="Network for the probe VM (self-provisioned path)",
    )
    parser.add_argument(
        "--skip-destroy",
        action="store_true",
        help=(
            "Preserve the probe fixtures this step creates (both probe service "
            "accounts, the probe VM, and both IAM bindings) instead of deleting "
            "them in the cleanup block (GCP_VM_SKIP_TEARDOWN passthrough). The "
            "same resolved preservation decision the terminal teardown step "
            "receives; reclaim the retained fixtures later with "
            "--reclaim-preserved."
        ),
    )
    parser.add_argument(
        "--reclaim-preserved",
        default=None,
        help=(
            "Path to a saved console_rbac JSON payload ('-' for stdin) whose "
            "preserved_fixtures record is replayed to delete the probe fixtures "
            "a run retained -- because --skip-destroy preserved them, because "
            "their ownership could not be reconciled, or because a delete or "
            "IAM revoke could not confirm they were gone. Every delete is "
            "marker-verified against that record. See the epilog below."
        ),
    )
    args = parser.parse_args()

    if args.reclaim_preserved:
        reclaim_result: dict[str, Any] = {
            "success": False,
            "platform": "vm",
            "test_name": "console_rbac_reclaim",
        }
        try:
            rc = _reclaim_preserved(source=args.reclaim_preserved, result=reclaim_result)
        except (OSError, ValueError) as e:
            reclaim_result["error_type"] = "configuration_error"
            reclaim_result["error"] = f"Unable to read console_rbac output {args.reclaim_preserved!r}: {e}"
            rc = 1
        print(json.dumps(reclaim_result, indent=2, default=str))
        return rc

    if not args.instance_id or not (args.zone or args.region):
        parser.error("--instance-id and one of --zone / --region are required unless --reclaim-preserved is given")

    project = resolve_project(args.project)
    zone = args.zone or narrow_region_to_zone(args.region)

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "test_name": "console_rbac",
        "instance_id": args.instance_id,
        "project": project,
        "zone": zone,
        "rbac_model": "gcp-iam",
        "access_restricted": False,
        "restricted_actions": list(_RESTRICTED_ACTIONS),
        "tests": {
            "denied_principal_cannot_access_console": {"passed": False, "principal": "", "evidence": ""},
            "allowed_principal_can_access_console": {"passed": False, "principal": "", "evidence": ""},
            "allowed_principal_is_resource_scoped": {"passed": False, "principal": "", "evidence": ""},
        },
    }

    denied_sa = os.environ.get(_DENIED_SA_ENV, "").strip()
    allowed_sa = os.environ.get(_ALLOWED_SA_ENV, "").strip()
    other_instance = os.environ.get(_OTHER_INSTANCE_ENV, "").strip()
    other_zone = os.environ.get(_OTHER_INSTANCE_ZONE_ENV, "").strip() or zone

    # The pre-provisioned fallback runs only when the operator supplies
    # all three env vars. Otherwise the default self-provisioned path
    # runs (unless explicitly disabled via _SELF_PROVISION_ENABLED_ENV=0).
    if denied_sa and allowed_sa and other_instance:
        result["mode"] = "preprovisioned"
        rc = _preprovisioned_probe(
            project=project,
            zone=zone,
            instance=args.instance_id,
            denied_sa=denied_sa,
            allowed_sa=allowed_sa,
            other_instance=other_instance,
            other_zone=other_zone,
            result=result,
        )
        print(json.dumps(result, indent=2, default=str))
        return rc

    if os.environ.get(_SELF_PROVISION_ENABLED_ENV, "1").strip() in {"0", "false", "no"}:
        # Intentional opt-out via env var. Treat as a clean policy-skip
        # (rc=0, success=True, skipped=True) — same shape as deploy_nim's
        # missing-NGC_API_KEY skip — so the orchestrator's
        # StepSuccessCheck reads this as "step short-circuited cleanly,"
        # not as a failed RBAC probe.
        result["mode"] = "skipped"
        result["skipped"] = True
        result["success"] = True
        result["skip_reason"] = (
            f"{_SELF_PROVISION_ENABLED_ENV} disables the self-provisioned probe and no "
            f"{_DENIED_SA_ENV} / {_ALLOWED_SA_ENV} / {_OTHER_INSTANCE_ENV} fallback was supplied"
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    result["mode"] = "self_provisioned"
    rc = _self_provisioned_probe(
        project=project,
        zone=zone,
        instance=args.instance_id,
        network=args.network,
        result=result,
        skip_destroy=args.skip_destroy,
    )
    print(json.dumps(result, indent=2, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())

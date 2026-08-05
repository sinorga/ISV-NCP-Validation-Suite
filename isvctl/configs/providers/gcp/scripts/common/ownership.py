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

"""Ownership transfer for GCP creates with ambiguous acknowledgements.

Create APIs can commit a resource and then lose the response.  A caller may
claim cleanup ownership only after either receiving the create acknowledgement
or reading back the exact resource with this invocation's marker.

Reconciling that readback is TRI-STATE, never a boolean.  "Found and marked by
this invocation" is ownership; "found without our marker" and "proven absent"
are both conclusive not-ours; but a lookup that was DENIED or kept failing
answered nothing at all.  Collapsing that last case into "not ours" is exactly
how a committed-but-unacknowledged resource is dropped from cleanup and leaks,
so it gets its own verdict and its own handoff channel (``on_unreconciled``)
that the caller persists for a later marker-verified reclamation pass.

An exact-identity 409 is likewise not a definite failure: it proves a candidate
EXISTS, not who created it.  It is reconciled against the invocation marker
before any generic reuse/adoption decision, so a retry of this invocation's own
create is recognised as ours instead of being adopted and preserved.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from secrets import token_hex
from typing import Any

from google.api_core import exceptions as gax

from common.errors import classify_gcp_error, is_transport_disconnect

INVOCATION_LABEL = "isv-invocation"
INVOCATION_DESCRIPTION_KEY = "isv-invocation"
CREATED_BY_LABEL = "created-by"
CREATED_BY_VALUE = "isvtest"
CREATED_BY_DESCRIPTION = f"{CREATED_BY_LABEL}={CREATED_BY_VALUE}"
DEFAULT_READBACK_ATTEMPTS = 3
DEFAULT_READBACK_BACKOFF_SECONDS = 1.0

_CONFLICT_CLASS_NAMES = {"Aborted", "AlreadyExists", "Conflict"}

# Create-outcome buckets. Only CREATE_FAILED proves nothing was committed (a
# conclusive pre-commit refusal: denied, unauthenticated, invalid argument,
# missing parent). CREATE_AMBIGUOUS is "the client cannot prove the server did
# NOT create it" — a dropped connection, a 429/5xx, or an unclassifiable error.
# CREATE_CONFLICT is the exact-identity 409: a candidate demonstrably exists,
# but the status alone does not say whether THIS invocation's earlier attempt
# committed it (a retried create whose first response was lost) or whether it
# belongs to another run. Both non-failed buckets must be reconciled against the
# per-invocation marker before the caller decides ownership.
CREATE_FAILED = "failed"
CREATE_CONFLICT = "conflict"
CREATE_AMBIGUOUS = "ambiguous"

# Reconciliation verdicts. Same vocabulary the console_rbac REST fixtures use so
# both reconcilers read identically. OWNED/FOREIGN/ABSENT are conclusive;
# INCONCLUSIVE means the lookup could not answer the existence question at all,
# which is NOT proof of absence and MUST retain a cleanup handoff.
RECONCILE_OWNED = "owned"
RECONCILE_FOREIGN = "foreign"
RECONCILE_ABSENT = "absent"
RECONCILE_INCONCLUSIVE = "inconclusive"

# Buckets where re-reading cannot change the answer: the caller is not permitted
# to observe the resource (or has no usable credentials), so further attempts
# only burn the enclosing step budget while staying inconclusive.
_UNOBSERVABLE_BUCKETS = frozenset({"access_denied", "credentials_invalid", "credentials_missing"})

# Field/record separators for the packed unresolved-candidate handoff. GCP
# resource names, project ids, and zones are [a-z0-9-] only, so neither
# separator can occur inside a real field; ``pack`` still scrubs them so a
# malformed input cannot shift the record layout.
_CANDIDATE_FIELD_SEP = "|"
_CANDIDATE_RECORD_SEP = ","
_CANDIDATE_FALSY = frozenset({"", "none", "null", "false"})


def new_invocation_id() -> str:
    """Return a GCP-label-safe invocation discriminator."""
    return token_hex(16)


def labels_with_invocation(labels: dict[str, str] | None, invocation_id: str) -> dict[str, str]:
    """Return ``labels`` with the invocation marker added without dropping ownership labels."""
    return {**(labels or {}), INVOCATION_LABEL: invocation_id}


def invocation_marker(invocation_id: str) -> str:
    """Return the literal marker text stamped into description-only resources.

    Single definition of the marker string so the writer (``description_with_
    invocation``) and every reader — typed SDK resources via
    ``has_invocation_description`` and raw REST JSON via
    ``description_carries_invocation`` — can never drift apart.
    """
    return f"{INVOCATION_DESCRIPTION_KEY}={invocation_id}"


def description_with_invocation(description: str, invocation_id: str) -> str:
    """Append an invocation marker to a description-only GCP resource."""
    return f"{description} ({invocation_marker(invocation_id)})"


def description_carries_invocation(description: str | None, invocation_id: str) -> bool:
    """Return whether a description STRING echoes ``invocation_id``.

    The string form is what a raw REST reader has: ``serviceAccounts.get`` /
    ``instances.get`` over HTTP return plain JSON dicts, not typed SDK objects,
    so ``getattr(resource, "description")`` would silently read ``""`` from a
    dict and turn every ownership question into "not ours".
    """
    return invocation_marker(invocation_id) in str(description or "")


def has_invocation_label(resource: Any, invocation_id: str) -> bool:
    """Return whether a label-capable resource echoes ``invocation_id``."""
    labels = dict(getattr(resource, "labels", None) or {})
    return labels.get(INVOCATION_LABEL) == invocation_id


def has_invocation_description(resource: Any, invocation_id: str) -> bool:
    """Return whether a description-only resource echoes ``invocation_id``."""
    return description_carries_invocation(getattr(resource, "description", ""), invocation_id)


@dataclass(frozen=True)
class UnreconciledCandidate:
    """One create candidate whose ownership could not be decided.

    Carries the full provenance a later reclamation pass needs to re-verify the
    invocation marker itself: resource kind, exact name, project, zone (empty
    for global resources such as firewall rules), and the invocation id whose
    marker the resource must echo before anything may delete it.

    ``pack`` renders the record as a single delimited token so it survives the
    scalar-only argv channel between an emitting step and a teardown step, and
    ``parse`` is its exact inverse — one definition of the wire shape, so the
    producer and the consumer cannot drift apart.
    """

    resource_type: str
    name: str
    project: str
    zone: str = ""
    invocation_id: str = ""

    def describe(self) -> str:
        """Return a human label for logs and structured warnings."""
        scope = f"@{self.zone}" if self.zone else ""
        return f"{self.resource_type} {self.name}{scope}"

    def pack(self) -> str:
        """Render this candidate as one separator-safe record."""
        return _CANDIDATE_FIELD_SEP.join(
            _scrub_candidate_field(value)
            for value in (self.resource_type, self.name, self.project, self.zone, self.invocation_id)
        )

    @classmethod
    def parse(cls, record: str) -> UnreconciledCandidate | None:
        """Rebuild a candidate from ``pack`` output; ``None`` when unusable.

        A record without a kind or a name names nothing deletable, so it is
        dropped rather than turned into a half-identified cleanup target.
        """
        fields = [field.strip() for field in str(record or "").split(_CANDIDATE_FIELD_SEP)]
        if len(fields) != 5 or not fields[0] or not fields[1]:
            return None
        return cls(
            resource_type=fields[0],
            name=fields[1],
            project=fields[2],
            zone=fields[3],
            invocation_id=fields[4],
        )


def _scrub_candidate_field(value: str) -> str:
    """Strip record/field separators out of one packed-candidate field."""
    text = str(value or "").strip()
    for separator in (_CANDIDATE_FIELD_SEP, _CANDIDATE_RECORD_SEP):
        text = text.replace(separator, "-")
    return text


def parse_unreconciled_records(text: str | None) -> list[UnreconciledCandidate]:
    """Parse the comma-joined packed-candidate form into candidates.

    Accepts exactly what the provider config forwards: the ``join(',')`` of an
    emitting step's packed records, or the ``'none'`` / empty sentinel meaning
    "no unresolved candidate was handed off". Unparseable records are skipped so
    one malformed entry cannot abort the reclamation of the others.
    """
    candidates: list[UnreconciledCandidate] = []
    for record in str(text or "").split(_CANDIDATE_RECORD_SEP):
        if record.strip().lower() in _CANDIDATE_FALSY:
            continue
        candidate = UnreconciledCandidate.parse(record)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def classify_create_outcome(error: Exception) -> str:
    """Classify a failed create as ``failed`` / ``conflict`` / ``ambiguous``.

    The create path branches on THIS, not on the two-state
    ``create_may_have_committed``: an exact-identity 409 is a reconcilable
    candidate (it may be this invocation's own committed retry), never a proven
    failure. Only a conclusive pre-commit refusal is ``failed``.
    """
    bucket = classify_gcp_error(error)[0]
    if isinstance(error, gax.Conflict) or bucket == "conflict" or type(error).__name__ in _CONFLICT_CLASS_NAMES:
        return CREATE_CONFLICT
    if is_transport_disconnect(error):
        return CREATE_AMBIGUOUS
    return CREATE_AMBIGUOUS if bucket in {"transient", "api_error", "unknown_error"} else CREATE_FAILED


def create_may_have_committed(error: Exception) -> bool:
    """Return whether a failed NON-CONFLICT mutation may have landed server-side.

    Two-state predicate for MUTATION paths (e.g. a metadata write) where an
    exact-identity 409 proves the write was rejected against the observed
    resource state, so it is correctly not "may have landed". Create paths must
    use ``classify_create_outcome`` instead, which keeps that 409 reconcilable.
    """
    return classify_create_outcome(error) == CREATE_AMBIGUOUS


def reconcile_owned[T](
    read_back: Callable[[], T],
    owns: Callable[[T], bool],
    *,
    attempts: int = DEFAULT_READBACK_ATTEMPTS,
    backoff_seconds: float = DEFAULT_READBACK_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, str]:
    """Boundedly read an exact resource and decide ownership; ``(verdict, detail)``.

    The verdict comes ONLY from the per-invocation marker, so a same-named
    resource from another run reports ``foreign`` and is never adopted for
    deletion:

      * ``owned`` — found, and it echoes this invocation's marker;
      * ``foreign`` — found, but the marker is another run's (or absent);
      * ``absent`` — EVERY bounded attempt returned NotFound. The exact
        project-scoped get is authoritative about existence, and the retry
        window covers a just-committed create that has not propagated yet;
      * ``inconclusive`` — the lookup itself was denied or kept failing, so it
        answered nothing. Callers MUST retain the cleanup handoff here rather
        than assume absence. Denied/credential buckets return immediately
        because re-reading cannot change the answer.

    The unobserved state is MONOTONIC across attempts, and that is the whole
    point of tracking it: a run of attempts that goes "transport failure, then
    NotFound" is a MIXED sequence, not proof of absence. The attempt that could
    not observe the resource never answered the existence question, so the later
    NotFound only proves the resource was missing on that one read — which is
    exactly the shape a just-committed create still propagating produces. Per
    the docstring contract above, ``absent`` therefore requires at least one
    attempt AND every attempt to have been a conclusive NotFound; any unobserved
    attempt anywhere in the sequence downgrades the verdict to ``inconclusive``
    so the caller's ``on_unreconciled`` handoff fires and the committed-but-
    unproven candidate stays in the cleanup set.
    """
    saw_unobserved = False
    not_found_attempts = 0
    detail = "readback not attempted"
    unobserved_detail = ""
    for attempt in range(1, attempts + 1):
        try:
            resource = read_back()
        except gax.NotFound:
            detail = "readback found nothing (NotFound)"
            not_found_attempts += 1
            resource = None
        except Exception as exc:
            bucket, message = classify_gcp_error(exc)
            detail = f"readback failed: {message}"
            unobserved_detail = detail
            saw_unobserved = True
            resource = None
            if bucket in _UNOBSERVABLE_BUCKETS:
                return RECONCILE_INCONCLUSIVE, detail

        if resource is not None:
            if owns(resource):
                return RECONCILE_OWNED, "readback marker matched"
            return RECONCILE_FOREIGN, "readback found a resource without this invocation's marker"
        if attempt < attempts:
            sleep(backoff_seconds * attempt)

    # Every loop exit here was either a NotFound or an unobserved attempt, so
    # "no unobserved attempt AND at least one NotFound" is exactly "all attempts
    # were conclusively NotFound". The `> 0` guard also keeps a zero-attempt
    # envelope from reporting absence it never looked for.
    if not_found_attempts > 0 and not saw_unobserved:
        return RECONCILE_ABSENT, detail
    return RECONCILE_INCONCLUSIVE, unobserved_detail or detail


def submit_owned_create[T](
    submit: Callable[[], T],
    read_back: Callable[[], Any],
    owns: Callable[[Any], bool],
    *,
    on_accepted: Callable[[], None] | None = None,
    on_unreconciled: Callable[[Exception, str], None] | None = None,
    readback_attempts: int = DEFAULT_READBACK_ATTEMPTS,
    readback_backoff_seconds: float = DEFAULT_READBACK_BACKOFF_SECONDS,
) -> T:
    """Submit a marked create and transfer cleanup ownership exactly once.

    A successful acknowledgement transfers ownership through ``on_accepted``
    before the caller waits on any asynchronous operation.

    An ambiguous error AND an exact-identity conflict both reconcile the exact
    candidate against this invocation's marker first:

      * marker matched — ``on_accepted`` fires, so a resource this invocation
        committed before losing the response (or before its own retry drew the
        409) is inside the caller's cleanup set;
      * conclusively not ours — nothing fires, and the caller's conflict/reuse
        path decides what to do with a resource it did not create;
      * inconclusive — ``on_unreconciled(error, detail)`` fires so the caller
        persists the candidate for a later marker-verified reclamation instead
        of silently dropping a handoff for a resource that may exist.

    The original error is always re-raised afterwards, so the caller's normal
    failure path runs with truthful ownership state. Callers whose cleanup gates
    on a ``*_created``-style flag MUST pass ``on_accepted``; callers whose
    teardown can reclaim later SHOULD also pass ``on_unreconciled``.
    """
    try:
        result = submit()
    except Exception as exc:
        # No ownership channel means the caller holds no ownership flag to
        # stamp and no handoff to retain (it deletes unconditionally by name),
        # so the reconciling lookup would cost an API call and inform nobody.
        if (on_accepted is None and on_unreconciled is None) or classify_create_outcome(exc) == CREATE_FAILED:
            raise
        verdict, detail = reconcile_owned(
            read_back,
            owns,
            attempts=readback_attempts,
            backoff_seconds=readback_backoff_seconds,
        )
        if verdict == RECONCILE_OWNED:
            if on_accepted is not None:
                on_accepted()
        elif verdict == RECONCILE_INCONCLUSIVE and on_unreconciled is not None:
            on_unreconciled(exc, detail)
        raise
    if on_accepted is not None:
        on_accepted()
    return result

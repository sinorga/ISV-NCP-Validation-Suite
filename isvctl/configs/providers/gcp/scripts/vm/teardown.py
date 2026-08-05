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

"""Teardown a Compute Engine VM + verified-reuse companions.

Mirrors the AWS oracle's teardown.py shape (instance + SG + key pair),
translated to Compute Engine:

  * ``instances.delete`` is zonal; ``firewalls.delete`` is project-global.
  * Instance, firewall, and local key pair are ALL verified-reuse —
    destruction MUST gate on the ``_created: bool`` flags forwarded
    from launch_instance via ``{{steps.launch_instance.X}}`` (cleanup
    contract). For the instance this
    means a run started with ``GCP_VM_INSTANCE_ID`` / ``GCP_VM_KEY_FILE``
    against an operator-supplied long-lived VM emits
    ``instance_created=False`` and teardown skips both the primary and
    every leaked-zone delete so the adopted VM survives.
  * ``--skip-destroy`` short-circuits to success BEFORE resolving the
    project, so an expired-ADC environment can still no-op cleanly —
    preservation-mode flags MUST be evaluated before any auth-resolving
    helper.
  * NotFound on the cloud-side preflight is idempotent SUCCESS for the
    instance read, but must NOT short-circuit local PEM/.pub cleanup —
    NotFound-on-cloud-read idempotency must not short-circuit
    local-artifact cleanup.
  * Each cleanup helper returns ``bool``; the final ``success`` is the
    AND of every per-resource bool — helpers that return ``bool`` for
    batch-cleanup safety MUST surface the bool into
    ``result['success']``.
  * ``success`` and ``resources_destroyed`` are DIFFERENT questions and are
    computed separately. ``success`` asks "did every gated cleanup do what it
    was asked to do"; ``resources_destroyed`` asks "is the instance gone".
    They diverge on every preservation path — ``--skip-destroy`` and the
    verified-reuse adoption gate below — where cleanup succeeds precisely by
    NOT deleting the instance. Deriving one from the other there would report
    a still-running, still-billable VM as destroyed, so
    ``resources_destroyed`` stays False whenever an instance is preserved and
    turns True only once the instance was deleted or confirmed absent AND
    every other gate succeeded.
  * ``--unreconciled-resources`` carries launch_instance's ambiguous-create
    candidates: resources that may have been committed before the response
    was lost, whose ownership its readback could not settle. They arrive
    WITHOUT a ``*_created`` bit because none could honestly be set, so this
    stub re-verifies the per-invocation marker on each exact resource before
    deleting it. A marker mismatch is another run's resource and is NEVER
    deleted; a lookup that stays denied or failing deletes nothing either and
    fails the teardown honestly so the operator can replay the reclamation.
    Discarding a candidate as absent takes the same bounded, monotonic
    readback the creating step used: EVERY attempt must return NotFound, since
    this step is the handoff's terminal consumer and a lone NotFound is what a
    committed-but-unpropagated create looks like.

Sentinel handling: the provider config wires bool / path args with the
non-empty defaults ``'none'`` / ``'false'`` — forwarded inter-step
Jinja args MUST use ``| default(<NON-EMPTY sentinel>)``. The stub
treats ``none`` / ``null`` / ``""`` / ``false`` as "no artifact
tracked". Every such flag defaults to ``None`` in ``argparse`` so an
unset flag stays distinguishable from an explicitly forwarded sentinel;
both resolve to "no artifact tracked" downstream.

Standalone operator cleanup: ``--from-launch-output`` replays a saved
``launch_instance`` JSON payload through these same gates, which is the
ONLY safe way to reclaim resources a preserved or aborted run left
behind. That payload is the sole record of the three ownership bits and
of the capacity walk's ``leaked_zones``; hand-written ``gcloud`` deletes
can read neither, so they destroy adopted operator resources on the
verified-reuse path and miss phantom instances outside the primary zone.
See the parser epilog (``teardown.py --help``) for the recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

from common.compute import (
    delete_local_keypair,
    get_instance,
    narrow_region_to_zone,
    resolve_project,
    wait_for_global_op,
    wait_for_zonal_op,
)
from common.errors import delete_with_retry, handle_gcp_errors, retry_idempotent
from common.ownership import (
    RECONCILE_ABSENT,
    RECONCILE_FOREIGN,
    RECONCILE_INCONCLUSIVE,
    RECONCILE_OWNED,
    UnreconciledCandidate,
    has_invocation_description,
    parse_unreconciled_records,
    reconcile_owned,
)
from google.api_core import exceptions as gax
from google.cloud import compute_v1

_FALSY_SENTINELS = {"", "none", "null", "false"}

# Bound the per-attempt wait so the 3-attempt delete_with_retry
# does not multiply zonal/global operation budgets into the
# enclosing teardown step timeout. The leaked-zones walk iterates over
# multiple zones with the same retry helper; without this bound, a
# transient throttle on a single zone could exhaust the step budget
# before later zones are even attempted. Firewall deletes have exceeded
# 120s in live GCP runs, so keep the global-op wait comfortably above
# that observed path while still inside the enclosing teardown budget.
_TEARDOWN_INSTANCE_WAIT_S = 180
_TEARDOWN_FIREWALL_WAIT_S = 300

# Bounded readback envelope for ambiguous-create candidate verification. The
# candidate arrived precisely BECAUSE its create response was lost, so it may be
# a committed resource that has not become readable yet: a single NotFound is a
# propagation artifact, not proof of absence. Three attempts with escalating
# backoff (2s + 4s) bound the reclamation of each candidate well inside the
# teardown step budget while still spanning the window a just-committed create
# needs to surface.
_CANDIDATE_RECONCILE_ATTEMPTS = 3
_CANDIDATE_RECONCILE_BACKOFF_S = 2.0

_STANDALONE_CLEANUP_EPILOG = """\
Standalone operator cleanup (ownership-gated)
---------------------------------------------
When a run preserved its resources (GCP_VM_SKIP_TEARDOWN=true) or aborted
before its teardown phase, reclaim them by replaying that run's
launch_instance step output through this stub:

    python3 teardown.py --from-launch-output launch_instance.json \\
        --delete-security-group --delete-key-pair

Save the launch_instance JSON to a file first, or pipe it in with
`--from-launch-output -`. Explicit flags still win over the payload.

Use this instead of hand-written delete commands, because the payload is the
only record of what the run actually owned:

  * every delete is gated on the matching ownership bit from that payload --
    instance_created, firewall_created, key_created -- so a run that ADOPTED
    an operator-supplied long-lived VM, firewall rule, or private key (the
    verified-reuse path, where all three are false) preserves them instead of
    destroying them;
  * the instance is deleted in the primary zone AND in every leaked_zones
    entry the multi-zone capacity walk touched, so a phantom record cannot be
    left behind billing in a zone the operator never sees;
  * every unreconciled_resources candidate -- a create whose acknowledgement
    was lost and whose ownership the run could not prove -- is re-checked
    against its recorded invocation marker and deleted only when the marker
    still matches, so the reclamation cannot destroy another run's resource;
  * NotFound is idempotent success, so re-running after a partial cleanup is
    safe.

`gcloud compute instances delete` / `firewall-rules delete` / `rm -f <key>`
typed out by hand see none of that: they cannot read the ownership bits, and
they only ever name one zone.
"""

# launch_instance output key -> teardown argparse attribute. Only the fields
# that carry identity, ownership, or zone scope are consumed; everything else
# in that payload is diagnostic and has no cleanup meaning.
_LAUNCH_OUTPUT_FIELDS: tuple[tuple[str, str], ...] = (
    ("instance_id", "instance_id"),
    ("zone", "zone"),
    ("region", "region"),
    ("instance_created", "instance_created"),
    ("firewall_name", "firewall_name"),
    ("firewall_created", "firewall_created"),
    ("key_file", "key_file"),
    ("key_created", "key_created"),
    ("leaked_zones", "leaked_zones"),
    # Packed ambiguous-create candidates. The payload holds the same list the
    # provider config comma-joins into --unreconciled-resources, so replay and
    # orchestrated teardown feed one parser.
    ("unreconciled_resources", "unreconciled_resources"),
)


def _truthy(arg: str | None) -> bool:
    """Per-arg sentinel check. Treats both "" / 'none' / 'null' / 'false' as falsy."""
    if arg is None:
        return False
    return arg.strip().lower() not in _FALSY_SENTINELS


def _as_arg_text(value: Any) -> str | None:
    """Render one launch_instance output value the way the config would pass it.

    Bools become the ``true`` / ``false`` sentinels the ownership gates parse,
    lists become the comma-joined form ``--leaked-zones`` expects, and a JSON
    null becomes ``None`` so the field stays "not supplied" rather than
    becoming the string ``"None"`` (which ``_truthy`` would read as a real
    artifact name).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return str(value)


def _apply_launch_output(args: argparse.Namespace, source: str) -> None:
    """Backfill unset teardown arguments from a saved launch_instance payload.

    The payload carries the identifiers, the three ownership bits
    (``instance_created`` / ``firewall_created`` / ``key_created``), and every
    ``leaked_zones`` entry the capacity walk touched — so feeding it back here
    reproduces exactly the gates the orchestrated teardown applies, with no
    operator transcription step that could drop one. Reads ``-`` from stdin.

    Only arguments still unset (``None``) are filled: an explicit flag is the
    operator overriding the record on purpose and always wins.
    """
    raw = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"launch output must be a JSON object, got {type(payload).__name__}")
    for key, attr in _LAUNCH_OUTPUT_FIELDS:
        if getattr(args, attr, None) is not None:
            continue
        rendered = _as_arg_text(payload.get(key))
        if rendered is not None:
            setattr(args, attr, rendered)


def _delete_instance_op(project: str, zone: str, name: str) -> None:
    """Delete an instance and wait on the zonal op (NotFound is idempotent)."""
    try:
        op = compute_v1.InstancesClient().delete(project=project, zone=zone, instance=name)
    except gax.NotFound:
        return
    op_name = getattr(op, "name", None) or getattr(op, "operation", "")
    if op_name:
        wait_for_zonal_op(project, zone, op_name, timeout=_TEARDOWN_INSTANCE_WAIT_S)


def _delete_firewall_op(project: str, name: str) -> None:
    """Delete a firewall rule and wait on the global op (NotFound is idempotent)."""
    try:
        op = compute_v1.FirewallsClient().delete(project=project, firewall=name)
    except gax.NotFound:
        return
    op_name = getattr(op, "name", None) or getattr(op, "operation", "")
    if op_name:
        wait_for_global_op(project, op_name, timeout=_TEARDOWN_FIREWALL_WAIT_S)


def _owned_resource_present(read_back: Callable[[], Any], *, desc: str) -> tuple[bool, str]:
    """Bounded existence probe for an ALREADY-OWNED resource; ``(present, detail)``.

    Ownership is settled before this is called — the forwarded ``*_created`` bit
    proves the run created the resource — so the only open question is whether it
    is still there, and the answer is skewed deliberately toward "present". A
    single read is not allowed to retire an owned resource: the same
    unpropagated-read window that makes a lone ``NotFound`` meaningless for an
    ambiguous candidate makes it meaningless here, and ``retry_idempotent`` does
    not retry ``NotFound``, so nothing else in the stack takes a second look.

    Absence is therefore reported only when EVERY bounded attempt returned a
    conclusive ``NotFound``. A read that failed for any other reason observed
    nothing at all, so the sequence stays "present" and the caller issues its
    delete — which is free when the resource really is gone, because
    ``delete_with_retry`` already counts an already-absent resource as success.
    Cheap in the common case too: a resource that IS present answers on the very
    first attempt, so only the genuinely-absent path pays the backoff.
    """
    detail = f"{desc}: presence not probed"
    for attempt in range(1, _CANDIDATE_RECONCILE_ATTEMPTS + 1):
        try:
            read_back()
        except gax.NotFound:
            detail = f"{desc} not found"
        except Exception as e:
            return True, f"{desc} presence read failed ({e}); treating as present"
        else:
            return True, f"{desc} present"
        if attempt < _CANDIDATE_RECONCILE_ATTEMPTS:
            time.sleep(_CANDIDATE_RECONCILE_BACKOFF_S * attempt)
    return False, f"{detail} on every bounded attempt"


def _verify_candidate_ownership(project: str, candidate: UnreconciledCandidate) -> tuple[str, str]:
    """Re-read one unresolved candidate and decide ownership; ``(verdict, detail)``.

    The verdict comes ONLY from the per-invocation marker the creating step
    stamped into the resource description, never from the name — a run-scoped
    name can still be reused by a later run, and deleting on the name alone is
    how a reclamation pass destroys someone else's VM. A candidate that arrives
    without a marker is reported inconclusive rather than deleted on faith.

    Absence is decided by the SAME bounded, monotonic envelope the creating step
    used (``reconcile_owned``), never by a single read. This is the terminal
    consumer of the ambiguous-create handoff: the verdict here either deletes the
    candidate or discards the only record that it might exist, so "absent" has to
    mean absent. One ``NotFound`` cannot carry that weight — the candidate exists
    precisely because a create response was lost, and a committed create that has
    not propagated yet answers a read with exactly that ``NotFound``. Worse,
    ``retry_idempotent`` deliberately does not retry ``NotFound``, so a single
    read gave the loop no second look either. ``reconcile_owned`` reports
    ``absent`` only when EVERY bounded attempt was a conclusive ``NotFound`` and
    downgrades any mixed or failed sequence to ``inconclusive``, which keeps the
    handoff (and fails the teardown honestly) instead of reporting a clean
    cleanup for a resource that can still surface afterwards.
    """
    if not candidate.invocation_id:
        return RECONCILE_INCONCLUSIVE, "no invocation marker recorded; ownership cannot be proven"

    if candidate.resource_type == "instance":
        if not candidate.zone:
            return RECONCILE_INCONCLUSIVE, "no zone recorded for a zonal candidate"
        zone = candidate.zone

        def read_back() -> Any:
            return get_instance(project, zone, candidate.name)
    elif candidate.resource_type == "firewall":

        def read_back() -> Any:
            return retry_idempotent(
                compute_v1.FirewallsClient().get,
                project=project,
                firewall=candidate.name,
                op_desc=f"firewalls.get {candidate.name} (marker verification)",
            )
    else:
        return RECONCILE_INCONCLUSIVE, f"unsupported candidate kind {candidate.resource_type!r}"

    return reconcile_owned(
        read_back,
        lambda resource: has_invocation_description(resource, candidate.invocation_id),
        attempts=_CANDIDATE_RECONCILE_ATTEMPTS,
        backoff_seconds=_CANDIDATE_RECONCILE_BACKOFF_S,
    )


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Teardown a Compute Engine VM + companions",
        epilog=_STANDALONE_CLEANUP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instance-id",
        default=None,
        help=(
            "Instance name, forwarded by the provider config or backfilled from "
            "--from-launch-output. Absent (or a falsy sentinel) means no instance "
            "was tracked, and the instance deletes are skipped."
        ),
    )
    parser.add_argument(
        "--region",
        default=None,
        help="GCP region or zone (not needed when --zone is known)",
    )
    parser.add_argument("--zone", default=None, help="GCP zone (overrides region)")
    parser.add_argument(
        "--from-launch-output",
        default=None,
        help=(
            "Path to a saved launch_instance JSON payload ('-' for stdin) used to "
            "backfill every unset identity/ownership/zone argument. This is the "
            "supported way to clean up a preserved or aborted run: it keeps the "
            "*_created ownership gates and the leaked-zone scope intact. See the "
            "epilog below."
        ),
    )
    parser.add_argument("--project", default=None, help="GCP project ID (ADC fallback)")
    parser.add_argument(
        "--delete-key-pair",
        action="store_true",
        help="Delete the local SSH key pair if --key-created is truthy",
    )
    parser.add_argument(
        "--delete-security-group",
        action="store_true",
        help="Delete the SSH firewall rule if --firewall-created is truthy",
    )
    parser.add_argument(
        "--skip-destroy",
        action="store_true",
        help="Short-circuit to success (preserve cloud state) BEFORE resolving auth",
    )
    parser.add_argument("--firewall-name", default=None, help="Firewall rule name")
    parser.add_argument(
        "--firewall-created",
        default=None,
        help="Bool sentinel forwarded from launch_instance.firewall_created",
    )
    parser.add_argument(
        "--instance-created",
        default=None,
        help=(
            "Bool sentinel forwarded from launch_instance.instance_created. "
            "False skips both the primary and every leaked-zone instance "
            "delete so a verified-reuse adoption of an operator-supplied "
            "long-lived VM is never destroyed by this teardown."
        ),
    )
    parser.add_argument(
        "--key-file",
        default=None,
        help="Local SSH PEM path forwarded from launch_instance.key_file",
    )
    parser.add_argument(
        "--key-created",
        default=None,
        help="Bool sentinel forwarded from launch_instance.key_created",
    )
    parser.add_argument(
        "--leaked-zones",
        default=None,
        help=(
            "Comma-separated zones the multi-zone walker accumulated "
            "partial-create leaks in. Teardown best-effort-deletes the "
            "instance in each before completing."
        ),
    )
    parser.add_argument(
        "--unreconciled-resources",
        default=None,
        help=(
            "Comma-separated 'kind|name|project|zone|invocation' records forwarded "
            "from launch_instance.unreconciled_resources: ambiguous creates whose "
            "ownership could not be proven. Each is deleted only after its "
            "invocation marker is re-verified on the exact resource; a mismatch is "
            "never deleted."
        ),
    )
    args = parser.parse_args()

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "resources_destroyed": False,
        "deleted": {
            "instances": [],
            "firewall_rules": [],
            "key_files": [],
        },
        "resources_deleted": [],  # flat list shape matching AWS oracle / my-isv
        "message": "",
    }

    # Standalone-cleanup backfill runs before everything else so the
    # ownership bits and leaked-zone scope recorded by launch_instance are in
    # place for every gate below. Reading the payload touches no cloud API, so
    # it stays ahead of the preservation short-circuit and of auth resolution.
    if args.from_launch_output:
        try:
            _apply_launch_output(args, args.from_launch_output)
        except (OSError, ValueError) as e:
            result["error_type"] = "configuration_error"
            result["error"] = f"Unable to read launch output {args.from_launch_output!r}: {e}"
            result["message"] = result["error"]
            print(json.dumps(result, indent=2, default=str))
            return 1

    # Preservation-mode flag short-circuits BEFORE any cloud / auth call
    # so an expired-credentials environment still no-ops cleanly.
    if args.skip_destroy:
        preserved = args.instance_id if _truthy(args.instance_id) else "(none recorded)"
        # Preservation is a successful outcome, not a destruction:
        # `resources_destroyed` keeps its False initializer here (same rule the
        # adoption gate applies below) because the instance is still running.
        result["success"] = True
        result["instance_id"] = args.instance_id
        pending = parse_unreconciled_records(args.unreconciled_resources)
        if pending:
            # Preservation covers unproven candidates too — they are part of the
            # fixture state the operator asked to keep, and the replay is what
            # marker-verifies them later.
            result.setdefault("warnings", []).append(
                "unreconciled candidates preserved (--skip-destroy): "
                + ", ".join(candidate.describe() for candidate in pending)
            )
        result["message"] = (
            f"Instance {preserved} preserved (--skip-destroy); reclaim it later with "
            "teardown.py --from-launch-output <launch_instance.json>, which reapplies "
            "the instance_created / firewall_created / key_created gates and the "
            "leaked-zone scope."
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    # A zone is the minimum scope a zonal delete needs. Fail with a structured
    # configuration error rather than letting narrow_region_to_zone(None) raise
    # an opaque type error from inside the cleanup path.
    if not _truthy(args.zone) and not _truthy(args.region):
        result["error_type"] = "configuration_error"
        result["error"] = (
            "No --zone or --region was supplied (and --from-launch-output did not "
            "provide one), so there is no zone scope to delete in."
        )
        result["message"] = result["error"]
        print(json.dumps(result, indent=2, default=str))
        return 1

    project = resolve_project(args.project)
    # Sentinel-aware, matching the validation above: a forwarded 'none' zone
    # means "upstream tracked no zone", not a zone literally named none.
    zone = args.zone if _truthy(args.zone) else narrow_region_to_zone(args.region)

    firewall_created = _truthy(args.firewall_created)
    key_created = _truthy(args.key_created)
    instance_created = _truthy(args.instance_created)
    instance_id = args.instance_id if _truthy(args.instance_id) else None
    fw_name = args.firewall_name if _truthy(args.firewall_name) else None
    key_file = args.key_file if _truthy(args.key_file) else None

    # Per-resource booleans surfaced into the final success.
    instance_ok = True
    firewall_ok = True
    key_ok = True
    candidates_ok = True

    # Destruction state of the instance, tracked SEPARATELY from the cleanup
    # bools above. A preserved instance is a successful cleanup outcome but is
    # not a destruction, so the two must never be derived from one another.
    instance_preserved = False

    # 1. Instance delete. Each preflight read is scoped narrow so a
    # transient probe error doesn't poison sibling cleanup blocks —
    # teardown preflight reads MUST NOT share an exception handler with
    # the cleanup blocks.
    #
    # The verified-reuse ownership gate (instance_created) bypasses
    # the preflight entirely — there is no observable difference
    # between "adopted, still present" and "we created and it's still
    # present", so we MUST trust the forwarded ownership bit rather
    # than the live state.
    instance_present = False
    if instance_id and not instance_created:
        # Preserved on purpose: the instance is still running after this step,
        # so it is recorded as preserved and NOT counted as destroyed below,
        # however cleanly the companion cleanup finishes.
        instance_preserved = True
        print(
            f"Skipping instance delete for {instance_id} (instance_created=false; "
            "verified-reuse adoption — never destroy resources this run did not create)",
            file=sys.stderr,
        )
        result.setdefault("warnings", []).append(
            f"instance {instance_id} preserved (verified-reuse adoption: instance_created=false)"
        )
    elif instance_id:
        print(f"Deleting instance {instance_id} in {zone}...", file=sys.stderr)
        # Bounded, monotonic preflight: only an all-NotFound sequence retires an
        # owned instance without a delete. Any other outcome (transient error,
        # mixed reads) stays "present" and falls through to delete_with_retry,
        # which absorbs NotFound idempotently.
        target_instance = instance_id
        instance_present, presence_detail = _owned_resource_present(
            lambda: get_instance(project, zone, target_instance),
            desc=f"instance {instance_id}",
        )
        print(f"  {presence_detail}", file=sys.stderr)
        if not instance_present:
            result.setdefault("warnings", []).append(f"instance {instance_id} not found at teardown — already deleted")
        elif "presence read failed" in presence_detail:
            result.setdefault("warnings", []).append(f"instance preflight read failed: {presence_detail}")
    else:
        print("Skipping instance delete (no instance id was produced)", file=sys.stderr)

    if instance_present:
        assert instance_id is not None
        instance_ok = delete_with_retry(
            _delete_instance_op,
            project,
            zone,
            instance_id,
            resource_desc=f"instance {instance_id}",
        )
        if instance_ok:
            result["deleted"]["instances"].append(instance_id)
            result["resources_deleted"].append(f"instance:{instance_id}")

    # 1b. The multi-zone walker may have accumulated zones where a partial
    # async insert leaked; best-effort delete in each so phantom records do
    # not survive the run. The producer records a zone here ONLY when it
    # accepted a create there and could not confirm the reclaim delete, so
    # every entry names a scope this run really allocated in — zones that
    # merely rejected the insert never reach this loop. Drop falsy sentinels
    # so the per-zone delete loop only walks real zone strings.
    leaked = [
        z.strip()
        for z in (args.leaked_zones or "").split(",")
        if z.strip() and z.strip().lower() not in _FALSY_SENTINELS
    ]
    if instance_id and not instance_created and leaked:
        # Verified-reuse adoption never invoked the multi-zone walker
        # (the walker runs only on the create path), so a leaked_zones
        # list arriving here is impossible under normal flow. If
        # something upstream wires it anyway, refuse to touch a
        # not-ours name in any zone.
        result.setdefault("warnings", []).append(
            f"leaked-zone cleanup skipped: instance_created=false (preserving adopted {instance_id})"
        )
    elif instance_id and instance_created:
        for leak_zone in leaked:
            if leak_zone == zone:
                continue  # already handled above
            print(f"Leaked-zone cleanup: instance {instance_id} in {leak_zone}", file=sys.stderr)
            leak_ok = delete_with_retry(
                _delete_instance_op,
                project,
                leak_zone,
                instance_id,
                resource_desc=f"instance {instance_id}@{leak_zone}",
            )
            # Leaked-zone failure surfaces into the aggregate success so the
            # operator sees an honest partial-cleanup verdict; the per-zone
            # delete is best-effort but its outcome is NOT swallowed.
            if not leak_ok:
                instance_ok = False
                result.setdefault("warnings", []).append(f"leaked-zone delete failed: {instance_id}@{leak_zone}")
            else:
                result["deleted"]["instances"].append(f"{instance_id}@{leak_zone}")
                result["resources_deleted"].append(f"instance:{instance_id}@{leak_zone}")
    elif leaked:
        result.setdefault("warnings", []).append(
            f"leaked zones ignored because no instance id was produced: {', '.join(leaked)}"
        )

    # 2. Firewall — gated on the verified-reuse flag forwarded by
    # launch_instance. NotFound is idempotent success; transient is
    # local-only (does not bypass key cleanup below).
    if args.delete_security_group:
        if firewall_created and fw_name:
            # Same bounded, monotonic preflight as the instance above: one
            # NotFound never retires an owned firewall rule.
            target_firewall = fw_name
            firewall_present, fw_presence_detail = _owned_resource_present(
                lambda: compute_v1.FirewallsClient().get(project=project, firewall=target_firewall),
                desc=f"firewall {fw_name}",
            )
            print(f"  {fw_presence_detail}", file=sys.stderr)
            if firewall_present and "presence read failed" in fw_presence_detail:
                result.setdefault("warnings", []).append(f"firewall preflight read failed: {fw_presence_detail}")

            if firewall_present:
                print(f"Deleting firewall rule {fw_name}...", file=sys.stderr)
                firewall_ok = delete_with_retry(
                    _delete_firewall_op,
                    project,
                    fw_name,
                    resource_desc=f"firewall {fw_name}",
                )
                if firewall_ok:
                    result["deleted"]["firewall_rules"].append(fw_name)
                    result["resources_deleted"].append(f"firewall_rule:{fw_name}")
        else:
            print(
                "  skipping firewall delete (firewall_created=false or no name)",
                file=sys.stderr,
            )

    # 3. Local SSH key pair — gated on key_created. Runs regardless of
    # the instance preflight outcome (cloud-side NotFound must NOT
    # short-circuit local cleanup). ``delete_local_keypair`` handles
    # both halves of the pair so the .pub is removed even when the PEM
    # was already gone from a prior run.
    if args.delete_key_pair:
        if key_created and key_file:
            pub_path = key_file + ".pub"
            priv_present = os.path.exists(key_file)
            pub_present = os.path.exists(pub_path)
            if priv_present or pub_present:
                print(
                    f"Deleting local SSH key pair: {key_file} (priv={priv_present}, pub={pub_present})",
                    file=sys.stderr,
                )
                key_ok = delete_local_keypair(key_file)
                if key_ok:
                    if priv_present:
                        result["deleted"]["key_files"].append(key_file)
                        result["resources_deleted"].append(f"key_file:{key_file}")
                    if pub_present:
                        result["deleted"]["key_files"].append(pub_path)
                        result["resources_deleted"].append(f"key_file:{pub_path}")
            else:
                print(f"  local SSH key pair already absent: {key_file} + .pub", file=sys.stderr)
        else:
            print(
                "  skipping local key cleanup (key_created=false or no path)",
                file=sys.stderr,
            )

    # 4. Ambiguous-create candidates. These carry no ownership bit — the
    # creating step could not prove one — so ownership is decided HERE, by
    # re-verifying the recorded invocation marker on the exact resource over a
    # bounded readback envelope. Only a marker match is deleted. A mismatch
    # belongs to another run and is preserved; an unverifiable candidate — and
    # any candidate whose reads did not ALL come back NotFound — is preserved
    # too and fails the teardown honestly, because "might still exist" is not
    # clean cleanup. This loop is the handoff's last reader: a verdict of
    # absent here throws the record away, so it has to be earned.
    for candidate in parse_unreconciled_records(args.unreconciled_resources):
        label = candidate.describe()
        candidate_project = candidate.project or project
        verdict, detail = _verify_candidate_ownership(candidate_project, candidate)
        if verdict == RECONCILE_OWNED:
            print(f"Unreconciled candidate {label}: {detail}; deleting", file=sys.stderr)
            if candidate.resource_type == "instance":
                deleted = delete_with_retry(
                    _delete_instance_op,
                    candidate_project,
                    candidate.zone,
                    candidate.name,
                    resource_desc=f"unreconciled instance {candidate.name}@{candidate.zone}",
                )
            else:
                deleted = delete_with_retry(
                    _delete_firewall_op,
                    candidate_project,
                    candidate.name,
                    resource_desc=f"unreconciled firewall {candidate.name}",
                )
            if deleted:
                result["resources_deleted"].append(f"{candidate.resource_type}:{candidate.name}")
                bucket = "instances" if candidate.resource_type == "instance" else "firewall_rules"
                result["deleted"][bucket].append(candidate.name)
            else:
                candidates_ok = False
                result.setdefault("warnings", []).append(f"unreconciled {label} delete failed after marker match")
        elif verdict == RECONCILE_FOREIGN:
            # Never delete a marker-mismatched candidate: the name matches but
            # the resource is another run's (or another generation's).
            print(f"Unreconciled candidate {label}: {detail}; preserving", file=sys.stderr)
            result.setdefault("warnings", []).append(f"unreconciled {label} preserved: {detail}")
        elif verdict == RECONCILE_ABSENT:
            print(
                f"Unreconciled candidate {label}: {detail} on every bounded attempt; "
                "nothing was committed, or it is already gone",
                file=sys.stderr,
            )
        else:
            candidates_ok = False
            result.setdefault("warnings", []).append(
                f"unreconciled {label} ownership unproven ({detail}); nothing deleted — retry with "
                "teardown.py --from-launch-output <launch_instance.json>"
            )

    # 5. Surface every per-resource bool into final success.
    result["success"] = bool(instance_ok and firewall_ok and key_ok and candidates_ok)

    # Destruction state is NOT recomputed from cleanup success. On the
    # verified-reuse adoption path the gate above deliberately leaves the
    # operator's instance running, and a clean companion cleanup would
    # otherwise flip this bool — publishing "destroyed" for a VM that is still
    # up and still billing, with a matching deletion message. So preservation
    # holds it False, and it is True only when the owned instance was deleted
    # or confirmed already absent AND every other gate succeeded. Companion
    # adoption (firewall_created / key_created false) is the "intentionally
    # skipped" case the cleanup contract allows to stay True: no run-owned
    # artifact is left behind by it.
    result["resources_destroyed"] = bool(result["success"] and not instance_preserved)

    if not result["success"]:
        result["message"] = (
            f"Cleanup partial: instance_ok={instance_ok}, firewall_ok={firewall_ok}, "
            f"key_ok={key_ok}, unreconciled_ok={candidates_ok}"
        )
    elif instance_preserved:
        result["message"] = (
            f"Cleanup complete; instance {instance_id} PRESERVED, not deleted "
            "(verified-reuse adoption: instance_created=false) — it is still running "
            "and still billable. Gated companion cleanup (firewall / local key / "
            "unreconciled candidates) succeeded."
        )
    elif instance_id:
        result["message"] = "Instance and verified-reuse companions deleted"
    else:
        result["message"] = "No instance was tracked; verified-reuse companions cleaned up"

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

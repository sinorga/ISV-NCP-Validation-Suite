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

"""Prove the specified SSH key reaches serial-over-LAN access on Compute Engine.

AUTH03-01. Two subtests, both derived from real probes — no literal passes:

``sol_access``
    Compute Engine has no managed key-pair store, so the "specified key" is
    the label the launch step CONFIRMED present in the instance's ``ssh-keys``
    metadata (forwarded here as ``--key-name`` from that step's
    ``instance_key_name`` readback). The probe proves that key authenticates
    to the interactive serial-console access path:

      1. re-read ``instances.get`` and re-confirm the public key paired with
         ``--key-file`` is in the instance's ``ssh-keys`` metadata;
      2. ensure interactive serial access is enabled for the instance
         (``serial-port-enable`` metadata), restoring the prior value when
         this step is the one that changed it;
      3. authenticate to the Compute Engine serial-console gateway for the
         instance's own region (``REGION-ssh-serialport.googleapis.com``
         port 9600) using ONLY that key, and require the OpenSSH client to
         report a completed publickey authentication.

    Retrieving serial output through ``instances.getSerialPortOutput`` is
    NOT accepted as evidence: that call is authorized by the caller's IAM
    credentials and proves nothing about the specified SSH key.

    Exactly ONE condition makes that path *unavailable* rather than failed and
    structured-skips the step (``success=true, skipped=true``) the way the
    reference provider skips when serial-console access is not available: the
    ``compute.disableSerialPortAccess`` organization policy, which removes the
    capability from the project entirely. That skip is still emitted only after
    the metadata restore has run: the constraint can refuse a later attempt of
    a write an earlier attempt already committed, and a restore that then fails
    demotes the skip to ``success=false`` with structured ``cleanup_errors``
    rather than reporting green over a mutated instance.

    A validation host whose network drops outbound TCP to the gateway port is
    NOT that case — the capability exists and only the local environment cannot
    reach it — so it never reads green. It fails with a structured
    ``configuration_error`` naming the egress the operator must open, and no
    environment setting converts that failure into a skip: this check is
    required to pass, so a probe that never ran stays a failure until the
    egress exists. A gateway that answers and then rejects the key is likewise
    always a failure — never a skip.

``network_device_access``
    Compute Engine networking is a software-defined fabric — tenants
    configure it exclusively through the Compute API and no physical switch
    or router exposes a tenant-reachable SSH endpoint. That is a
    provider-owned plane, so the subtest is reported
    ``passed=true, provider_hidden=true`` — but only after a real
    project/resource identity probe (``instances.get`` for the target VM's
    network-interface inventory, plus a Cloud Router enumeration for the
    instance's region) establishes the surface actually observed.

Usage:
    python component_key_access.py --instance-id <name> --key-file <path> \\
        --key-name <name> --region <region> [--zone <zone>] [--ssh-user <user>]
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

from common.compute import (
    get_instance,
    instance_has_pubkey,
    narrow_region_to_zone,
    pubkey_from_private_key,
    resolve_project,
    short_name,
    wait_for_zonal_op,
    zone_to_region,
)
from common.errors import classify_gcp_error, handle_gcp_errors, retry_idempotent_list
from common.ownership import create_may_have_committed
from common.result import preserve_success_after_cleanup
from common.ssh_utils import SSH_CANONICAL_OPTS
from google.api_core import exceptions as gax
from google.cloud import compute_v1

# Sentinel the provider config renders when an upstream step emitted a null /
# empty value. Boolean-mode Jinja defaults keep the flag/value argv pair
# aligned; the stub treats the literal as "upstream did not supply".
_NONE_SENTINEL = "none"

# Compute Engine interactive serial-console gateway. The connection endpoint is
# REGIONAL — `REGION-ssh-serialport.googleapis.com` — and the legacy global
# `ssh-serialport.googleapis.com` name is documented as unavailable to projects
# that have not already used it, so a hardcoded global host makes this probe
# unreachable (and therefore silently skippable) on new projects. The host is
# always derived from the region of the resolved instance zone.
# https://cloud.google.com/compute/docs/troubleshooting/troubleshooting-using-serial-console#connecting_to_a_serial_console
_GATEWAY_HOST_SUFFIX = "ssh-serialport.googleapis.com"
_GATEWAY_PORT = 9600

# Instance metadata key that enables interactive serial-console access.
_SERIAL_ENABLE_KEY = "serial-port-enable"
_SERIAL_ENABLE_TRUE = "TRUE"
_TRUTHY_METADATA = frozenset({"1", "true", "yes", "on"})

# Org-policy constraint that removes the interactive serial console from the
# project entirely. The AWS reference structured-skips when serial-console
# access is disabled account-wide; this is the Compute Engine counterpart.
_SERIAL_DISABLED_CONSTRAINT = "compute.disableSerialPortAccess"
_SERIAL_DISABLED_SKIP_REASON = (
    "Interactive serial console is disabled for this project by the "
    f"constraints/{_SERIAL_DISABLED_CONSTRAINT} organization policy; "
    "the specified-key SOL path cannot be exercised."
)

# The other way the specified-key SOL handshake never runs: the gateway's TCP
# port is not reachable from the validation host. This is deliberately NOT
# treated as the same class as the org-policy skip above. The reference
# provider structured-skips only when the platform itself has removed the
# serial-console capability; here the capability exists and only the local
# validation host cannot reach it. Letting that read green would make the one
# probe that proves AUTH03 self-disabling: a host with no egress to tcp/9600
# would report "specified-key SOL access satisfied" forever without a single
# completed publickey handshake.
#
# The posture is therefore a structured configuration error that names the
# egress to open, and that is the ONLY posture: this step carries no
# environment-controlled opt-out, deliberately. A stub that reports
# `skipped=true` is turned into a pytest skip by the validation engine, and the
# orchestrator counts a skipped validation as passed — so an env-gated skip
# here would be indistinguishable, in the final evidence set, from a completed
# publickey handshake. The check is admitted as required-to-pass precisely
# because absent key-authenticated serial-console evidence means the probe or
# the operator's egress is missing — not that the project lacks the capability.
# Unreachable egress therefore stays failed until the real probe runs.
#
# `console_rbac`'s self-provision opt-out is NOT the same shape: its subtests
# need IAM mutation rights the project itself may forbid, which is a genuine
# environment capability limit the check is allowed to record as a coverage
# loss. Reaching a public Google endpoint on tcp/9600 is not.

# Observed facts behind the failure text: the name resolves, TCP never
# connected on either bracketing probe, and no attempt reached a denial — so
# the key was never offered and no verdict exists.
_GATEWAY_UNREACHABLE_DETAIL = (
    "The Compute Engine interactive serial-console gateway for this instance's region "
    "{host}:{port} is not reachable from this validation host: the name resolves "
    "({addresses}) but TCP connections to port {port} time out ({error}), so the "
    "specified-key SOL handshake was never attempted and no verdict about the key exists."
)
_GATEWAY_UNREACHABLE_ERROR = (
    _GATEWAY_UNREACHABLE_DETAIL + " AUTH03 specified-key serial-console access is "
    "therefore UNPROVEN, not satisfied. Operator action: allow outbound TCP {port} to "
    "{host} from the validation host, then re-run. There is no waiver: this handshake "
    "is the only evidence for the released AUTH03 property, so the step stays failed "
    "until it actually runs."
)

# OpenSSH client verbose markers. `Authenticated to <host>` is emitted only
# after the publickey exchange completes, so it is the precise signal that the
# supplied key — and nothing else — was accepted by the gateway.
_AUTH_SUCCESS_TOKENS = ("Authenticated to ", "Authentication succeeded")
_AUTH_DENIED_TOKENS = (
    "Permission denied",
    "No supported authentication methods",
    "Too many authentication failures",
)

# Serial-gateway probe budget. Metadata (ssh-keys / serial-port-enable)
# propagates to the gateway asynchronously, so a first-attempt denial is
# retried with backoff before it is reported as a real failure.
_GATEWAY_ATTEMPTS = 3
_GATEWAY_BACKOFFS: tuple[int, ...] = (15, 30)
_GATEWAY_TIMEOUT = 40
_GATEWAY_CONNECT_TIMEOUT = 15

# Transport reachability probe. The SSH client collapses "gateway unreachable"
# and "gateway rejected the key" into the same non-zero exit, so a failed
# handshake alone cannot tell an environment problem (DNS, blocked egress on
# the gateway port, a middlebox that answers the TCP connect but never speaks
# SSH) apart from a genuine key-authorization failure. A direct socket probe
# separates them BEFORE the SSH attempts run: it records whether the gateway
# name resolves, whether the port accepts a connection, and whether the peer
# emits an OpenSSH identification banner.
_GATEWAY_BANNER_TIMEOUT = 10
_GATEWAY_BANNER_PREFIX = "SSH-"

# Diagnostic bounds. The SSH client trace and the JSON contract both stay
# small; only the tail matters because the failure is always at the end.
_TRACE_TAIL_EVIDENCE = 1200
_TRACE_TAIL_ERROR = 480

# Bounded read-modify-write budget for the metadata mutation (the fingerprint
# is an optimistic-concurrency token, so a stale token is re-read and retried).
_METADATA_ATTEMPTS = 3
_METADATA_BACKOFF = 3.0
_METADATA_OP_TIMEOUT = 120

# In-process wall-clock budget, and why the writes above are deadline-bounded
# rather than merely attempt-bounded.
#
# The orchestrator runs this stub under `subprocess.run(timeout=...)`, which
# KILLS the child when the provider-config `timeout:` expires — no signal
# handler, no `finally`. This step mutates the target instance
# (`serial-port-enable`) and reverts it in its `finally` block, so a kill
# landing before that revert leaves interactive serial access switched ON. On a
# run that ADOPTED an operator instance (`GCP_VM_INSTANCE_ID`), teardown
# preserves the VM by design and never cleans that flag up, so the leak is
# durable on a long-lived host.
#
# Attempt bounds alone do not prevent that: the enable write and the restore
# write each carry their own `_METADATA_ATTEMPTS` x retry ladder ending in a
# `_METADATA_OP_TIMEOUT` op-wait, and those two ladders sum on TOP of the
# serial-gateway probe budget (165s of auth attempts + backoffs) and the
# transport probes. Under a throttled control plane that sum can pass the
# configured cap, and the step is killed mid-revert.
#
# So the stub bounds its own wall-clock instead of trusting arithmetic in a
# comment. `_STEP_WALL_BUDGET` is the total it allows itself; every metadata
# write runs under a deadline carved from what is left of it, and
# `_RESTORE_RESERVE` is withheld from the enable write so the `finally` restore
# still has a window of its own. `_METADATA_MIN_WRITE_BUDGET` and
# `_METADATA_MIN_OP_WAIT` are floors that keep a restore attempt possible even
# when the probe path overran, and they are the only way total time can exceed
# `_STEP_WALL_BUDGET`. The bound is therefore
# `_STEP_WALL_BUDGET + _METADATA_MIN_WRITE_BUDGET + _METADATA_MIN_OP_WAIT`
# (525s); the provider config sizes `timeout:` above it, so the kill can only
# ever arrive after the revert has already run.
_STEP_WALL_BUDGET = 480.0
_RESTORE_RESERVE = 120.0
_METADATA_WRITE_BUDGET = 150.0
_METADATA_MIN_WRITE_BUDGET = 30.0
_METADATA_MIN_OP_WAIT = 15.0

# Bounded fresh-readback budget used to reconcile an AMBIGUOUS submit failure
# (see `_write_metadata_key`). An accepted mutation can become visible a moment
# after the response was lost, so the readback polls a few times before it
# concludes the value never landed — and only an ALL-conclusive sequence may
# reach that conclusion (see `_requested_value_landed`).
_METADATA_RECONCILE_ATTEMPTS = 3
_METADATA_RECONCILE_DELAY = 4.0

# `setMetadata` submit failures split into two dispositions, and conflating
# them is what loses cleanup ownership:
#
#  * REJECTED — the server refused the request and committed NOTHING, so the
#    only correct response is to re-read the live fingerprint and resubmit.
#    `PreconditionFailed` (HTTP 412) is the status Compute Engine returns for a
#    STALE fingerprint; it is a sibling of `Conflict` (HTTP 409), not a
#    subclass, so it must be listed explicitly or the read-modify-write retry is
#    dead for the exact race it exists to absorb. `Aborted` (a `Conflict`
#    subclass) is the concurrent-mutation sibling.
#  * AMBIGUOUS — the request may have been committed before the failure
#    surfaced: a lost response, a deadline, a throttle, an exhausted internal
#    retry, or a raw transport drop all report failure for a mutation the
#    backend may already hold. These are NOT assertions that nothing changed,
#    so they are reconciled by fresh readback instead of assumed clean.
#
# Ambiguity is decided by the shared ownership taxonomy
# (`common.ownership.create_may_have_committed`) rather than a hand-copied
# local tuple, so this mutation reads "may already be committed" from the same
# classifier every create path in this provider uses, and inherits later
# additions instead of drifting. The one deliberate difference is the
# exact-identity conflict: a create reconciles it by invocation marker (it can
# be the create's own committed retry), whereas for `setMetadata` it is the
# stale-fingerprint refusal excluded above, so this two-state predicate keeps
# reporting it as "nothing landed". That taxonomy already spans the whole
# documented ambiguous set:
# the `transient` bucket (`ServiceUnavailable`, `InternalServerError`,
# `GatewayTimeout` and its `DeadlineExceeded` subclass, `TooManyRequests` and
# its `ResourceExhausted` subclass, `RetryError`), raw transport disconnects
# (which are not `google.api_core` types at all), and the uncategorized
# `api_error` / `unknown_error` buckets a lost response can land in.
_METADATA_FRESH_FINGERPRINT_ERRORS: tuple[type[BaseException], ...] = (
    gax.PreconditionFailed,
    gax.Aborted,
    gax.Conflict,
)


class _VerdictRecorded(Exception):
    """Internal signal: the step verdict is already written into ``result``.

    Raised INSTEAD of printing the JSON and returning from inside the block
    that the metadata restore's ``finally`` protects. Printing a terminal
    verdict there is what makes a mutated instance look clean: Python runs
    ``finally`` while holding the handler's return value, so a restore that
    fails afterwards can append ``cleanup_errors`` and demote ``success`` in a
    ``result`` object that has ALREADY been serialized to stdout, and the step
    still exits with the handler's status. The org-policy skip is the sharpest
    case — the enable write can be accepted on one attempt and refused by the
    constraint on the next, so restoration is armed at the moment the skip is
    decided — but the rule is the same for every early verdict inside that
    block: record it, let cleanup run, and emit exactly once afterwards from
    the single exit path at the bottom of ``main``.
    """


def _is_ambiguous_submit_failure(exc: BaseException) -> bool:
    """True when ``exc`` may have surfaced AFTER the server committed the write.

    Only meaningful once ``_METADATA_FRESH_FINGERPRINT_ERRORS`` has been
    excluded by the caller. The shared helper was written for creates, which
    carry no fingerprint token, so it buckets HTTP 412 as a possibly-committed
    ``api_error``; for ``setMetadata`` a 412 is the documented STALE-TOKEN
    REFUSAL and nothing was committed.
    """
    return isinstance(exc, Exception) and create_may_have_committed(exc)


def _is_serial_access_disabled(exc: Exception) -> bool:
    """True when ``exc`` is the org-policy denial of interactive serial access."""
    return _SERIAL_DISABLED_CONSTRAINT in str(exc)


def _metadata_value(instance: compute_v1.Instance, key: str) -> str | None:
    """Return the value of instance metadata ``key``, or None when absent."""
    metadata = getattr(instance, "metadata", None)
    for item in getattr(metadata, "items", None) or []:
        if item.key == key:
            return item.value or ""
    return None


def _same_metadata_value(observed: str | None, requested: str | None) -> bool:
    """Whether a live metadata value matches the one a write requested.

    ``None`` on either side means "key absent", so a removal request is
    satisfied only by the key being gone. Present values compare on their
    stripped text: Compute Engine stores metadata verbatim, and only
    surrounding whitespace is not meaningful.
    """
    if observed is None or requested is None:
        return observed == requested
    return observed.strip() == requested.strip()


def _metadata_write_budget(started: float, *, reserve: float = 0.0) -> float:
    """Wall-clock one metadata write may spend, carved from the step budget.

    ``started`` is the monotonic stamp taken when the step began its work, and
    ``reserve`` is the tail of the budget this call must NOT consume — the
    enable write passes ``_RESTORE_RESERVE`` so the `finally` restore keeps a
    window even when the enable ladder ran long.

    The floor (``_METADATA_MIN_WRITE_BUDGET``) is deliberate: when the probe
    path has already eaten the budget, a restore that is given no time at all
    is the same leak the budget exists to prevent, so a short window is still
    granted. That floor is the ONLY way the step can exceed
    ``_STEP_WALL_BUDGET``, and the provider-config cap is sized above the sum.
    """
    remaining = _STEP_WALL_BUDGET - (time.monotonic() - started) - reserve
    return max(_METADATA_MIN_WRITE_BUDGET, min(_METADATA_WRITE_BUDGET, remaining))


def _op_wait_timeout(deadline: float) -> int:
    """Op-wait window that fits inside what is left of a write's deadline.

    An accepted `setMetadata` has ALREADY changed the instance, so the op-wait
    only confirms the asynchronous operation reached DONE. Truncating that
    confirmation is therefore safe (a timeout is reported, never silently
    swallowed) while overrunning it is not — the overrun is what gets the step
    killed before it can revert the mutation.
    """
    return int(max(_METADATA_MIN_OP_WAIT, min(_METADATA_OP_TIMEOUT, deadline - time.monotonic())))


def _requested_value_landed(
    project: str,
    zone: str,
    instance_name: str,
    key: str,
    value: str | None,
) -> bool | None:
    """Bounded fresh readback: did the requested metadata write become live?

    Returns True once a fresh ``instances.get`` shows ``key`` holding
    ``value``, False when EVERY completed readback proved the instance still
    carries something else, and None when the sequence proved nothing.

    None is deliberately NOT collapsed into False. "I could not look" is not
    evidence that nothing changed, and the caller treats it the same way it
    treats an observed change — arming restoration — because re-writing the
    prior value is a no-op when the mutation never landed, while skipping
    restoration on a mutation that DID land leaves interactive serial access
    switched on.

    The unobserved state is MONOTONIC across attempts, the same rule the shared
    SDK reconciler (``common.ownership.reconcile_owned``) and the console-RBAC
    reconciler apply, so the three readers cannot drift. A run of attempts that
    goes "readback failed, then the prior value is still there" is a MIXED
    sequence, not proof that nothing landed: the failed attempt answered
    nothing, and the later read only proves the mutation was not visible on
    that one poll — which is exactly what an accepted ``setMetadata`` whose
    response was lost looks like while it is still converging. Letting the
    later read overwrite the earlier unobserved one would return False, leave
    ``restore_armed`` clear in the caller, and let a request that commits after
    the bounded window leave ``serial-port-enable`` switched on — on an adopted
    operator VM that teardown preserves, permanently. So a False verdict
    requires at least one conclusive differing observation AND no unobserved
    attempt anywhere in the sequence; anything else downgrades to None and arms
    restoration.
    """
    saw_unobserved = False
    differing_attempts = 0
    for attempt in range(1, _METADATA_RECONCILE_ATTEMPTS + 1):
        try:
            observed = _metadata_value(get_instance(project, zone, instance_name), key)
        except Exception as exc:  # readback is evidence-only; never masks the submit failure
            print(f"  warn: metadata readback for {key} failed: {exc}", file=sys.stderr)
            saw_unobserved = True
        else:
            if _same_metadata_value(observed, value):
                return True
            differing_attempts += 1
        if attempt < _METADATA_RECONCILE_ATTEMPTS:
            time.sleep(_METADATA_RECONCILE_DELAY)
    # Every loop exit here was either a conclusive differing observation or an
    # attempt that observed nothing. The `> 0` guard also stops a zero-attempt
    # envelope from reporting a "nothing landed" proof it never gathered.
    if differing_attempts > 0 and not saw_unobserved:
        return False
    return None


def _write_metadata_key(
    client: compute_v1.InstancesClient,
    project: str,
    zone: str,
    instance_name: str,
    key: str,
    value: str | None,
    *,
    budget: float,
    on_accepted: Callable[[], None] | None = None,
) -> None:
    """Set (or remove, when ``value`` is None) one instance metadata key.

    ``budget`` is the wall-clock this whole read-modify-write ladder may spend
    (see ``_metadata_write_budget``). It caps the retry ladder AND the op-wait,
    so the two writes this step performs — the enable and the `finally`
    restore — cannot sum past the provider-config step timeout and get the
    process killed with ``serial-port-enable`` still switched on.

    ``instances.setMetadata`` replaces the WHOLE metadata block and is guarded
    by a fingerprint, so every attempt re-reads the live instance and rebuilds
    the item list from it. That preserves every other key (notably
    ``ssh-keys``, which downstream SSH steps depend on) and lets a stale
    fingerprint be resolved by re-reading rather than by clobbering.

    ``on_accepted`` fires the moment the mutation is known to have reached the
    instance, which is EARLIER than the asynchronous operation completing. Two
    paths reach it:

      * the synchronous ``set_metadata`` returns — the API accepted the write,
        so the instance has already changed even if the op-wait later times out
        or the operation reports an error;
      * an AMBIGUOUS submit failure (see ``_is_ambiguous_submit_failure``) is
        reconciled by fresh readback and the requested value is observable, or
        no readback could be completed at all. A lost response, a deadline, a
        throttle, an exhausted internal retry, or a raw transport drop can all
        surface after the backend committed the mutation, so an exception here
        is NOT proof that nothing was accepted — treating it as proof is what
        would leave ``serial-port-enable`` switched on with restoration
        disarmed.

    Cleanup ownership is therefore recorded on every path where the instance
    may have changed, never only on the clean-return path. When the readback
    proves the requested value is live, the write has achieved its purpose and
    the call returns normally instead of propagating a failure the instance
    does not reflect.
    """
    deadline = time.monotonic() + budget
    for attempt in range(1, _METADATA_ATTEMPTS + 1):
        instance = get_instance(project, zone, instance_name)
        metadata = getattr(instance, "metadata", None)
        items = [
            compute_v1.Items(key=item.key, value=item.value)
            for item in (getattr(metadata, "items", None) or [])
            if item.key != key
        ]
        if value is not None:
            items.append(compute_v1.Items(key=key, value=value))
        request = compute_v1.SetMetadataInstanceRequest(
            project=project,
            zone=zone,
            instance=instance_name,
            metadata_resource=compute_v1.Metadata(
                fingerprint=getattr(metadata, "fingerprint", "") or "",
                items=items,
            ),
        )
        try:
            operation = client.set_metadata(request=request)
        except Exception as exc:
            if _is_serial_access_disabled(exc):
                # An org-policy constraint denial is a definitive refusal
                # whichever HTTP status carries it: the platform removed the
                # capability, so nothing was committed and neither a readback
                # nor a retry can change that. Raise straight away so the
                # caller's structured-skip arm stays fast.
                raise
            rejected = isinstance(exc, _METADATA_FRESH_FINGERPRINT_ERRORS)
            ambiguous = not rejected and _is_ambiguous_submit_failure(exc)
            if ambiguous:
                # The submit failed, but not in a way that proves the backend
                # refused it. Reconcile against live state before deciding
                # anything: if the requested value is observable (or cannot be
                # observed at all), the instance may already carry this step's
                # mutation and restoration MUST be armed with the original
                # prior value before the failure propagates.
                landed = _requested_value_landed(project, zone, instance_name, key, value)
                if landed is not False and on_accepted is not None:
                    on_accepted()
                if landed is True:
                    print(
                        f"  {key} write reported {type(exc).__name__} but readback shows it "
                        "committed; restoration armed",
                        file=sys.stderr,
                    )
                    return
            elif not rejected:
                # A definitive refusal (permission, org policy, bad request,
                # not found): the backend committed nothing and retrying
                # cannot change that, so ownership stays disarmed.
                raise
            # Rejected outright (stale fingerprint / concurrent mutation) or an
            # ambiguous failure the readback did not resolve: re-read the live
            # fingerprint and resubmit while budget remains. "Budget" is both
            # the attempt count AND the wall-clock deadline — a resubmit that
            # cannot finish inside the deadline would only push the kill closer
            # to the restore that has to follow it, so the failure is raised
            # now while there is still time to revert.
            backoff = _METADATA_BACKOFF * attempt
            if attempt >= _METADATA_ATTEMPTS or time.monotonic() + backoff >= deadline:
                raise
            time.sleep(backoff)
            continue
        # Accepted: the instance has changed. Arm cleanup ownership BEFORE the
        # async wait so a wait timeout / operation error still reverts it.
        if on_accepted is not None:
            on_accepted()
        op_name = getattr(operation, "name", None) or getattr(operation, "operation", "")
        if op_name:
            wait_for_zonal_op(project, zone, op_name, timeout=_op_wait_timeout(deadline))
        return


def _ensure_serial_port_enabled(
    client: compute_v1.InstancesClient,
    project: str,
    zone: str,
    instance_name: str,
    instance: compute_v1.Instance,
    *,
    budget: float,
    on_mutation_accepted: Callable[[str | None], None],
) -> str:
    """Ensure interactive serial access is on. Returns the observed state.

    ``already-enabled`` when the instance already carried a truthy
    ``serial-port-enable`` (nothing is written, so nothing is reverted), else
    ``enabled-by-step``.

    ``budget`` is the wall-clock the enable write may spend; the caller has
    already withheld the restore reserve from it, so this write cannot consume
    the window the `finally` revert needs.

    ``on_mutation_accepted`` is invoked with the EXACT prior value (None when
    the key was absent) the moment Compute Engine accepts the write, before
    the async operation wait — that is the point after which the instance has
    changed, so it is the point at which restore ownership must be recorded.
    """
    current = _metadata_value(instance, _SERIAL_ENABLE_KEY)
    if current is not None and current.strip().lower() in _TRUTHY_METADATA:
        return "already-enabled"
    _write_metadata_key(
        client,
        project,
        zone,
        instance_name,
        _SERIAL_ENABLE_KEY,
        _SERIAL_ENABLE_TRUE,
        budget=budget,
        on_accepted=lambda: on_mutation_accepted(current),
    )
    return "enabled-by-step"


def _gateway_username(project: str, zone: str, instance_name: str, ssh_user: str) -> str:
    """Build the serial-console gateway login name for the target instance."""
    return f"{project}.{zone}.{instance_name}.{ssh_user}"


def _gateway_host(zone: str) -> str:
    """Return the regional serial-console gateway host serving ``zone``.

    Compute Engine serves the interactive serial console from a per-region
    endpoint, so an instance in ``us-central1-a`` must be reached through
    ``us-central1-ssh-serialport.googleapis.com``. The bare global name is a
    legacy endpoint that new projects cannot use at all, so it is never
    emitted: an unresolvable zone raises instead of silently falling back to a
    host that would report the access path as "unreachable".
    """
    region = zone_to_region((zone or "").strip())
    if not region:
        raise ValueError("Cannot derive the regional serial-console gateway host: no instance zone was resolved")
    return f"{region}-{_GATEWAY_HOST_SUFFIX}"


def _probe_gateway_transport(host: str) -> dict[str, Any]:
    """Resolve, connect to, and read the SSH banner from the serial gateway.

    Runs BEFORE the authenticated probes so the failure classification below
    can name the stage that broke: name resolution, TCP reachability on the
    gateway port, or the SSH identification exchange. Every field is observed
    (``getaddrinfo`` / ``connect`` / ``recv``); nothing is assumed. ``host`` is
    the regional gateway derived from the target instance's zone, and it is
    recorded on the probe so the evidence names the endpoint actually tried.
    """
    info: dict[str, Any] = {"host": host, "port": _GATEWAY_PORT, "stage": "dns"}
    started = time.monotonic()
    try:
        addrinfo = socket.getaddrinfo(host, _GATEWAY_PORT, type=socket.SOCK_STREAM)
    except OSError as err:
        info["dns_resolved"] = False
        info["error"] = f"{type(err).__name__}: {err}"
        info["elapsed_s"] = round(time.monotonic() - started, 1)
        return info
    info["dns_resolved"] = True
    info["addresses"] = sorted({str(entry[4][0]) for entry in addrinfo})[:4]
    info["stage"] = "tcp"
    try:
        with socket.create_connection((host, _GATEWAY_PORT), timeout=_GATEWAY_CONNECT_TIMEOUT) as sock:
            info["tcp_connected"] = True
            info["stage"] = "banner"
            sock.settimeout(_GATEWAY_BANNER_TIMEOUT)
            banner = sock.recv(256).decode("utf-8", "replace").strip()
            info["banner"] = banner[:120]
            info["speaks_ssh"] = banner.startswith(_GATEWAY_BANNER_PREFIX)
            if info["speaks_ssh"]:
                info["stage"] = "ok"
    except OSError as err:
        info.setdefault("tcp_connected", False)
        info["error"] = f"{type(err).__name__}: {err}"
    info["elapsed_s"] = round(time.monotonic() - started, 1)
    return info


def _gateway_argv(gateway_user: str, key_file: str, host: str) -> list[str]:
    """Build the ``ssh`` argv for the regional serial-console gateway probe."""
    opts = list(SSH_CANONICAL_OPTS)
    try:
        opts[opts.index("ConnectTimeout=5")] = f"ConnectTimeout={_GATEWAY_CONNECT_TIMEOUT}"
    except ValueError:  # pragma: no cover - canonical set always carries it
        opts.extend(["-o", f"ConnectTimeout={_GATEWAY_CONNECT_TIMEOUT}"])
    return [
        "ssh",
        "-vv",  # client-side trace we classify on; -vv also names the auth stage
        "-T",  # no PTY: the gateway streams serial output, we only need auth
        "-n",  # stdin from /dev/null so the session cannot block on input
        "-F",
        "/dev/null",  # ignore operator ssh_config: no inherited ProxyCommand/Host rules
        "-p",
        str(_GATEWAY_PORT),
        *opts,
        "-o",
        "PreferredAuthentications=publickey",
        "-i",
        key_file,
        f"{gateway_user}@{host}",
    ]


def _run_gateway_probe(gateway_user: str, key_file: str, host: str) -> dict[str, Any]:
    """One serial-gateway auth attempt, recorded as a structured attempt.

    The gateway keeps the session open streaming serial output, so a timeout
    AFTER a completed authentication is a success, not an error — the verdict
    comes from the client trace, never from the exit code. ``elapsed_s`` and
    ``outcome`` are what separate "rejected the key fast" from "stalled in the
    handshake", which the exit code alone cannot express.
    """
    argv = _gateway_argv(gateway_user, key_file, host)
    stderr = ""
    outcome = "completed"
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_GATEWAY_TIMEOUT,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as err:
        outcome = "timeout"
        raw = err.stderr
        if isinstance(raw, bytes):
            stderr = raw.decode("utf-8", "replace")
        elif isinstance(raw, str):
            stderr = raw
    except OSError as err:
        outcome = "spawn_error"
        stderr = f"OSError: {err}"
    authenticated = any(token in stderr for token in _AUTH_SUCCESS_TOKENS)
    denied = any(token in stderr for token in _AUTH_DENIED_TOKENS)
    return {
        "authenticated": authenticated,
        "denied": denied,
        "outcome": outcome,
        "elapsed_s": round(time.monotonic() - started, 1),
        "stderr": stderr,
    }


def _probe_gateway_with_retry(gateway_user: str, key_file: str, host: str) -> tuple[bool, str, list[dict[str, Any]]]:
    """Authenticate to the serial gateway, retrying metadata-propagation lag.

    Instance ``ssh-keys`` / ``serial-port-enable`` metadata reaches the gateway
    asynchronously, so a first-attempt ``Permission denied`` is retried with
    backoff before it is reported as a real key-access failure. Returns the
    verdict, the last client trace, and one bounded record per attempt so a
    stalled handshake is distinguishable from a rejected key after the fact.
    """
    stderr = ""
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, _GATEWAY_ATTEMPTS + 1):
        probe = _run_gateway_probe(gateway_user, key_file, host)
        stderr = probe.pop("stderr")
        probe["attempt"] = attempt
        probe["trace_tail"] = _stderr_tail(stderr, _TRACE_TAIL_ERROR)
        attempts.append(probe)
        if probe["authenticated"]:
            return True, stderr, attempts
        if attempt >= _GATEWAY_ATTEMPTS:
            break
        if not probe["denied"]:
            # Transport-level failure (unreachable gateway, DNS, timeout before
            # the key was ever offered). Retry it under the same budget.
            print(f"  serial gateway {host} unreachable (attempt {attempt}/{_GATEWAY_ATTEMPTS})", file=sys.stderr)
        else:
            print(
                f"  serial gateway {host} rejected the key (attempt {attempt}/{_GATEWAY_ATTEMPTS}); "
                "retrying for metadata propagation",
                file=sys.stderr,
            )
        time.sleep(_GATEWAY_BACKOFFS[min(attempt - 1, len(_GATEWAY_BACKOFFS) - 1)])
    return False, stderr, attempts


def _stderr_tail(stderr: str, limit: int = 400) -> str:
    """Return the tail of an SSH trace, bounded for the JSON contract."""
    text = stderr.strip()
    return text[-limit:] if len(text) > limit else text


def _gateway_unreachable(probes: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> bool:
    """True only when the gateway PORT was provably never reachable.

    Every one of these must hold, so a rejected key can never be laundered
    into "unreachable":

      * the gateway name resolved on every transport probe (so this is not a
        DNS problem being reported as a port problem);
      * no transport probe ever completed a TCP connection — ``probes`` brackets
        the SSH attempts, so a single blip cannot satisfy it;
      * no SSH attempt saw an authentication denial, i.e. the client never got
        far enough to offer the key.
    """
    if not probes or not attempts:
        return False
    if not all(probe.get("dns_resolved") for probe in probes):
        return False
    if any(probe.get("tcp_connected") for probe in probes):
        return False
    return not any(attempt.get("denied") or attempt.get("authenticated") for attempt in attempts)


def _diagnose_gateway_failure(transport: dict[str, Any], attempts: list[dict[str, Any]]) -> str:
    """Summarize WHY the gateway handshake failed, from observed signals only.

    Folded into the step's ``error`` string because only that string reaches
    the validation verdict — an evidence-only diagnosis is invisible to the
    reader who has to act on the failure.
    """
    parts: list[str] = []
    host = transport.get("host", "?")
    if not transport.get("dns_resolved"):
        parts.append(f"gateway name {host} did not resolve: {transport.get('error', 'unknown')}")
    elif not transport.get("tcp_connected"):
        parts.append(f"{host} tcp/{_GATEWAY_PORT} unreachable: {transport.get('error', 'unknown')}")
    elif not transport.get("speaks_ssh"):
        parts.append(f"peer on {host} tcp/{_GATEWAY_PORT} sent no SSH banner: {transport.get('error', 'unknown')}")
    else:
        parts.append(f"transport to {host} OK (banner {transport.get('banner', '')!r})")
    last = attempts[-1] if attempts else {}
    parts.append(
        "attempts="
        + ",".join(f"{a.get('outcome')}@{a.get('elapsed_s')}s" for a in attempts)
        + (" denied" if last.get("denied") else "")
    )
    tail = last.get("trace_tail") or ""
    if tail:
        parts.append(f"ssh trace tail: {tail[-_TRACE_TAIL_ERROR:]}")
    return " | ".join(parts)


def _probe_network_device_access(project: str, zone: str, instance: compute_v1.Instance) -> dict[str, Any]:
    """Enumerate the tenant-visible network surface for the target instance.

    Compute Engine's network fabric is software-defined: the only tenant-side
    devices are virtual NICs on the instance and API-managed Cloud Routers,
    neither of which exposes an SSH/console endpoint a specified key could
    authenticate against. The subtest is therefore provider-hidden — but the
    verdict is taken only after these real API reads resolve the project's own
    resources, never from a constant.
    """
    interfaces = [
        {
            "name": nic.name,
            "network": short_name(nic.network),
            "subnetwork": short_name(nic.subnetwork),
            "nic_type": nic.nic_type or "VIRTIO_NET",
        }
        for nic in (instance.network_interfaces or [])
    ]
    evidence: dict[str, Any] = {
        "project": project,
        "zone": zone,
        "instance_self_link": instance.self_link,
        "network_interfaces": interfaces,
    }
    probes = ["instances.get"]
    region = zone_to_region(zone)
    try:
        routers = retry_idempotent_list(
            compute_v1.RoutersClient().list,
            op_desc=f"routers.list {region}",
            project=project,
            region=region,
        )
        evidence["cloud_routers"] = [router.name for router in routers]
        probes.append("routers.list")
    except (gax.GoogleAPICallError, gax.RetryError) as exc:
        # A denied/failed router enumeration is recorded as-is; it is
        # supplementary evidence, and the instance readback above already
        # resolved the project's own network surface.
        _, detail = classify_gcp_error(exc)
        evidence["cloud_routers_error"] = detail
    return {
        "passed": True,
        "provider_hidden": True,
        "message": (
            "Compute Engine networking is software-defined: the instance's network "
            "devices are virtual NICs and the only tenant-configurable network "
            "appliance is the API-managed Cloud Router. Neither exposes a "
            "tenant-reachable SSH or console endpoint, so key-based network-device "
            "access is a provider-owned plane on this platform."
        ),
        "probes": probes,
        "evidence": evidence,
    }


@handle_gcp_errors
def main() -> int:
    """Run AUTH03 specified-key component-access probes and emit JSON."""
    parser = argparse.ArgumentParser(description="Prove specified-key access to SOL / network devices")
    parser.add_argument("--instance-id", required=True, help="Compute Engine instance name")
    parser.add_argument("--key-file", required=True, help="Path to the instance private key")
    parser.add_argument(
        "--key-name",
        required=True,
        help="SSH key identifier confirmed on the instance at launch (instance_key_name)",
    )
    parser.add_argument("--region", required=True, help="GCP region or zone")
    parser.add_argument("--zone", default=None, help="GCP zone (overrides region)")
    parser.add_argument("--ssh-user", default="ubuntu", help="Guest user the launch key was injected for")
    parser.add_argument("--project", default=None, help="GCP project ID (ADC fallback)")
    args = parser.parse_args()

    # Start of the step's self-imposed wall-clock budget (see
    # `_STEP_WALL_BUDGET`). Stamped before any cloud call so every deadline
    # derived from it accounts for the work already done.
    started = time.monotonic()

    project = resolve_project(args.project)
    zone = args.zone or narrow_region_to_zone(args.region)
    key_name = "" if args.key_name == _NONE_SENTINEL else args.key_name
    key_file = "" if args.key_file == _NONE_SENTINEL else args.key_file

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "test_name": "component_key_access",
        "instance_id": args.instance_id,
        "key_name": key_name,
        "region": args.region,
        "zone": zone,
        "project": project,
        "tests": {},
    }

    # The launch step emits `instance_key_name` only when its metadata
    # readback CONFIRMED the injected key. An absent value therefore means
    # there is no verified key identity to prove access with — fail honestly
    # rather than substituting the requested label.
    if not key_name:
        result["error"] = (
            "No confirmed SSH key identity forwarded from launch_instance "
            "(instance_key_name was null): the launch readback did not prove the "
            "requested key is present in the instance ssh-keys metadata."
        )
        result["tests"]["sol_access"] = {"passed": False, "error": result["error"]}
        print(json.dumps(result, indent=2, default=str))
        return 1

    if not key_file or not Path(key_file).is_file():
        result["error"] = f"Key file not found: {args.key_file}"
        result["tests"]["sol_access"] = {"passed": False, "error": result["error"]}
        print(json.dumps(result, indent=2, default=str))
        return 1

    pubkey = pubkey_from_private_key(key_file)
    if not pubkey:
        result["error"] = f"Unable to derive an OpenSSH public key from {key_file}"
        result["tests"]["sol_access"] = {"passed": False, "error": result["error"]}
        print(json.dumps(result, indent=2, default=str))
        return 1

    client = compute_v1.InstancesClient()
    serial_state = "unknown"
    restore_armed = False
    prior_serial_value: str | None = None
    cleanup_errors: list[str] = []

    def _arm_serial_restore(prior: str | None) -> None:
        """Record restore ownership the instant the metadata write is accepted.

        Called from inside ``_write_metadata_key`` before the async operation
        wait, so an accepted mutation whose op-wait later times out or errors
        is still reverted by the ``finally`` block below.
        """
        nonlocal restore_armed, prior_serial_value
        restore_armed = True
        prior_serial_value = prior

    try:
        instance = get_instance(project, zone, args.instance_id)

        # 1. Re-confirm the specified key is the one on the instance.
        key_confirmed = instance_has_pubkey(instance, pubkey)

        sol: dict[str, Any] = {
            "passed": False,
            "probes": ["instances.get:ssh-keys"],
            "evidence": {
                "key_name": key_name,
                "key_file": key_file,
                "ssh_keys_metadata_confirmed": key_confirmed,
            },
        }

        if not key_confirmed:
            sol["error"] = (
                f"Public key paired with {key_file} is not present in the {args.instance_id} ssh-keys metadata"
            )
            result["tests"]["sol_access"] = sol
            result["tests"]["network_device_access"] = _probe_network_device_access(project, zone, instance)
            result["error"] = sol["error"]
            # Verdict recorded, not emitted: every exit from inside this block
            # goes through the cleanup `finally` and the single print below it.
            raise _VerdictRecorded

        # 2. Enable interactive serial access when the instance does not
        #    already carry it, remembering the prior value for restoration.
        try:
            serial_state = _ensure_serial_port_enabled(
                client,
                project,
                zone,
                args.instance_id,
                instance,
                # Withhold the restore reserve: whatever this write spends, the
                # `finally` revert below still gets a window inside the
                # provider-config cap.
                budget=_metadata_write_budget(started, reserve=_RESTORE_RESERVE),
                on_mutation_accepted=_arm_serial_restore,
            )
        except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as exc:
            if _is_serial_access_disabled(exc):
                # Platform-level removal of the capability, not a key failure.
                # Mirrors the reference provider's serial-access-disabled skip:
                # a skipped run reports no subtest verdicts, because the path
                # that would have produced them never ran.
                #
                # The constraint can also refuse a LATER attempt of the
                # read-modify-write ladder, after an earlier attempt was
                # accepted and armed restoration. So the skip is only
                # RECORDED here and emitted after the `finally` restore has
                # run: a restore failure must be able to turn this green skip
                # into a failing result before anything reaches stdout.
                result["success"] = True
                result["skipped"] = True
                result["skip_reason"] = _SERIAL_DISABLED_SKIP_REASON
                result["skip_evidence"] = {
                    "ssh_keys_metadata_confirmed": key_confirmed,
                    "constraint": _SERIAL_DISABLED_CONSTRAINT,
                    "probes": [*sol["probes"], f"instances.setMetadata:{_SERIAL_ENABLE_KEY}"],
                }
                result["tests"] = {}
                raise _VerdictRecorded from exc
            raise

        sol["probes"].append(f"instances.setMetadata:{_SERIAL_ENABLE_KEY}")
        sol["evidence"]["serial_port_enable"] = serial_state

        # 3. Authenticate to the serial-console gateway with ONLY that key.
        #    The gateway is the one serving the instance's own region, derived
        #    from the resolved zone. The transport probe runs first so a failure
        #    names the stage that broke instead of collapsing every cause into
        #    "did not authenticate".
        gateway_host = _gateway_host(zone)
        transport = _probe_gateway_transport(gateway_host)
        gateway_user = _gateway_username(project, zone, args.instance_id, args.ssh_user)
        authenticated, stderr, attempts = _probe_gateway_with_retry(gateway_user, key_file, gateway_host)
        transport_probes = [transport]
        if not authenticated:
            # Re-probe AFTER the attempts so "the port never accepted a
            # connection" is a bracketed observation rather than one sample.
            transport_probes.append(_probe_gateway_transport(gateway_host))
        sol["probes"].extend(["serial_console_gateway_tcp", "serial_console_gateway_ssh"])
        sol["evidence"].update(
            {
                "gateway_host": gateway_host,
                "gateway_port": _GATEWAY_PORT,
                "gateway_user": gateway_user,
                "gateway_transport": transport_probes,
                "gateway_attempts": attempts,
                "authenticated": authenticated,
                "ssh_trace_tail": _stderr_tail(stderr, _TRACE_TAIL_EVIDENCE),
            }
        )

        # `unreachable` means the gateway port was provably never reachable
        # from this host. Unlike the org-policy arm above, the platform still
        # offers the capability, so that is a defect in the validation
        # environment — reported as a failing configuration error, with no
        # environment-controlled path to a skip or a pass.
        unreachable = _gateway_unreachable(transport_probes, attempts)
        last = transport_probes[-1]
        unreachable_fields = {
            "host": gateway_host,
            "port": _GATEWAY_PORT,
            "addresses": ",".join(last.get("addresses") or []),
            "error": last.get("error", "connect timed out"),
        }

        sol["passed"] = authenticated
        if authenticated:
            sol["message"] = (
                f"Specified key {key_name!r} completed publickey authentication to the "
                "Compute Engine serial-console gateway"
            )
        elif unreachable:
            # Not a key verdict and not a skip: the environment could not run
            # the probe at all, which is operator-actionable.
            sol["message"] = _GATEWAY_UNREACHABLE_ERROR.format(**unreachable_fields)
            sol["error"] = sol["message"]
            result["error_type"] = "configuration_error"
        else:
            sol["message"] = (
                f"Specified key {key_name!r} did not authenticate to the Compute Engine "
                f"serial-console gateway ({_diagnose_gateway_failure(transport_probes[-1], attempts)})"
            )
            sol["error"] = sol["message"]

        result["tests"]["sol_access"] = sol
        result["tests"]["network_device_access"] = _probe_network_device_access(project, zone, instance)
        result["success"] = all(bool(test.get("passed")) for test in result["tests"].values())
        if not result["success"] and sol.get("error"):
            result["error"] = sol["error"]

    except _VerdictRecorded:
        # The verdict (structured skip, or unconfirmed-key failure) is already
        # in `result`; nothing to add. The raise exists so restoration below
        # runs BEFORE the single emit, and so cleanup can still demote what
        # was recorded. Must precede the generic handler: `_VerdictRecorded`
        # is an `Exception` and would otherwise be classified as a cloud
        # error and overwrite the recorded verdict.
        pass
    except Exception as exc:
        bucket, detail = classify_gcp_error(exc)
        result["success"] = False
        result["error_type"] = bucket
        result["error"] = detail
        result["tests"].setdefault("sol_access", {"passed": False, "error": detail})
    finally:
        # Restore the instance to the exact metadata posture this step found.
        # Only a write THIS step had ACCEPTED is reverted — an operator
        # instance that already had interactive serial access on keeps it, and
        # a write that was never accepted (stale fingerprint, retry budget
        # exhausted) leaves nothing to revert.
        #
        # The revert runs under whatever is left of the step budget (floored,
        # never zero), so it starts and finishes before the orchestrator's
        # kill instead of racing it. A truncated op-wait here is reported as a
        # cleanup error rather than hidden: the write itself was accepted, so
        # the instance is restored even when its operation is not confirmed.
        if restore_armed:
            try:
                _write_metadata_key(
                    client,
                    project,
                    zone,
                    args.instance_id,
                    _SERIAL_ENABLE_KEY,
                    prior_serial_value,
                    budget=_metadata_write_budget(started),
                )
            except Exception as exc:
                # Recorded, never masked: a failed restore demotes success
                # below via preserve_success_after_cleanup. The message carries
                # the full replay instruction — instance, zone, and the exact
                # prior value (or "unset" when the key was absent) — because
                # that prior value exists only in this process, so an error
                # naming just the key would leave the operator unable to put
                # the instance back the way this step found it.
                _, detail = classify_gcp_error(exc)
                target = "unset" if prior_serial_value is None else repr(prior_serial_value)
                cleanup_errors.append(
                    f"restore {_SERIAL_ENABLE_KEY} to {target} on {args.instance_id}@{zone}: {detail}"
                )
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
            if result.pop("skipped", False):
                # A structured skip is a POST-cleanup gate: it may be emitted
                # only when cleanup was clean. Cleanup here is the metadata
                # restore, so a skip surviving a failed restore would tell the
                # operator "the capability is unavailable, nothing was done"
                # about an instance that still carries this step's mutation —
                # and on an adopted operator VM nothing downstream would ever
                # clear it. The skip is withdrawn; its reason is kept as the
                # evidence of why no subtest verdict exists.
                result["skip_evidence"] = {
                    **result.pop("skip_evidence", {}),
                    "withdrawn_skip_reason": result.pop("skip_reason", ""),
                }
            if not result.get("error"):
                # A run whose only failure is the restore has no other error to
                # report, and that is precisely the run an operator must act
                # on. Say so in `error` as well as in `cleanup_errors`.
                result["error"] = "; ".join(cleanup_errors)
        preserve_success_after_cleanup(result)

    # The ONLY emit. Reached on every path — pass, failure, structured skip,
    # unhandled cloud error — so the JSON on stdout is always the result AFTER
    # cleanup has had its say, and the exit status always matches it.
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

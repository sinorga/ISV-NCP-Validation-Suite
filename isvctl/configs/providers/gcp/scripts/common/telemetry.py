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

"""Shared GCP observability telemetry / log signal helpers.

GCP analog of the AWS oracle's ``common/cloudwatch.py``: the primitives the
observability telemetry and log-availability probes share, kept SDK-only
(``google.cloud.logging_v2`` / ``google.cloud.resourcemanager_v3``) behind the
shared ``common.errors`` envelope so callers never re-implement scope-binding or
error classification.

Two primitives live here:

* ``probe_project_identity`` — the Resource Manager ``get_project`` identity
  probe that precedes every PROVIDER-HIDDEN result. A managed physical plane
  (BMC, NVLink switch fabric, subnet manager, ...) exposes no tenant telemetry
  endpoint on Compute Engine, so the honest evidence is "the tenant control
  plane is reachable but this plane is provider-owned". The probe proves the
  first half with a real API call; a probe failure is a real control-plane
  error, never silently swallowed into a provider-hidden pass.

* ``query_vpc_flows`` — a scope-bound Cloud Logging query against the project
  ``compute.googleapis.com/vpc_flows`` log. VPC Flow Logs are the tenant-visible
  GCP network-telemetry signal (per-flow ``bytes_sent`` / ``packets_sent`` with
  inbound + outbound coverage), so the concrete network-plane and
  telemetry-delivery probes derive their real sample counts and freshest sample
  timestamp from it rather than a provider-hidden literal. The query is bound to
  the target region and the caller-supplied run-owned subnetwork / instance
  filters so unrelated same-named, cross-region, or future-dated entries can
  never stand in as evidence.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from google.cloud import logging_v2, resourcemanager_v3

from common.compute import short_name, unique_suffix
from common.errors import classify_gcp_error, retry_idempotent
from common.ssh_utils import quote, ssh_run

# Compute Engine writes VPC Flow Logs to this project Cloud Logging log on
# gce_subnetwork resources. ``%2F`` is the URL-encoded slash used in logName.
VPC_FLOWS_LOG_ID = "compute.googleapis.com%2Fvpc_flows"
VPC_FLOWS_LOG_DESTINATION = "compute.googleapis.com/vpc_flows"
# The provider-neutral telemetry source label emitted for the vpc_flows signal.
VPC_FLOWS_TELEMETRY_SOURCE = f"cloud_logging:{VPC_FLOWS_LOG_DESTINATION}"
# The per-flow jsonPayload byte/packet fields the count-based network telemetry
# probes report as their metric names (real vpc_flows fields, not invented).
VPC_FLOWS_METRIC_FIELDS = ["bytes_sent", "packets_sent"]

DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_SAMPLE_MAX = 20


def probe_project_identity(project: str) -> str:
    """Prove the active GCP project is reachable; return its project id.

    The idempotent Resource Manager read is retried on the typed transient
    bucket and raw transport drops via ``retry_idempotent`` (a single 429/5xx no
    longer aborts a provider-hidden path). Any non-transient error, or exhausted
    transient budget, raises the underlying Resource Manager error so the caller
    records a real control-plane failure instead of emitting provider-hidden
    evidence for an unreachable project.
    """
    proj = retry_idempotent(
        resourcemanager_v3.ProjectsClient().get_project,
        name=f"projects/{project}",
        op_desc=f"resourcemanager get_project {project}",
    )
    return proj.project_id or project


@dataclass
class FlowQueryResult:
    """Result of a scope-bound Cloud Logging vpc_flows query.

    ``ok`` is True when the scoped query executes (a successful query, even with
    zero entries, proves the Cloud Logging endpoint and vpc_flows destination are
    accessible). ``sample_count`` and ``latest_timestamp`` are REAL bounded
    samples, never fabricated. ``metric_fields`` is the subset of the requested
    byte/packet fields that the returned entries actually carried in their
    ``jsonPayload`` — the read-back metric surface, so a consumer emits only
    fields the probe observed instead of a preloaded literal. On failure
    ``error`` carries the classified ``[bucket=<name>]`` message and ``error_type``
    the bare bucket so callers preserve WHY the query failed rather than
    collapsing it to a generic string.
    """

    ok: bool
    sample_count: int = 0
    latest_timestamp: str = ""
    metric_fields: list[str] = field(default_factory=list)
    error: str = ""
    error_type: str = ""


def subnetwork_scope_filter(subnets: list) -> str:
    """Build a region-safe subnetwork-identity clause for the vpc_flows query.

    Prefers the stable numeric ``subnetwork_id`` when every target exposes one
    (an id cannot collide across regions); otherwise falls back to
    ``subnetwork_name`` which the query already region-binds via
    ``resource.labels.location``. Returns ``""`` when no subnets are supplied so
    the caller can refuse an unscoped query.
    """
    if not subnets:
        return ""
    subnet_ids = [str(s.id) for s in subnets if getattr(s, "id", 0)]
    if len(subnet_ids) == len(subnets):
        clause = " OR ".join(f'"{sid}"' for sid in subnet_ids)
        return f"resource.labels.subnetwork_id=({clause})"
    name_clause = " OR ".join(f'"{short_name(s.name)}"' for s in subnets)
    return f"resource.labels.subnetwork_name=({name_clause})"


def query_vpc_flows(
    project: str,
    *,
    region: str,
    extra_filters: list[str] | None = None,
    max_results: int = DEFAULT_SAMPLE_MAX,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    start_time: datetime | None = None,
) -> FlowQueryResult:
    """Run a scope-bound Cloud Logging query for the vpc_flows log.

    Scope-binding (so unrelated same-named, cross-region, future-dated, or
    pre-fixture entries can never stand in as evidence for the target network):

      * Bind a fixed start/end window. The upper bound is a single ``now``
        snapshot so a future-dated entry can never be admitted. The lower bound
        is the caller-supplied ``start_time`` fixture marker when given — so a
        freshness consumer only sees records emitted at/after its own traffic
        fixture and a stale pre-fixture flow cannot masquerade as the newly
        emitted sample — or the freshness lookback window otherwise.
      * Bind ``resource.type=gce_subnetwork`` to the target region
        (``resource.labels.location``).
      * The caller supplies ``extra_filters`` binding the query to the exact
        run-owned subnetwork identity and/or instance and traffic direction.

    The idempotent list is retried on the typed transient bucket and raw
    transport drops via ``retry_idempotent``.
    """
    log_name = f"projects/{project}/logs/{VPC_FLOWS_LOG_ID}"
    now = datetime.now(UTC)
    if start_time is not None:
        lower = min(start_time.astimezone(UTC), now)
    else:
        lower = now - timedelta(hours=max(lookback_hours, 1))
    start = lower.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    filter_parts = [
        f'logName="{log_name}"',
        'resource.type="gce_subnetwork"',
        f'resource.labels.location="{region}"',
        f'timestamp >= "{start}"',
        f'timestamp <= "{end}"',
    ]
    filter_parts.extend(extra_filters or [])
    filter_str = " AND ".join(filter_parts)
    try:
        client = logging_v2.Client(project=project)
        entries = retry_idempotent(
            lambda: list(
                client.list_entries(
                    resource_names=[f"projects/{project}"],
                    filter_=filter_str,
                    # Newest-first so a freshness consumer (delivery latency)
                    # reads the actual freshest record rather than the newest of
                    # the oldest ``max_results``; count consumers are unaffected.
                    order_by=logging_v2.DESCENDING,
                    max_results=max_results,
                )
            ),
            op_desc=f"cloud logging list_entries {VPC_FLOWS_LOG_DESTINATION}",
        )
        return FlowQueryResult(
            ok=True,
            sample_count=len(entries),
            latest_timestamp=newest_entry_timestamp(entries),
            metric_fields=observed_metric_fields(entries, VPC_FLOWS_METRIC_FIELDS),
        )
    except Exception as e:
        error_type, error_msg = classify_gcp_error(e)
        return FlowQueryResult(ok=False, error=error_msg, error_type=error_type)


def observed_metric_fields(entries: list, candidate_fields: list[str]) -> list[str]:
    """Return which ``candidate_fields`` the returned vpc_flows entries carry.

    Reads back each entry's ``jsonPayload`` (the ``StructEntry.payload`` dict
    Cloud Logging returns for a vpc_flows record) and reports the subset of
    ``candidate_fields`` that appears with a concrete value in at least one
    sampled entry. This is the GCP analog of the AWS oracle gating metric
    availability on the descriptors CloudWatch returns for the requested metric
    names: the metric-surface evidence is DERIVED from the readback, never a
    preloaded literal, so a scoped entry that lacks the byte/packet fields can
    never make a consumer report a metric the probe did not observe. Order
    follows ``candidate_fields`` for a stable probe.
    """
    observed: set[str] = set()
    for entry in entries:
        payload = getattr(entry, "payload", None)
        if not isinstance(payload, dict):
            continue
        for candidate in candidate_fields:
            if payload.get(candidate) not in (None, "", [], {}):
                observed.add(candidate)
    return [candidate for candidate in candidate_fields if candidate in observed]


def newest_entry_timestamp(entries: list) -> str:
    """Return the newest Cloud Logging entry timestamp as an ISO string, or ""."""
    newest: datetime | None = None
    for entry in entries:
        ts = getattr(entry, "timestamp", None)
        if not isinstance(ts, datetime):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if newest is None or ts > newest:
            newest = ts
    return newest.isoformat() if newest is not None else ""


def age_seconds(iso_timestamp: str) -> int:
    """Return the age in seconds of an ISO timestamp, or -1 when unparseable/empty."""
    if not iso_timestamp:
        return -1
    try:
        parsed = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return -1
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(int((datetime.now(UTC) - parsed).total_seconds()), 0)


# ── vpc_flows sample polling ───────────────────────────────────────────────
# Compute Engine exports VPC Flow Logs to Cloud Logging on a batching cadence, so
# a freshly generated flow is not immediately queryable. The concrete network /
# delivery probes poll the scope-bound query until a sample appears (the AWS
# oracle polls CloudWatch for the identical ingestion-delay reason).
DEFAULT_FLOW_POLL_TIMEOUT_SECONDS = 300
DEFAULT_FLOW_POLL_INTERVAL_SECONDS = 20


def poll_vpc_flows(
    project: str,
    *,
    region: str,
    extra_filters: list[str] | None = None,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    poll_timeout_seconds: int = DEFAULT_FLOW_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: int = DEFAULT_FLOW_POLL_INTERVAL_SECONDS,
    start_time: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> FlowQueryResult:
    """Poll the scope-bound vpc_flows query until a sample appears or the deadline.

    Polling stops early on a query error (``ok=False``) so a real API / auth /
    transient-exhausted failure surfaces as itself rather than being masked as
    "no samples". Each iteration re-runs the fully scoped, retry-wrapped
    ``query_vpc_flows`` (its own bounded transient retry is inside).

    When ``start_time`` is supplied every iteration is lower-bounded at that
    fixture marker, so the poll only returns once a record emitted at/after the
    caller's own traffic fixture is queryable — a pre-fixture flow already in the
    log cannot satisfy the poll on the first iteration.
    """
    result = query_vpc_flows(
        project, region=region, extra_filters=extra_filters, lookback_hours=lookback_hours, start_time=start_time
    )
    deadline = time.monotonic() + max(poll_timeout_seconds, 0)
    while result.ok and result.sample_count == 0 and time.monotonic() < deadline:
        sleep(poll_interval_seconds)
        result = query_vpc_flows(
            project, region=region, extra_filters=extra_filters, lookback_hours=lookback_hours, start_time=start_time
        )
    return result


# ── Tenant-visible telemetry fixtures (SSH-driven, run-scoped) ─────────────
# VPC Flow Logs and hypervisor persistent-disk metrics record only ACTUAL flows /
# I-O — unlike the AWS oracle, whose CloudWatch packet/EBS metrics report
# continuously for any running instance. So the concrete GCP probes deterministically
# generate a bounded, run-scoped fixture on the launched host (over the same
# injected-key SSH boundary the host_syslogs probe uses) before they query. Every
# fixture's OBS_FIXTURE_OK marker is gated on the traffic / I-O actually
# completing (not merely on SSH connectivity), so a caller that requires the
# fixture reflects a real run-generated flow / I-O. A fixture that cannot run — or
# ran but moved no traffic — is surfaced to the caller as a failure, which then
# honestly fails on absent samples; it is never turned into a success-shaped skip.
_FIXTURE_MARK = "OBS_FIXTURE_OK"

# IP protocol numbers the vpc_flows ``jsonPayload.connection.protocol`` field
# carries, keyed by the external fixture's probe kind so a north-south / delivery
# query pins the EXACT protocol the successful probe used (an ICMP fixture can
# never be satisfied by a TCP flow, and vice versa).
_PROTO_NUMBERS = {"icmp": 1, "tcp": 6, "udp": 17}

# The external fixture's PRIMARY candidate endpoint — the first target the
# generate_external_traffic script probes (ICMP to 8.8.8.8). When a SUCCESSFUL
# fixture marker does not carry a parseable proto/dip/dport tuple (e.g. a replayed
# or simulated guest that emits only the bare OBS_FIXTURE_OK without the echoed
# tuple), the fixture correlates against THIS deterministic endpoint so the caller
# still binds an EXACT external tuple (icmp/8.8.8.8) rather than an under-scoped
# subnet+vm_name query the host's own SSH control flow could self-satisfy. A live
# guest running the script always echoes the reached tuple, so this default is only
# ever used for a marker-only replay — never in place of a real generated flow.
_EXTERNAL_FIXTURE_PRIMARY_IP = "8.8.8.8"


@dataclass
class ExternalFlowFixture:
    """The exact external endpoint a successful north-south traffic fixture reached.

    The plain ``ran_ok`` / ``detail`` gate the internal / disk fixtures use is not
    enough for the external planes: a caller must bind its VPC Flow Logs query to
    the SPECIFIC flow this fixture generated, otherwise the fixture's own inbound
    SSH control session (tcp/22 — an internet VPC Flow Logs record that involves
    the run host) can self-satisfy a query scoped only by subnet + host identity.
    So the fixture also reports which tuple actually succeeded:

    * ``protocol`` — the IP protocol number of the successful probe (ICMP=1 for
      ping, TCP=6 for the HTTPS fallback); 0 when the fixture did not run.
    * ``dest_ip`` — the LITERAL external endpoint IP the probe reached (never a
      hostname, so the queryable ``connection.dest_ip`` / ``src_ip`` is known and
      can be pinned).
    * ``dest_port`` — the destination port for the TCP probe (443); 0 for ICMP,
      where a port does not apply.
    """

    ok: bool
    detail: str = ""
    protocol: int = 0
    dest_ip: str = ""
    dest_port: int = 0


def _ssh_fixture(host: str, ssh_user: str, key_file: str, script: str, *, timeout: int) -> tuple[bool, str, str]:
    """Run a bounded fixture shell script over SSH; return ``(ran_ok, detail, stdout)``."""
    if not (host and ssh_user and key_file):
        return False, "no SSH host/user/key forwarded from launch_host", ""
    exit_code, stdout, stderr = ssh_run(host, ssh_user, key_file, script, timeout=timeout)
    if exit_code == 0 and _FIXTURE_MARK in stdout:
        return True, "", stdout
    return False, (stderr.strip()[:200] or f"fixture exited rc={exit_code}"), stdout


def _run_fixture(host: str, ssh_user: str, key_file: str, script: str, *, timeout: int) -> tuple[bool, str]:
    """Run a bounded fixture shell script over SSH; return ``(ran_ok, detail)``."""
    ran_ok, detail, _ = _ssh_fixture(host, ssh_user, key_file, script, timeout=timeout)
    return ran_ok, detail


def _parse_external_flow_tuple(stdout: str) -> tuple[int, str, int]:
    """Parse the ``proto=/dip=/dport=`` tuple the external fixture echoes on its marker line.

    Returns ``(protocol_number, dest_ip, dest_port)`` for the FIRST probe that
    succeeded, or ``(0, "", 0)`` when the marker carries no parseable tuple.
    """
    for line in stdout.splitlines():
        if _FIXTURE_MARK not in line:
            continue
        fields: dict[str, str] = {}
        for token in line.split():
            key, sep, value = token.partition("=")
            if sep:
                fields[key] = value
        protocol = _PROTO_NUMBERS.get(fields.get("proto", ""), 0)
        dest_ip = fields.get("dip", "")
        try:
            dest_port = int(fields.get("dport", "0"))
        except ValueError:
            dest_port = 0
        if protocol and dest_ip:
            return protocol, dest_ip, dest_port
    return 0, "", 0


def generate_external_traffic(host: str, ssh_user: str, key_file: str) -> ExternalFlowFixture:
    """Generate bounded north-south (VM<->external) traffic and report the exact flow.

    Drives ICMP (ping) then a TCP/443 HTTPS fallback to LITERAL public anycast IPs
    (never hostnames, so the queryable destination IP is known) so the run host's
    NIC emits an outbound + inbound flow that vpc_flows records with a KNOWN
    external peer. Egress is allowed by the GCP default egress firewall; only SSH
    ingress is operator-restricted.

    The probe stops at the FIRST endpoint it reaches and echoes that endpoint's
    exact tuple on the ``OBS_FIXTURE_OK`` marker line (``proto=/dip=/dport=``), so
    the caller can pin its VPC Flow Logs query to the precise protocol + external
    IP (+ port) THIS fixture generated rather than any post-marker flow that merely
    involves the host. The marker is emitted (with ``rc`` propagated as the SSH
    exit status) ONLY when a probe succeeded; a host that reached nothing external
    returns ``ok=False`` instead of a spurious success.

    A SUCCESSFUL marker whose tuple is not parseable (a replayed / simulated guest
    that emits only the bare marker) does NOT abandon the query: the traffic ran
    (the marker is gated on a probe succeeding), so the fixture correlates against
    the deterministic PRIMARY candidate endpoint (icmp/``8.8.8.8``) instead. That
    keeps the caller bound to an EXACT external tuple — never an under-scoped
    subnet+vm_name query the host's own SSH control flow could self-satisfy — and
    lets the API result decide the verdict, while a live guest running this script
    always echoes the reached tuple so the fallback is never used for a real flow.
    """
    script = (
        "rc=1; proto=; dip=; dport=0; "
        "for t in 8.8.8.8 1.1.1.1; do "
        'if ping -c 15 -i 0.2 -W 1 "$t" >/dev/null 2>&1; then rc=0; proto=icmp; dip="$t"; dport=0; break; fi; '
        "done; "
        'if [ "$rc" -ne 0 ]; then '
        "for t in 1.1.1.1 8.8.8.8; do "
        'if curl -sk -o /dev/null --max-time 8 "https://$t"; then rc=0; proto=tcp; dip="$t"; dport=443; break; fi; '
        "done; fi; "
        f'[ "$rc" -eq 0 ] && echo "{_FIXTURE_MARK} proto=$proto dip=$dip dport=$dport"; exit "$rc"'
    )
    ran_ok, detail, stdout = _ssh_fixture(host, ssh_user, key_file, script, timeout=90)
    if not ran_ok:
        return ExternalFlowFixture(ok=False, detail=detail)
    protocol, dest_ip, dest_port = _parse_external_flow_tuple(stdout)
    if not (protocol and dest_ip):
        # Marker present (traffic ran) but no parseable tuple: correlate against the
        # fixture's deterministic primary candidate endpoint so the query stays an
        # EXACT external tuple rather than a self-satisfiable subnet-only fallback.
        protocol, dest_ip, dest_port = _PROTO_NUMBERS["icmp"], _EXTERNAL_FIXTURE_PRIMARY_IP, 0
    return ExternalFlowFixture(ok=True, protocol=protocol, dest_ip=dest_ip, dest_port=dest_port)


def external_flow_tuple_filter(fixture: ExternalFlowFixture, instance_name: str) -> str:
    """Bind a vpc_flows query to the EXACT external flow a fixture generated, both directions.

    Pins, in EITHER flow direction:

    * ``connection.protocol`` — the fixture's IP protocol number;
    * the fixture's external endpoint IP as the NON-host end
      (``connection.dest_ip`` on the run host's outbound record, ``connection.src_ip``
      on the inbound / return record), plus the fixture's destination port for TCP;
    * the run host as the internal endpoint (``src_instance`` / ``dest_instance``
      ``vm_name``) — i.e. the run host bound in both directions.

    Because the external endpoint IP and protocol are pinned to the tuple the
    fixture actually reached, the fixture's own inbound SSH control flow — tcp/22
    whose external end is the operator trust IP, not the fixture endpoint — cannot
    satisfy the clause even though it shares the run host's ``vm_name``. Returns ""
    when the tuple or host identity is unknown so the caller refuses an
    under-scoped query rather than falling back to a self-satisfiable one.
    """
    if not (fixture.protocol and fixture.dest_ip and instance_name):
        return ""
    out_port = f" AND jsonPayload.connection.dest_port={fixture.dest_port}" if fixture.dest_port else ""
    in_port = f" AND jsonPayload.connection.src_port={fixture.dest_port}" if fixture.dest_port else ""
    outbound = (
        f'(jsonPayload.connection.dest_ip="{fixture.dest_ip}"{out_port} '
        f'AND jsonPayload.src_instance.vm_name="{instance_name}")'
    )
    inbound = (
        f'(jsonPayload.connection.src_ip="{fixture.dest_ip}"{in_port} '
        f'AND jsonPayload.dest_instance.vm_name="{instance_name}")'
    )
    return f"jsonPayload.connection.protocol={fixture.protocol} AND ({outbound} OR {inbound})"


def matches_external_fixture(record: dict, fixture: ExternalFlowFixture, instance_name: str) -> bool:
    """Executable model of ``external_flow_tuple_filter`` for the negative-world check.

    Reads the SAME vpc_flows fields the Cloud Logging clause constrains, so a
    synthetic flow record is admitted here iff the real scoped query would admit
    it. Kept next to the clause builder so the two renderings of the tuple stay in
    lock-step. ``record`` is a flat dict of the constrained fields:
    ``protocol`` / ``src_ip`` / ``dest_ip`` / ``src_port`` / ``dest_port`` /
    ``src_vm`` / ``dest_vm``.
    """
    if not (fixture.protocol and fixture.dest_ip and instance_name):
        return False
    if record.get("protocol") != fixture.protocol:
        return False
    outbound = (
        record.get("dest_ip") == fixture.dest_ip
        and (not fixture.dest_port or record.get("dest_port") == fixture.dest_port)
        and record.get("src_vm") == instance_name
    )
    inbound = (
        record.get("src_ip") == fixture.dest_ip
        and (not fixture.dest_port or record.get("src_port") == fixture.dest_port)
        and record.get("dest_vm") == instance_name
    )
    return bool(outbound or inbound)


# A canonical inbound SSH control flow: tcp/22 FROM the operator trust IP TO the run
# host. This is the "internet" VPC Flow Logs record the fixture's own SSH session
# emits after the fixture-start marker — the exact record that can self-satisfy a
# subnet+vm_name-only external query. The negative-world check proves the
# tuple-bound query rejects it. (TEST-NET-3 / RFC 1918 literals; never a real host.)
_SSH_ONLY_OPERATOR_IP = "203.0.113.9"
_SSH_ONLY_HOST_PRIVATE_IP = "10.128.0.9"


def external_fixture_discriminates_ssh(fixture: ExternalFlowFixture, instance_name: str) -> tuple[bool, str]:
    """Negative-world check: prove the tuple query admits the fixture flow, rejects SSH-only.

    Evaluates the exact tuple the caller is about to issue against two synthetic
    worlds over ``matches_external_fixture``:

    * POSITIVE world — the run host's own external flow to the fixture endpoint MUST
      be admitted, so the clause is not trivially empty (a query that matches
      nothing fails closed, but for the wrong reason).
    * NEGATIVE world — a post-marker inbound tcp/22 SSH-only control flow (operator
      trust IP -> run host) MUST be rejected, proving a required tenant-visible
      validator cannot pass on the control session instead of the operation it
      claims to measure.

    Returns ``(ok, reason)``; ``ok=False`` with a reason when either world is
    misclassified, so the caller fails the plane honestly instead of shipping a
    self-satisfiable query.
    """
    positive = {
        "protocol": fixture.protocol,
        "src_ip": _SSH_ONLY_HOST_PRIVATE_IP,
        "dest_ip": fixture.dest_ip,
        "src_port": 51000,
        "dest_port": fixture.dest_port,
        "src_vm": instance_name,
        "dest_vm": "",
    }
    ssh_only = {
        "protocol": _PROTO_NUMBERS["tcp"],
        "src_ip": _SSH_ONLY_OPERATOR_IP,
        "dest_ip": _SSH_ONLY_HOST_PRIVATE_IP,
        "src_port": 40000,
        "dest_port": 22,
        "src_vm": "",
        "dest_vm": instance_name,
    }
    if not matches_external_fixture(positive, fixture, instance_name):
        return False, "tuple filter would not match the fixture's own external flow"
    if matches_external_fixture(ssh_only, fixture, instance_name):
        return False, "tuple filter would admit an inbound SSH-only control flow"
    return True, ""


def generate_internal_traffic(
    host: str,
    ssh_user: str,
    key_file: str,
    peer_private_ip: str,
) -> tuple[bool, str]:
    """Generate proven ICMP traffic to one distinct run-owned internal peer."""
    if not peer_private_ip:
        return False, "no run-owned peer private IP was forwarded"
    script = f"ping -c 40 -i 0.2 -W 1 {quote(peer_private_ip)} >/dev/null 2>&1 && echo {_FIXTURE_MARK}"
    return _run_fixture(host, ssh_user, key_file, script, timeout=90)


def generate_disk_io(host: str, ssh_user: str, key_file: str, *, size_mb: int = 256) -> tuple[bool, str]:
    """Generate bounded write+read persistent-disk I-O on the host.

    ``conv=fdatasync`` forces the write burst to the disk (real write bytes / ops
    the hypervisor persistent-disk metrics record); the read pass uses direct I-O
    where supported to bypass the page cache. The run-scoped scratch file is
    always removed so the fixture leaves no artifact behind.

    The success marker is gated on the write+read actually completing: the write
    and read are AND-chained into an ``rc`` flag, cleanup (``rm -f``) runs
    unconditionally so no scratch file leaks, and ``OBS_FIXTURE_OK`` is emitted
    (with ``rc`` propagated as the SSH exit status) ONLY when both bounded ``dd``
    passes succeeded. A failed write or read therefore returns ``(False, ...)``
    instead of a spurious success, so the bounded I-O the caller measures is
    proven to have run before the fixture claims it did.
    """
    path = quote(f"/tmp/{unique_suffix('obs-io')}.bin")
    script = (
        "rc=0; "
        f"dd if=/dev/zero of={path} bs=1M count={int(size_mb)} conv=fdatasync 2>/dev/null "
        f"&& ( dd if={path} of=/dev/null bs=1M iflag=direct 2>/dev/null "
        f"|| dd if={path} of=/dev/null bs=1M 2>/dev/null ) || rc=1; "
        f"rm -f {path}; "
        f'[ "$rc" -eq 0 ] && echo {_FIXTURE_MARK}; exit "$rc"'
    )
    return _run_fixture(host, ssh_user, key_file, script, timeout=150)


def read_guest_capacity(
    host: str, ssh_user: str, key_file: str, device_names: list[str]
) -> tuple[bool, int, int, int, int, list[str]]:
    """Read guest used/free/total bytes joined to the enumerated persistent disks.

    ``device_names`` are the Compute Engine attached-disk ``device_name`` values
    the API enumerated for the launched instance. On GCE the guest exposes every
    attached disk (whole device and each partition) at the stable
    ``/dev/disk/by-id/google-<device_name>`` udev symlink — the same by-id join the
    AWS oracle uses for EBS — so the API disks are resolved to their canonical
    guest block devices and ONLY mounted filesystems backed by one of THOSE devices
    are summed. Unrelated loop, local-SSD, tmpfs, overlay, or auxiliary ``/dev/*``
    mounts can never contribute, and when no enumerated disk can be joined to a
    mounted filesystem the read fails (``ok=False``) rather than returning a
    device-set that describes different storage or a provisioned-size substitution.

    Returns ``(ok, total_bytes, used_bytes, free_bytes, mounts_checked,
    joined_device_names)`` — ``joined_device_names`` is the subset of API disks
    actually matched to guest capacity evidence.
    """
    if not (host and ssh_user and key_file) or not device_names:
        return False, 0, 0, 0, 0, []
    devs = " ".join(quote(d) for d in device_names)
    # ENUM lines map each enumerated disk (whole device + every partition) to its
    # canonical guest block device; DF lines carry each mount's canonicalised
    # source device plus byte columns. The identity join itself is done in Python
    # (below) so it stays deterministic and unit-testable, not buried in shell.
    script = (
        f"for d in {devs}; do "
        'for link in "/dev/disk/by-id/google-$d" "/dev/disk/by-id/google-$d"-part*; do '
        '[ -e "$link" ] || continue; '
        'real=$(readlink -f "$link" 2>/dev/null); '
        '[ -n "$real" ] && printf "ENUM\\t%s\\t%s\\n" "$d" "$real"; '
        "done; done; "
        "df -B1 --output=source,size,used,avail 2>/dev/null | tail -n +2 | "
        "while IFS= read -r row; do set -- $row; "
        '[ "$#" -ge 4 ] || continue; '
        'real=$(readlink -f "$1" 2>/dev/null); [ -n "$real" ] || real="$1"; '
        'printf "DF\\t%s\\t%s\\t%s\\t%s\\n" "$real" "$2" "$3" "$4"; '
        "done"
    )
    exit_code, stdout, _ = ssh_run(host, ssh_user, key_file, script, timeout=30)
    if exit_code != 0 or not stdout.strip():
        return False, 0, 0, 0, 0, []
    enum_device_to_name: dict[str, str] = {}
    df_rows: list[tuple[str, str, str, str]] = []
    for line in stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "ENUM" and len(parts) >= 3:
            enum_device_to_name[parts[2]] = parts[1]
        elif parts[0] == "DF" and len(parts) >= 5:
            df_rows.append((parts[1], parts[2], parts[3], parts[4]))
    total = used = free = mounts = 0
    joined: set[str] = set()
    for source, size_s, used_s, avail_s in df_rows:
        device_name = enum_device_to_name.get(source)
        if device_name is None:
            # Mount is not backed by any API-enumerated disk — exclude it so the
            # totals describe only the joined persistent disks.
            continue
        try:
            total += int(size_s)
            used += int(used_s)
            free += int(avail_s)
        except ValueError:
            continue
        joined.add(device_name)
        mounts += 1
    return mounts > 0, total, used, free, mounts, sorted(joined)

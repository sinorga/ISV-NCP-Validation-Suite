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

"""GCP telemetry delivery latency probe (observability test phase).

The GCP port of the AWS oracle ``telemetry_delivery_test.py``. AWS measures the
age of the freshest CloudWatch datapoint for the launched host; GCP measures the
age of the freshest queryable ``compute.googleapis.com/vpc_flows`` Cloud Logging
record scoped to the run-owned subnetworks. A newest-record age within the
operator budget proves the tenant-visible telemetry pipeline is delivering
timestamped records end to end; the observed latency is derived from a REAL entry
timestamp, never fabricated.

Honest, non-skipping contract (mirrors the AWS oracle exactly — the oracle NEVER
emits a success-shaped skip):

  * ``telemetry_endpoint_reachable`` passes when the scoped Cloud Logging query
    executes (real endpoint evidence).
  * The probe captures a fixture-start marker, then generates a bounded,
    run-scoped external-traffic fixture on the launched host (Compute Engine VPC
    Flow Logs record only ACTUAL flows, unlike CloudWatch which reports
    continuously for any running host). A fixture that cannot move real traffic is
    TERMINAL (rc=1) — the pass must reflect a flow this run actually generated. It
    then polls the query bound to the run subnetwork AND the launched host's
    instance identity — lower bounded at the fixture marker — until a host flow
    emitted at/after the fixture is delivered. A stale pre-fixture
    launch/cloud-init/SSH flow, or an unrelated same-subnet peer flow, can never
    satisfy the poll, so the measured latency is always the newly emitted
    operation's real emission-to-query age, and a missing correlated record fails
    (rc=1).
  * ``delivery_sample_present`` / ``delivery_within_threshold`` are gated on the
    REAL freshest record and its measured age; a missing record or an over-budget
    age is an honest ``_failed`` (rc=1), never a skip. Overall ``success`` is the
    AND of every subtest.

AWS reference implementation:
    ../../aws/scripts/observability/telemetry_delivery_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

from common.compute import resolve_project, short_name
from common.errors import classify_gcp_error, handle_gcp_errors
from common.network import list_subnetworks_for_network
from common.telemetry import (
    DEFAULT_FLOW_POLL_INTERVAL_SECONDS,
    DEFAULT_FLOW_POLL_TIMEOUT_SECONDS,
    VPC_FLOWS_TELEMETRY_SOURCE,
    age_seconds,
    external_fixture_discriminates_ssh,
    external_flow_tuple_filter,
    generate_external_traffic,
    poll_vpc_flows,
    subnetwork_scope_filter,
)

TEST_NAME = "telemetry_delivery_latency"
ASPECT_TESTS = [
    "telemetry_endpoint_reachable",
    "delivery_sample_present",
    "delivery_within_threshold",
]

# GCP VPC Flow Logs are exported to Cloud Logging on a batching cadence, so the
# freshest queryable record is routinely minutes old — the OBS05 NSRG 120s
# target is not achievable via this pipeline. 600s is the portable budget that
# still flags a stalled telemetry pipeline; the provider config forwards
# max_delivery_seconds (suite default 600).
DEFAULT_MAX_DELIVERY_SECONDS = 600

# The delivery poll ceiling MUST exceed the pass threshold, mirroring the AWS
# oracle's scan window (``max_delivery + SAMPLE_WINDOW_SECONDS``). A ceiling below
# ``max_delivery_seconds`` is a parity defect: (a) it abandons a slow-but-in-budget
# delivery (roughly poll_ceiling..max_delivery) before its record is queryable,
# falsely reporting a HEALTHY pipeline as "no correlated record", and (b) it caps
# the observable age below the threshold, so the ``observed > max_delivery_seconds``
# failure branch can never fire. Polling a bounded margin PAST the threshold lets a
# just-over-budget delivery be observed and fail the threshold honestly (the
# over-budget branch becomes reachable) while a delivery beyond the ceiling still
# fails closed via the no-correlated-record path.
DELIVERY_MEASURE_MARGIN_SECONDS = 60


def _base_result() -> dict[str, Any]:
    """Build the common observability result envelope."""
    return {
        "success": False,
        "platform": "observability",
        "test_name": TEST_NAME,
        "tests": {name: {"passed": False} for name in ASPECT_TESTS},
    }


def _passed(message: str, probes: dict[str, Any]) -> dict[str, Any]:
    """Build a passing subtest result."""
    return {"passed": True, "message": message, "probes": probes}


def _failed(error: str, probes: dict[str, Any]) -> dict[str, Any]:
    """Build a failing subtest result."""
    return {"passed": False, "error": error, "probes": probes}


def check_telemetry_delivery_latency(
    project: str,
    *,
    region: str,
    network_id: str,
    instance_id: str = "",
    host: str = "",
    ssh_user: str = "",
    key_file: str = "",
    max_delivery_seconds: int = DEFAULT_MAX_DELIVERY_SECONDS,
    poll_timeout_seconds: int = DEFAULT_FLOW_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: int = DEFAULT_FLOW_POLL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Measure vpc_flows delivery latency for the run-owned network."""
    result = _base_result()
    probes: dict[str, Any] = {
        "telemetry_source": VPC_FLOWS_TELEMETRY_SOURCE,
        "observed_delivery_seconds": -1,
        "max_delivery_seconds": max_delivery_seconds,
        "sample_count": 0,
        "latest_timestamp": "",
        "fixture_generated": False,
        "fixture_started_at": "",
        "probe_resource_id": instance_id or network_id,
    }

    try:
        subnets = list_subnetworks_for_network(project, region, network_id)
    except Exception as e:
        error_type, error_msg = classify_gcp_error(e)
        result["error_type"] = error_type
        result["error"] = error_msg
        for name in ASPECT_TESTS:
            result["tests"][name] = _failed(error_msg, probes)
        return result

    scope = subnetwork_scope_filter(subnets)
    if not scope:
        error = f"no subnetworks resolved for {network_id}; refusing an unscoped vpc_flows query"
        for name in ASPECT_TESTS:
            result["tests"][name] = _failed(error, probes)
        result["error"] = error
        return result

    # Capture the fixture-start marker BEFORE generating traffic so the readback is
    # correlated to THIS run's own flow. Every launch, cloud-init, and SSH flow
    # that already populated the subnet's vpc_flows predates this marker; binding
    # the poll's lower timestamp bound to it stops a stale pre-fixture record from
    # passing delivery latency without observing the operation we measure here.
    fixture_start = datetime.now(UTC)
    probes["fixture_started_at"] = fixture_start.isoformat()

    # Generate a bounded run-scoped external-traffic fixture so a fresh flow is
    # produced for the pipeline to deliver. Fixture failure is terminal (mirrors
    # network_telemetry_test): without a proven run-owned flow the poll could time
    # delivery latency off an unrelated scope-bound record (e.g. the east-west peer
    # or another run), so an unsuccessful fixture fails here instead of measuring a
    # flow this run did not generate.
    fixture = generate_external_traffic(host, ssh_user, key_file)
    probes["fixture_generated"] = fixture.ok
    if not fixture.ok:
        probes["fixture_detail"] = fixture.detail
        error = f"Failed to generate the telemetry-delivery traffic fixture: {fixture.detail}"
        result["error"] = error
        for name in ASPECT_TESTS:
            result["tests"][name] = _failed(error, probes)
        return result
    probes["fixture_protocol"] = fixture.protocol
    probes["fixture_dest_ip"] = fixture.dest_ip
    probes["fixture_dest_port"] = fixture.dest_port

    # Bind the delivery query to the EXACT external flow the fixture generated —
    # its protocol + external endpoint IP (+ port) AND the run host in both
    # directions — mirroring the AWS oracle scoping the CloudWatch probe to
    # InstanceId. The freshest record's age is only ever measured off THIS run's
    # generated flow: an unrelated same-subnet peer flow, another run, OR the
    # fixture's own inbound tcp/22 SSH control flow (whose external end is the
    # operator trust IP, not the fixture endpoint) can no longer time delivery
    # latency. A missing host identity or fixture tuple is terminal rather than a
    # self-satisfiable subnet-only fallback.
    tuple_clause = external_flow_tuple_filter(fixture, instance_id)
    if not tuple_clause:
        error = (
            "cannot bind telemetry-delivery latency to the exact external fixture flow "
            "(missing host identity or fixture tuple); refusing an under-scoped query "
            "its own SSH control flow could satisfy"
        )
        result["error"] = error
        for name in ASPECT_TESTS:
            result["tests"][name] = _failed(error, probes)
        return result
    discriminates, reason = external_fixture_discriminates_ssh(fixture, instance_id)
    if not discriminates:
        error = f"telemetry-delivery external-flow filter failed its SSH-only negative-world check: {reason}"
        result["error"] = error
        for name in ASPECT_TESTS:
            result["tests"][name] = _failed(error, probes)
        return result
    extra_filters = [scope, tuple_clause]

    # Never abandon the poll before the pass threshold (AWS oracle parity): the
    # effective ceiling scans a window WIDER than max_delivery_seconds, so a
    # slow-but-in-budget delivery is still caught and a just-over-budget one is
    # measured and fails the threshold branch instead of the poll giving up first.
    # The caller-supplied poll_timeout_seconds is only ever a floor.
    effective_poll_timeout = max(poll_timeout_seconds, max_delivery_seconds + DELIVERY_MEASURE_MARGIN_SECONDS)

    # start_time binds the poll's lower bound to the fixture marker, so observed
    # latency is only ever computed from a correlated (post-fixture) record and a
    # stale pre-fixture flow can never return immediately as a pass. An over-budget
    # delivery beyond the ceiling surfaces honestly as "no post-fixture record within
    # the poll budget" (rc=1) rather than an over-lookback record masking the stall.
    query = poll_vpc_flows(
        project,
        region=region,
        extra_filters=extra_filters,
        poll_timeout_seconds=effective_poll_timeout,
        poll_interval_seconds=poll_interval_seconds,
        start_time=fixture_start,
    )
    if not query.ok:
        probes = {**probes, "sample_count": query.sample_count}
        result["error_type"] = query.error_type
        result["error"] = query.error
        for name in ASPECT_TESTS:
            result["tests"][name] = _failed(f"Cloud Logging vpc_flows query failed: {query.error}", probes)
        return result

    observed = age_seconds(query.latest_timestamp)
    probes = {
        **probes,
        "observed_delivery_seconds": max(observed, 0),
        "sample_count": query.sample_count,
        "latest_timestamp": query.latest_timestamp,
    }

    result["tests"]["telemetry_endpoint_reachable"] = _passed(
        f"Cloud Logging vpc_flows query succeeded ({query.sample_count} sample entry(ies))", probes
    )

    # Honest pass/fail on the REAL freshest record — mirrors the AWS oracle, which
    # fails (never skips) when no recent datapoint is deliverable.
    if observed < 0 or query.sample_count == 0:
        error = (
            "No vpc_flows record emitted at/after the fixture start "
            f"({probes['fixture_started_at']}) was deliverable to time delivery latency within the "
            f"{effective_poll_timeout}s poll budget (fixture_generated={probes['fixture_generated']}); "
            "a stale pre-fixture flow is not accepted as a correlated delivery sample"
        )
        result["tests"]["delivery_sample_present"] = _failed(error, probes)
        result["tests"]["delivery_within_threshold"] = _failed(
            "Cannot measure delivery latency without a correlated post-fixture record", probes
        )
        result["error"] = error
        return result

    result["tests"]["delivery_sample_present"] = _passed(
        f"{query.sample_count} vpc_flows record(s) emitted at/after the fixture start found", probes
    )
    if observed <= max_delivery_seconds:
        result["tests"]["delivery_within_threshold"] = _passed(
            f"Telemetry delivery latency {observed}s within {max_delivery_seconds}s", probes
        )
    else:
        result["tests"]["delivery_within_threshold"] = _failed(
            f"Telemetry delivery latency {observed}s exceeds {max_delivery_seconds}s", probes
        )

    result["success"] = all(test.get("passed") for test in result["tests"].values())
    if not result["success"]:
        result["error"] = "Telemetry delivery latency checks failed"
    return result


@handle_gcp_errors
def main() -> int:
    """Run the GCP telemetry delivery latency probe and emit structured JSON."""
    parser = argparse.ArgumentParser(description="GCP telemetry delivery latency test")
    parser.add_argument("--region", default="us-central1", help="GCP region of the run subnetworks")
    parser.add_argument("--network-id", default="", help="Compute Engine network name")
    parser.add_argument("--instance-id", default="", help="Launched host instance name (probe scope label)")
    parser.add_argument("--host", default="", help="Launched host address for the traffic fixture (SSH)")
    parser.add_argument("--key-file", default="", help="Local SSH private-key path for the traffic fixture")
    parser.add_argument("--ssh-user", default="", help="Guest SSH user for the traffic fixture")
    parser.add_argument("--max-delivery-seconds", type=int, default=DEFAULT_MAX_DELIVERY_SECONDS)
    parser.add_argument("--poll-timeout-seconds", type=int, default=DEFAULT_FLOW_POLL_TIMEOUT_SECONDS)
    parser.add_argument("--poll-interval-seconds", type=int, default=DEFAULT_FLOW_POLL_INTERVAL_SECONDS)
    parser.add_argument("--project", default=None, help="GCP project ID (ADC fallback)")
    args = parser.parse_args()

    if args.max_delivery_seconds <= 0:
        print(
            json.dumps(
                {
                    "success": False,
                    "platform": "observability",
                    "test_name": TEST_NAME,
                    "error": "--max-delivery-seconds must be greater than 0",
                },
                indent=2,
            )
        )
        return 1

    instance_id = "" if args.instance_id in ("", "none") else args.instance_id
    host = "" if args.host in ("", "none") else args.host
    key_file = "" if args.key_file in ("", "none") else args.key_file
    ssh_user = "" if args.ssh_user in ("", "none") else args.ssh_user
    result = check_telemetry_delivery_latency(
        resolve_project(args.project),
        region=args.region,
        network_id=args.network_id,
        instance_id=short_name(instance_id) if instance_id else "",
        host=host,
        ssh_user=ssh_user,
        key_file=key_file,
        max_delivery_seconds=args.max_delivery_seconds,
        poll_timeout_seconds=args.poll_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

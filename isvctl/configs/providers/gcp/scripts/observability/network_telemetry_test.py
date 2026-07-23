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

"""GCP network-plane telemetry probes (observability test phase).

The GCP port of the AWS oracle ``network_telemetry_test.py``. Each aspect emits
the validator-named subtests derived from REAL Google Cloud signals:

  * ``north_south_network_telemetry`` / ``east_west_network_telemetry`` /
    ``host_nic_network_telemetry`` — CONCRETE, tenant-visible planes. Compute
    Engine VPC Flow Logs are the customer-facing network-telemetry signal, so
    these query the project ``compute.googleapis.com/vpc_flows`` Cloud Logging
    log, scoped to the run-owned subnetworks (and, for host NIC, the launched
    instance's real interface inventory). North-south is bound to flows with an
    external endpoint; east-west to intra-VPC flows (BOTH endpoints carry a
    ``src_vpc``/``dest_vpc`` annotation, so no internet endpoint is involved).
    These planes are tenant-visible, so they are NEVER emitted as provider_hidden.
  * ``management_network_telemetry`` / ``nvswitch_fabric_telemetry`` —
    PROVIDER-HIDDEN physical planes. Compute Engine does not expose the
    provider-owned physical management / NVSwitch fabric plane to the tenant, so
    after a real ``ProjectsClient.get_project`` identity probe these emit
    provider-hidden evidence (every subtest passed=true + provider_hidden=true,
    sample_count=0). Guest / VPC counters are never relabeled as physical-plane
    evidence.

Honest, non-skipping contract for the concrete planes (mirrors the AWS oracle,
which gates ``success`` on EVERY subtest and NEVER emits a success-shaped skip):
because VPC Flow Logs record only ACTUAL flows (unlike CloudWatch's continuous
instance metrics), the probe first generates a bounded, run-scoped traffic
fixture on the launched host — external traffic for north-south / host-NIC,
in-subnet traffic for east-west — then polls the scope-bound vpc_flows query
until the (non-immediate) record is delivered. ``telemetry_endpoint_reachable``
passes on the real query; ``plane_metrics_present`` / ``nic_metrics_present`` on
the byte/packet metric surface READ BACK from the returned flow records (both
``bytes_sent`` and ``packets_sent`` must actually appear, and for host NIC a real
NIC inventory); and ``samples_recent`` on a REAL non-zero sample count. Overall ``success`` is the AND
of all three — a plane that produces no recorded flow honestly FAILS (rc=1), it
is never decoupled from ``samples_recent`` or skipped.

AWS reference implementation:
    ../../aws/scripts/observability/network_telemetry_test.py
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
from common.errors import classify_gcp_error, handle_gcp_errors, retry_idempotent
from common.network import list_subnetworks_for_network
from common.telemetry import (
    DEFAULT_FLOW_POLL_INTERVAL_SECONDS,
    DEFAULT_FLOW_POLL_TIMEOUT_SECONDS,
    VPC_FLOWS_METRIC_FIELDS,
    VPC_FLOWS_TELEMETRY_SOURCE,
    ExternalFlowFixture,
    external_fixture_discriminates_ssh,
    external_flow_tuple_filter,
    generate_external_traffic,
    generate_internal_traffic,
    poll_vpc_flows,
    probe_project_identity,
    subnetwork_scope_filter,
)
from google.cloud import compute_v1

_PLANE_TESTS = ["telemetry_endpoint_reachable", "plane_metrics_present", "samples_recent"]

ASPECT_TESTS: dict[str, list[str]] = {
    "north_south_network_telemetry": _PLANE_TESTS,
    "east_west_network_telemetry": _PLANE_TESTS,
    "management_network_telemetry": _PLANE_TESTS,
    "nvswitch_fabric_telemetry": _PLANE_TESTS,
    "host_nic_network_telemetry": ["telemetry_endpoint_reachable", "nic_metrics_present", "samples_recent"],
}

# Physical planes Compute Engine does not expose to tenants.
HIDDEN_ASPECTS = {"management_network_telemetry", "nvswitch_fabric_telemetry"}

GCP_NO_CUSTOMER_FABRIC_MESSAGE = (
    "Compute Engine does not expose the provider-owned physical management / NVSwitch fabric plane "
    "as a tenant telemetry endpoint (guest and VPC Flow Log counters are not physical-plane evidence)"
)

# Direction clauses on the vpc_flows jsonPayload, keyed on the POSITIVE VPC-
# membership annotation Compute Engine writes onto each flow record:
#   * ``src_vpc``/``dest_vpc`` are populated for every endpoint that sits inside a
#     VPC network; an EXTERNAL (internet) endpoint carries ``src_location`` /
#     ``dest_location`` instead and never a ``*_vpc`` block. So a flow with BOTH a
#     ``src_vpc`` and a ``dest_vpc`` annotation is genuinely intra-VPC (east-west,
#     both endpoints internal), and a flow missing either is north-south (it has
#     an external endpoint). The same ``src_vpc``/``dest_vpc`` scoping is used by
#     ``network/sdn_logging_test.py``.
# East-west is gated on this POSITIVE both-endpoints-internal signal, NOT on the
# absence of ``internet_routing_details``. That field is populated ONLY on
# egress-to-internet flows (GCP VPC Flow Logs record format,
# https://cloud.google.com/vpc/docs/flow-logs), so its absence does NOT prove a
# flow is intra-VPC: an INGRESS internet flow (e.g. the inbound SSH session that
# delivers the traffic fixture, or host_syslogs) also lacks it and would
# self-satisfy an absence-keyed east-west filter — a guaranteed false pass in the
# minimal single-host lifecycle, which has no second run VM. Requiring both a
# ``src_vpc`` and a ``dest_vpc`` excludes every internet flow (ingress and
# egress), so an east-west pass genuinely observes an internal-to-internal flow
# (the run host to its subnet gateway / an in-subnet peer); when no such flow is
# produced the aspect honestly fails on absent samples instead of passing on the
# delivering north-south SSH flow.
_INTRA_VPC = "jsonPayload.src_vpc.vpc_name:* AND jsonPayload.dest_vpc.vpc_name:*"


def _base_result(aspect: str) -> dict[str, Any]:
    """Build the common observability result envelope."""
    return {
        "success": False,
        "platform": "observability",
        "test_name": aspect,
        "tests": {name: {"passed": False} for name in ASPECT_TESTS[aspect]},
    }


def _passed(message: str, probes: dict[str, Any]) -> dict[str, Any]:
    """Build a passing subtest result."""
    return {"passed": True, "message": message, "probes": probes}


def _failed(error: str, probes: dict[str, Any]) -> dict[str, Any]:
    """Build a failing subtest result."""
    return {"passed": False, "error": error, "probes": probes}


def _provider_hidden(test_name: str, *, project: str) -> dict[str, Any]:
    """Build a passing provider-hidden subtest result for a physical plane."""
    return {
        "passed": True,
        "provider_hidden": True,
        "probes": {"sample_count": 0, "telemetry_source": "", "metric_names": []},
        "message": f"{test_name}: {GCP_NO_CUSTOMER_FABRIC_MESSAGE} (project {project} reachable).",
    }


def _east_west_pair_filter(
    instance_name: str,
    primary_private_ip: str,
    peer_instance_name: str,
    peer_private_ip: str,
) -> str:
    """Bind east-west evidence to the exact run-owned ICMP endpoint pair."""
    forward = (
        f'(jsonPayload.connection.src_ip="{primary_private_ip}" '
        f'AND jsonPayload.connection.dest_ip="{peer_private_ip}" '
        f'AND jsonPayload.src_instance.vm_name="{instance_name}" '
        f'AND jsonPayload.dest_instance.vm_name="{peer_instance_name}")'
    )
    reverse = (
        f'(jsonPayload.connection.src_ip="{peer_private_ip}" '
        f'AND jsonPayload.connection.dest_ip="{primary_private_ip}" '
        f'AND jsonPayload.src_instance.vm_name="{peer_instance_name}" '
        f'AND jsonPayload.dest_instance.vm_name="{instance_name}")'
    )
    return f"jsonPayload.connection.protocol=1 AND ({forward} OR {reverse})"


def _direction_filters(
    aspect: str,
    instance_name: str,
    ext_fixture: ExternalFlowFixture | None = None,
    primary_private_ip: str = "",
    peer_instance_name: str = "",
    peer_private_ip: str = "",
) -> list[str]:
    """Return the vpc_flows traffic-direction clauses for a concrete aspect."""
    if aspect == "east_west_network_telemetry":
        return [
            _INTRA_VPC,
            _east_west_pair_filter(
                instance_name,
                primary_private_ip,
                peer_instance_name,
                peer_private_ip,
            ),
        ]
    if aspect == "north_south_network_telemetry":
        # North-south is bound to the EXACT external flow the fixture generated —
        # its protocol + external endpoint IP (+ port) AND the run host in both
        # directions (src_instance/dest_instance vm_name). The earlier vm_name-only
        # binding excluded peer and other-run traffic but NOT the fixture's own
        # inbound tcp/22 SSH control flow, which shares the host vm_name; pinning
        # the external endpoint IP + protocol excludes it because SSH reaches the
        # operator trust IP, not the fixture endpoint. The positive NOT(intra-VPC)
        # membership guard is kept for defense in depth (the fixture endpoint is
        # external, so the flow carries no *_vpc annotation). An empty tuple clause
        # (no fixture tuple or host identity) is refused by the caller, never
        # silently downgraded to the self-satisfiable scope.
        tuple_clause = external_flow_tuple_filter(ext_fixture, instance_name) if ext_fixture else ""
        clauses = []
        if tuple_clause:
            clauses.append(tuple_clause)
        clauses.append(f"NOT ({_INTRA_VPC})")
        return clauses
    if aspect == "host_nic_network_telemetry":
        # Host-NIC joins the launched instance's NIC inventory to the SAME exact
        # external fixture flow (both directions), so the host's own SSH control
        # flow cannot stand in for the generated NIC traffic.
        tuple_clause = external_flow_tuple_filter(ext_fixture, instance_name) if ext_fixture else ""
        return [tuple_clause] if tuple_clause else []
    return []


def _generate_plane_fixture(
    aspect: str,
    host: str,
    ssh_user: str,
    key_file: str,
    peer_private_ip: str = "",
) -> tuple[bool, str, ExternalFlowFixture | None]:
    """Generate the bounded traffic fixture matching the plane's direction.

    Returns ``(ran_ok, detail, ext_fixture)`` — ``ext_fixture`` carries the exact
    external endpoint tuple for north-south / host-NIC (so the caller can bind its
    query to that flow), and is ``None`` for east-west, which is already bound to
    its exact ICMP endpoint pair.
    """
    if aspect == "east_west_network_telemetry":
        ok, detail = generate_internal_traffic(host, ssh_user, key_file, peer_private_ip)
        return ok, detail, None
    # north-south and host-NIC both need a flow with an external peer.
    fixture = generate_external_traffic(host, ssh_user, key_file)
    return fixture.ok, fixture.detail, fixture


def _count_instance_nics(project: str, zone: str, instance_name: str) -> int:
    """Return the number of network interfaces attached to the launched instance."""
    if not (zone and instance_name):
        return 0
    instance = retry_idempotent(
        compute_v1.InstancesClient().get,
        project=project,
        zone=zone,
        instance=instance_name,
        op_desc=f"compute get instance {instance_name}",
    )
    return len(list(getattr(instance, "network_interfaces", []) or []))


def check_plane_telemetry(
    project: str,
    *,
    aspect: str,
    region: str,
    network_id: str,
    instance_name: str = "",
    zone: str = "",
    host: str = "",
    ssh_user: str = "",
    key_file: str = "",
    primary_private_ip: str = "",
    peer_instance_name: str = "",
    peer_private_ip: str = "",
    poll_timeout_seconds: int = DEFAULT_FLOW_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: int = DEFAULT_FLOW_POLL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Validate tenant-visible VPC Flow Log telemetry for a concrete network plane."""
    result = _base_result(aspect)
    host_nic = aspect == "host_nic_network_telemetry"
    metrics_present_test = ASPECT_TESTS[aspect][1]
    probes: dict[str, Any] = {
        "telemetry_source": VPC_FLOWS_TELEMETRY_SOURCE,
        # Populated from the flow-record readback below, never a preloaded literal:
        # a subtest reports only the byte/packet fields the returned entries carried.
        "metric_names": [],
        "sample_count": 0,
        "latest_timestamp": "",
        "fixture_generated": False,
        "fixture_started_at": "",
        "probe_resource_id": instance_name or network_id,
    }
    if aspect == "east_west_network_telemetry":
        probes.update(
            {
                "primary_instance_id": instance_name,
                "primary_private_ip": primary_private_ip,
                "peer_instance_id": peer_instance_name,
                "peer_private_ip": peer_private_ip,
            }
        )
        if (
            not all(
                (
                    instance_name,
                    primary_private_ip,
                    peer_instance_name,
                    peer_private_ip,
                )
            )
            or instance_name == peer_instance_name
            or primary_private_ip == peer_private_ip
        ):
            error = "east-west telemetry requires two distinct run-owned endpoint identities and private IPs"
            for name in ASPECT_TESTS[aspect]:
                result["tests"][name] = _failed(error, probes)
            result["error"] = error
            return result
    if host_nic:
        probes["nics_checked"] = 0

    try:
        subnets = list_subnetworks_for_network(project, region, network_id)
        if host_nic:
            probes["nics_checked"] = _count_instance_nics(project, zone, instance_name)
    except Exception as e:
        error_type, error_msg = classify_gcp_error(e)
        result["error_type"] = error_type
        result["error"] = error_msg
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(error_msg, probes)
        return result

    scope = subnetwork_scope_filter(subnets)
    if not scope:
        error = f"no subnetworks resolved for {network_id}; refusing an unscoped vpc_flows query"
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(error, probes)
        result["error"] = error
        return result

    # Capture the fixture-start marker BEFORE generating traffic so the readback is
    # correlated to THIS run's own flow. Every launch, cloud-init, and pre-fixture
    # SSH management flow (e.g. host_syslogs) that already populated the subnet's
    # vpc_flows predates this marker; binding the poll's lower timestamp bound to it
    # stops a stale pre-fixture but scope-bound record from passing
    # samples_recent/success when the fixture itself failed to run.
    fixture_start = datetime.now(UTC)
    probes["fixture_started_at"] = fixture_start.isoformat()

    # Generate the bounded run-scoped traffic fixture matching the plane so a real
    # flow is produced for the pipeline to deliver. Fixture failure is terminal:
    # querying after a failed probe could accept a different flow for the same pair.
    fixture_ok, fixture_detail, ext_fixture = _generate_plane_fixture(aspect, host, ssh_user, key_file, peer_private_ip)
    probes["fixture_generated"] = fixture_ok
    if not fixture_ok:
        probes["fixture_detail"] = fixture_detail
        error = f"Failed to generate the exact {aspect} traffic fixture: {fixture_detail}"
        result["error"] = error
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(error, probes)
        return result

    # For the external planes (north-south / host-NIC), record the exact fixture
    # tuple and refuse to query unless the tuple-bound clause discriminates the
    # fixture flow from a post-marker SSH-only control flow (the finding's
    # self-satisfaction world). East-west is already bound to its exact ICMP
    # endpoint pair, so it needs no external-tuple guard.
    if ext_fixture is not None:
        probes["fixture_protocol"] = ext_fixture.protocol
        probes["fixture_dest_ip"] = ext_fixture.dest_ip
        probes["fixture_dest_port"] = ext_fixture.dest_port
        if not external_flow_tuple_filter(ext_fixture, instance_name):
            error = (
                f"cannot bind {aspect} telemetry to the exact external fixture flow "
                f"(missing host identity or fixture tuple); refusing an under-scoped query "
                f"its own SSH control flow could satisfy"
            )
            result["error"] = error
            for name in ASPECT_TESTS[aspect]:
                result["tests"][name] = _failed(error, probes)
            return result
        discriminates, reason = external_fixture_discriminates_ssh(ext_fixture, instance_name)
        if not discriminates:
            error = f"{aspect} external-flow filter failed its SSH-only negative-world check: {reason}"
            result["error"] = error
            for name in ASPECT_TESTS[aspect]:
                result["tests"][name] = _failed(error, probes)
            return result

    extra_filters = [
        scope,
        *_direction_filters(
            aspect,
            instance_name,
            ext_fixture=ext_fixture,
            primary_private_ip=primary_private_ip,
            peer_instance_name=peer_instance_name,
            peer_private_ip=peer_private_ip,
        ),
    ]
    # start_time binds every poll iteration's lower bound to the fixture marker, so
    # samples_recent/success can only ever pass on a correlated (post-fixture)
    # record; a stale pre-fixture flow inside the 24h lookback can no longer satisfy
    # the probe when the fixture failed to run (mirrors the sibling
    # telemetry_delivery_test / storage_telemetry_test fixture-correlation).
    query = poll_vpc_flows(
        project,
        region=region,
        extra_filters=extra_filters,
        poll_timeout_seconds=poll_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        start_time=fixture_start,
    )
    if not query.ok:
        result["error_type"] = query.error_type
        result["error"] = query.error
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(f"Cloud Logging vpc_flows query failed: {query.error}", probes)
        return result

    # Read the observed metric-field names back from the returned flow records
    # (never a preloaded literal), so the probe emits only fields the query
    # actually surfaced.
    observed_fields = query.metric_fields
    probes = {
        **probes,
        "sample_count": query.sample_count,
        "latest_timestamp": query.latest_timestamp,
        "metric_names": observed_fields,
    }
    result["tests"]["telemetry_endpoint_reachable"] = _passed(
        f"Cloud Logging vpc_flows query succeeded ({query.sample_count} sample entry(ies))", probes
    )

    # Gate the metric-surface subtest on the readback proving BOTH the required
    # byte and packet fields (mirrors the AWS oracle gating plane_metrics_present
    # on the descriptors returned for the requested metric names). A scoped entry
    # that lacks these fields can no longer report metric availability the probe
    # never observed.
    missing_fields = [f for f in VPC_FLOWS_METRIC_FIELDS if f not in observed_fields]
    if host_nic and not probes["nics_checked"]:
        result["tests"][metrics_present_test] = _failed("No network interfaces resolved for the launched host", probes)
    elif missing_fields:
        result["tests"][metrics_present_test] = _failed(
            f"VPC Flow Log records did not expose the required packet/byte field(s) "
            f"{missing_fields} (observed {observed_fields or 'none'})",
            probes,
        )
    else:
        result["tests"][metrics_present_test] = _passed(
            f"VPC Flow Log packet/byte telemetry {observed_fields} is available for the {aspect} plane", probes
        )

    if query.sample_count > 0:
        result["tests"]["samples_recent"] = _passed(f"{query.sample_count} recent VPC Flow Log sample(s) found", probes)
    else:
        result["tests"]["samples_recent"] = _failed(
            f"No recent {aspect} VPC Flow Log samples were delivered within the {poll_timeout_seconds}s poll budget "
            f"(fixture_generated={probes['fixture_generated']})",
            probes,
        )

    # Honest gating mirroring the AWS oracle: overall success is the AND of ALL
    # subtests, samples_recent included. A plane that produces no recorded flow
    # fails; success is never decoupled from the observed sample.
    result["success"] = all(test.get("passed") for test in result["tests"].values())
    if not result["success"]:
        result["error"] = f"{aspect} telemetry checks failed"
    return result


def check_hidden_plane_telemetry(project: str, *, aspect: str) -> dict[str, Any]:
    """Emit provider-hidden evidence after a real project identity probe."""
    result = _base_result(aspect)
    try:
        project_id = probe_project_identity(project)
    except Exception as e:
        error_type, error_msg = classify_gcp_error(e)
        result["error_type"] = error_type
        result["error"] = error_msg
        probes = {"sample_count": 0, "telemetry_source": "", "metric_names": []}
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(f"GCP project identity probe failed: {error_msg}", probes)
        return result

    result["tests"] = {name: _provider_hidden(name, project=project_id) for name in ASPECT_TESTS[aspect]}
    result["success"] = True
    return result


@handle_gcp_errors
def main() -> int:
    """Run the selected GCP network telemetry probe and emit structured JSON."""
    parser = argparse.ArgumentParser(description="GCP network telemetry test")
    parser.add_argument("--region", default="us-central1", help="GCP region of the run subnetworks")
    parser.add_argument("--network-id", default="", help="Compute Engine network name")
    parser.add_argument("--instance-id", default="", help="Launched host instance name (host_nic scope)")
    parser.add_argument("--zone", default="", help="Launched host zone (host_nic NIC inventory)")
    parser.add_argument("--host", default="", help="Launched host address for the traffic fixture (SSH)")
    parser.add_argument("--key-file", default="", help="Local SSH private-key path for the traffic fixture")
    parser.add_argument("--ssh-user", default="", help="Guest SSH user for the traffic fixture")
    parser.add_argument("--private-ip", default="", help="Launched host internal IPv4")
    parser.add_argument("--peer-instance-id", default="", help="Run-owned internal peer instance name")
    parser.add_argument("--peer-private-ip", default="", help="Run-owned internal peer IPv4")
    parser.add_argument("--aspect", required=True, choices=sorted(ASPECT_TESTS))
    parser.add_argument("--poll-timeout-seconds", type=int, default=DEFAULT_FLOW_POLL_TIMEOUT_SECONDS)
    parser.add_argument("--poll-interval-seconds", type=int, default=DEFAULT_FLOW_POLL_INTERVAL_SECONDS)
    parser.add_argument("--project", default=None, help="GCP project ID (ADC fallback)")
    args = parser.parse_args()

    project = resolve_project(args.project)
    if args.aspect in HIDDEN_ASPECTS:
        result = check_hidden_plane_telemetry(project, aspect=args.aspect)
    else:
        instance_name = "" if args.instance_id in ("", "none") else short_name(args.instance_id)
        zone = "" if args.zone in ("", "none") else short_name(args.zone)
        host = "" if args.host in ("", "none") else args.host
        key_file = "" if args.key_file in ("", "none") else args.key_file
        ssh_user = "" if args.ssh_user in ("", "none") else args.ssh_user
        private_ip = "" if args.private_ip in ("", "none") else args.private_ip
        peer_instance_name = "" if args.peer_instance_id in ("", "none") else short_name(args.peer_instance_id)
        peer_private_ip = "" if args.peer_private_ip in ("", "none") else args.peer_private_ip
        result = check_plane_telemetry(
            project,
            aspect=args.aspect,
            region=args.region,
            network_id=args.network_id,
            instance_name=instance_name,
            zone=zone,
            host=host,
            ssh_user=ssh_user,
            key_file=key_file,
            primary_private_ip=private_ip,
            peer_instance_name=peer_instance_name,
            peer_private_ip=peer_private_ip,
            poll_timeout_seconds=args.poll_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

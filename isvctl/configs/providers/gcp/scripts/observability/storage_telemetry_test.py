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

"""GCP storage telemetry probes (observability test phase).

The GCP port of the AWS oracle ``storage_telemetry_test.py``. Two aspects, each a
tenant-visible plane (never provider-hidden), each honestly gated on real
evidence — the AWS oracle NEVER emits a success-shaped skip, and neither do these:

  * ``storage_performance_telemetry`` — bandwidth / IOPS / latency from real Cloud
    Monitoring persistent-disk time series. The probe first drives a bounded,
    run-scoped write+read I-O fixture on the launched host (VPC/PD metrics record
    only ACTUAL I-O), then polls the instance-scoped Cloud Monitoring series:

        compute.googleapis.com/instance/disk/read_bytes_count   -> bandwidth
        compute.googleapis.com/instance/disk/write_bytes_count  -> bandwidth
        compute.googleapis.com/instance/disk/read_ops_count     -> iops
        compute.googleapis.com/instance/disk/write_ops_count    -> iops
        compute.googleapis.com/instance/disk/average_io_latency -> latency

    The query is scope-bound to the run-owned instance's numeric ``instance_id``
    (``resource.type=gce_instance``) so no unrelated host's metrics stand in as
    evidence, AND time-bound to a fixture-start marker captured before the I-O:
    only points whose interval end is at/after that marker are accepted, so boot
    or pre-fixture disk activity already inside the lookback window can never
    satisfy the check. ``performance_metrics_present`` requires all of
    bandwidth/IOPS/latency; ``samples_recent`` a real non-zero sample; a missing
    kind or sample is an honest ``_failed`` (rc=1).

  * ``storage_capacity_telemetry`` — used / free / total from REAL guest
    filesystem evidence read over SSH (``df``) for the attached persistent disks,
    joined to the Compute Engine disk enumeration by each disk's
    ``/dev/disk/by-id/google-<device_name>`` guest identity so only mounts backed
    by an API-enumerated disk contribute. Provisioned disk size is never
    substituted for observed capacity, unrelated guest mounts never contribute,
    and when no enumerated disk can be joined to guest evidence the capacity
    subtests honestly ``_failed`` rather than skip.

Cloud Monitoring is queried through ``google-api-python-client``
(``googleapiclient``) — the official Google client for GCP's own Monitoring REST
API; ``google-cloud-monitoring`` is not a workspace dependency, and googleapiclient
accesses only the GCP Monitoring API (the target NCP is reached only through its
own API surface). Genuine API / authentication / malformed-input failures still
return rc=1.

AWS reference implementation:
    ../../aws/scripts/observability/storage_telemetry_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

import google.auth
from common.compute import resolve_project, short_name
from common.errors import classify_gcp_error, handle_gcp_errors, retry_idempotent
from common.telemetry import generate_disk_io, read_guest_capacity
from google.cloud import compute_v1
from googleapiclient.discovery import build

ASPECT_TESTS: dict[str, list[str]] = {
    "storage_capacity_telemetry": [
        "telemetry_endpoint_reachable",
        "capacity_metrics_present",
        "samples_recent",
    ],
    "storage_performance_telemetry": [
        "telemetry_endpoint_reachable",
        "performance_metrics_present",
        "samples_recent",
    ],
}

# Provider-neutral kinds each aspect's validator expects.
REQUIRED_KINDS: dict[str, list[str]] = {
    "storage_capacity_telemetry": ["used", "free", "total"],
    "storage_performance_telemetry": ["bandwidth", "iops", "latency"],
}

METRICS_PRESENT_TEST: dict[str, str] = {
    "storage_capacity_telemetry": "capacity_metrics_present",
    "storage_performance_telemetry": "performance_metrics_present",
}

KIND_PROBE_FIELD: dict[str, str] = {
    "storage_capacity_telemetry": "capacity_kinds",
    "storage_performance_telemetry": "performance_kinds",
}

# Real, tenant-visible GCE hypervisor persistent-disk metrics (no guest agent
# required) mapped to the provider-neutral performance kinds the validator
# expects. Verified against the project's live metricDescriptors catalog.
PERFORMANCE_METRIC_KINDS: dict[str, str] = {
    "read_bytes_count": "bandwidth",
    "write_bytes_count": "bandwidth",
    "read_ops_count": "iops",
    "write_ops_count": "iops",
    "average_io_latency": "latency",
}
GCE_DISK_METRIC_PREFIX = "compute.googleapis.com/instance/disk/"
PERFORMANCE_TELEMETRY_SOURCE = "cloud_monitoring:compute.googleapis.com/instance/disk"
CAPACITY_TELEMETRY_SOURCE = "guest_filesystem:df"

# The generated I-O fixture lands within the first minute; GCE hypervisor DELTA
# metrics carry a few minutes of ingestion latency, so a wide lookback reliably
# spans it while the query window end stays ``now``. The wide window is only the
# API query span — a fixture-start marker still gates which points are ACCEPTED,
# so pre-fixture samples in the window are rejected rather than treated as recent.
# Poll mirrors the AWS oracle: wait for real samples to become queryable.
DEFAULT_PERF_LOOKBACK_SECONDS = 1800
DEFAULT_POLL_TIMEOUT_SECONDS = 300
DEFAULT_POLL_INTERVAL_SECONDS = 20


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


def _get_instance(project: str, zone: str, instance_name: str) -> compute_v1.Instance:
    """Read the launched instance (numeric id + attached disks) under idempotent retry."""
    return retry_idempotent(
        compute_v1.InstancesClient().get,
        project=project,
        zone=zone,
        instance=instance_name,
        op_desc=f"compute get instance {instance_name}",
    )


def _instance_volume_ids(instance: compute_v1.Instance) -> list[str]:
    """Return the persistent-disk names attached to the launched instance."""
    volume_ids: list[str] = []
    for disk in getattr(instance, "disks", []) or []:
        source = getattr(disk, "source", "") or ""
        name = short_name(source) if source else (getattr(disk, "device_name", "") or "")
        if name:
            volume_ids.append(name)
    return volume_ids


def _instance_disk_device_names(instance: compute_v1.Instance) -> list[str]:
    """Return the guest ``device_name`` of each attached persistent disk.

    The ``device_name`` is the join key between the Compute Engine disk enumeration
    and the guest: GCE materialises each attached disk at
    ``/dev/disk/by-id/google-<device_name>``. Capacity evidence is aggregated only
    over the mounted filesystems backed by these enumerated devices, so unrelated
    guest block devices cannot stand in for a persistent disk.
    """
    device_names: list[str] = []
    for disk in getattr(instance, "disks", []) or []:
        device_name = getattr(disk, "device_name", "") or ""
        if device_name and device_name not in device_names:
            device_names.append(device_name)
    return device_names


def _monitoring_service() -> Any:
    """Build the Cloud Monitoring v3 client from Application Default Credentials."""
    creds, _ = google.auth.default()
    return build("monitoring", "v3", credentials=creds, cache_discovery=False)


def _parse_point_timestamp(value: str) -> datetime | None:
    """Parse a Cloud Monitoring point endTime (RFC3339) to an aware datetime, or None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _query_disk_performance(
    service: Any,
    project: str,
    instance_id: str,
    *,
    lookback_seconds: int,
    not_before: datetime | None = None,
) -> tuple[list[str], list[str], int, str]:
    """Query GCE persistent-disk performance time series for the run-owned instance.

    Returns ``(metric_names, performance_kinds, sample_count, latest_timestamp)``
    where ``metric_names`` are the GCE disk metrics that actually reported points
    (real evidence, never a static list). Scope-binding: every series is filtered
    to ``resource.type=gce_instance`` and the exact numeric ``instance_id`` so no
    unrelated host's disk telemetry can stand in as evidence.

    ``not_before`` is the fixture-start marker: the query still spans a wide
    ingestion-latency lookback window, but a point is only ACCEPTED when its
    interval end is at or after this marker. That way boot or pre-fixture disk
    activity already sitting in the lookback window cannot satisfy the check —
    only telemetry whose interval could contain the bounded run-generated I-O is
    counted, so a pass genuinely observes the operation the aspect claims to
    measure. A point whose interval end is missing or unparseable is rejected
    (it cannot be placed at/after the marker).
    """
    now = datetime.now(UTC)
    start = now - timedelta(seconds=max(lookback_seconds, 60))
    start_iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    metric_names: list[str] = []
    sample_count = 0
    newest: datetime | None = None

    for metric_suffix in PERFORMANCE_METRIC_KINDS:
        metric_type = f"{GCE_DISK_METRIC_PREFIX}{metric_suffix}"
        flt = (
            f'metric.type="{metric_type}" '
            f'AND resource.type="gce_instance" '
            f'AND resource.labels.instance_id="{instance_id}"'
        )
        response = retry_idempotent(
            lambda flt=flt: (
                service.projects()
                .timeSeries()
                .list(
                    name=f"projects/{project}",
                    filter=flt,
                    interval_startTime=start_iso,
                    interval_endTime=end_iso,
                    view="FULL",
                )
                .execute()
            ),
            op_desc=f"monitoring timeSeries.list {metric_suffix}",
        )
        points_for_metric = 0
        for series in response.get("timeSeries", []) or []:
            for point in series.get("points", []) or []:
                ts = _parse_point_timestamp(point.get("interval", {}).get("endTime", ""))
                if ts is None:
                    continue
                if not_before is not None and ts < not_before:
                    # Pre-fixture sample already in the lookback window; not
                    # attributable to the run-generated I-O, so reject it.
                    continue
                points_for_metric += 1
                if newest is None or ts > newest:
                    newest = ts
        if points_for_metric > 0:
            metric_names.append(metric_suffix)
            sample_count += points_for_metric

    kinds: list[str] = []
    for metric_suffix in metric_names:
        kind = PERFORMANCE_METRIC_KINDS[metric_suffix]
        if kind not in kinds:
            kinds.append(kind)

    latest = newest.isoformat() if newest is not None else ""
    return sorted(metric_names), kinds, sample_count, latest


def check_storage_performance_telemetry(
    project: str,
    *,
    zone: str,
    instance_name: str,
    host: str = "",
    ssh_user: str = "",
    key_file: str = "",
    lookback_seconds: int = DEFAULT_PERF_LOOKBACK_SECONDS,
    poll_timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Validate customer-visible GCE persistent-disk performance telemetry."""
    aspect = "storage_performance_telemetry"
    result = _base_result(aspect)
    probes: dict[str, Any] = {
        "telemetry_source": PERFORMANCE_TELEMETRY_SOURCE,
        "metric_names": [],
        "performance_kinds": [],
        "volumes_checked": 0,
        "sample_count": 0,
        "latest_timestamp": "",
        "fixture_generated": False,
        "probe_resource_id": instance_name,
    }

    if not (zone and instance_name):
        error = "no launched instance/zone forwarded to scope the storage probe"
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(error, probes)
        result["error"] = error
        return result

    try:
        instance = _get_instance(project, zone, instance_name)
        volume_ids = _instance_volume_ids(instance)
        instance_id = str(getattr(instance, "id", "") or "")
    except Exception as e:
        error_type, error_msg = classify_gcp_error(e)
        result["error_type"] = error_type
        result["error"] = error_msg
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(error_msg, probes)
        return result

    probes = {**probes, "volumes_checked": len(volume_ids), "probe_resource_id": instance_id or instance_name}

    if not volume_ids or not instance_id:
        error = f"No persistent disks / numeric id resolved for instance {instance_name}"
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(error, probes)
        result["error"] = error
        return result

    # Drive a bounded run-scoped I-O fixture so the hypervisor disk metrics report
    # real bandwidth/IOPS/latency. The fixture is a REQUIRED precondition, not a
    # diagnostic: this aspect claims to observe telemetry produced by the bounded
    # I-O it generates, and the 1,800-second lookback can already hold boot or
    # earlier disk activity, so passing without a proven write+read would attribute
    # unrelated samples to run-generated I-O. Fail closed when the fixture did not
    # complete rather than accepting unattributable metrics.
    #
    # Capture the marker BEFORE the fixture runs: every subsequent query accepts
    # only points whose interval end is at/after this instant, so pre-fixture
    # samples in the lookback window can never satisfy the check and a pass
    # genuinely observes the run-generated I-O.
    fixture_start = datetime.now(UTC)
    probes["fixture_start"] = fixture_start.isoformat()
    fixture_ok, fixture_detail = generate_disk_io(host, ssh_user, key_file)
    probes["fixture_generated"] = fixture_ok
    if not fixture_ok:
        probes["fixture_detail"] = fixture_detail
        error = (
            "Bounded storage I-O fixture did not complete its write+read, so persistent-disk "
            f"performance telemetry cannot be attributed to run-generated I-O: {fixture_detail}"
        )
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(error, probes)
        result["error"] = error
        return result

    try:
        service = _monitoring_service()
        metric_names, kinds, sample_count, latest = _query_disk_performance(
            service, project, instance_id, lookback_seconds=lookback_seconds, not_before=fixture_start
        )
    except Exception as e:
        error_type, error_msg = classify_gcp_error(e)
        result["error_type"] = error_type
        result["error"] = error_msg
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(error_msg, probes)
        return result

    probes = {
        **probes,
        "metric_names": metric_names,
        "performance_kinds": kinds,
        "sample_count": sample_count,
        "latest_timestamp": latest,
    }
    result["tests"]["telemetry_endpoint_reachable"] = _passed(
        f"Cloud Monitoring persistent-disk telemetry reachable "
        f"({len(volume_ids)} volume(s), {len(metric_names)} metric(s) reporting)",
        probes,
    )

    # Poll until ALL required kinds are queryable (fresh I-O needs a few minutes of
    # ingestion latency), mirroring the AWS oracle polling CloudWatch. The DELTA
    # count metrics (bandwidth/iops) ingest first; the GAUGE average_io_latency
    # lags, so poll on the full required-kinds set, not sample_count>0 alone.
    required_kinds = set(REQUIRED_KINDS[aspect])
    deadline = time.monotonic() + max(poll_timeout_seconds, 0)
    while not required_kinds.issubset(set(kinds)) and time.monotonic() < deadline:
        sleep(poll_interval_seconds)
        try:
            metric_names, kinds, sample_count, latest = _query_disk_performance(
                service, project, instance_id, lookback_seconds=lookback_seconds, not_before=fixture_start
            )
        except Exception as e:
            # _query_disk_performance has ALREADY exhausted the bounded idempotent
            # retry budget for its transient class, so any exception escaping here
            # is a genuine operational failure (auth / exhausted-transient /
            # implementation) — fail closed with a classified error (rc=1), never
            # swallow it into a false pass.
            error_type, error_msg = classify_gcp_error(e)
            result["error_type"] = error_type
            result["error"] = error_msg
            probes = {
                **probes,
                "metric_names": metric_names,
                "performance_kinds": kinds,
                "sample_count": sample_count,
                "latest_timestamp": latest,
            }
            for name in (METRICS_PRESENT_TEST[aspect], "samples_recent"):
                result["tests"][name] = _failed(error_msg, probes)
            return result

    probes = {
        **probes,
        "metric_names": metric_names,
        "performance_kinds": kinds,
        "sample_count": sample_count,
        "latest_timestamp": latest,
    }

    if required_kinds.issubset(set(kinds)):
        result["tests"]["performance_metrics_present"] = _passed(
            "GCE persistent-disk performance telemetry present for bandwidth, IOPS, and latency", probes
        )
    else:
        missing = sorted(required_kinds - set(kinds))
        result["tests"]["performance_metrics_present"] = _failed(
            f"Missing observed performance metric kinds: {', '.join(missing)} "
            f"(fixture_generated={probes['fixture_generated']})",
            probes,
        )

    if sample_count > 0:
        result["tests"]["samples_recent"] = _passed(
            f"{sample_count} recent persistent-disk telemetry sample(s) found", probes
        )
    else:
        result["tests"]["samples_recent"] = _failed(
            f"No recent persistent-disk telemetry samples became queryable within the {poll_timeout_seconds}s "
            f"poll budget (fixture_generated={probes['fixture_generated']})",
            probes,
        )

    # Honest gating mirroring the AWS oracle: success is the AND of every subtest.
    result["success"] = all(test.get("passed") for test in result["tests"].values())
    if not result["success"]:
        result["error"] = "storage performance telemetry checks failed"
    return result


def check_storage_capacity_telemetry(
    project: str,
    *,
    zone: str,
    instance_name: str,
    host: str = "",
    ssh_user: str = "",
    key_file: str = "",
) -> dict[str, Any]:
    """Validate customer-visible storage capacity (used/free/total) telemetry.

    The Compute Engine disk enumeration proves the storage surface is reachable;
    the used/free/total values come from REAL guest filesystem evidence read over
    SSH (``df``) for the attached persistent disks. Provisioned disk size is NEVER
    substituted for observed capacity — absent guest evidence is an honest failure.
    """
    aspect = "storage_capacity_telemetry"
    result = _base_result(aspect)
    kind_field = KIND_PROBE_FIELD[aspect]
    probes: dict[str, Any] = {
        "telemetry_source": CAPACITY_TELEMETRY_SOURCE,
        "metric_names": [],
        kind_field: [],
        "volumes_checked": 0,
        "sample_count": 0,
        "latest_timestamp": "",
        "probe_resource_id": instance_name,
    }

    if not (zone and instance_name):
        error = "no launched instance/zone forwarded to scope the storage probe"
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(error, probes)
        result["error"] = error
        return result

    try:
        instance = _get_instance(project, zone, instance_name)
        volume_ids = _instance_volume_ids(instance)
        device_names = _instance_disk_device_names(instance)
    except Exception as e:
        error_type, error_msg = classify_gcp_error(e)
        result["error_type"] = error_type
        result["error"] = error_msg
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(error_msg, probes)
        return result

    probes = {**probes, "volumes_checked": len(volume_ids), "enumerated_disks": device_names}

    if not volume_ids:
        error = f"No persistent disks attached to instance {instance_name}"
        for name in ASPECT_TESTS[aspect]:
            result["tests"][name] = _failed(error, probes)
        result["error"] = error
        return result

    result["tests"]["telemetry_endpoint_reachable"] = _passed(
        f"Compute Engine disk telemetry surface reachable ({len(volume_ids)} volume(s) enumerated)", probes
    )

    # Read REAL used/free/total from guest filesystem evidence, JOINED to the
    # enumerated persistent disks by their ``/dev/disk/by-id/google-<device_name>``
    # guest identity so only mounts backed by an API-enumerated disk contribute.
    # A missing reading — or capacity that cannot be joined to any enumerated disk —
    # is an honest failure, never a provisioned-size substitution, a skip, or a
    # total that describes a different device set.
    ok, total, used, free, mounts, joined = read_guest_capacity(host, ssh_user, key_file, device_names)
    if ok:
        probes = {
            **probes,
            "metric_names": ["used", "free", "total"],
            kind_field: REQUIRED_KINDS[aspect],
            "used_bytes": used,
            "free_bytes": free,
            "total_bytes": total,
            "mounts_checked": mounts,
            "joined_disks": joined,
            "sample_count": mounts,
            "latest_timestamp": datetime.now(UTC).isoformat(),
        }
        result["tests"]["capacity_metrics_present"] = _passed(
            f"Guest filesystem capacity telemetry present for used/free/total across {mounts} mount(s) "
            f"joined to enumerated disk(s) {', '.join(joined)}",
            probes,
        )
        result["tests"]["samples_recent"] = _passed(
            f"Guest used/free/total capacity read for {mounts} mount(s) on enumerated disk(s) "
            f"{', '.join(joined)} (total={total}B, used={used}B, free={free}B)",
            probes,
        )
    else:
        error = (
            "No guest filesystem used/free/total evidence could be joined to an enumerated persistent disk "
            f"({', '.join(device_names) or 'no device_name resolved'}) over SSH; unrelated guest mounts and "
            "provisioned disk size are not observed capacity telemetry for the enumerated disks"
        )
        result["tests"]["capacity_metrics_present"] = _failed(error, probes)
        result["tests"]["samples_recent"] = _failed(error, probes)

    # Honest gating mirroring the AWS oracle: success is the AND of every subtest.
    result["success"] = all(test.get("passed") for test in result["tests"].values())
    if not result["success"]:
        result["error"] = "storage capacity telemetry checks failed"
    return result


@handle_gcp_errors
def main() -> int:
    """Run the selected GCP storage telemetry probe and emit structured JSON."""
    parser = argparse.ArgumentParser(description="GCP storage telemetry test")
    parser.add_argument("--region", default="us-central1", help="GCP region (contextual)")
    parser.add_argument("--instance-id", default="", help="Launched host instance name")
    parser.add_argument("--zone", default="", help="Launched host zone")
    parser.add_argument("--host", default="", help="Launched host address for the I-O / df fixture (SSH)")
    parser.add_argument("--key-file", default="", help="Local SSH private-key path for the fixture")
    parser.add_argument("--ssh-user", default="", help="Guest SSH user for the fixture")
    parser.add_argument("--aspect", required=True, choices=sorted(ASPECT_TESTS))
    parser.add_argument("--project", default=None, help="GCP project ID (ADC fallback)")
    parser.add_argument(
        "--poll-timeout-seconds",
        type=int,
        default=DEFAULT_POLL_TIMEOUT_SECONDS,
        help="Seconds to wait for Cloud Monitoring samples to appear before giving up",
    )
    parser.add_argument("--poll-interval-seconds", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS)
    args = parser.parse_args()

    instance_name = "" if args.instance_id in ("", "none") else short_name(args.instance_id)
    zone = "" if args.zone in ("", "none") else short_name(args.zone)
    host = "" if args.host in ("", "none") else args.host
    key_file = "" if args.key_file in ("", "none") else args.key_file
    ssh_user = "" if args.ssh_user in ("", "none") else args.ssh_user
    project = resolve_project(args.project)

    if args.aspect == "storage_performance_telemetry":
        result = check_storage_performance_telemetry(
            project,
            zone=zone,
            instance_name=instance_name,
            host=host,
            ssh_user=ssh_user,
            key_file=key_file,
            poll_timeout_seconds=args.poll_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    else:
        result = check_storage_capacity_telemetry(
            project,
            zone=zone,
            instance_name=instance_name,
            host=host,
            ssh_user=ssh_user,
            key_file=key_file,
        )

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

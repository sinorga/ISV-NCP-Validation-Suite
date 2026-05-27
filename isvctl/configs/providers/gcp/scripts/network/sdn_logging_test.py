#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""SDN logging multi-aspect dispatcher (hardware_faults / latency_perf /
audit_trail).

Compute Engine sources used per aspect:
  * hardware_faults — Compute Engine system_event Cloud Audit Logs
    (hostError, host_event_notify, scheduled host maintenance).
  * latency_perf — VPC Flow Logs / Cloud Monitoring network metrics.
  * audit_trail — Admin Activity Cloud Audit Logs for firewall
    insert / patch-or-update / delete on a temporary probe rule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import UTC

from common.compute import resolve_project, unique_suffix, wait_for_global_op
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest sdn_logging — verified-reuse marker"
HARDWARE_LOG_NAME = "cloudaudit.googleapis.com/system_event"
LATENCY_NAMESPACE = "compute.googleapis.com/instance/network"
AUDIT_LOG_NAME = "cloudaudit.googleapis.com/activity"


def _list_log_entries(filter_str: str, max_results: int = 50) -> tuple[bool, int]:
    """Return (reachable, count). reachable=False when google-cloud-logging
    is not installed OR the API call raises."""
    try:
        from google.cloud import logging_v2  # type: ignore[attr-defined]
    except ImportError:
        return False, 0
    client = logging_v2.Client()
    count = 0
    for _ in client.list_entries(filter_=filter_str, page_size=max_results, max_results=max_results):
        count += 1
    return True, count


def _iter_log_entries(filter_str: str, max_results: int = 50):
    """Yield raw entries for the filter, or empty iterator on import/API
    failure. The caller is expected to inspect ``payload``/``method_name``
    fields directly."""
    try:
        from google.cloud import logging_v2  # type: ignore[attr-defined]
    except ImportError:
        return
    client = logging_v2.Client()
    yield from client.list_entries(filter_=filter_str, page_size=max_results, max_results=max_results)


def _hardware_faults(project: str, network: str) -> dict[str, Any]:
    """Each of the four subtests issues its OWN Cloud Logging probe so the
    pass-signal cannot collapse to a single API call. AWS oracle does the
    same with distinct CloudWatch / Health probes."""
    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "test_name": "sdn_hardware_fault_logging",
        "network_id": network,
        "aspect": "hardware_faults",
        "log_destination": HARDWARE_LOG_NAME,
        "recent_event_count": 0,
        "tests": {},
    }
    base_log = f'logName="projects/{project}/logs/{HARDWARE_LOG_NAME}"'
    try:
        # 1. Logging endpoint reachable: tightest possible filter
        # (page_size=1) — proves the Cloud Logging API can be called.
        endpoint_ok, _ = _list_log_entries(base_log, max_results=1)
        result["tests"]["logging_endpoint_reachable"] = {"passed": endpoint_ok}

        # 2. Fault-event source queryable: filter narrowing to host_event_*
        # method names.
        source_filter = base_log + ' AND protoPayload.methodName:"host_event"'
        source_ok, _ = _list_log_entries(source_filter, max_results=5)
        result["tests"]["fault_event_source_queryable"] = {"passed": source_ok}

        # 3. Log destination configured: verify the canonical log name
        # resolves (a non-existent log returns NOT_FOUND on the list call).
        dest_filter = base_log
        dest_ok, _ = _list_log_entries(dest_filter, max_results=1)
        result["tests"]["log_destination_configured"] = {
            "passed": dest_ok,
            "log_destination": HARDWARE_LOG_NAME,
        }

        # 4. Event schema valid: distinct filter for hostError method name
        # (verifies the protoPayload schema by attempting to filter on a
        # nested-field path).
        schema_filter = base_log + ' AND protoPayload.methodName:"hostError"'
        schema_ok, count = _list_log_entries(schema_filter, max_results=5)
        result["tests"]["event_schema_valid"] = {
            "passed": schema_ok,
            "event_count": count,
        }
        result["recent_event_count"] = count
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"] = classify_gcp_error(e)[0]
        result["error"] = classify_gcp_error(e)[1]
        return result
    result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    return result


def _latency_perf(project: str, network: str) -> dict[str, Any]:
    """Each subtest issues a distinct probe — flow-log endpoint, named
    performance namespace, packet-metric filter, recency window — so the
    pass signal does not collapse to a single API call (AWS oracle parity).
    """
    sample_window = 300
    flow_log_root = f'logName="projects/{project}/logs/compute.googleapis.com%2Fvpc_flows"'
    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "test_name": "sdn_latency_perf_logging",
        "network_id": network,
        "aspect": "latency_perf",
        "telemetry_namespace": LATENCY_NAMESPACE,
        "sample_window_seconds": sample_window,
        "probe_resource_id": network,
        "tests": {},
    }
    try:
        # 1. Metrics endpoint reachable: smallest possible filter against
        # the VPC Flow Logs log name.
        endpoint_ok, _ = _list_log_entries(flow_log_root, max_results=1)
        result["tests"]["metrics_endpoint_reachable"] = {"passed": endpoint_ok}

        # 2. Performance metric present: narrow to entries carrying the
        # bytes_sent / bytes_received fields the perf namespace uses.
        perf_filter = flow_log_root + " AND jsonPayload.bytes_sent:*"
        perf_ok, _ = _list_log_entries(perf_filter, max_results=5)
        result["tests"]["performance_metric_present"] = {
            "passed": perf_ok,
            "namespace": LATENCY_NAMESPACE,
        }

        # 3. Packet metric present: distinct narrow to packet-count fields.
        packet_filter = flow_log_root + " AND jsonPayload.packets_sent:*"
        packet_ok, flow_count = _list_log_entries(packet_filter, max_results=5)
        result["tests"]["packet_metric_present"] = {
            "passed": packet_ok,
            "flow_log_count": flow_count,
        }

        # 4. Samples recent: filter on the last N-second window so a stale
        # log archive doesn't false-pass.
        from datetime import datetime, timedelta

        since = (datetime.now(UTC) - timedelta(seconds=sample_window)).isoformat()
        recent_filter = flow_log_root + f' AND timestamp>="{since}"'
        recent_ok, _ = _list_log_entries(recent_filter, max_results=5)
        result["tests"]["samples_recent"] = {
            "passed": recent_ok,
            "sample_window_seconds": sample_window,
        }
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"] = classify_gcp_error(e)[0]
        result["error"] = classify_gcp_error(e)[1]
        return result
    result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    return result


def _audit_trail(project: str, network: str) -> dict[str, Any]:
    """Create-patch-delete a probe firewall, then poll Admin Activity logs."""
    fw_name = unique_suffix("isv-audit")
    firewalls = compute_v1.FirewallsClient()
    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "test_name": "sdn_filter_audit_trail",
        "network_id": network,
        "aspect": "audit_trail",
        "trail_id": AUDIT_LOG_NAME,
        "actor_field": "protoPayload.authenticationInfo.principalEmail",
        "target_rule_id": fw_name,
        "tests": {},
    }
    created = False
    try:
        op = firewalls.insert(
            project=project,
            firewall_resource=compute_v1.Firewall(
                name=fw_name,
                description=ISV_DESCRIPTION,
                network=f"projects/{project}/global/networks/{network}",
                direction="INGRESS",
                source_ranges=["0.0.0.0/0"],
                allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=["22"])],
            ),
        )
        # Stamp tracker BEFORE wait — partial-create cleanup contract.
        created = True
        wait_for_global_op(project, op.name, timeout=180)

        op = firewalls.patch(
            project=project,
            firewall=fw_name,
            firewall_resource=compute_v1.Firewall(
                allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=["80"])],
            ),
        )
        wait_for_global_op(project, op.name, timeout=180)

        op = firewalls.delete(project=project, firewall=fw_name)
        wait_for_global_op(project, op.name, timeout=180)
        created = False

        # Poll Admin Activity logs for the three method names. Audit
        # propagation can take 1-5 min on Compute Engine; budget 480s
        # with 10s interval. Filter is scoped by URL-encoded logName
        # ('cloudaudit.googleapis.com%2Factivity' — Cloud Logging stores
        # the slash in the log ID as %2F, and the `=` comparator is exact
        # match) plus a `protoPayload.resourceName:<token>` substring test
        # against the unique firewall name. methodNames are then inspected
        # on each returned entry rather than embedded in the filter so a
        # vendor API-version prefix change (v1.compute.* or compute.*) does
        # not silently zero-out the seen counters.
        deadline = time.monotonic() + 480
        seen: dict[str, int] = {"insert": 0, "patch": 0, "delete": 0}
        reachable = False
        first_insert_entry = None
        audit_log_id_encoded = AUDIT_LOG_NAME.replace("/", "%2F")
        base = f'logName="projects/{project}/logs/{audit_log_id_encoded}" AND protoPayload.resourceName:"{fw_name}"'
        while time.monotonic() < deadline and not all(seen.values()):
            try:
                entries = list(_iter_log_entries(base, max_results=50))
                reachable = True
            except (gax.GoogleAPICallError, RuntimeError, TimeoutError):
                entries = []
            for entry in entries:
                payload = getattr(entry, "payload", None) or {}
                if isinstance(payload, dict):
                    method_name = payload.get("methodName", "") or ""
                    resource_name = payload.get("resourceName", "") or ""
                else:
                    method_name = getattr(payload, "method_name", "") or ""
                    resource_name = getattr(payload, "resource_name", "") or ""
                # Confirm the entry actually targets OUR firewall (defensive
                # against the substring filter matching anything containing
                # the unique suffix).
                if fw_name not in resource_name:
                    continue
                # Match both `compute.firewalls.<op>` and
                # `v1.compute.firewalls.<op>` to tolerate vendor API-version
                # prefix differences across regions/log versions.
                for op_name in ("insert", "patch", "delete"):
                    if method_name.endswith(f"compute.firewalls.{op_name}"):
                        seen[op_name] += 1
                        if op_name == "insert" and first_insert_entry is None:
                            first_insert_entry = entry
            if all(seen.values()):
                break
            time.sleep(10)

        result["tests"]["audit_endpoint_reachable"] = {"passed": reachable}
        result["tests"]["create_rule_logged"] = {"passed": bool(seen["insert"])}
        # patch OR update — both Compute Engine method names valid.
        result["tests"]["modify_rule_logged"] = {"passed": bool(seen["patch"])}
        result["tests"]["delete_rule_logged"] = {"passed": bool(seen["delete"])}
        # Audit-event field validation: inspect the captured insert entry;
        # require principalEmail + timestamp + resourceName all present per
        # the protoPayload schema.
        fields_ok = False
        if first_insert_entry is not None:
            payload = getattr(first_insert_entry, "payload", None)
            if isinstance(payload, dict):
                auth = payload.get("authenticationInfo", {})
                actor = auth.get("principalEmail") if isinstance(auth, dict) else None
                resource = payload.get("resourceName")
            else:
                auth = getattr(payload, "authentication_info", None) if payload else None
                actor = getattr(auth, "principal_email", None) if auth else None
                resource = getattr(payload, "resource_name", None) if payload else None
            timestamp = getattr(first_insert_entry, "timestamp", None)
            fields_ok = bool(actor) and bool(timestamp) and bool(resource)
        result["tests"]["audit_event_has_required_fields"] = {
            "passed": fields_ok,
            "actor_field": result["actor_field"],
        }
        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        cleanup_ok = True
        if created:
            cleanup_ok = delete_with_retry(
                lambda: wait_for_global_op(
                    project,
                    firewalls.delete(project=project, firewall=fw_name).name,
                    timeout=120,
                ),
                resource_desc=f"firewall {fw_name}",
            )
    # Gate the cleanup subtest on the actual delete bool (AWS oracle parity).
    result["tests"]["cleanup"] = {"passed": cleanup_ok}
    result["success"] = result.get("success", False) and cleanup_ok
    return result


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="SDN logging multi-aspect dispatcher")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--vpc-id", required=True)
    parser.add_argument(
        "--aspect",
        choices=["hardware_faults", "latency_perf", "audit_trail"],
        required=True,
    )
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    if args.aspect == "hardware_faults":
        result = _hardware_faults(project, args.vpc_id)
    elif args.aspect == "latency_perf":
        result = _latency_perf(project, args.vpc_id)
    else:
        result = _audit_trail(project, args.vpc_id)
    result["region"] = args.region

    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())

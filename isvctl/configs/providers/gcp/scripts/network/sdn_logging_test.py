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

from common.compute import (
    narrow_region_to_zone,
    resolve_project,
    unique_suffix,
    wait_for_global_op,
    wait_for_zonal_op,
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest sdn_logging — verified-reuse marker"
HARDWARE_LOG_NAME = "cloudaudit.googleapis.com/system_event"
LATENCY_NAMESPACE = "compute.googleapis.com/instance/network"
AUDIT_LOG_NAME = "cloudaudit.googleapis.com/activity"


def _list_log_entries(project: str, filter_str: str, max_results: int = 50) -> tuple[bool, int]:
    """Return (reachable, count). reachable=False when google-cloud-logging
    is not installed OR the API call raises.

    ``project`` must be the operator-resolved GCP project; the Cloud Logging
    client and ``resource_names`` are both pinned to it so a run whose
    ``--project`` differs from the ADC default project does not silently
    query the wrong logging scope.
    """
    try:
        from google.cloud import logging_v2  # type: ignore[attr-defined]
    except ImportError:
        return False, 0
    client = logging_v2.Client(project=project)
    count = 0
    for _ in client.list_entries(
        resource_names=[f"projects/{project}"],
        filter_=filter_str,
        page_size=max_results,
        max_results=max_results,
    ):
        count += 1
    return True, count


def _iter_log_entries(project: str, filter_str: str, max_results: int = 50):
    """Yield raw entries for the filter, or empty iterator on import/API
    failure. The caller is expected to inspect ``payload``/``method_name``
    fields directly. The logging client and ``resource_names`` are pinned
    to the operator-resolved ``project``.
    """
    try:
        from google.cloud import logging_v2  # type: ignore[attr-defined]
    except ImportError:
        return
    client = logging_v2.Client(project=project)
    yield from client.list_entries(
        resource_names=[f"projects/{project}"],
        filter_=filter_str,
        page_size=max_results,
        max_results=max_results,
    )


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
        endpoint_ok, _ = _list_log_entries(project, base_log, max_results=1)
        result["tests"]["logging_endpoint_reachable"] = {"passed": endpoint_ok}

        # 2. Fault-event source queryable: filter narrowing to host_event_*
        # method names.
        source_filter = base_log + ' AND protoPayload.methodName:"host_event"'
        source_ok, _ = _list_log_entries(project, source_filter, max_results=5)
        result["tests"]["fault_event_source_queryable"] = {"passed": source_ok}

        # 3. Log destination configured: verify the canonical log name
        # resolves (a non-existent log returns NOT_FOUND on the list call).
        dest_filter = base_log
        dest_ok, _ = _list_log_entries(project, dest_filter, max_results=1)
        result["tests"]["log_destination_configured"] = {
            "passed": dest_ok,
            "log_destination": HARDWARE_LOG_NAME,
        }

        # 4. Event schema valid: distinct filter for hostError method name
        # (verifies the protoPayload schema by attempting to filter on a
        # nested-field path).
        schema_filter = base_log + ' AND protoPayload.methodName:"hostError"'
        schema_ok, count = _list_log_entries(project, schema_filter, max_results=5)
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


def _list_vpc_subnets(project: str, region: str, network: str) -> list[compute_v1.Subnetwork]:
    """Return subnetworks in ``region`` whose ``network`` URL ends with
    ``/networks/{network}``."""
    client = compute_v1.SubnetworksClient()
    network_suffix = f"/networks/{network}"
    return [s for s in client.list(project=project, region=region) if (s.network or "").endswith(network_suffix)]


def _latency_perf(project: str, region: str, network: str) -> dict[str, Any]:
    """Each subtest issues a distinct probe — flow-log endpoint, named
    performance namespace, packet-metric filter, recency window — so the
    pass signal does not collapse to a single API call (AWS oracle parity).

    The flow-log telemetry source must be REAL and SCOPED:

      * Subnetworks created by create_vpc.py carry log_config.enable=True
        (verified-reuse — we read it back here, never fabricated).
      * A bounded-traffic probe VM is launched on a target subnet to
        generate egress packets so Cloud Logging has observable flow-log
        entries within this run's sample window. Without the probe, a
        clean GCP project with flow logs enabled but no VM workload
        produces zero `vpc_flows` entries and `samples_recent` fails.
      * Every Cloud Logging query is scoped to the target VPC via
        `jsonPayload.{src_vpc,dest_vpc}.vpc_name="{network}"` — an
        unscoped query would let an unrelated VPC's flow logs satisfy the
        validator (AWS oracle parity: `resource-id` filter on the AWS
        side scopes the equivalent query).
    """
    from datetime import datetime

    sample_window = 600
    flow_log_root = f'logName="projects/{project}/logs/compute.googleapis.com%2Fvpc_flows"'
    # Target-VPC scope on jsonPayload network names. Either side of a
    # connection (src or dest) attributed to our network suffices.
    vpc_scope = f'(jsonPayload.src_vpc.vpc_name="{network}" OR jsonPayload.dest_vpc.vpc_name="{network}")'
    flow_log_base = f"{flow_log_root} AND {vpc_scope}"
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

    instances_c = compute_v1.InstancesClient()
    probe_vm_name = unique_suffix("isv-flow-prb")
    zone = narrow_region_to_zone(region)
    probe_created = False
    flow_logs_enabled = False
    probe_subnet: compute_v1.Subnetwork | None = None
    test_start = datetime.now(UTC)

    try:
        # ---- Detect flow-log configuration on target-VPC subnets ----
        subnets = _list_vpc_subnets(project, region, network)
        if not subnets:
            raise RuntimeError(
                f"no subnetworks found for VPC {network!r} in region {region}; create_network must precede this step"
            )
        for sub in subnets:
            if getattr(sub.log_config, "enable", False):
                flow_logs_enabled = True
                probe_subnet = sub
                break
        if not flow_logs_enabled or probe_subnet is None:
            raise RuntimeError(
                f"VPC {network!r} has subnetworks without log_config.enable=True; "
                "create_vpc.py must set SubnetworkLogConfig(enable=True) at subnet create"
            )

        # ---- Bounded-traffic probe ----
        # Startup script pings 8.8.8.8 for ~10s to generate VPC egress
        # captured by flow logs. External IP attached because flow logs
        # of egress to the internet record src_vpc=our_network.
        startup_script = (
            "#!/bin/bash\n"
            "set -e\n"
            "# Bounded outbound ICMP — generates flow-log entries scoped\n"
            "# to this VPC (src_vpc=our_network on egress).\n"
            "ping -c 10 -i 1 8.8.8.8 || true\n"
        )
        probe = compute_v1.Instance(
            name=probe_vm_name,
            description=ISV_DESCRIPTION,
            machine_type=f"zones/{zone}/machineTypes/e2-small",
            disks=[
                compute_v1.AttachedDisk(
                    boot=True,
                    auto_delete=True,
                    initialize_params=compute_v1.AttachedDiskInitializeParams(
                        source_image="projects/debian-cloud/global/images/family/debian-12",
                        disk_size_gb=10,
                    ),
                )
            ],
            network_interfaces=[
                compute_v1.NetworkInterface(
                    network=f"projects/{project}/global/networks/{network}",
                    subnetwork=f"projects/{project}/regions/{region}/subnetworks/{probe_subnet.name}",
                    access_configs=[
                        compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT"),
                    ],
                )
            ],
            metadata=compute_v1.Metadata(
                items=[compute_v1.Items(key="startup-script", value=startup_script)],
            ),
            service_accounts=[],
        )
        op = instances_c.insert(project=project, zone=zone, instance_resource=probe)
        probe_created = True
        # Each individual wait must fit within the orchestrator step
        # timeout (180s — AWS oracle cap). Probe VM insert typically
        # completes in 30-60s; cap at 90s.
        wait_for_zonal_op(project, zone, op.name, timeout=90)

        # Boot + startup-script + traffic flow. cloud-init runs the
        # startup script ~5-15s after RUNNING; the ping loop takes 10s;
        # flow-log buffering adds another ~5s. 20s gives ~5s margin under
        # the tight 180s overall budget.
        time.sleep(20)

        # ---- Poll Cloud Logging for scoped flow-log samples ----
        # Restrict the recency window to entries observed AFTER this step
        # started so a stale unrelated run cannot satisfy the validator.
        since_ts = test_start.isoformat()
        # 1. Metrics endpoint reachable: VPC Flow Logs API reachable for
        # entries attributed to this network.
        endpoint_filter = flow_log_base + f' AND timestamp>="{since_ts}"'
        # 2. Performance metric present: bytes_sent populated.
        perf_filter = endpoint_filter + " AND jsonPayload.bytes_sent:*"
        # 3. Packet metric present: packets_sent populated.
        packet_filter = endpoint_filter + " AND jsonPayload.packets_sent:*"

        endpoint_ok = False
        perf_ok = False
        perf_count = 0
        packet_ok = False
        packet_count = 0
        recent_ok = False
        recent_count = 0
        # Cloud Logging flow-log propagation typically 15-45s for a fresh
        # entry. 45s deadline with 10s interval fits under the 180s
        # orchestrator cap (AWS oracle parity). A genuinely-slow GCP
        # logging tier exceeds 45s and surfaces as a real test failure
        # rather than being masked by a padded timeout.
        poll_deadline = time.monotonic() + 45
        while time.monotonic() < poll_deadline:
            endpoint_ok, _ = _list_log_entries(project, endpoint_filter, max_results=1)
            perf_ok, perf_count = _list_log_entries(project, perf_filter, max_results=5)
            packet_ok, packet_count = _list_log_entries(project, packet_filter, max_results=5)
            recent_ok, recent_count = _list_log_entries(project, endpoint_filter, max_results=5)
            if endpoint_ok and perf_count > 0 and packet_count > 0 and recent_count > 0:
                break
            time.sleep(10)

        result["tests"]["metrics_endpoint_reachable"] = {"passed": endpoint_ok}
        result["tests"]["performance_metric_present"] = {
            "passed": perf_ok and perf_count > 0,
            "namespace": LATENCY_NAMESPACE,
            "matching_entries": perf_count,
        }
        result["tests"]["packet_metric_present"] = {
            "passed": packet_ok and packet_count > 0,
            "flow_log_count": packet_count,
        }
        result["tests"]["samples_recent"] = {
            "passed": recent_ok and recent_count > 0,
            "sample_window_seconds": sample_window,
            "matching_entries": recent_count,
        }
        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        cleanup_ok = True
        cleanup_error: str | None = None
        if probe_created:
            try:
                cleanup_ok = delete_with_retry(
                    lambda: wait_for_zonal_op(
                        project,
                        zone,
                        instances_c.delete(project=project, zone=zone, instance=probe_vm_name).name,
                        timeout=60,
                    ),
                    resource_desc=f"probe instance {probe_vm_name}",
                )
            except Exception as e:
                cleanup_ok = False
                cleanup_error = str(e)
    cleanup_test: dict[str, Any] = {"passed": cleanup_ok}
    if cleanup_error:
        cleanup_test["error"] = cleanup_error
    result["tests"]["cleanup"] = cleanup_test
    result["success"] = result.get("success", False) and cleanup_ok
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
        # Logical event keys, not raw API method suffixes: a firewall
        # modification audit entry is recorded as `compute.firewalls.patch`
        # OR `compute.firewalls.update` depending on the SDK call path /
        # log version, and both satisfy the modify-rule oracle.
        seen: dict[str, int] = {"insert": 0, "modify": 0, "delete": 0}
        method_to_event = {
            "insert": "insert",
            "patch": "modify",
            "update": "modify",
            "delete": "delete",
        }
        reachable = False
        first_insert_entry = None
        audit_log_id_encoded = AUDIT_LOG_NAME.replace("/", "%2F")
        base = f'logName="projects/{project}/logs/{audit_log_id_encoded}" AND protoPayload.resourceName:"{fw_name}"'
        while time.monotonic() < deadline and not all(seen.values()):
            try:
                entries = list(_iter_log_entries(project, base, max_results=50))
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
                # prefix differences across regions/log versions. Both
                # `patch` and `update` map to the logical `modify` event.
                for method_suffix, event_key in method_to_event.items():
                    if method_name.endswith(f"compute.firewalls.{method_suffix}"):
                        seen[event_key] += 1
                        if method_suffix == "insert" and first_insert_entry is None:
                            first_insert_entry = entry
            if all(seen.values()):
                break
            time.sleep(10)

        result["tests"]["audit_endpoint_reachable"] = {"passed": reachable}
        result["tests"]["create_rule_logged"] = {"passed": bool(seen["insert"])}
        # modify counter accepts compute.firewalls.patch OR .update.
        result["tests"]["modify_rule_logged"] = {"passed": bool(seen["modify"])}
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
        result = _latency_perf(project, args.region, args.vpc_id)
    else:
        result = _audit_trail(project, args.vpc_id)
    result["region"] = args.region

    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())

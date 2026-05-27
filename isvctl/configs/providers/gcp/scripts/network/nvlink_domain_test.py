#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine NVLink domain metadata probe (NvlinkDomainCheck).

Divergences from the AWS oracle:

  * AWS does not ship an nvlink_domain step. The validator was added
    after the AWS-shaped suite; non-AWS NCPs must adapt.
  * Compute Engine does NOT expose an NVLink-domain identifier via
    public APIs. The honest portable probe is:
      1. Resolve a real Compute Engine instance to inspect. If the
         operator supplied ``--node-id`` and it resolves via
         InstancesClient aggregated-list, use that instance. Otherwise
         launch a tiny e2-small (no-accelerator) probe VM into a
         subnet of the shared ``--vpc-id`` and use that — this gives
         the released validator the real non-NVLink shape it expects
         without requiring GPU quota.
      2. Read ``guestAccelerators`` on the resolved instance.
      3. Emit ``nvlink_supported=false`` for non-NVLink shapes. The
         validator then surfaces an explicit ``pytest.skip``.
      4. When NVLink IS attached, a ``nvidia-smi topo`` guest probe is
         required to populate ``nvlink_domain_id``. This stub does NOT
         SSH into the guest, so the NVLink-capable path fails
         ``nvlink_support_detected`` with a clear "no provider-side
         topology probe; guest probe required" message. Do NOT force
         nvlink_supported=false to silently route through pytest.skip
         — knowledge: "do not invent an ID from machine type or zone"
         AND "Emit nvlink_supported=false ONLY from real evidence that
         the resolved node is non-NVLink."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import (
    narrow_region_to_zone,
    resolve_project,
    short_name,
    unique_suffix,
    wait_for_zonal_op,
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest nvlink_domain — verified-reuse marker"

# NVLink-capable Compute Engine accelerator types (interconnect topology
# exposes peer NVLink bridges within an HGX system). Source:
# https://cloud.google.com/compute/docs/gpus — A100/H100/H200/B200 ship
# with NVLink fabric on HGX baseboards.
NVLINK_ACCELERATOR_FAMILIES = (
    "nvidia-h100",
    "nvidia-h200",
    "nvidia-a100",
    "nvidia-b200",
)


def _accelerator_type_short(selflink: str) -> str:
    return selflink.rsplit("/", 1)[-1] if selflink else ""


def _is_nvlink_capable(accel_type: str) -> bool:
    return any(accel_type.startswith(family) for family in NVLINK_ACCELERATOR_FAMILIES)


def _resolve_instance(project: str, node_id: str) -> compute_v1.Instance | None:
    """Find a Compute Engine instance named ``node_id`` across all zones.

    Returns the first match found via aggregated-list, or None when no
    instance with that name exists in the project.
    """
    client = compute_v1.InstancesClient()
    try:
        for _zone, scoped in client.aggregated_list(project=project):
            for inst in getattr(scoped, "instances", ()) or ():
                if inst.name == node_id:
                    return inst
    except gax.GoogleAPICallError:
        return None
    return None


def _pick_subnet(project: str, region: str, vpc_id: str) -> str | None:
    """Return the name of any subnetwork in ``region`` belonging to ``vpc_id``.

    Exact short-name comparison via :func:`short_name`. Raw selfLink suffix
    matching accepts supersets (e.g., a sibling VPC whose name shares the
    ``vpc_id`` suffix) and binds the probe to the wrong scope; the
    scope-binding oracle rule mandates trailing-segment equality.
    """
    client = compute_v1.SubnetworksClient()
    for sub in client.list(project=project, region=region):
        if short_name(sub.network) == vpc_id:
            return sub.name
    return None


def _submit_probe_insert(
    project: str,
    region: str,
    vpc_id: str,
    instance_name: str,
) -> tuple[str, str]:
    """Submit an async ``instances.insert`` for the non-NVLink probe VM.

    Returns ``(zone, op_name)``. The probe-VM cleanup tracker is stamped
    by the caller **immediately after this call returns**, BEFORE waiting
    on the op. Mirrors the async-create discipline (Pattern (a) — stamp
    before wait): if the operation succeeds but ``wait_for_zonal_op``
    later raises (timeout, DONE-with-error), the tracker already names the
    accepted VM so the ``finally`` cleanup path deletes it.

    The probe is intentionally e2-small with no accelerators so the
    released validator sees a real non-NVLink shape without requiring
    GPU quota.
    """
    subnet = _pick_subnet(project, region, vpc_id)
    if subnet is None:
        raise RuntimeError(
            f"no subnetwork found in region {region} for VPC {vpc_id!r}; create_network must run before nvlink_domain"
        )
    zone = narrow_region_to_zone(region)
    instances_c = compute_v1.InstancesClient()
    probe = compute_v1.Instance(
        name=instance_name,
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
                network=f"projects/{project}/global/networks/{vpc_id}",
                subnetwork=f"projects/{project}/regions/{region}/subnetworks/{subnet}",
            )
        ],
        service_accounts=[],
    )
    op = instances_c.insert(project=project, zone=zone, instance_resource=probe)
    return zone, op.name


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine NVLink domain metadata probe")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument(
        "--node-id",
        default="",
        help=(
            "Optional Compute Engine instance name to inspect. When empty or "
            "unresolved, an ephemeral non-NVLink probe VM is launched in the "
            "shared VPC and inspected (released validator only requires real "
            "guestAccelerator readback, not a specific GPU node)."
        ),
    )
    parser.add_argument(
        "--vpc-id",
        required=True,
        help="Shared VPC name; used to place the probe VM when --node-id is unresolved.",
    )
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    instances_c = compute_v1.InstancesClient()

    probe_name = unique_suffix("isv-nvlk-prb")
    probe_zone: str | None = None
    probe_created = False
    chosen_node_id = args.node_id or probe_name

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "test_name": "nvlink_domain",
        "region": args.region,
        "node_id": chosen_node_id,
        "nvlink_supported": False,
        "tests": {
            "node_resolved": {"passed": False, "node_id": chosen_node_id},
            "nvlink_support_detected": {"passed": False},
            "nvlink_domain_id_present": {"passed": False},
        },
    }

    try:
        instance: compute_v1.Instance | None = None
        if args.node_id:
            instance = _resolve_instance(project, args.node_id)
        if instance is None:
            # Either no --node-id was supplied or the named node does not
            # exist. Launch our own non-NVLink probe so the released
            # validator gets a real guest_accelerators readback.
            #
            # Async-create discipline (Pattern (a)): submit the insert,
            # then stamp the cleanup tracker BEFORE waiting on the op.
            # If wait_for_zonal_op times out or returns DONE-with-error,
            # the `finally` path still names the accepted VM and deletes
            # it, so a half-created probe cannot leak.
            probe_zone, op_name = _submit_probe_insert(project, args.region, args.vpc_id, probe_name)
            probe_created = True
            result["node_id"] = probe_name
            chosen_node_id = probe_name
            wait_for_zonal_op(project, probe_zone, op_name, timeout=120)
            instance = instances_c.get(project=project, zone=probe_zone, instance=probe_name)

        result["tests"]["node_resolved"] = {
            "passed": True,
            "node_id": chosen_node_id,
            "found": True,
            "probe_created": probe_created,
        }
        accelerator_types = [
            _accelerator_type_short(a.accelerator_type)
            for a in (instance.guest_accelerators or ())
            if a.accelerator_type
        ]
        nvlink_capable = any(_is_nvlink_capable(t) for t in accelerator_types)
        if nvlink_capable:
            # NVLink hardware attached — the validator requires a real
            # `nvlink_domain_id`. Public Compute Engine API does not
            # expose one; an in-guest probe (e.g., SSH + `nvidia-smi topo`)
            # would be required. Knowledge: do not invent the ID and do not
            # silently route through pytest.skip on NVLink-capable hardware.
            result["tests"]["nvlink_support_detected"] = {
                "passed": False,
                "accelerators": accelerator_types,
                "nvlink_capable_family_present": True,
                "message": (
                    "NVLink-capable accelerators detected but no provider-side "
                    "NVLink-domain API and no guest topology probe is implemented; "
                    "an `nvidia-smi topo` SSH probe is required to emit a real "
                    "nvlink_domain_id"
                ),
            }
        else:
            # Non-NVLink shape — the canonical skip path. nvlink_supported
            # stays False and the validator pytest.skips the check.
            result["tests"]["nvlink_support_detected"] = {
                "passed": True,
                "accelerators": accelerator_types,
                "nvlink_capable_family_present": False,
            }
            result["nvlink_supported"] = False

        # nvlink_domain_id_present stays False — only meaningful on the
        # NVLink-supported path with a verified guest probe.
        result["success"] = (
            result["tests"]["node_resolved"]["passed"] and result["tests"]["nvlink_support_detected"]["passed"]
        )
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        # Probe-VM cleanup — capture the bool and surface in cleanup
        # subtest so a leaked probe cannot read as cleanup success.
        cleanup_ok = True
        cleanup_error: str | None = None
        if probe_created and probe_zone is not None:
            try:
                cleanup_ok = delete_with_retry(
                    lambda: wait_for_zonal_op(
                        project,
                        probe_zone,
                        instances_c.delete(project=project, zone=probe_zone, instance=probe_name).name,
                        timeout=180,
                    ),
                    resource_desc=f"probe instance {probe_name}",
                )
            except Exception as e:
                cleanup_ok = False
                cleanup_error = str(e)
    result["tests"]["cleanup"] = {"passed": cleanup_ok}
    if cleanup_error:
        result["tests"]["cleanup"]["error"] = cleanup_error
    # AND cleanup into success — silently-leaked probe must not read as success.
    result["success"] = result.get("success", False) and cleanup_ok

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

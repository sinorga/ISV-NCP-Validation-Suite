#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine stable private IP test (StablePrivateIpCheck).

Divergences from the AWS oracle:

  * Instance states: RUNNING / TERMINATED (use canonical_state).
  * stop / start return Operations; wait via ZoneOperationsClient.wait.
  * Compute Engine preserves internal IP across stop/start by default,
    so ip_unchanged passes naturally on the same Instance resource.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import (
    first_internal_ip,
    narrow_region_to_zone,
    poll_instance_state,
    resolve_project,
    unique_suffix,
    wait_for_global_op,
    wait_for_zonal_op,
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest stable_ip — verified-reuse marker"
DEFAULT_IMAGE = "projects/debian-cloud/global/images/family/debian-12"


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine stable private IP")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr", default="10.91.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = narrow_region_to_zone(args.region)
    network = unique_suffix("isv-stab")
    subnet = unique_suffix("isv-stab-sn")
    sub_cidr = str(next(iter(ipaddress.ip_network(args.cidr).subnets(new_prefix=24))))
    instance = unique_suffix("isv-stab-i")

    networks_c = compute_v1.NetworksClient()
    subnets_c = compute_v1.SubnetworksClient()
    instances_c = compute_v1.InstancesClient()

    result: dict[str, Any] = {"success": False, "platform": "network", "tests": {}}
    cleanup: list[tuple[str, str]] = []
    try:
        op = networks_c.insert(
            project=project,
            network_resource=compute_v1.Network(
                name=network,
                description=ISV_DESCRIPTION,
                auto_create_subnetworks=False,
            ),
        )
        wait_for_global_op(project, op.name, timeout=300)
        cleanup.append(("network", network))
        op = subnets_c.insert(
            project=project,
            region=args.region,
            subnetwork_resource=compute_v1.Subnetwork(
                name=subnet,
                description=ISV_DESCRIPTION,
                ip_cidr_range=sub_cidr,
                network=f"projects/{project}/global/networks/{network}",
                region=args.region,
            ),
        )
        compute_v1.RegionOperationsClient().wait(
            project=project,
            region=args.region,
            operation=op.name,
            timeout=180,
        )
        cleanup.append(("subnet", subnet))

        inst = compute_v1.Instance(
            name=instance,
            description=ISV_DESCRIPTION,
            machine_type=f"zones/{zone}/machineTypes/e2-small",
            disks=[
                compute_v1.AttachedDisk(
                    boot=True,
                    auto_delete=True,
                    initialize_params=compute_v1.AttachedDiskInitializeParams(
                        source_image=DEFAULT_IMAGE,
                        disk_size_gb=10,
                    ),
                )
            ],
            network_interfaces=[
                compute_v1.NetworkInterface(
                    network=f"projects/{project}/global/networks/{network}",
                    subnetwork=f"projects/{project}/regions/{args.region}/subnetworks/{subnet}",
                )
            ],
            service_accounts=[],
        )
        op = instances_c.insert(project=project, zone=zone, instance_resource=inst)
        cleanup.append(("instance", instance))
        wait_for_zonal_op(project, zone, op.name, timeout=600)
        poll_instance_state(project, zone, instance, target_canonical="running", timeout=600)
        result["tests"]["create_instance"] = {"passed": True, "instance_id": instance}

        obj = instances_c.get(project=project, zone=zone, instance=instance)
        ip_before = first_internal_ip(obj)
        result["tests"]["record_ip"] = {"passed": bool(ip_before), "private_ip": ip_before}

        op = instances_c.stop(project=project, zone=zone, instance=instance)
        wait_for_zonal_op(project, zone, op.name, timeout=300)
        poll_instance_state(project, zone, instance, target_canonical="stopped", timeout=300)
        result["tests"]["stop_instance"] = {"passed": True}

        op = instances_c.start(project=project, zone=zone, instance=instance)
        wait_for_zonal_op(project, zone, op.name, timeout=300)
        poll_instance_state(project, zone, instance, target_canonical="running", timeout=300)
        result["tests"]["start_instance"] = {"passed": True}

        obj = instances_c.get(project=project, zone=zone, instance=instance)
        ip_after = first_internal_ip(obj)
        result["tests"]["ip_unchanged"] = {
            "passed": ip_before == ip_after and bool(ip_before),
            "ip_before": ip_before,
            "ip_after": ip_after,
        }

        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        for kind, n in reversed(cleanup):
            try:
                if kind == "instance":
                    delete_with_retry(
                        lambda nn=n: wait_for_zonal_op(
                            project,
                            zone,
                            instances_c.delete(project=project, zone=zone, instance=nn).name,
                            timeout=300,
                        ),
                        resource_desc=f"instance {n}",
                    )
                elif kind == "subnet":
                    delete_with_retry(
                        lambda nn=n: compute_v1.RegionOperationsClient().wait(
                            project=project,
                            region=args.region,
                            operation=subnets_c.delete(project=project, region=args.region, subnetwork=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"subnet {n}",
                    )
                else:
                    delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            networks_c.delete(project=project, network=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"network {n}",
                    )
            except Exception:
                pass

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

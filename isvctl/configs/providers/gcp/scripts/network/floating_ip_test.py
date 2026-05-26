#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine floating IP test (FloatingIpCheck).

Divergences from the AWS oracle:

  * Reassociate is two API calls — delete_access_config on A +
    add_access_config(natIP=<ip>) on B; time both Operations.
  * verify_on_a / verify_on_b read
    networkInterfaces[*].accessConfigs[*].natIP.
  * Provider config raises max_switch_seconds to 20 — Compute Engine
    ephemeral→static promotion measured ~12s.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import (
    first_external_ip,
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

ISV_DESCRIPTION = "isvtest floating_ip — verified-reuse marker"
DEFAULT_IMAGE = "projects/debian-cloud/global/images/family/debian-12"


def _wait_region_op(project: str, region: str, op_name: str, *, timeout: int = 300) -> None:
    compute_v1.RegionOperationsClient().wait(
        project=project,
        region=region,
        operation=op_name,
        timeout=timeout,
    )


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine floating IP")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr", default="10.92.0.0/16")
    parser.add_argument("--max-switch-seconds", type=int, default=20)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = narrow_region_to_zone(args.region)

    network = unique_suffix("isv-fip")
    subnet = unique_suffix("isv-fip-sn")
    sub_cidr = str(next(iter(ipaddress.ip_network(args.cidr).subnets(new_prefix=24))))
    addr_name = unique_suffix("isv-fip-addr")
    inst_a = unique_suffix("isv-fip-a")
    inst_b = unique_suffix("isv-fip-b")

    networks_c = compute_v1.NetworksClient()
    subnets_c = compute_v1.SubnetworksClient()
    instances_c = compute_v1.InstancesClient()
    addresses_c = compute_v1.AddressesClient()

    result: dict[str, Any] = {"success": False, "platform": "network", "tests": {}}
    cleanup: list[tuple[str, str]] = []
    address_value: str | None = None
    try:
        # Setup network + subnet + 2 instances.
        op = networks_c.insert(
            project=project,
            network_resource=compute_v1.Network(
                name=network,
                description=ISV_DESCRIPTION,
                auto_create_subnetworks=False,
            ),
        )
        cleanup.append(("network", network))
        wait_for_global_op(project, op.name, timeout=300)
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
        cleanup.append(("subnet", subnet))
        _wait_region_op(project, args.region, op.name, timeout=180)

        for n in (inst_a, inst_b):
            op = instances_c.insert(
                project=project,
                zone=zone,
                instance_resource=compute_v1.Instance(
                    name=n,
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
                            access_configs=[compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")],
                        )
                    ],
                    service_accounts=[],
                ),
            )
            cleanup.append(("instance", n))
            wait_for_zonal_op(project, zone, op.name, timeout=600)
            poll_instance_state(project, zone, n, target_canonical="running", timeout=600)

        # allocate_eip — reserve a regional static external IP.
        op = addresses_c.insert(
            project=project,
            region=args.region,
            address_resource=compute_v1.Address(
                name=addr_name,
                description=ISV_DESCRIPTION,
                address_type="EXTERNAL",
                labels={"createdby": "isvtest"},
            ),
        )
        cleanup.append(("address", addr_name))
        _wait_region_op(project, args.region, op.name, timeout=180)
        addr_obj = addresses_c.get(project=project, region=args.region, address=addr_name)
        address_value = addr_obj.address
        result["tests"]["allocate_eip"] = {
            "passed": True,
            "allocation_id": addr_name,
            "public_ip": address_value,
        }

        # associate_to_a — delete A's ephemeral accessConfig then add static.
        a_obj = instances_c.get(project=project, zone=zone, instance=inst_a)
        nic_name = a_obj.network_interfaces[0].name or "nic0"
        if a_obj.network_interfaces[0].access_configs:
            ac_name = a_obj.network_interfaces[0].access_configs[0].name
            op = instances_c.delete_access_config(
                project=project,
                zone=zone,
                instance=inst_a,
                access_config=ac_name,
                network_interface=nic_name,
            )
            wait_for_zonal_op(project, zone, op.name, timeout=180)
        op = instances_c.add_access_config(
            project=project,
            zone=zone,
            instance=inst_a,
            network_interface=nic_name,
            access_config_resource=compute_v1.AccessConfig(
                name="External NAT",
                type_="ONE_TO_ONE_NAT",
                nat_i_p=address_value,
            ),
        )
        wait_for_zonal_op(project, zone, op.name, timeout=180)
        result["tests"]["associate_to_a"] = {"passed": True, "association_id": "access-config:External NAT"}

        a_obj = instances_c.get(project=project, zone=zone, instance=inst_a)
        a_pub = first_external_ip(a_obj)
        result["tests"]["verify_on_a"] = {"passed": a_pub == address_value, "public_ip": a_pub}

        # reassociate_to_b — delete on A, add on B; time both ops.
        t0 = time.time()
        op = instances_c.delete_access_config(
            project=project,
            zone=zone,
            instance=inst_a,
            access_config="External NAT",
            network_interface=nic_name,
        )
        wait_for_zonal_op(project, zone, op.name, timeout=180)

        b_obj = instances_c.get(project=project, zone=zone, instance=inst_b)
        b_nic = b_obj.network_interfaces[0].name or "nic0"
        if b_obj.network_interfaces[0].access_configs:
            ac_name = b_obj.network_interfaces[0].access_configs[0].name
            op = instances_c.delete_access_config(
                project=project,
                zone=zone,
                instance=inst_b,
                access_config=ac_name,
                network_interface=b_nic,
            )
            wait_for_zonal_op(project, zone, op.name, timeout=180)
        op = instances_c.add_access_config(
            project=project,
            zone=zone,
            instance=inst_b,
            network_interface=b_nic,
            access_config_resource=compute_v1.AccessConfig(
                name="External NAT",
                type_="ONE_TO_ONE_NAT",
                nat_i_p=address_value,
            ),
        )
        wait_for_zonal_op(project, zone, op.name, timeout=180)
        switch_seconds = round(time.time() - t0, 1)
        result["tests"]["reassociate_to_b"] = {
            "passed": switch_seconds <= args.max_switch_seconds,
            "switch_seconds": switch_seconds,
        }

        b_obj = instances_c.get(project=project, zone=zone, instance=inst_b)
        b_pub = first_external_ip(b_obj)
        result["tests"]["verify_on_b"] = {"passed": b_pub == address_value, "public_ip": b_pub}

        a_obj2 = instances_c.get(project=project, zone=zone, instance=inst_a)
        a_pub2 = first_external_ip(a_obj2)
        result["tests"]["verify_not_on_a"] = {"passed": a_pub2 != address_value, "public_ip": a_pub2}

        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        # Cleanup ordering matters here: the static external address is held
        # by whichever instance currently owns its accessConfig. Delete order
        # MUST be instances first (releases the address) → address → subnet
        # → network. Default reversed() over the tracker list runs in
        # reverse-append order: address → instance_b → instance_a → subnet
        # → network, which leaks the address (deletion fails with non-
        # retryable BadRequest while the address is in use).
        priority = {"instance": 0, "address": 1, "subnet": 2, "network": 3}
        for kind, n in sorted(cleanup, key=lambda kv: priority.get(kv[0], 99)):
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
                elif kind == "address":
                    delete_with_retry(
                        lambda nn=n: _wait_region_op(
                            project,
                            args.region,
                            addresses_c.delete(project=project, region=args.region, address=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"address {n}",
                    )
                elif kind == "subnet":
                    delete_with_retry(
                        lambda nn=n: _wait_region_op(
                            project,
                            args.region,
                            subnets_c.delete(project=project, region=args.region, subnetwork=nn).name,
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

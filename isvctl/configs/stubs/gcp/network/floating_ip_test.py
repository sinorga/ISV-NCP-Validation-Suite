#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test GCP floating (static external) IP: allocate, associate, reassociate.

GCP's equivalent of an AWS Elastic IP is a regional static external IP
(``compute_v1.AddressesClient``). Reassociation requires deleting the
current VM's ``access_config`` and inserting a new one pointing at the
reserved address — GCP has no single-call atomic reassign like AWS
``AssociateAddress(AllowReassociation=True)``. Per docs/gcp.yaml this
typically takes ~12s, which exceeds the oracle's default 10s threshold;
the provider config overrides ``max_switch_seconds`` to 20.

Subtests (match oracle schema):
  - allocate_eip         : insert a static regional address, capture public IP
  - associate_to_a       : add accessConfig(nat_ip=address) to instance A
  - verify_on_a          : describe A → external IP == address
  - reassociate_to_b     : detach from A, attach to B, measure elapsed
  - verify_on_b          : describe B → external IP == address
  - verify_not_on_a      : describe A → external IP != address

Usage:
    python floating_ip_test.py --region asia-east1-a --cidr 10.92.0.0/16 \\
        --max-switch-seconds 20

Output JSON matches the oracle floating_ip schema.
"""

import argparse
import ipaddress
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import (
    get_instance_external_ip,
    resolve_project,
    zone_to_region,
)
from common.errors import handle_gcp_errors
from common.vpc import (
    create_subnet,
    create_vpc,
    delete_subnet,
    delete_vpc,
    wait_operation,
)
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1

_IMAGE_PROJECT = "ubuntu-os-cloud"
_IMAGE_FAMILY = "ubuntu-2204-lts"
_INSTANCE_TYPE = "e2-small"
_ACCESS_CONFIG_NAME = "External NAT"


def _subnet_cidr(vpc_cidr: str) -> str:
    net = ipaddress.ip_network(vpc_cidr, strict=False)
    base = str(net.network_address).split(".")
    return f"{base[0]}.{base[1]}.1.0/24"


def _build_instance(
    name: str,
    zone: str,
    subnet_self_link: str,
    source_image: str,
) -> compute_v1.Instance:
    disk = compute_v1.AttachedDisk()
    disk.boot = True
    disk.auto_delete = True
    init = compute_v1.AttachedDiskInitializeParams()
    init.source_image = source_image
    init.disk_size_gb = 20
    init.disk_type = f"zones/{zone}/diskTypes/pd-balanced"
    disk.initialize_params = init

    access = compute_v1.AccessConfig()
    access.type_ = "ONE_TO_ONE_NAT"
    access.name = _ACCESS_CONFIG_NAME
    access.network_tier = "PREMIUM"

    nic = compute_v1.NetworkInterface()
    nic.subnetwork = subnet_self_link
    nic.access_configs = [access]

    instance = compute_v1.Instance()
    instance.name = name
    instance.machine_type = f"zones/{zone}/machineTypes/{_INSTANCE_TYPE}"
    instance.disks = [disk]
    instance.network_interfaces = [nic]
    instance.labels = {"created-by": "isvtest", "purpose": "floating-ip"}
    return instance


def _attach_static_ip(
    instances_client: compute_v1.InstancesClient,
    project: str,
    zone: str,
    instance_name: str,
    static_ip: str,
) -> None:
    """Replace the instance's access_config with one pointing at static_ip.

    GCP requires: delete the current ephemeral access_config, then insert
    a new one with nat_i_p set. Order matters — inserting first errors
    with "access config already exists".
    """
    try:
        op = instances_client.delete_access_config(
            project=project,
            zone=zone,
            instance=instance_name,
            access_config=_ACCESS_CONFIG_NAME,
            network_interface="nic0",
        )
        wait_operation(op, timeout=120)
    except gax_exc.NotFound:
        # Already detached, fine.
        pass

    new_access = compute_v1.AccessConfig()
    new_access.type_ = "ONE_TO_ONE_NAT"
    new_access.name = _ACCESS_CONFIG_NAME
    new_access.nat_i_p = static_ip
    new_access.network_tier = "PREMIUM"

    op = instances_client.add_access_config(
        project=project,
        zone=zone,
        instance=instance_name,
        network_interface="nic0",
        access_config_resource=new_access,
    )
    wait_operation(op, timeout=120)


def _detach_static_ip(
    instances_client: compute_v1.InstancesClient,
    project: str,
    zone: str,
    instance_name: str,
) -> None:
    """Delete the instance's access_config (strips the external IP)."""
    try:
        op = instances_client.delete_access_config(
            project=project,
            zone=zone,
            instance=instance_name,
            access_config=_ACCESS_CONFIG_NAME,
            network_interface="nic0",
        )
        wait_operation(op, timeout=120)
    except gax_exc.NotFound:
        pass


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Test GCP floating IP switch")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--cidr", default="10.92.0.0/16")
    parser.add_argument("--max-switch-seconds", type=int, default=20)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = args.region
    region = zone_to_region(zone)

    networks_client = compute_v1.NetworksClient()
    subnets_client = compute_v1.SubnetworksClient()
    instances_client = compute_v1.InstancesClient()
    images_client = compute_v1.ImagesClient()
    addresses_client = compute_v1.AddressesClient()

    suffix = str(uuid.uuid4())[:8]
    vpc_name = f"isv-fip-vpc-{suffix}"
    subnet_name = f"isv-fip-sn-{suffix}"
    instance_a = f"isv-fip-a-{suffix}"
    instance_b = f"isv-fip-b-{suffix}"
    address_name = f"isv-fip-addr-{suffix}"

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "status": "failed",
        "tests": {},
    }

    created = {
        "vpc": False,
        "subnet": False,
        "instance_a": False,
        "instance_b": False,
        "address": False,
    }

    try:
        # Setup VPC + subnet + 2 VMs.
        if not create_vpc(networks_client, project, vpc_name)["passed"]:
            raise RuntimeError("VPC create failed")
        created["vpc"] = True
        net = networks_client.get(project=project, network=vpc_name)

        sn = create_subnet(subnets_client, project, region, subnet_name, net.self_link, _subnet_cidr(args.cidr))
        if not sn["passed"]:
            raise RuntimeError(f"Subnet create failed: {sn.get('error')}")
        created["subnet"] = True
        subnet_self_link = f"projects/{project}/regions/{region}/subnetworks/{subnet_name}"

        image = images_client.get_from_family(project=_IMAGE_PROJECT, family=_IMAGE_FAMILY)
        source_image = f"projects/{_IMAGE_PROJECT}/global/images/{image.name}"

        for name, flag in [(instance_a, "instance_a"), (instance_b, "instance_b")]:
            inst = _build_instance(name, zone, subnet_self_link, source_image)
            op = instances_client.insert(project=project, zone=zone, instance_resource=inst)
            op.result(timeout=600)
            created[flag] = True

        # ── Test 1: Allocate static address ───────────────────────────
        alloc: dict[str, Any] = {"passed": False}
        try:
            addr = compute_v1.Address()
            addr.name = address_name
            addr.address_type = "EXTERNAL"
            addr.network_tier = "PREMIUM"
            op = addresses_client.insert(project=project, region=region, address_resource=addr)
            wait_operation(op, timeout=120)
            created["address"] = True

            got = addresses_client.get(project=project, region=region, address=address_name)
            alloc["passed"] = True
            alloc["allocation_id"] = address_name
            alloc["public_ip"] = got.address
            alloc["message"] = f"Allocated static IP {got.address}"
            static_ip = got.address
        except Exception as e:
            alloc["error"] = str(e)
            static_ip = None
        result["tests"]["allocate_eip"] = alloc

        if not static_ip:
            raise RuntimeError("allocate_eip failed")

        # ── Test 2: Associate with instance A ─────────────────────────
        assoc_a: dict[str, Any] = {"passed": False}
        try:
            _attach_static_ip(instances_client, project, zone, instance_a, static_ip)
            assoc_a["passed"] = True
            assoc_a["association_id"] = f"{instance_a}-access"
            assoc_a["message"] = f"Associated {static_ip} with {instance_a}"
        except Exception as e:
            assoc_a["error"] = str(e)
            raise
        result["tests"]["associate_to_a"] = assoc_a

        # ── Test 3: Verify on A ───────────────────────────────────────
        verify_a: dict[str, Any] = {"passed": False}
        desc_a = instances_client.get(project=project, zone=zone, instance=instance_a)
        a_ip = get_instance_external_ip(desc_a)
        if a_ip == static_ip:
            verify_a["passed"] = True
            verify_a["public_ip"] = a_ip
            verify_a["message"] = f"{static_ip} confirmed on {instance_a}"
        else:
            verify_a["error"] = f"Expected {static_ip}, got {a_ip}"
        result["tests"]["verify_on_a"] = verify_a

        # ── Test 4: Reassociate to B (timed) ──────────────────────────
        reassoc: dict[str, Any] = {"passed": False}
        try:
            start_t = time.monotonic()
            _detach_static_ip(instances_client, project, zone, instance_a)
            _attach_static_ip(instances_client, project, zone, instance_b, static_ip)
            elapsed = time.monotonic() - start_t
            reassoc["switch_seconds"] = round(elapsed, 2)
            if elapsed <= args.max_switch_seconds:
                reassoc["passed"] = True
                reassoc["message"] = f"Reassociated in {elapsed:.2f}s (limit {args.max_switch_seconds}s)"
            else:
                reassoc["error"] = f"Switch took {elapsed:.2f}s, limit is {args.max_switch_seconds}s"
        except Exception as e:
            reassoc["error"] = str(e)
        result["tests"]["reassociate_to_b"] = reassoc

        # ── Test 5: Verify on B ───────────────────────────────────────
        verify_b: dict[str, Any] = {"passed": False}
        desc_b = instances_client.get(project=project, zone=zone, instance=instance_b)
        b_ip = get_instance_external_ip(desc_b)
        if b_ip == static_ip:
            verify_b["passed"] = True
            verify_b["public_ip"] = b_ip
            verify_b["message"] = f"{static_ip} confirmed on {instance_b}"
        else:
            verify_b["error"] = f"Expected {static_ip}, got {b_ip}"
        result["tests"]["verify_on_b"] = verify_b

        # ── Test 6: Verify not on A ───────────────────────────────────
        verify_not_a: dict[str, Any] = {"passed": False}
        desc_a_after = instances_client.get(project=project, zone=zone, instance=instance_a)
        a_ip_after = get_instance_external_ip(desc_a_after)
        # A should either have a different IP or no external IP at all.
        if a_ip_after != static_ip:
            verify_not_a["passed"] = True
            verify_not_a["message"] = f"{instance_a} external IP no longer {static_ip} (now {a_ip_after})"
        else:
            verify_not_a["error"] = f"{instance_a} still has {static_ip}"
        result["tests"]["verify_not_on_a"] = verify_not_a

        all_passed = all(t.get("passed", False) for t in result["tests"].values())
        result["success"] = all_passed
        result["status"] = "passed" if all_passed else "failed"

    except Exception as e:
        result.setdefault("error", str(e))
    finally:
        # Cleanup: must detach addresses from instances before releasing.
        for inst_name, flag in [(instance_a, "instance_a"), (instance_b, "instance_b")]:
            if created[flag]:
                try:
                    _detach_static_ip(instances_client, project, zone, inst_name)
                except gax_exc.GoogleAPIError:
                    pass
                try:
                    op = instances_client.delete(project=project, zone=zone, instance=inst_name)
                    op.result(timeout=300)
                except gax_exc.GoogleAPIError:
                    pass
        if created["address"]:
            try:
                op = addresses_client.delete(project=project, region=region, address=address_name)
                op.result(timeout=120)
            except gax_exc.GoogleAPIError:
                pass
        if created["subnet"]:
            delete_subnet(subnets_client, project, region, subnet_name)
        if created["vpc"]:
            delete_vpc(networks_client, project, vpc_name)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

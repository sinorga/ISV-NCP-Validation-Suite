#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test that GCP private IPs survive stop/start.

GCP preserves a VM's internal IP across stop (TERMINATED) and start by
default — the IP is bound to the VM, not to the power state. This stub:

  1. Creates a VPC + subnet in the test region
  2. Launches a minimal VM
  3. Records the internal IP
  4. Stops the VM (state: TERMINATED — see docs/gcp.yaml)
  5. Starts it again
  6. Verifies the internal IP is unchanged

Usage:
    python stable_ip_test.py --region asia-east1-a --cidr 10.91.0.0/16

Output JSON matches the oracle stable_ip schema.
"""

import argparse
import ipaddress
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import (
    canonical_state,
    get_instance_internal_ip,
    resolve_project,
    zone_to_region,
)
from common.errors import handle_gcp_errors
from common.vpc import (
    create_subnet,
    create_vpc,
    delete_subnet,
    delete_vpc,
)
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1

_IMAGE_PROJECT = "ubuntu-os-cloud"
_IMAGE_FAMILY = "ubuntu-2204-lts"
_INSTANCE_TYPE = "e2-small"


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

    nic = compute_v1.NetworkInterface()
    nic.subnetwork = subnet_self_link
    # No access_configs → no external IP; the test only cares about internal IP.

    instance = compute_v1.Instance()
    instance.name = name
    instance.machine_type = f"zones/{zone}/machineTypes/{_INSTANCE_TYPE}"
    instance.disks = [disk]
    instance.network_interfaces = [nic]
    instance.labels = {"created-by": "isvtest", "purpose": "stable-ip"}
    return instance


def _poll_state(
    instances_client: compute_v1.InstancesClient,
    project: str,
    zone: str,
    name: str,
    expected: str,
    timeout_s: int = 300,
) -> str:
    """Poll instance.status until it matches ``expected`` or times out.

    ``expected`` uses the canonical oracle state name (``running`` /
    ``stopped``). Translates to GCP status via canonical_state before
    comparing so the loop works on both sides of the stop/start.
    """
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        inst = instances_client.get(project=project, zone=zone, instance=name)
        if canonical_state(inst.status) == expected:
            return inst.status
        time.sleep(3)
    raise RuntimeError(f"Timed out waiting for {name} to reach {expected}")


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Test GCP stable private IP")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--cidr", default="10.91.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = args.region
    region = zone_to_region(zone)

    networks_client = compute_v1.NetworksClient()
    subnets_client = compute_v1.SubnetworksClient()
    instances_client = compute_v1.InstancesClient()
    images_client = compute_v1.ImagesClient()

    suffix = str(uuid.uuid4())[:8]
    vpc_name = f"isv-stable-vpc-{suffix}"
    subnet_name = f"isv-stable-sn-{suffix}"
    instance_name = f"isv-stable-inst-{suffix}"

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "status": "failed",
        "tests": {},
    }

    created = {"vpc": False, "subnet": False, "instance": False}

    try:
        # Setup VPC + subnet.
        vpc_result = create_vpc(networks_client, project, vpc_name)
        if not vpc_result["passed"]:
            result["tests"]["setup_vpc"] = {"passed": False, "error": vpc_result.get("error")}
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

        # ── Test 1: Create instance ────────────────────────────────────
        try:
            instance = _build_instance(instance_name, zone, subnet_self_link, source_image)
            op = instances_client.insert(project=project, zone=zone, instance_resource=instance)
            op.result(timeout=600)
            created["instance"] = True
            result["tests"]["create_instance"] = {
                "passed": True,
                "instance_id": instance_name,
                "message": f"Launched {instance_name}",
            }
        except Exception as e:
            result["tests"]["create_instance"] = {"passed": False, "error": str(e)}
            raise

        # ── Test 2: Record IP ──────────────────────────────────────────
        desc = instances_client.get(project=project, zone=zone, instance=instance_name)
        original_ip = get_instance_internal_ip(desc)
        if original_ip:
            result["tests"]["record_ip"] = {"passed": True, "private_ip": original_ip}
        else:
            result["tests"]["record_ip"] = {"passed": False, "error": "No internal IP assigned"}
            raise RuntimeError("No internal IP")

        # ── Test 3: Stop (GCP state: TERMINATED) ───────────────────────
        try:
            op = instances_client.stop(project=project, zone=zone, instance=instance_name)
            op.result(timeout=300)
            _poll_state(instances_client, project, zone, instance_name, "stopped", timeout_s=300)
            result["tests"]["stop_instance"] = {"passed": True, "message": "Instance TERMINATED"}
        except Exception as e:
            result["tests"]["stop_instance"] = {"passed": False, "error": str(e)}
            raise

        # ── Test 4: Start ──────────────────────────────────────────────
        try:
            op = instances_client.start(project=project, zone=zone, instance=instance_name)
            op.result(timeout=300)
            _poll_state(instances_client, project, zone, instance_name, "running", timeout_s=300)
            result["tests"]["start_instance"] = {"passed": True, "message": "Instance RUNNING"}
        except Exception as e:
            result["tests"]["start_instance"] = {"passed": False, "error": str(e)}
            raise

        # ── Test 5: Verify IP unchanged ────────────────────────────────
        desc_after = instances_client.get(project=project, zone=zone, instance=instance_name)
        current_ip = get_instance_internal_ip(desc_after)
        if current_ip == original_ip:
            result["tests"]["ip_unchanged"] = {
                "passed": True,
                "ip_before": original_ip,
                "ip_after": current_ip,
                "message": f"Private IP {current_ip} unchanged across stop/start",
            }
        else:
            result["tests"]["ip_unchanged"] = {
                "passed": False,
                "ip_before": original_ip,
                "ip_after": current_ip,
                "error": f"IP changed: {original_ip} → {current_ip}",
            }

        all_passed = all(t.get("passed", False) for t in result["tests"].values())
        result["success"] = all_passed
        result["status"] = "passed" if all_passed else "failed"

    except Exception as e:
        # Most subtests already have their specific error captured; only
        # add a top-level error if something upstream exploded.
        result.setdefault("error", str(e))
    finally:
        if created["instance"]:
            try:
                op = instances_client.delete(project=project, zone=zone, instance=instance_name)
                op.result(timeout=300)
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

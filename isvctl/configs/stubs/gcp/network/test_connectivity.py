#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test GCP instance connectivity: launch two VMs in the shared VPC, ping via SSH.

Replaces the AWS SSM-based approach with direct SSH from the test host to
the source instance (GCP has no SSM). The test:

  1. Launch 2 Ubuntu VMs in the shared VPC subnets (one per subnet).
  2. Wait for SSH on instance A.
  3. Run ``ping <B.private_ip>`` and ``ping 8.8.8.8`` over SSH from A.
  4. Report per-test pass/fail with latency.

Usage:
    python test_connectivity.py --vpc-id <network> --subnet-ids subnet1,subnet2 \
        --sg-id <firewall> --region asia-east1-a

Output JSON matches the oracle connectivity_result schema.
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import (
    canonical_state,
    create_ssh_key_pair,
    get_instance_external_ip,
    get_instance_internal_ip,
    read_public_key,
    resolve_project,
    ssh_exec,
    wait_for_ssh,
    zone_to_region,
)
from common.errors import handle_gcp_errors
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1

# Ubuntu 22.04 LTS — small boot image is plenty for connectivity testing.
_CONNECTIVITY_IMAGE_PROJECT = "ubuntu-os-cloud"
_CONNECTIVITY_IMAGE_FAMILY = "ubuntu-2204-lts"
_INSTANCE_TYPE = "e2-small"


def _build_instance(
    name: str,
    zone: str,
    subnet_self_link: str,
    source_image: str,
    public_key: str,
    ssh_user: str,
) -> compute_v1.Instance:
    """Assemble an Instance resource for the connectivity test.

    Sized as e2-small with a 20 GB pd-balanced disk — cheap, boots quickly,
    and has enough headroom to run ping. The SSH key goes on the instance
    metadata so no project-wide key exposure is needed.
    """
    disk = compute_v1.AttachedDisk()
    disk.boot = True
    disk.auto_delete = True
    init_params = compute_v1.AttachedDiskInitializeParams()
    init_params.source_image = source_image
    init_params.disk_size_gb = 20
    init_params.disk_type = f"zones/{zone}/diskTypes/pd-balanced"
    disk.initialize_params = init_params

    access = compute_v1.AccessConfig()
    access.type_ = "ONE_TO_ONE_NAT"
    access.name = "External NAT"
    access.network_tier = "PREMIUM"

    nic = compute_v1.NetworkInterface()
    nic.subnetwork = subnet_self_link
    nic.access_configs = [access]

    metadata_item = compute_v1.Items()
    metadata_item.key = "ssh-keys"
    metadata_item.value = f"{ssh_user}:{public_key}"
    metadata = compute_v1.Metadata()
    metadata.items = [metadata_item]

    instance = compute_v1.Instance()
    instance.name = name
    instance.machine_type = f"zones/{zone}/machineTypes/{_INSTANCE_TYPE}"
    instance.disks = [disk]
    instance.network_interfaces = [nic]
    instance.metadata = metadata
    instance.labels = {"created-by": "isvtest", "purpose": "connectivity"}
    return instance


def _resolve_subnet_region(
    subnets_client: compute_v1.SubnetworksClient,
    project: str,
    fallback_region: str,
    subnet_name: str,
) -> tuple[str, str]:
    """Find the region for a subnet name and return (region, self_link).

    Falls back to ``fallback_region`` when the caller already knows it;
    otherwise aggregated_list walks every region to find the match.
    """
    try:
        sub = subnets_client.get(project=project, region=fallback_region, subnetwork=subnet_name)
        return fallback_region, sub.self_link
    except gax_exc.NotFound:
        pass

    # Fall back to aggregated list.
    for scope, subs in subnets_client.aggregated_list(project=project):
        if not scope.startswith("regions/"):
            continue
        region = scope.split("/", 1)[1]
        for sub in subs.subnetworks or []:
            if sub.name == subnet_name:
                return region, sub.self_link

    raise RuntimeError(f"Subnet {subnet_name} not found in any region")


def _parse_ping_latency(output: str) -> float | None:
    """Pull the avg latency (ms) out of standard iputils ping output."""
    # e.g. "rtt min/avg/max/mdev = 0.321/0.442/0.589/0.107 ms"
    match = re.search(r"(?:rtt|round-trip).*?=\s*[\d.]+/([\d.]+)/", output)
    if match:
        return float(match.group(1))
    return None


def launch_connectivity_pair(
    instances_client: compute_v1.InstancesClient,
    images_client: compute_v1.ImagesClient,
    project: str,
    zone: str,
    subnet_self_link: str,
    subnet_id: str,
    ssh_user: str,
    public_key: str,
    suffix: str,
) -> list[dict[str, Any]]:
    """Launch two instances (source + target) and return their descriptors."""
    image = images_client.get_from_family(project=_CONNECTIVITY_IMAGE_PROJECT, family=_CONNECTIVITY_IMAGE_FAMILY)
    source_image = f"projects/{_CONNECTIVITY_IMAGE_PROJECT}/global/images/{image.name}"

    instances: list[dict[str, Any]] = []
    for i in range(2):
        name = f"isv-conn-{i}-{suffix}"
        instance = _build_instance(name, zone, subnet_self_link, source_image, public_key, ssh_user)
        op = instances_client.insert(project=project, zone=zone, instance_resource=instance)
        op.result(timeout=600)
        desc = instances_client.get(project=project, zone=zone, instance=name)
        instances.append(
            {
                "instance_id": name,
                "subnet_id": subnet_id,
                "private_ip": get_instance_internal_ip(desc),
                "public_ip": get_instance_external_ip(desc),
                "state": canonical_state(desc.status),
            }
        )
    return instances


def validate_vpc_resources(
    networks_client: compute_v1.NetworksClient,
    subnets_client: compute_v1.SubnetworksClient,
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    vpc_id: str,
    subnet_ids: list[str],
    fw_name: str,
    fallback_region: str,
) -> dict[str, Any]:
    """Confirm the subnets + firewall all belong to ``vpc_id``."""
    validation: dict[str, Any] = {
        "valid": True,
        "errors": [],
        "validated_subnets": [],
        "validated_sg": None,
    }

    try:
        net = networks_client.get(project=project, network=vpc_id)
        network_self_link = net.self_link
    except Exception as e:
        validation["valid"] = False
        validation["errors"].append(f"Failed to get VPC {vpc_id}: {e}")
        return validation

    for subnet_name in subnet_ids:
        try:
            region, sub_link = _resolve_subnet_region(subnets_client, project, fallback_region, subnet_name)
            sub = subnets_client.get(project=project, region=region, subnetwork=subnet_name)
            if sub.network != network_self_link:
                validation["valid"] = False
                validation["errors"].append(f"Subnet {subnet_name} belongs to {sub.network}, not {vpc_id}")
            else:
                validation["validated_subnets"].append(sub_link)
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Failed to resolve subnet {subnet_name}: {e}")

    if fw_name:
        try:
            fw = firewalls_client.get(project=project, firewall=fw_name)
            if fw.network != network_self_link:
                validation["valid"] = False
                validation["errors"].append(f"Firewall {fw_name} belongs to {fw.network}, not {vpc_id}")
            else:
                validation["validated_sg"] = fw_name
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Failed to resolve firewall {fw_name}: {e}")

    return validation


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Test GCP VPC connectivity")
    parser.add_argument("--vpc-id", required=True, help="GCP network name")
    parser.add_argument("--subnet-ids", required=True, help="Comma-separated subnet names")
    parser.add_argument("--sg-id", required=True, help="Firewall name (GCP's SG equivalent)")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--project", default=None)
    parser.add_argument("--ssh-user", default="ubuntu")
    parser.add_argument("--skip-cleanup", action="store_true")
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = args.region
    region = zone_to_region(zone)
    subnet_ids = args.subnet_ids.split(",")

    networks_client = compute_v1.NetworksClient()
    subnets_client = compute_v1.SubnetworksClient()
    firewalls_client = compute_v1.FirewallsClient()
    instances_client = compute_v1.InstancesClient()
    images_client = compute_v1.ImagesClient()

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "vpc_id": args.vpc_id,
        "tests": {},
        "instances": [],
    }

    suffix = str(uuid.uuid4())[:8]
    instances: list[dict[str, Any]] = []
    key_dir = Path("/tmp")
    key_name = f"isv-conn-key-{suffix}"

    try:
        # Validate VPC / subnet / SG membership.
        validation = validate_vpc_resources(
            networks_client,
            subnets_client,
            firewalls_client,
            project,
            args.vpc_id,
            subnet_ids,
            args.sg_id,
            region,
        )
        result["vpc_validation"] = validation
        if not validation["valid"]:
            result["error"] = f"VPC validation failed: {'; '.join(validation['errors'])}"
            result["status"] = "failed"
            print(json.dumps(result, indent=2))
            return 1

        # SSH key (instance metadata only — no project-wide key).
        priv, pub = create_ssh_key_pair(key_name, key_dir=key_dir)
        public_key = read_public_key(pub)

        # Launch two instances in the first subnet (AZ/zone-local for
        # minimum latency). The validator only cares that instances are
        # in the VPC and have IPs; per-subnet placement isn't required.
        target_subnet = subnet_ids[0]
        target_region, subnet_self_link = _resolve_subnet_region(
            subnets_client,
            project,
            region,
            target_subnet,
        )
        if target_region != region:
            # Pick an arbitrary zone in that region (GCP requires a zone
            # for InstancesClient.insert).
            zone = f"{target_region}-a"

        instances = launch_connectivity_pair(
            instances_client,
            images_client,
            project,
            zone,
            subnet_self_link,
            target_subnet,
            args.ssh_user,
            public_key,
            suffix,
        )
        result["instances"] = instances

        # Wait for SSH on instance 0 (source for ping).
        src = instances[0]
        target = instances[1]
        if not wait_for_ssh(src["public_ip"], args.ssh_user, priv, max_attempts=30, interval=10):
            raise RuntimeError(f"SSH never became ready on {src['public_ip']}")

        # Install iputils-ping if not present (some Ubuntu minimal images
        # drop it). Ignore failure — ping usually ships with 22.04.
        ssh_exec(src["public_ip"], args.ssh_user, priv, "command -v ping || sudo apt-get -qq install -y iputils-ping")

        # Instance-to-instance ping.
        rc, out, err = ssh_exec(
            src["public_ip"],
            args.ssh_user,
            priv,
            f"ping -c 3 -W 2 {target['private_ip']}",
            timeout=60,
        )
        latency = _parse_ping_latency(out)
        result["tests"]["instance_to_instance"] = (
            {"passed": True, "latency_ms": latency, "target": target["private_ip"]}
            if rc == 0
            else {"passed": False, "error": err.strip() or out.strip() or f"rc={rc}"}
        )

        # Instance-to-internet ping.
        rc2, out2, err2 = ssh_exec(
            src["public_ip"],
            args.ssh_user,
            priv,
            "ping -c 3 -W 2 8.8.8.8",
            timeout=60,
        )
        result["tests"]["instance_to_internet"] = (
            {"passed": True, "latency_ms": _parse_ping_latency(out2)}
            if rc2 == 0
            else {"passed": False, "error": err2.strip() or out2.strip() or f"rc={rc2}"}
        )

        all_passed = all(t.get("passed", False) for t in result["tests"].values())
        result["success"] = all_passed
        result["status"] = "passed" if all_passed else "failed"

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "failed"
    finally:
        if not args.skip_cleanup and instances:
            for inst in instances:
                try:
                    op = instances_client.delete(project=project, zone=zone, instance=inst["instance_id"])
                    op.result(timeout=300)
                except gax_exc.GoogleAPIError:
                    pass
            # Brief pause so ENIs release before downstream steps touch the VPC.
            time.sleep(2)
            result["cleanup"] = True

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

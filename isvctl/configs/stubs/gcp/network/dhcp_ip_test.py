#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Launch a short-lived GCP VM in the shared VPC and emit SSH details.

Downstream, ``DhcpIpManagementCheck`` SSHes into the instance to inspect
the DHCP lease, the interface IPs, and /etc/resolv.conf. This stub's job
is just to provision the VM and report the SSH triple (host + user + key).

We reuse the shared VPC's subnet + firewall (which already allows SSH
from 0.0.0.0/0), so the only new resource this stub creates is the VM
itself.

Usage:
    python dhcp_ip_test.py --vpc-id <network> --subnet-id <subnet> \\
        --sg-id <firewall> --region asia-east1-a

Output JSON matches the oracle dhcp_ip schema.
"""

import argparse
import json
import os
import sys
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
    wait_for_ssh,
    zone_to_region,
)
from common.errors import classify_gcp_error, handle_gcp_errors
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1

_IMAGE_PROJECT = "ubuntu-os-cloud"
_IMAGE_FAMILY = "ubuntu-2204-lts"
_INSTANCE_TYPE = "e2-small"


def _build_instance(
    name: str,
    zone: str,
    subnet_self_link: str,
    source_image: str,
    public_key: str,
    ssh_user: str,
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
    access.name = "External NAT"
    access.network_tier = "PREMIUM"

    nic = compute_v1.NetworkInterface()
    nic.subnetwork = subnet_self_link
    nic.access_configs = [access]

    meta = compute_v1.Metadata()
    meta_item = compute_v1.Items()
    meta_item.key = "ssh-keys"
    meta_item.value = f"{ssh_user}:{public_key}"
    meta.items = [meta_item]

    instance = compute_v1.Instance()
    instance.name = name
    instance.machine_type = f"zones/{zone}/machineTypes/{_INSTANCE_TYPE}"
    instance.disks = [disk]
    instance.network_interfaces = [nic]
    instance.metadata = meta
    instance.labels = {"created-by": "isvtest", "purpose": "dhcp-ip"}
    return instance


def _resolve_subnet_region(
    subnets_client: compute_v1.SubnetworksClient,
    project: str,
    fallback_region: str,
    subnet_name: str,
) -> str:
    """Return the region the named subnet lives in (aggregated fallback)."""
    try:
        subnets_client.get(project=project, region=fallback_region, subnetwork=subnet_name)
        return fallback_region
    except gax_exc.NotFound:
        pass

    for scope, subs in subnets_client.aggregated_list(project=project):
        if not scope.startswith("regions/"):
            continue
        region = scope.split("/", 1)[1]
        for sub in subs.subnetworks or []:
            if sub.name == subnet_name:
                return region
    raise RuntimeError(f"Subnet {subnet_name} not found in any region")


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Launch VM for DHCP/IP management check")
    parser.add_argument("--vpc-id", required=True, help="GCP network name (unused — subnet carries the link)")
    parser.add_argument("--subnet-id", required=True, help="Subnet name to launch the VM into")
    parser.add_argument("--sg-id", required=True, help="Firewall name (unused — rule exists on the VPC)")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--project", default=None)
    parser.add_argument("--ssh-user", default="ubuntu")
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = args.region
    region = zone_to_region(zone)

    instances_client = compute_v1.InstancesClient()
    images_client = compute_v1.ImagesClient()
    subnets_client = compute_v1.SubnetworksClient()

    suffix = str(uuid.uuid4())[:8]
    key_name = f"isv-dhcp-key-{suffix}"
    instance_name = f"isv-dhcp-{suffix}"

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "test_name": "dhcp_ip",
        "public_ip": None,
        "private_ip": None,
        "key_file": None,
        "key_name": key_name,
        "ssh_user": args.ssh_user,
        "instance_id": None,
    }

    created_instance = False

    try:
        # Locate the subnet's region (for the instance, zone must match).
        subnet_region = _resolve_subnet_region(subnets_client, project, region, args.subnet_id)
        if subnet_region != region:
            # Use the provided zone only if it's in the subnet's region;
            # otherwise pick -a in that region.
            zone_parent = zone_to_region(zone)
            if zone_parent != subnet_region:
                zone = f"{subnet_region}-a"

        subnet_self_link = f"projects/{project}/regions/{subnet_region}/subnetworks/{args.subnet_id}"

        # SSH key pair.
        priv, pub = create_ssh_key_pair(key_name)
        result["key_file"] = priv
        public_key = read_public_key(pub)

        # Image.
        image = images_client.get_from_family(project=_IMAGE_PROJECT, family=_IMAGE_FAMILY)
        source_image = f"projects/{_IMAGE_PROJECT}/global/images/{image.name}"

        # Launch.
        inst = _build_instance(
            instance_name,
            zone,
            subnet_self_link,
            source_image,
            public_key,
            args.ssh_user,
        )
        op = instances_client.insert(project=project, zone=zone, instance_resource=inst)
        op.result(timeout=600)
        created_instance = True

        desc = instances_client.get(project=project, zone=zone, instance=instance_name)
        result["instance_id"] = instance_name
        result["public_ip"] = get_instance_external_ip(desc)
        result["private_ip"] = get_instance_internal_ip(desc)
        result["instance_state"] = canonical_state(desc.status)

        if not result["public_ip"]:
            raise RuntimeError("Instance has no external IP after provisioning")

        # Wait for SSH so downstream DhcpIpManagementCheck can connect immediately.
        if not wait_for_ssh(result["public_ip"], args.ssh_user, priv, max_attempts=30, interval=10):
            raise RuntimeError(f"SSH never became ready on {result['public_ip']}")

        result["success"] = True
    except gax_exc.GoogleAPIError as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    except Exception as e:
        result["error"] = str(e)

    # NOTE: we deliberately do NOT tear down the instance here — the
    # downstream ``DhcpIpManagementCheck`` validation SSHes into it after
    # this stub emits JSON. The shared-VPC teardown step removes the VM
    # at the end of the suite (via describe_instances on the VPC).

    if not result["success"] and created_instance:
        # Clean up if we created the VM but bailed before success.
        try:
            op = instances_client.delete(project=project, zone=zone, instance=instance_name)
            op.result(timeout=300)
        except gax_exc.GoogleAPIError as cleanup_err:
            print(f"  Cleanup warning: {cleanup_err}", file=sys.stderr)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine DHCP/IP management test (DhcpIpManagementCheck).

Divergences from the AWS oracle:

  * No EC2 key pair resource — keypair is generated locally via
    common.compute.generate_ssh_keypair and pushed via instance
    metadata ``ssh-keys``. Track key_created so teardown removes the
    local PEM via the verified-reuse cleanup contract.
  * External IP attached via accessConfigs[].natIP at launch.
  * GCP fully supports the SSH probe pattern — do NOT skip; do NOT
    add the ssh marker exclusion.
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
    first_external_ip,
    first_internal_ip,
    generate_ssh_keypair,
    narrow_region_to_zone,
    poll_instance_state,
    read_ssh_pubkey,
    resolve_project,
    unique_suffix,
    wait_for_zonal_op,
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from common.ssh_utils import wait_for_ssh
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest dhcp_ip — verified-reuse marker"
DEFAULT_IMAGE = "projects/debian-cloud/global/images/family/debian-12"
SSH_USER = "isvtest"


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine DHCP/IP probe")
    parser.add_argument("--vpc-id", required=True)
    parser.add_argument("--subnet-id", required=True)
    parser.add_argument("--sg-id", required=True)
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = narrow_region_to_zone(args.region)
    instances_c = compute_v1.InstancesClient()

    key_stem = unique_suffix("dhcp-key")
    name = unique_suffix("isv-dhcp")
    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "instance_id": name,
        "key_name": key_stem,
        "key_file": None,
        "key_created": False,
        "ssh_user": SSH_USER,
        "private_ip": None,
        "public_ip": None,
        "region": args.region,
    }

    key_path: str | None = None
    instance_created = False
    try:
        kp = generate_ssh_keypair(key_stem)
        if isinstance(kp, tuple):
            key_path, key_created = kp[0], bool(kp[1])
        else:
            key_path, key_created = kp, True
        result["key_created"] = key_created
        result["key_file"] = key_path
        pubkey = read_ssh_pubkey(key_path)

        inst = compute_v1.Instance(
            name=name,
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
                    network=f"projects/{project}/global/networks/{args.vpc_id}",
                    subnetwork=f"projects/{project}/regions/{args.region}/subnetworks/{args.subnet_id}",
                    access_configs=[compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT")],
                )
            ],
            metadata=compute_v1.Metadata(
                items=[
                    compute_v1.Items(key="ssh-keys", value=f"{SSH_USER}:{pubkey}"),
                ]
            ),
            service_accounts=[],
        )
        op = instances_c.insert(project=project, zone=zone, instance_resource=inst)
        instance_created = True
        wait_for_zonal_op(project, zone, op.name, timeout=600)
        poll_instance_state(project, zone, name, target_canonical="running", timeout=600)
        obj = instances_c.get(project=project, zone=zone, instance=name)
        result["private_ip"] = first_internal_ip(obj)
        public_ip = first_external_ip(obj)
        result["public_ip"] = public_ip
        if not public_ip:
            raise RuntimeError("instance has no external IP — accessConfigs missing")
        wait_for_ssh(public_ip, SSH_USER, key_path, max_attempts=30, interval=10)
        result["success"] = True
    except gax.GoogleAPICallError as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        if instance_created:
            delete_with_retry(
                lambda: wait_for_zonal_op(
                    project,
                    zone,
                    instances_c.delete(project=project, zone=zone, instance=name).name,
                    timeout=300,
                ),
                resource_desc=f"instance {name}",
            )
        # Local key file is forwarded to teardown via key_file/key_name so we
        # do NOT delete it here in the success path. On error, also keep the
        # file so the operator can debug; teardown's --key-created gate
        # cleans up.

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

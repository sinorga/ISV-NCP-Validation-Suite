#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine connectivity test (NetworkConnectivityCheck).

Divergences from the AWS oracle:

  * Replace SSM with SSH (paramiko-based via providers/gcp/scripts/common/ssh_utils).
  * IAM/SSM role is a no-op (Compute Engine attaches service accounts at
    launch; no SSM-equivalent role is required). Emit ``iam_profile=None``.
  * VPC validation: subnetwork.network and firewall.network must equal
    the supplied network selfLink (exact tail match — vendor-API contract).
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

from common.compute import (
    first_external_ip,
    first_internal_ip,
    generate_ssh_keypair,
    narrow_region_to_zone,
    poll_instance_state,
    read_ssh_pubkey,
    resolve_project,
    short_name,
    unique_suffix,
    wait_for_zonal_op,
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from common.ssh_utils import ssh_run, wait_for_ssh
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest connectivity — verified-reuse marker"
DEFAULT_IMAGE = "projects/debian-cloud/global/images/family/debian-12"
SSH_USER = "isvtest"
# Must match create_vpc.py PROVENANCE_TAG — create_vpc emits a firewall
# whose `target_tags` is gated on this tag, so probe VMs MUST carry it
# for the SSH-22 ingress rule to apply. Untagged VMs are firewall-blocked
# and wait_for_ssh times out (run 28492c99 evidence).


def _validate_vpc(
    project: str,
    network: str,
    subnet_ids: list[str],
    sg_id: str,
    region: str,
) -> dict[str, Any]:
    errors: list[str] = []
    validated_subnets: list[str] = []
    subnets_client = compute_v1.SubnetworksClient()
    firewalls = compute_v1.FirewallsClient()
    for sn in subnet_ids:
        try:
            sub = subnets_client.get(project=project, region=region, subnetwork=sn)
            if short_name(sub.network) != network:
                errors.append(f"subnet {sn} belongs to network {short_name(sub.network)} not {network}")
            else:
                validated_subnets.append(sn)
        except gax.NotFound:
            errors.append(f"subnet {sn} not found")
    validated_sg = False
    try:
        fw = firewalls.get(project=project, firewall=sg_id)
        if short_name(fw.network) == network:
            validated_sg = True
        else:
            errors.append(f"firewall {sg_id} belongs to network {short_name(fw.network)} not {network}")
    except gax.NotFound:
        errors.append(f"firewall {sg_id} not found")
    return {
        "valid": not errors,
        "errors": errors,
        "validated_subnets": validated_subnets,
        "validated_sg": validated_sg,
    }


def _build_instance(
    project: str,
    zone: str,
    name: str,
    network: str,
    subnet: str,
    region: str,
    pubkey: str,
    public_ip: bool = True,
) -> compute_v1.Instance:
    nic = compute_v1.NetworkInterface(
        network=f"projects/{project}/global/networks/{network}",
        subnetwork=f"projects/{project}/regions/{region}/subnetworks/{subnet}",
    )
    if public_ip:
        nic.access_configs = [
            compute_v1.AccessConfig(name="External NAT", type_="ONE_TO_ONE_NAT"),
        ]
    return compute_v1.Instance(
        name=name,
        description=ISV_DESCRIPTION,
        machine_type=f"zones/{zone}/machineTypes/e2-small",
        tags=compute_v1.Tags(items=["isvtest"]),
        disks=[
            compute_v1.AttachedDisk(
                boot=True,
                auto_delete=True,
                initialize_params=compute_v1.AttachedDiskInitializeParams(
                    source_image=DEFAULT_IMAGE,
                    disk_size_gb=10,
                ),
            ),
        ],
        network_interfaces=[nic],
        metadata=compute_v1.Metadata(
            items=[compute_v1.Items(key="ssh-keys", value=f"{SSH_USER}:{pubkey}")],
        ),
        service_accounts=[],
    )


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine connectivity test")
    parser.add_argument("--vpc-id", required=True)
    parser.add_argument("--subnet-ids", required=True)
    parser.add_argument("--sg-id", required=True)
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    subnet_ids = [s.strip() for s in args.subnet_ids.split(",") if s.strip()]

    instances_client = compute_v1.InstancesClient()
    zone = narrow_region_to_zone(args.region)

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "vpc_id": args.vpc_id,
        "iam_profile": None,
        "tests": {},
        "instances": [],
    }

    # vpc_validation
    val = _validate_vpc(project, args.vpc_id, subnet_ids, args.sg_id, args.region)
    result["vpc_validation"] = val
    if not val["valid"]:
        result["error"] = f"VPC validation failed: {val['errors']}"
        print(json.dumps(result, indent=2))
        return 1

    created_names: list[str] = []
    keypair_paths: list[tuple[str, str]] = []  # (priv, pub)
    cleanup_errors: list[str] = []
    try:
        # Two instances on the same subnet so we can ping.
        keypair = generate_ssh_keypair(unique_suffix("conn-key"))
        if isinstance(keypair, tuple):
            priv_path = keypair[0]
        else:
            priv_path = keypair
        pubkey = read_ssh_pubkey(priv_path)
        keypair_paths.append((priv_path, priv_path + ".pub"))

        instance_metadata: list[dict[str, Any]] = []
        for i in range(2):
            name = unique_suffix(f"isv-conn-{i}")
            inst = _build_instance(project, zone, name, args.vpc_id, subnet_ids[0], args.region, pubkey)
            op = instances_client.insert(project=project, zone=zone, instance_resource=inst)
            created_names.append(name)
            wait_for_zonal_op(project, zone, op.name, timeout=600)
            poll_instance_state(project, zone, name, target_canonical="running", timeout=600)
            inst_read = instances_client.get(project=project, zone=zone, instance=name)
            ipub = first_external_ip(inst_read)
            ipriv = first_internal_ip(inst_read)
            instance_metadata.append(
                {
                    "instance_id": name,
                    "subnet_id": subnet_ids[0],
                    "private_ip": ipriv,
                    "public_ip": ipub,
                    "vpc_id": args.vpc_id,
                }
            )
        result["instances"] = instance_metadata

        # SSH-ready
        a_ip = instance_metadata[0]["public_ip"]
        b_priv = instance_metadata[1]["private_ip"]
        if not a_ip or not wait_for_ssh(a_ip, SSH_USER, priv_path, max_attempts=30, interval=10):
            result["error"] = "ssh did not come up on instance A"
            print(json.dumps(result, indent=2))
            return 1

        # instance_to_instance ping. ssh_run returns (rc, stdout, stderr).
        t0 = time.time()
        rc, _, _ = ssh_run(a_ip, SSH_USER, priv_path, f"ping -c 3 -W 3 {b_priv}")
        latency_ms = round((time.time() - t0) * 1000, 1)
        result["tests"]["instance_to_instance"] = {
            "passed": rc == 0,
            "latency_ms": latency_ms,
        }

        # instance_to_internet
        rc2, _, _ = ssh_run(
            a_ip, SSH_USER, priv_path, "curl -s -m 5 -o /dev/null -w %{http_code} https://www.google.com"
        )
        result["tests"]["instance_to_internet"] = {"passed": rc2 == 0}

        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        for name in created_names:
            ok = delete_with_retry(
                lambda n=name: wait_for_zonal_op(
                    project,
                    zone,
                    instances_client.delete(project=project, zone=zone, instance=n).name,
                    timeout=300,
                ),
                resource_desc=f"instance {name}",
            )
            if not ok:
                cleanup_errors.append(f"instance {name}: delete_with_retry returned False")
        for priv, pub in keypair_paths:
            for p in (priv, pub):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
    result["tests"]["cleanup"] = {"passed": not cleanup_errors, "errors": cleanup_errors}
    result["success"] = result.get("success", False) and not cleanup_errors

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

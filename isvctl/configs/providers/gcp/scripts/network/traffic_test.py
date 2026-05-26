#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine traffic validation (TrafficFlowCheck).

Divergences from the AWS oracle:

  * No IGW resource — emit ``create_igw.passed=True`` with a note about
    Compute Engine's implicit default-internet-gateway.
  * No SSM — use SSH; preserve the JSON key ``ssm_ready`` with
    ``message='ssh-ready'``.
  * Two SGs (allow/deny ICMP) implemented as two firewalls with
    distinct targetTags; tag instances accordingly.
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
    canonical_state,
    first_external_ip,
    first_internal_ip,
    generate_ssh_keypair,
    narrow_region_to_zone,
    poll_instance_state,
    read_ssh_pubkey,
    resolve_project,
    unique_suffix,
    wait_for_global_op,
    wait_for_zonal_op,
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from common.ssh_utils import ssh_run, wait_for_ssh
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest traffic_validation — verified-reuse marker"
DEFAULT_IMAGE = "projects/debian-cloud/global/images/family/debian-12"
SSH_USER = "isvtest"


def _insert_network(project: str, name: str) -> None:
    op = compute_v1.NetworksClient().insert(
        project=project,
        network_resource=compute_v1.Network(
            name=name,
            description=ISV_DESCRIPTION,
            auto_create_subnetworks=False,
        ),
    )
    wait_for_global_op(project, op.name, timeout=300)


def _insert_subnet(project: str, region: str, network: str, name: str, cidr: str) -> None:
    op = compute_v1.SubnetworksClient().insert(
        project=project,
        region=region,
        subnetwork_resource=compute_v1.Subnetwork(
            name=name,
            description=ISV_DESCRIPTION,
            ip_cidr_range=cidr,
            network=f"projects/{project}/global/networks/{network}",
            region=region,
        ),
    )
    compute_v1.RegionOperationsClient().wait(
        project=project,
        region=region,
        operation=op.name,
        timeout=180,
    )


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine traffic validation")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr", default="10.93.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = narrow_region_to_zone(args.region)
    network = unique_suffix("isv-traffic")
    subnet = unique_suffix("isv-tsn")
    sg_allow = unique_suffix("isv-allow")
    sg_deny = unique_suffix("isv-deny")
    sub_cidr = str(next(iter(ipaddress.ip_network(args.cidr).subnets(new_prefix=24))))

    networks = compute_v1.NetworksClient()
    subnets_c = compute_v1.SubnetworksClient()
    firewalls = compute_v1.FirewallsClient()
    instances_c = compute_v1.InstancesClient()

    result: dict[str, Any] = {"success": False, "platform": "network", "tests": {}}
    cleanup: list[tuple[str, str]] = []
    keypair_paths: list[str] = []
    try:
        _insert_network(project, network)
        cleanup.append(("network", network))
        result["tests"]["create_vpc"] = {"passed": True, "vpc_id": network}
        result["tests"]["create_igw"] = {
            "passed": True,
            "message": "default-internet-gateway implicit on Compute Engine",
        }

        _insert_subnet(project, args.region, network, subnet, sub_cidr)
        cleanup.append(("subnet", subnet))
        result["tests"]["network_setup"] = {"passed": True, "subnet_id": subnet}

        result["tests"]["create_iam"] = {
            "passed": True,
            "message": "no-op on Compute Engine — service-account model",
        }

        # Two firewalls: allow-icmp on isv-allow tag, ssh-only on isv-deny tag.
        op = firewalls.insert(
            project=project,
            firewall_resource=compute_v1.Firewall(
                name=sg_allow,
                description=ISV_DESCRIPTION,
                network=f"projects/{project}/global/networks/{network}",
                direction="INGRESS",
                source_ranges=["0.0.0.0/0"],
                allowed=[
                    compute_v1.Allowed(I_p_protocol="tcp", ports=["22"]),
                    compute_v1.Allowed(I_p_protocol="icmp"),
                ],
                target_tags=[sg_allow],
            ),
        )
        wait_for_global_op(project, op.name, timeout=180)
        cleanup.append(("firewall", sg_allow))
        op = firewalls.insert(
            project=project,
            firewall_resource=compute_v1.Firewall(
                name=sg_deny,
                description=ISV_DESCRIPTION,
                network=f"projects/{project}/global/networks/{network}",
                direction="INGRESS",
                source_ranges=["0.0.0.0/0"],
                allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=["22"])],
                target_tags=[sg_deny],
            ),
        )
        wait_for_global_op(project, op.name, timeout=180)
        cleanup.append(("firewall", sg_deny))
        result["tests"]["create_security_groups"] = {
            "passed": True,
            "sg_allow": sg_allow,
            "sg_deny": sg_deny,
        }

        # Three instances: A (allow-tag), B (allow-tag), C (deny-tag).
        keypair = generate_ssh_keypair(unique_suffix("traf-key"))
        priv = keypair[0] if isinstance(keypair, tuple) else keypair
        keypair_paths.extend([priv, priv + ".pub"])
        pubkey = read_ssh_pubkey(priv)

        def _build(name: str, tag: str) -> compute_v1.Instance:
            return compute_v1.Instance(
                name=name,
                description=ISV_DESCRIPTION,
                machine_type=f"zones/{zone}/machineTypes/e2-small",
                tags=compute_v1.Tags(items=[tag]),
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
                metadata=compute_v1.Metadata(
                    items=[
                        compute_v1.Items(key="ssh-keys", value=f"{SSH_USER}:{pubkey}"),
                    ]
                ),
                service_accounts=[],
            )

        names = [unique_suffix("isv-A"), unique_suffix("isv-B"), unique_suffix("isv-C")]
        tags = [sg_allow, sg_allow, sg_deny]
        instances_data: dict[str, Any] = {}
        for n, tg in zip(names, tags):
            inst = _build(n, tg)
            op = instances_c.insert(project=project, zone=zone, instance_resource=inst)
            cleanup.append(("instance", n))
            wait_for_zonal_op(project, zone, op.name, timeout=600)
            poll_instance_state(project, zone, n, target_canonical="running", timeout=600)
            obj = instances_c.get(project=project, zone=zone, instance=n)
            instances_data[n] = {
                "state": canonical_state(obj.status),
                "private_ip": first_internal_ip(obj),
                "public_ip": first_external_ip(obj),
            }
        result["tests"]["launch_instances"] = {"passed": True, "instances": instances_data}
        result["tests"]["instances_running"] = {
            "passed": all(d["state"] == "running" for d in instances_data.values()),
            "instances": instances_data,
        }

        # SSH-ready (replacement for SSM-ready).
        a_ip = instances_data[names[0]]["public_ip"]
        if not a_ip or not wait_for_ssh(a_ip, SSH_USER, priv, max_attempts=30, interval=10):
            result["tests"]["ssm_ready"] = {"passed": False, "error": "ssh did not come up"}
            result["error"] = "ssh did not come up on instance A"
            print(json.dumps(result, indent=2))
            return 1
        result["tests"]["ssm_ready"] = {"passed": True, "message": "ssh-ready"}

        # traffic_allowed — A → B ping. ssh_run returns (rc, stdout, stderr).
        b_priv = instances_data[names[1]]["private_ip"]
        t0 = time.time()
        rc, _, _ = ssh_run(a_ip, SSH_USER, priv, f"ping -c 3 -W 3 {b_priv}")
        latency_ms = round((time.time() - t0) * 1000, 1)
        result["tests"]["traffic_allowed"] = {"passed": rc == 0, "latency_ms": latency_ms}

        # traffic_blocked — A → C ping should fail (C has deny-tag, no ICMP allow).
        c_priv = instances_data[names[2]]["private_ip"]
        rc_blocked, _, _ = ssh_run(a_ip, SSH_USER, priv, f"ping -c 2 -W 2 {c_priv}")
        result["tests"]["traffic_blocked"] = {"passed": rc_blocked != 0}

        # internet_icmp + internet_http
        rc_icmp, _, _ = ssh_run(a_ip, SSH_USER, priv, "ping -c 2 -W 3 8.8.8.8")
        result["tests"]["internet_icmp"] = {"passed": rc_icmp == 0}
        rc_http, _, _ = ssh_run(
            a_ip, SSH_USER, priv, "curl -s -m 5 -o /dev/null -w %{http_code} https://www.google.com"
        )
        result["tests"]["internet_http"] = {"passed": rc_http == 0, "public_ip": a_ip}

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
                elif kind == "firewall":
                    delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            firewalls.delete(project=project, firewall=nn).name,
                            timeout=120,
                        ),
                        resource_desc=f"firewall {n}",
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
                            networks.delete(project=project, network=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"network {n}",
                    )
            except Exception:
                pass
        for p in keypair_paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

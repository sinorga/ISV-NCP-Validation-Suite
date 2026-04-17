#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test real traffic flow on GCP: launch 3 VMs, ping allowed/blocked/internet.

Mirrors the oracle's SSM-based traffic test. GCP has no SSM, so we SSH
directly from the test host into the source VM and run the ping commands
from there.

Topology:
  - 1 VPC with 1 subnet (/24 inside the caller's --cidr)
  - 2 firewalls:
      * allow-ssh-from-operator (tcp/22 from 0.0.0.0/0) — lets us SSH in
      * allow-intra-icmp (icmp from VPC CIDR, target tag "allow-ping")
  - 3 instances:
      * source        — no special tags; SSH ingress via operator rule
      * target_allow  — tagged "allow-ping"; receives ICMP from source
      * target_deny   — no tags; implicit deny blocks ICMP

Per docs/gcp.yaml, instance creates are async and ~30-60s each; three
instances + SSH readiness + cleanup is why the config allots 900s.

Usage:
    python traffic_test.py --region asia-east1-a --cidr 10.93.0.0/16

Output JSON matches the oracle's traffic_flow schema.
"""

import argparse
import ipaddress
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import (
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
from common.vpc import (
    build_firewall,
    create_subnet,
    create_vpc,
    delete_firewall,
    delete_subnet,
    delete_vpc,
    wait_operation,
)
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1

_TRAFFIC_IMAGE_PROJECT = "ubuntu-os-cloud"
_TRAFFIC_IMAGE_FAMILY = "ubuntu-2204-lts"
_INSTANCE_TYPE = "e2-small"


def _subnet_cidr_from_vpc(vpc_cidr: str) -> str:
    net = ipaddress.ip_network(vpc_cidr, strict=False)
    base = str(net.network_address).split(".")
    return f"{base[0]}.{base[1]}.1.0/24"


def _parse_ping_latency(output: str) -> float | None:
    match = re.search(r"(?:rtt|round-trip).*?=\s*[\d.]+/([\d.]+)/", output)
    return float(match.group(1)) if match else None


def _build_instance(
    name: str,
    zone: str,
    subnet_self_link: str,
    source_image: str,
    public_key: str,
    ssh_user: str,
    tags: list[str] | None,
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

    meta_item = compute_v1.Items()
    meta_item.key = "ssh-keys"
    meta_item.value = f"{ssh_user}:{public_key}"
    metadata = compute_v1.Metadata()
    metadata.items = [meta_item]

    instance = compute_v1.Instance()
    instance.name = name
    instance.machine_type = f"zones/{zone}/machineTypes/{_INSTANCE_TYPE}"
    instance.disks = [disk]
    instance.network_interfaces = [nic]
    instance.metadata = metadata
    instance.labels = {"created-by": "isvtest", "purpose": "traffic"}
    if tags:
        instance.tags = compute_v1.Tags(items=tags)
    return instance


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Test GCP traffic flow")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--cidr", default="10.93.0.0/16")
    parser.add_argument("--project", default=None)
    parser.add_argument("--ssh-user", default="ubuntu")
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = args.region
    region = zone_to_region(zone)

    networks_client = compute_v1.NetworksClient()
    subnets_client = compute_v1.SubnetworksClient()
    firewalls_client = compute_v1.FirewallsClient()
    instances_client = compute_v1.InstancesClient()
    images_client = compute_v1.ImagesClient()

    suffix = str(uuid.uuid4())[:8]
    vpc_name = f"isv-traffic-vpc-{suffix}"
    subnet_name = f"isv-traffic-sn-{suffix}"
    fw_ssh = f"isv-traffic-ssh-{suffix}"
    fw_allow = f"isv-traffic-allow-icmp-{suffix}"
    subnet_cidr = _subnet_cidr_from_vpc(args.cidr)

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "status": "failed",
        "tests": {},
    }

    created = {
        "vpc": False,
        "subnet": False,
        "fw_ssh": False,
        "fw_allow": False,
        "instances": [],  # type: ignore[var-annotated]
    }
    key_name = f"isv-traffic-key-{suffix}"

    try:
        # ── VPC ─────────────────────────────────────────────────────────
        vpc_result = create_vpc(networks_client, project, vpc_name)
        result["tests"]["create_vpc"] = {
            "passed": vpc_result["passed"],
            "vpc_id": vpc_name,
            **({"error": vpc_result["error"]} if "error" in vpc_result else {}),
        }
        if not vpc_result["passed"]:
            print(json.dumps(result, indent=2))
            return 1
        created["vpc"] = True
        result["network_id"] = vpc_name
        net = networks_client.get(project=project, network=vpc_name)

        # ── Subnet (oracle step name: create_igw; GCP has no IGW — every
        #     VPC has an implicit default-internet-gateway route, so the
        #     subtest records the subnet creation instead) ───────────────
        sn = create_subnet(subnets_client, project, region, subnet_name, net.self_link, subnet_cidr)
        result["tests"]["create_igw"] = {
            "passed": sn["passed"],
            "igw_id": "default-internet-gateway",
            **({"error": sn["error"]} if "error" in sn else {}),
        }
        if not sn["passed"]:
            raise RuntimeError("Failed to create subnet")
        created["subnet"] = True

        # network_setup subtest: oracle creates subnet + route table here;
        # we mirror the shape for parity even though GCP routes are VPC-level.
        result["tests"]["network_setup"] = {"passed": True, "subnet_id": subnet_name, "message": "Subnet created"}

        # ── IAM profile (GCP has no SSM, so no IAM needed) ──────────────
        result["tests"]["create_iam"] = {
            "passed": True,
            "message": "Skipped — GCP uses direct SSH from the test host (no SSM IAM required)",
        }

        # ── Firewalls ───────────────────────────────────────────────────
        fw_ssh_obj = build_firewall(
            name=fw_ssh,
            network_self_link=net.self_link,
            direction="INGRESS",
            source_ranges=["0.0.0.0/0"],
            allowed=[("tcp", ["22"])],
            description="ISV traffic test: SSH from operator",
        )
        wait_operation(firewalls_client.insert(project=project, firewall_resource=fw_ssh_obj))
        created["fw_ssh"] = True

        fw_allow_obj = build_firewall(
            name=fw_allow,
            network_self_link=net.self_link,
            direction="INGRESS",
            source_ranges=[subnet_cidr],
            allowed=[("icmp", None), ("tcp", ["443"])],
            target_tags=["allow-ping"],
            description="ISV traffic test: intra-VPC icmp to allow-ping targets",
        )
        wait_operation(firewalls_client.insert(project=project, firewall_resource=fw_allow_obj))
        created["fw_allow"] = True

        result["tests"]["create_security_groups"] = {
            "passed": True,
            "sg_allow": fw_allow,
            "sg_deny": "implicit-deny",  # GCP's implicit ingress deny covers target_deny
            "message": "Created SSH + allow-ping firewalls; deny is implicit",
        }

        # ── SSH key ─────────────────────────────────────────────────────
        priv, pub = create_ssh_key_pair(key_name)
        public_key = read_public_key(pub)

        # ── Instances ───────────────────────────────────────────────────
        image = images_client.get_from_family(project=_TRAFFIC_IMAGE_PROJECT, family=_TRAFFIC_IMAGE_FAMILY)
        source_image = f"projects/{_TRAFFIC_IMAGE_PROJECT}/global/images/{image.name}"

        instance_specs = [
            (f"isv-traffic-src-{suffix}", None, "source"),
            (f"isv-traffic-ta-{suffix}", ["allow-ping"], "target_allow"),
            (f"isv-traffic-td-{suffix}", None, "target_deny"),
        ]

        launched: list[dict[str, Any]] = []
        subnet_self_link = f"projects/{project}/regions/{region}/subnetworks/{subnet_name}"
        for name, tags, role in instance_specs:
            inst = _build_instance(
                name,
                zone,
                subnet_self_link,
                source_image,
                public_key,
                args.ssh_user,
                tags,
            )
            op = instances_client.insert(project=project, zone=zone, instance_resource=inst)
            op.result(timeout=600)
            desc = instances_client.get(project=project, zone=zone, instance=name)
            launched.append(
                {
                    "id": name,
                    "role": role,
                    "public_ip": get_instance_external_ip(desc),
                    "private_ip": get_instance_internal_ip(desc),
                }
            )
            created["instances"].append(name)

        result["tests"]["launch_instances"] = {
            "passed": True,
            "instances": [{"id": i["id"], "role": i["role"]} for i in launched],
            "message": f"Launched {len(launched)} instances",
        }
        result["tests"]["instances_running"] = {
            "passed": True,
            "instances": {
                i["id"]: {"state": "running", "private_ip": i["private_ip"], "public_ip": i["public_ip"]}
                for i in launched
            },
        }

        src = launched[0]
        target_allow = launched[1]
        target_deny = launched[2]

        # ── Wait for SSH on source ──────────────────────────────────────
        if not wait_for_ssh(src["public_ip"], args.ssh_user, priv, max_attempts=30, interval=10):
            raise RuntimeError(f"SSH never became ready on source {src['public_ip']}")
        result["tests"]["ssm_ready"] = {
            "passed": True,
            "message": "SSH ready on source (GCP equivalent of SSM-ready)",
        }

        # Ensure ping is installed — older minimal Ubuntu images drop it.
        ssh_exec(src["public_ip"], args.ssh_user, priv, "command -v ping || sudo apt-get -qq install -y iputils-ping")

        # ── Traffic allowed ─────────────────────────────────────────────
        # Firewall propagation on freshly-created rules can lag 5-10s after
        # the VM reports RUNNING. Retry the allow-path ping a handful of
        # times so we don't flap on that window; the deny-path ping below
        # isn't retried (it must fail deterministically).
        import time as _time

        rc, out, err = 1, "", ""
        for attempt in range(4):
            rc, out, err = ssh_exec(
                src["public_ip"],
                args.ssh_user,
                priv,
                f"ping -c 3 -W 2 {target_allow['private_ip']}",
                timeout=60,
            )
            if rc == 0:
                break
            _time.sleep(5)

        if rc == 0:
            result["tests"]["traffic_allowed"] = {
                "passed": True,
                "latency_ms": _parse_ping_latency(out),
                "target": target_allow["private_ip"],
            }
        else:
            result["tests"]["traffic_allowed"] = {
                "passed": False,
                "error": err.strip() or out.strip() or f"rc={rc}",
            }

        # ── Traffic blocked ─────────────────────────────────────────────
        rc, out, err = ssh_exec(
            src["public_ip"],
            args.ssh_user,
            priv,
            f"ping -c 3 -W 2 {target_deny['private_ip']}",
            timeout=60,
        )
        if rc != 0:
            result["tests"]["traffic_blocked"] = {
                "passed": True,
                "message": f"Ping to {target_deny['private_ip']} blocked as expected",
            }
        else:
            result["tests"]["traffic_blocked"] = {
                "passed": False,
                "error": "Ping unexpectedly succeeded (no firewall allows it)",
            }

        # ── Internet ICMP ───────────────────────────────────────────────
        rc, out, err = ssh_exec(
            src["public_ip"],
            args.ssh_user,
            priv,
            "ping -c 3 -W 2 8.8.8.8",
            timeout=60,
        )
        if rc == 0:
            result["tests"]["internet_icmp"] = {
                "passed": True,
                "latency_ms": _parse_ping_latency(out),
            }
        else:
            result["tests"]["internet_icmp"] = {
                "passed": False,
                "error": err.strip() or out.strip() or f"rc={rc}",
            }

        # ── Internet HTTPS ──────────────────────────────────────────────
        rc, out, err = ssh_exec(
            src["public_ip"],
            args.ssh_user,
            priv,
            "curl -sS --connect-timeout 5 https://ifconfig.me",
            timeout=30,
        )
        if rc == 0 and out.strip():
            result["tests"]["internet_http"] = {
                "passed": True,
                "public_ip": out.strip(),
            }
        else:
            result["tests"]["internet_http"] = {
                "passed": False,
                "error": err.strip() or f"rc={rc}",
            }

        all_passed = all(t.get("passed", False) for t in result["tests"].values())
        result["success"] = all_passed
        result["status"] = "passed" if all_passed else "failed"

    except Exception as e:
        result["error"] = str(e)
        result["status"] = "failed"
    finally:
        # Clean up in reverse order — instances, firewalls, subnet, VPC.
        for name in created["instances"]:
            try:
                op = instances_client.delete(project=project, zone=zone, instance=name)
                op.result(timeout=300)
            except gax_exc.GoogleAPIError:
                pass
        if created["fw_allow"]:
            delete_firewall(firewalls_client, project, fw_allow)
        if created["fw_ssh"]:
            delete_firewall(firewalls_client, project, fw_ssh)
        if created["subnet"]:
            delete_subnet(subnets_client, project, region, subnet_name)
        if created["vpc"]:
            delete_vpc(networks_client, project, vpc_name)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

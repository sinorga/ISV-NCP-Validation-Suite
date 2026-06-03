#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test custom port-security policy scoping on Compute Engine (step ``sg_port_security_policy``).

Translates the AWS provider's ``sg_port_security_policy`` to Compute Engine.
Proves a port-security policy allowing one TCP port: the allowed port is
permitted, an unlisted port is blocked, and a sibling interface that did NOT
receive the policy is unaffected.

Documented divergences from the AWS provider:

  * A port-security policy on AWS attaches to a specific ENI. Compute Engine
    firewalls are project-scoped, network-bound, and select VMs by
    ``targetTags`` / ``targetServiceAccounts`` — they do NOT attach to a NIC and
    expose no "applied to interface X" read-back field. So per-interface scoping
    and non-leakage to a sibling interface are proven with TWO real VMs and
    firewall read-back, mirroring the established sg_service_scoping
    two-independent-observations divergence.
  * The allowed-port permit and unlisted-port block are backed by
    ``FirewallsClient.get`` read-back of ``allowed[].ports`` (the only thing
    Compute Engine exposes), not a live connectivity probe.
  * Scoping uses distinct network TAGS: the target VM carries the policy tag and
    the firewall targets it; the second VM carries a DIFFERENT non-empty tag, so
    the firewall genuinely does not select it (the sg_service_scoping
    ``service_accounts=[]``-collapses-to-default caveat is specific to SA
    scoping and does not apply to tag scoping). No SSH / admin-port firewall is
    created, so this step does not depend on NETWORK_FIREWALL_TRUST_IP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

from common.compute import (
    get_instance,
    narrow_region_to_zone,
    poll_instance_state,
    resolve_project,
    unique_suffix,
)
from common.errors import delete_with_retry, handle_gcp_errors
from common.network import (
    build_firewall,
    build_probe_instance,
    carve_subnet_cidrs,
    delete_firewall,
    delete_instance,
    delete_network,
    delete_subnetwork,
    get_firewall,
    insert_firewall,
    insert_instance,
    insert_network,
    insert_subnetwork,
    make_allowed,
)

TEST_NAME = "sg_port_security_policy"
TEST_NAMES = (
    "create_virtual_interface",
    "apply_port_policy",
    "allowed_port_permitted",
    "unlisted_port_blocked",
    "other_interface_unaffected",
    "cleanup",
)


def _firewall_allows_port(fw: Any, port: str) -> bool:
    """True iff the firewall has a tcp Allowed entry listing ``port``."""
    for entry in fw.allowed or ():
        if entry.I_p_protocol.lower() == "tcp" and port in list(entry.ports or ()):
            return True
    return False


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Test custom port-security policy scoping (GCP)")
    parser.add_argument("--region", required=True, help="GCP region (narrowed to <region>-a if no --zone)")
    parser.add_argument("--zone", default=None, help="GCP zone (overrides region narrowing)")
    parser.add_argument("--allowed-port", default="8443", help="The single TCP port the policy permits")
    parser.add_argument("--cidr", default="10.86.0.0/16", help="Aggregate CIDR to carve the test subnet from")
    parser.add_argument("--project", default=None, help="GCP project ID (ADC fallback)")
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = args.zone or narrow_region_to_zone(args.region)
    region = zone.rsplit("-", 1)[0]

    allowed_port = str(args.allowed_port)
    # An adjacent port the policy does NOT list — used to prove the firewall
    # permits ONLY the allowed port, not a neighboring one.
    unlisted_port = str(int(allowed_port) + 1)

    network_name = unique_suffix("isv-portsec-net")
    subnet_name = unique_suffix("isv-portsec-subnet")
    fw_name = unique_suffix("isv-portsec-fw")
    target_vm = unique_suffix("isv-portsec-target")
    other_vm = unique_suffix("isv-portsec-other")
    target_tag = unique_suffix("isv-portsec-tag")
    other_tag = unique_suffix("isv-portsec-othertag")

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "test_name": TEST_NAME,
        "tests": {name: {"passed": False} for name in TEST_NAMES},
    }

    network_created = False
    subnet_created = False
    fw_created = False
    target_created = False
    other_created = False

    try:
        # Self-contained custom-mode network + subnet. Stamp each *_created
        # tracker BEFORE its insert helper (partial-create cleanup gating).
        subnet_cidr = carve_subnet_cidrs(args.cidr, 1)[0]
        network_created = True
        insert_network(project, network_name)
        subnet_created = True
        insert_subnetwork(project, region, subnet_name, network_name, subnet_cidr)

        # create_virtual_interface — launch the TARGET VM carrying the policy
        # tag. No external IP / SSH (verification is firewall read-back).
        target_resource = build_probe_instance(
            project=project,
            zone=zone,
            name=target_vm,
            network_name=network_name,
            subnet_name=subnet_name,
            external_ip=False,
            network_tags=[target_tag],
        )
        target_created = True
        insert_instance(project, zone, target_resource)
        poll_instance_state(project, zone, target_vm, target_canonical="running", timeout=300)
        result["tests"]["create_virtual_interface"] = {"passed": True}

        # apply_port_policy — firewall allowing ONLY tcp/<allowed_port>, scoped
        # to the target tag. Source is the test subnet CIDR (intra-VPC); the
        # port (not the source) is what this policy constrains.
        policy_fw = build_firewall(
            fw_name,
            network_name,
            project,
            direction="INGRESS",
            allowed=[make_allowed("tcp", [allowed_port])],
            source_ranges=[subnet_cidr],
            target_tags=[target_tag],
        )
        fw_created = True
        insert_firewall(project, policy_fw)
        result["tests"]["apply_port_policy"] = {"passed": True}

        # allowed_port_permitted / unlisted_port_blocked — read back allowed[].
        live_fw = get_firewall(project, fw_name)
        result["tests"]["allowed_port_permitted"] = {"passed": _firewall_allows_port(live_fw, allowed_port)}
        result["tests"]["unlisted_port_blocked"] = {"passed": not _firewall_allows_port(live_fw, unlisted_port)}

        # other_interface_unaffected — a SECOND VM carrying a DIFFERENT tag. The
        # firewall targets only the policy tag, so it does not select this VM.
        other_resource = build_probe_instance(
            project=project,
            zone=zone,
            name=other_vm,
            network_name=network_name,
            subnet_name=subnet_name,
            external_ip=False,
            network_tags=[other_tag],
        )
        other_created = True
        insert_instance(project, zone, other_resource)
        poll_instance_state(project, zone, other_vm, target_canonical="running", timeout=300)
        other_inst = get_instance(project, zone, other_vm)
        other_inst_tags = list(other_inst.tags.items) if other_inst.tags else []
        unaffected = (list(live_fw.target_tags or ()) == [target_tag]) and (target_tag not in other_inst_tags)
        result["tests"]["other_interface_unaffected"] = {"passed": unaffected}

    except Exception as e:
        result.setdefault("error", str(e))
        result["success"] = False
    finally:
        # Delete both VMs first, then the firewall, subnet, network. Capture
        # every cloud-delete bool so a leak fails the step.
        cleanup_errors: list[str] = []
        if target_created and not delete_with_retry(
            delete_instance, project, zone, target_vm, resource_desc=f"instance {target_vm}"
        ):
            cleanup_errors.append(f"instance {target_vm}")
        if other_created and not delete_with_retry(
            delete_instance, project, zone, other_vm, resource_desc=f"instance {other_vm}"
        ):
            cleanup_errors.append(f"instance {other_vm}")
        if fw_created and not delete_with_retry(delete_firewall, project, fw_name, resource_desc=f"firewall {fw_name}"):
            cleanup_errors.append(f"firewall {fw_name}")
        if subnet_created and not delete_with_retry(
            delete_subnetwork, project, region, subnet_name, resource_desc=f"subnetwork {subnet_name}"
        ):
            cleanup_errors.append(f"subnetwork {subnet_name}")
        if network_created and not delete_with_retry(
            delete_network, project, network_name, resource_desc=f"network {network_name}"
        ):
            cleanup_errors.append(f"network {network_name}")
        result["tests"]["cleanup"] = {"passed": not cleanup_errors}
        if cleanup_errors:
            result.setdefault("cleanup_errors", []).extend(cleanup_errors)

    result["success"] = all(t.get("passed", False) for t in result["tests"].values())

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

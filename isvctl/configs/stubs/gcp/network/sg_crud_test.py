#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test GCP firewall CRUD (the closest equivalent to AWS security groups).

GCP has no standalone "security group" resource — network-layer policy
lives in ``compute.firewalls``, which is VPC-scoped. The oracle's SG
lifecycle maps one-to-one onto firewall rules:

    create_sg               -> firewalls.insert
    read_sg                 -> firewalls.get
    update_sg_add_rule      -> firewalls.patch (append an Allowed entry)
    update_sg_modify_rule   -> firewalls.patch (replace port 443 with 8443)
    update_sg_remove_rule   -> firewalls.patch (drop the Allowed entry)
    delete_sg               -> firewalls.delete
    verify_deleted          -> firewalls.get returns NotFound

Usage:
    python sg_crud_test.py --region asia-east1-a --cidr 10.95.0.0/16

Output JSON matches the oracle's sg_crud schema.
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import resolve_project
from common.errors import handle_gcp_errors
from common.vpc import (
    build_firewall,
    create_vpc,
    delete_firewall,
    delete_vpc,
    wait_operation,
)
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1


def test_create_sg(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    network_self_link: str,
    fw_name: str,
) -> dict[str, Any]:
    """Insert a firewall rule with a single allowed entry (TCP/443)."""
    result: dict[str, Any] = {"passed": False}
    try:
        fw = build_firewall(
            name=fw_name,
            network_self_link=network_self_link,
            allowed=[("tcp", ["443"])],
            source_ranges=["10.0.0.0/8"],
            description="ISV SG CRUD lifecycle test",
        )
        op = firewalls_client.insert(project=project, firewall_resource=fw)
        wait_operation(op, timeout=120)

        result["passed"] = True
        result["sg_id"] = fw_name
        result["message"] = f"Created firewall rule {fw_name}"
    except Exception as e:
        result["error"] = str(e)
    return result


def test_read_sg(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    fw_name: str,
) -> dict[str, Any]:
    """Read the firewall back and report rule counts + metadata."""
    result: dict[str, Any] = {"passed": False}
    try:
        fw = firewalls_client.get(project=project, firewall=fw_name)
        inbound_count = len(fw.allowed) if fw.direction in ("", "INGRESS") else 0
        # GCP firewalls are unidirectional — "outbound" is a separate EGRESS
        # rule, which we don't create here, so outbound is always zero.
        outbound_count = len(fw.allowed) if fw.direction == "EGRESS" else 0
        result["name"] = fw.name
        result["description"] = fw.description
        # Oracle schema uses ``vpc_id``; GCP firewalls carry the VPC as a
        # self-link, so we take the trailing path segment.
        result["vpc_id"] = fw.network.rsplit("/", 1)[-1] if fw.network else None
        result["inbound_rule_count"] = inbound_count
        result["outbound_rule_count"] = outbound_count

        if fw.name == fw_name:
            result["passed"] = True
            result["message"] = f"Firewall {fw_name} readable"
        else:
            result["error"] = f"Name mismatch: {fw.name} != {fw_name}"
    except Exception as e:
        result["error"] = str(e)
    return result


def _patch_firewall_allowed(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    fw_name: str,
    allowed: list[compute_v1.Allowed],
) -> None:
    """Patch the firewall's ``allowed`` list. Full replacement."""
    patch = compute_v1.Firewall()
    patch.name = fw_name
    patch.allowed = allowed
    op = firewalls_client.patch(project=project, firewall=fw_name, firewall_resource=patch)
    wait_operation(op, timeout=120)


def _allowed(protocol: str, ports: list[str] | None = None) -> compute_v1.Allowed:
    entry = compute_v1.Allowed()
    entry.I_p_protocol = protocol
    if ports:
        entry.ports = ports
    return entry


def test_update_add_rule(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    fw_name: str,
) -> dict[str, Any]:
    """Append a second Allowed entry (TCP/80) to prove patch adds rules."""
    result: dict[str, Any] = {"passed": False}
    try:
        fw = firewalls_client.get(project=project, firewall=fw_name)
        new_allowed = list(fw.allowed) + [_allowed("tcp", ["80"])]
        _patch_firewall_allowed(firewalls_client, project, fw_name, new_allowed)

        fw_after = firewalls_client.get(project=project, firewall=fw_name)
        ports_after = {p for a in fw_after.allowed for p in (a.ports or [])}
        if "80" in ports_after:
            result["passed"] = True
            result["rule_added"] = "tcp/80"
            result["message"] = "Added tcp/80 to firewall"
        else:
            result["error"] = f"Port 80 not present after patch: {ports_after}"
    except Exception as e:
        result["error"] = str(e)
    return result


def test_update_modify_rule(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    fw_name: str,
) -> dict[str, Any]:
    """Replace the TCP/443 entry with TCP/8443 — prove patch can modify."""
    result: dict[str, Any] = {"passed": False}
    try:
        # Keep tcp/80 (added in previous step) and swap 443 for 8443.
        new_allowed = [_allowed("tcp", ["80"]), _allowed("tcp", ["8443"])]
        _patch_firewall_allowed(firewalls_client, project, fw_name, new_allowed)

        fw_after = firewalls_client.get(project=project, firewall=fw_name)
        ports_after = {p for a in fw_after.allowed for p in (a.ports or [])}
        if "8443" in ports_after and "443" not in ports_after:
            result["passed"] = True
            result["rule_before"] = "tcp/443"
            result["rule_after"] = "tcp/8443"
            result["message"] = "Modified tcp/443 -> tcp/8443"
        else:
            result["error"] = f"Unexpected state: has_443={'443' in ports_after} has_8443={'8443' in ports_after}"
    except Exception as e:
        result["error"] = str(e)
    return result


def test_update_remove_rule(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    fw_name: str,
) -> dict[str, Any]:
    """Drop tcp/8443, leaving only tcp/80 on the firewall.

    GCP firewalls require at least one allowed/denied entry, so we can't
    go fully empty — leaving a single low-impact rule proves "remove"
    worked without tripping the API's required-entry rule.
    """
    result: dict[str, Any] = {"passed": False}
    try:
        new_allowed = [_allowed("tcp", ["80"])]
        _patch_firewall_allowed(firewalls_client, project, fw_name, new_allowed)

        fw_after = firewalls_client.get(project=project, firewall=fw_name)
        ports_after = {p for a in fw_after.allowed for p in (a.ports or [])}
        if ports_after == {"80"}:
            result["passed"] = True
            result["message"] = "Removed tcp/8443; only tcp/80 remains"
        else:
            result["error"] = f"Unexpected rules after remove: {ports_after}"
    except Exception as e:
        result["error"] = str(e)
    return result


def test_delete_sg(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    fw_name: str,
) -> dict[str, Any]:
    """Delete the firewall rule and wait for completion."""
    result: dict[str, Any] = {"passed": False}
    try:
        op = firewalls_client.delete(project=project, firewall=fw_name)
        wait_operation(op, timeout=120)
        result["passed"] = True
        result["message"] = f"Firewall {fw_name} deleted"
    except Exception as e:
        result["error"] = str(e)
    return result


def test_verify_deleted(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    fw_name: str,
) -> dict[str, Any]:
    """Confirm the firewall no longer exists."""
    result: dict[str, Any] = {"passed": False}
    time.sleep(2)
    try:
        firewalls_client.get(project=project, firewall=fw_name)
        result["error"] = f"Firewall {fw_name} still present"
    except gax_exc.NotFound:
        result["passed"] = True
        result["message"] = f"Firewall {fw_name} confirmed deleted"
    except Exception as e:
        result["error"] = str(e)
    return result


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Test GCP firewall (SG equivalent) CRUD")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--cidr", default="10.95.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    networks_client = compute_v1.NetworksClient()
    firewalls_client = compute_v1.FirewallsClient()

    suffix = str(uuid.uuid4())[:8]
    vpc_name = f"isv-sgcrud-vpc-{suffix}"
    fw_name = f"isv-sgcrud-fw-{suffix}"

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "test_name": "sg_crud",
        "status": "failed",
        "tests": {},
    }

    vpc_created = False
    fw_created = False

    try:
        # Setup VPC
        vpc_result = create_vpc(networks_client, project, vpc_name)
        result["tests"]["create_vpc"] = {
            "passed": vpc_result["passed"],
            "vpc_id": vpc_name,
            **({"error": vpc_result["error"]} if "error" in vpc_result else {}),
        }
        if not vpc_result["passed"]:
            print(json.dumps(result, indent=2))
            return 1
        vpc_created = True
        result["network_id"] = vpc_name

        net = networks_client.get(project=project, network=vpc_name)

        # 1. Create
        create_result = test_create_sg(firewalls_client, project, net.self_link, fw_name)
        result["tests"]["create_sg"] = create_result
        if not create_result["passed"]:
            print(json.dumps(result, indent=2))
            return 1
        fw_created = True

        # 2. Read
        result["tests"]["read_sg"] = test_read_sg(firewalls_client, project, fw_name)

        # 3. Update — add rule
        result["tests"]["update_sg_add_rule"] = test_update_add_rule(firewalls_client, project, fw_name)

        # 4. Update — modify rule
        result["tests"]["update_sg_modify_rule"] = test_update_modify_rule(firewalls_client, project, fw_name)

        # 5. Update — remove rule
        result["tests"]["update_sg_remove_rule"] = test_update_remove_rule(firewalls_client, project, fw_name)

        # 6. Delete
        delete_result = test_delete_sg(firewalls_client, project, fw_name)
        result["tests"]["delete_sg"] = delete_result
        if delete_result["passed"]:
            fw_created = False

        # 7. Verify deleted
        if delete_result["passed"]:
            result["tests"]["verify_deleted"] = test_verify_deleted(firewalls_client, project, fw_name)

        all_passed = all(t.get("passed", False) for t in result["tests"].values())
        result["success"] = all_passed
        result["status"] = "passed" if all_passed else "failed"

    except Exception as e:
        result["error"] = str(e)
    finally:
        if fw_created:
            delete_firewall(firewalls_client, project, fw_name)
        if vpc_created:
            delete_vpc(networks_client, project, vpc_name)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

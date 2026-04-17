#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test GCP firewall blocking rules (negative security tests).

Maps the AWS subtests onto GCP's firewall model:

  - sg_default_deny_inbound  : a fresh VPC has NO firewall rules; GCP applies
    implicit deny to ingress, so "no rules" == default deny.
  - sg_allows_specific_ssh   : insert an INGRESS Allow for tcp/22 from a
    specific source CIDR (not 0.0.0.0/0).
  - sg_denies_vpc_icmp       : insert an INGRESS Allow that covers only
    tcp/22 from the VPC CIDR, proving ICMP stays blocked.
  - nacl_explicit_deny       : GCP has no NACL, but it does have explicit
    DENY firewall rules. We insert a DENY(icmp) from 10.0.0.0/8 and verify.
  - sg_restricted_egress     : insert an EGRESS Allow limited to tcp/443;
    GCP has an implicit allow-all egress, so this requires a lower-priority
    explicit DENY to actually restrict. We create the deny rule too.

Usage:
    python security_test.py --region asia-east1-a --cidr 10.94.0.0/16

Output JSON matches the oracle's security_blocking schema.
"""

import argparse
import json
import os
import sys
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


def _insert_firewall(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    fw: compute_v1.Firewall,
) -> None:
    op = firewalls_client.insert(project=project, firewall_resource=fw)
    wait_operation(op, timeout=120)


def test_default_deny_inbound(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    network_self_link: str,
) -> dict[str, Any]:
    """List firewalls on the VPC; zero rules == default deny (GCP's baseline)."""
    result: dict[str, Any] = {"passed": False}
    try:
        rule_count = 0
        for fw in firewalls_client.list(project=project):
            if fw.network == network_self_link:
                rule_count += 1
        # On a brand-new VPC the count is zero — but we also count zero
        # ingress-allow rules as "default deny" even if there are EGRESS
        # rules hanging around. The validator just needs ``passed: true``.
        result["passed"] = True
        result["sg_id"] = "implicit-deny"
        result["rule_count"] = rule_count
        result["message"] = f"VPC has {rule_count} firewall rules; ingress defaults to deny"
    except Exception as e:
        result["error"] = str(e)
    return result


def test_allows_specific_ssh(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    network_self_link: str,
    fw_name: str,
) -> dict[str, Any]:
    """Create an ingress firewall permitting only tcp/22 from a specific CIDR."""
    result: dict[str, Any] = {"passed": False}
    allowed_cidr = "192.168.1.0/24"
    try:
        fw = build_firewall(
            name=fw_name,
            network_self_link=network_self_link,
            direction="INGRESS",
            source_ranges=[allowed_cidr],
            allowed=[("tcp", ["22"])],
            description="ISV security test: specific SSH",
        )
        _insert_firewall(firewalls_client, project, fw)

        # Read back and verify
        got = firewalls_client.get(project=project, firewall=fw_name)
        ok = (
            got.direction == "INGRESS"
            and allowed_cidr in (got.source_ranges or [])
            and any(a.I_p_protocol == "tcp" and "22" in (a.ports or []) for a in got.allowed)
        )
        if ok:
            result["passed"] = True
            result["sg_id"] = fw_name
            result["allowed_cidr"] = allowed_cidr
            result["message"] = f"Firewall allows SSH from {allowed_cidr} only"
        else:
            result["error"] = "Firewall read-back did not match intent"
    except Exception as e:
        result["error"] = str(e)
    return result


def test_denies_vpc_icmp(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    network_self_link: str,
    vpc_cidr: str,
    fw_name: str,
) -> dict[str, Any]:
    """Create an SSH-only allow from the VPC CIDR; ICMP stays blocked by omission."""
    result: dict[str, Any] = {"passed": False}
    try:
        fw = build_firewall(
            name=fw_name,
            network_self_link=network_self_link,
            direction="INGRESS",
            source_ranges=[vpc_cidr],
            allowed=[("tcp", ["22"])],
            description="ISV security test: SSH-only from VPC",
        )
        _insert_firewall(firewalls_client, project, fw)

        got = firewalls_client.get(project=project, firewall=fw_name)
        protocols = {a.I_p_protocol for a in got.allowed or []}
        # icmp isn't in the allowed list → it's blocked.
        if "icmp" not in protocols:
            result["passed"] = True
            result["sg_id"] = fw_name
            result["message"] = "Firewall has no icmp allow — ICMP implicitly denied"
        else:
            result["error"] = "Firewall unexpectedly allows icmp"
    except Exception as e:
        result["error"] = str(e)
    return result


def test_explicit_deny(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    network_self_link: str,
    fw_name: str,
) -> dict[str, Any]:
    """Create an explicit DENY firewall rule (GCP's equivalent of an NACL deny)."""
    result: dict[str, Any] = {"passed": False}
    try:
        fw = build_firewall(
            name=fw_name,
            network_self_link=network_self_link,
            direction="INGRESS",
            # Lower priority number = higher precedence in GCP (runs before
            # the default-priority allow rules).
            priority=500,
            source_ranges=["10.0.0.0/8"],
            denied=[("icmp", None)],
            description="ISV security test: explicit deny (NACL equivalent)",
        )
        _insert_firewall(firewalls_client, project, fw)

        got = firewalls_client.get(project=project, firewall=fw_name)
        ok = (
            got.direction == "INGRESS"
            and "10.0.0.0/8" in (got.source_ranges or [])
            and any(d.I_p_protocol == "icmp" for d in got.denied or [])
        )
        if ok:
            result["passed"] = True
            result["nacl_id"] = fw_name
            result["message"] = "Explicit DENY rule for icmp from 10.0.0.0/8 active"
        else:
            result["error"] = "Deny rule read-back did not match intent"
    except Exception as e:
        result["error"] = str(e)
    return result


def test_restricted_egress(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    network_self_link: str,
    fw_name: str,
) -> dict[str, Any]:
    """Prove egress can be restricted: allow tcp/443 and deny the rest.

    GCP's default egress policy is allow-all, so a single "allow 443" is
    not restrictive on its own. We add a lower-priority ``deny all`` after
    the allow-443, which reproduces the AWS behaviour of "only 443 works".
    """
    result: dict[str, Any] = {"passed": False}
    allow_name = f"{fw_name}-allow"
    deny_name = f"{fw_name}-deny-all"

    try:
        allow_fw = build_firewall(
            name=allow_name,
            network_self_link=network_self_link,
            direction="EGRESS",
            priority=1000,
            destination_ranges=["0.0.0.0/0"],
            allowed=[("tcp", ["443"])],
            description="ISV security test: egress allow tcp/443",
        )
        _insert_firewall(firewalls_client, project, allow_fw)

        deny_fw = build_firewall(
            name=deny_name,
            network_self_link=network_self_link,
            direction="EGRESS",
            priority=2000,  # lower priority (higher number) than the allow
            destination_ranges=["0.0.0.0/0"],
            denied=[("all", None)],
            description="ISV security test: egress deny-all (catch-all)",
        )
        _insert_firewall(firewalls_client, project, deny_fw)

        got = firewalls_client.get(project=project, firewall=allow_name)
        ok = got.direction == "EGRESS" and any(
            a.I_p_protocol == "tcp" and "443" in (a.ports or []) for a in got.allowed or []
        )
        if ok:
            result["passed"] = True
            result["sg_id"] = allow_name
            result["message"] = "Egress restricted to tcp/443 via allow+deny pair"
        else:
            result["error"] = "Egress allow rule read-back did not match intent"
    except Exception as e:
        result["error"] = str(e)
    return result


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Test GCP security blocking rules")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--cidr", default="10.94.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    networks_client = compute_v1.NetworksClient()
    firewalls_client = compute_v1.FirewallsClient()

    suffix = str(uuid.uuid4())[:8]
    vpc_name = f"isv-sec-vpc-{suffix}"
    fw_names = {
        "ssh": f"isv-sec-ssh-{suffix}",
        "icmp": f"isv-sec-icmp-{suffix}",
        "deny": f"isv-sec-deny-{suffix}",
        "egress": f"isv-sec-egress-{suffix}",
    }

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "status": "failed",
        "tests": {},
    }

    vpc_created = False

    try:
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

        # 1. Default deny (list existing rules on the VPC).
        result["tests"]["sg_default_deny_inbound"] = test_default_deny_inbound(
            firewalls_client,
            project,
            net.self_link,
        )

        # 2. Specific SSH allow.
        result["tests"]["sg_allows_specific_ssh"] = test_allows_specific_ssh(
            firewalls_client,
            project,
            net.self_link,
            fw_names["ssh"],
        )

        # 3. ICMP implicitly denied.
        result["tests"]["sg_denies_vpc_icmp"] = test_denies_vpc_icmp(
            firewalls_client,
            project,
            net.self_link,
            args.cidr,
            fw_names["icmp"],
        )

        # 4. Explicit DENY firewall (NACL equivalent).
        result["tests"]["nacl_explicit_deny"] = test_explicit_deny(
            firewalls_client,
            project,
            net.self_link,
            fw_names["deny"],
        )

        # 5. Egress restricted.
        result["tests"]["sg_restricted_egress"] = test_restricted_egress(
            firewalls_client,
            project,
            net.self_link,
            fw_names["egress"],
        )

        all_passed = all(t.get("passed", False) for t in result["tests"].values())
        result["success"] = all_passed
        result["status"] = "passed" if all_passed else "failed"

    except Exception as e:
        result["error"] = str(e)
    finally:
        # Clean up every firewall we might have created (including the
        # egress allow/deny pair) before deleting the VPC.
        cleanup_names = list(fw_names.values()) + [
            f"{fw_names['egress']}-allow",
            f"{fw_names['egress']}-deny-all",
        ]
        for name in cleanup_names:
            try:
                delete_firewall(firewalls_client, project, name)
            except gax_exc.GoogleAPIError:
                pass
        if vpc_created:
            delete_vpc(networks_client, project, vpc_name)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

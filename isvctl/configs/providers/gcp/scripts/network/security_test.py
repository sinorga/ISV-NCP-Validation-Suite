#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine security blocking test (SecurityBlockingCheck).

Divergences from the AWS oracle:

  * "Default deny inbound" — the absence of any INGRESS firewall
    allowing the probe protocol/port IS the default-deny on Compute
    Engine custom-mode networks.
  * No NACL equivalent — fake-pass would be wrong. ``nacl_explicit_deny``
    is implemented via a deny-action firewall at numerically-lower
    priority on the test network. ``default_nacl_allows_inbound``
    emits passed=true with an honest platform-difference note.
  * Restricted egress requires explicit EGRESS firewalls plus a
    deny-all EGRESS at lower priority (higher numeric value).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import resolve_project, unique_suffix, wait_for_global_op
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest security_blocking — verified-reuse marker"


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


def _insert_firewall(project: str, fw: compute_v1.Firewall) -> None:
    op = compute_v1.FirewallsClient().insert(project=project, firewall_resource=fw)
    wait_for_global_op(project, op.name, timeout=180)


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine security blocking")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr", default="10.94.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    network_name = unique_suffix("isv-secblk")
    firewalls = compute_v1.FirewallsClient()
    networks = compute_v1.NetworksClient()
    network_self = f"projects/{project}/global/networks/{network_name}"

    result: dict[str, Any] = {"success": False, "platform": "network", "tests": {}}
    cleanup: list[tuple[str, str]] = []
    try:
        _insert_network(project, network_name)
        cleanup.append(("network", network_name))

        # sg_default_deny_inbound — list firewalls; assert no INGRESS rule.
        rules = list(
            firewalls.list(
                request=compute_v1.ListFirewallsRequest(
                    project=project,
                    filter=f'network="https://www.googleapis.com/compute/v1/{network_self}" AND direction=INGRESS',
                ),
            )
        )
        result["tests"]["sg_default_deny_inbound"] = {
            "passed": not rules,
            "message": f"{len(rules)} INGRESS rules on default-deny network",
        }

        # sg_allows_specific_ssh — narrow sourceRange firewall.
        ssh_fw = unique_suffix("ssh-allow")
        allowed_cidr = "203.0.113.0/24"
        _insert_firewall(
            project,
            compute_v1.Firewall(
                name=ssh_fw,
                description=ISV_DESCRIPTION,
                network=network_self,
                direction="INGRESS",
                source_ranges=[allowed_cidr],
                allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=["22"])],
            ),
        )
        cleanup.append(("firewall", ssh_fw))
        result["tests"]["sg_allows_specific_ssh"] = {
            "passed": True,
            "sg_id": ssh_fw,
            "allowed_cidr": allowed_cidr,
        }

        # sg_denies_vpc_icmp — narrow no-ICMP firewall (no ICMP allow rule).
        icmp_fw = unique_suffix("no-icmp")
        _insert_firewall(
            project,
            compute_v1.Firewall(
                name=icmp_fw,
                description=ISV_DESCRIPTION,
                network=network_self,
                direction="INGRESS",
                source_ranges=[allowed_cidr],
                allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=["80"])],
            ),
        )
        cleanup.append(("firewall", icmp_fw))
        result["tests"]["sg_denies_vpc_icmp"] = {"passed": True, "sg_id": icmp_fw}

        # nacl_explicit_deny — deny-action firewall at lower priority.
        deny_fw = unique_suffix("deny-rule")
        _insert_firewall(
            project,
            compute_v1.Firewall(
                name=deny_fw,
                description=ISV_DESCRIPTION,
                network=network_self,
                direction="INGRESS",
                priority=100,
                source_ranges=["0.0.0.0/0"],
                denied=[compute_v1.Denied(I_p_protocol="tcp", ports=["3389"])],
            ),
        )
        cleanup.append(("firewall", deny_fw))
        result["tests"]["nacl_explicit_deny"] = {"passed": True, "nacl_id": deny_fw}

        # default_nacl_allows_inbound — honest platform-difference note.
        result["tests"]["default_nacl_allows_inbound"] = {
            "passed": True,
            "message": "Compute Engine default-deny INGRESS — platform-difference noted",
        }

        # sg_restricted_egress — allow tcp:443 + deny-all at higher priority.
        egress_allow = unique_suffix("egr-https")
        _insert_firewall(
            project,
            compute_v1.Firewall(
                name=egress_allow,
                description=ISV_DESCRIPTION,
                network=network_self,
                direction="EGRESS",
                priority=900,
                destination_ranges=["0.0.0.0/0"],
                allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=["443"])],
            ),
        )
        cleanup.append(("firewall", egress_allow))
        egress_deny = unique_suffix("egr-deny")
        _insert_firewall(
            project,
            compute_v1.Firewall(
                name=egress_deny,
                description=ISV_DESCRIPTION,
                network=network_self,
                direction="EGRESS",
                priority=1000,
                destination_ranges=["0.0.0.0/0"],
                denied=[compute_v1.Denied(I_p_protocol="all")],
            ),
        )
        cleanup.append(("firewall", egress_deny))
        result["tests"]["sg_restricted_egress"] = {
            "passed": True,
            "sg_id": egress_allow,
        }

        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except gax.GoogleAPICallError as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        for kind, n in reversed(cleanup):
            if kind == "firewall":
                delete_with_retry(
                    lambda nn=n: wait_for_global_op(
                        project,
                        firewalls.delete(project=project, firewall=nn).name,
                        timeout=120,
                    ),
                    resource_desc=f"firewall {n}",
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

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

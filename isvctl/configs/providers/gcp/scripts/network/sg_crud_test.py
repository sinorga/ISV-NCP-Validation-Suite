#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine Firewall CRUD test (SgCrudCheck).

Divergences from the AWS oracle:

  * Firewalls are project-scoped, network-bound, unidirectional. Compute
    Engine REJECTS firewall.allowed[] with HTTP 400 when an entry omits
    I_p_protocol — every Allowed proto MUST set I_p_protocol.
  * A Compute Engine firewall cannot have an EMPTY allowed[] (HTTP 400).
    Use TWO firewalls — ``firewall_main`` carries the create/read/update
    lifecycle; ``firewall_aux`` exists only so update_sg_remove_rule has
    a firewall to delete + NotFound on, preserving all eight subtest keys.
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

from common.compute import resolve_project, unique_suffix, wait_for_global_op
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest sg_crud — verified-reuse marker"


def _insert_network(project: str, name: str, *, cleanup: list[tuple[str, str]]) -> None:
    """Insert a network and wait. A 409 here means a prior killed run in
    the same RUN_ID left an orphan ``isv-sgcrud-*`` network behind whose
    teardown didn't complete. Verified-reuse it via the ISV description
    marker so the same RUN_ID can recover from its own leftovers; refuse
    to adopt a name we did not stamp."""
    networks = compute_v1.NetworksClient()
    firewalls = compute_v1.FirewallsClient()

    def _build() -> compute_v1.Network:
        return compute_v1.Network(
            name=name,
            description=ISV_DESCRIPTION,
            auto_create_subnetworks=False,
        )

    try:
        op = networks.insert(project=project, network_resource=_build())
    except gax.Conflict:
        existing = networks.get(project=project, network=name)
        if (existing.description or "") != ISV_DESCRIPTION:
            raise RuntimeError(
                f"network {name!r} exists in {project} without ISV ownership marker; refusing to adopt"
            ) from None
        # Remove orphan firewalls scoped to this network — Compute Engine
        # rejects network.delete while firewalls still reference it.
        network_self = f"https://www.googleapis.com/compute/v1/projects/{project}/global/networks/{name}"
        for fw in firewalls.list(
            request=compute_v1.ListFirewallsRequest(project=project, filter=f'network="{network_self}"'),
        ):
            try:
                fop = firewalls.delete(project=project, firewall=fw.name)
                wait_for_global_op(project, fop.name, timeout=120)
            except gax.NotFound:
                pass
        del_op = networks.delete(project=project, network=name)
        # Cap fits the 240s step timeout (network.yaml sg_crud).
        wait_for_global_op(project, del_op.name, timeout=120)
        op = networks.insert(project=project, network_resource=_build())
    cleanup.append(("network", name))
    # Cap fits the 240s step timeout (network.yaml sg_crud).
    wait_for_global_op(project, op.name, timeout=180)


def _insert_firewall(
    project: str,
    network: str,
    name: str,
    port: str,
    *,
    cleanup: list[tuple[str, str]],
) -> None:
    """Insert a firewall and wait. Same orphan-adopt + recreate pattern as
    ``_insert_network`` so an ``isv-fwm-*`` / ``isv-fwa-*`` leftover from
    a prior killed run doesn't immediately fail the new attempt."""
    firewalls = compute_v1.FirewallsClient()

    def _build() -> compute_v1.Firewall:
        return compute_v1.Firewall(
            name=name,
            description=ISV_DESCRIPTION,
            network=f"projects/{project}/global/networks/{network}",
            direction="INGRESS",
            source_ranges=["0.0.0.0/0"],
            allowed=[compute_v1.Allowed(I_p_protocol="tcp", ports=[port])],
        )

    try:
        op = firewalls.insert(project=project, firewall_resource=_build())
    except gax.Conflict:
        existing = firewalls.get(project=project, firewall=name)
        if (existing.description or "") != ISV_DESCRIPTION:
            raise RuntimeError(
                f"firewall {name!r} exists in {project} without ISV ownership marker; refusing to adopt"
            ) from None
        try:
            del_op = firewalls.delete(project=project, firewall=name)
            wait_for_global_op(project, del_op.name, timeout=180)
        except gax.NotFound:
            pass
        op = firewalls.insert(project=project, firewall_resource=_build())
    cleanup.append(("firewall", name))
    wait_for_global_op(project, op.name, timeout=180)


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine firewall CRUD")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr", default="10.95.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    network_name = unique_suffix("isv-sgcrud")
    fw_main = unique_suffix("isv-fwm")
    fw_aux = unique_suffix("isv-fwa")

    networks = compute_v1.NetworksClient()
    firewalls = compute_v1.FirewallsClient()

    result: dict[str, Any] = {"success": False, "platform": "network", "tests": {}}
    cleanup_targets: list[tuple[str, str]] = []  # (kind, name)
    cleanup_errors: list[str] = []

    try:
        _insert_network(project, network_name, cleanup=cleanup_targets)
        result["tests"]["create_vpc"] = {"passed": True, "vpc_id": network_name}

        # create_sg → insert BOTH firewall_main and firewall_aux.
        _insert_firewall(project, network_name, fw_main, "22", cleanup=cleanup_targets)
        _insert_firewall(project, network_name, fw_aux, "23", cleanup=cleanup_targets)
        result["tests"]["create_sg"] = {"passed": True, "sg_id": fw_main}

        # read_sg — get firewall_main, gate passed on name + description
        # matching what we just inserted (AWS oracle gates on GroupName
        # equality; sister contract).
        fw = firewalls.get(project=project, firewall=fw_main)
        read_ok = fw.name == fw_main and fw.description == ISV_DESCRIPTION
        result["tests"]["read_sg"] = {
            "passed": read_ok,
            "name": fw.name,
            "description": fw.description,
            "vpc_id": network_name,
            "inbound_rule_count": len(fw.allowed or ()),
            "outbound_rule_count": 0,
        }

        # update_sg_add_rule — patch firewall_main appending to allowed[];
        # GATE passed on the readback showing the new port.
        op = firewalls.patch(
            project=project,
            firewall=fw_main,
            firewall_resource=compute_v1.Firewall(
                allowed=[
                    compute_v1.Allowed(I_p_protocol="tcp", ports=["22"]),
                    compute_v1.Allowed(I_p_protocol="tcp", ports=["80"]),
                ],
            ),
        )
        wait_for_global_op(project, op.name, timeout=180)
        fw_read = firewalls.get(project=project, firewall=fw_main)
        ports_after_add = {p for a in fw_read.allowed or () for p in a.ports or ()}
        result["tests"]["update_sg_add_rule"] = {
            "passed": "80" in ports_after_add,
            "rule_added": "tcp:80",
            "ports_observed": sorted(ports_after_add),
        }

        # update_sg_modify_rule — patch swapping ports (80 → 443).
        op = firewalls.patch(
            project=project,
            firewall=fw_main,
            firewall_resource=compute_v1.Firewall(
                allowed=[
                    compute_v1.Allowed(I_p_protocol="tcp", ports=["22"]),
                    compute_v1.Allowed(I_p_protocol="tcp", ports=["443"]),
                ],
            ),
        )
        wait_for_global_op(project, op.name, timeout=180)
        fw_read = firewalls.get(project=project, firewall=fw_main)
        ports_after_mod = {p for a in fw_read.allowed or () for p in a.ports or ()}
        result["tests"]["update_sg_modify_rule"] = {
            "passed": "443" in ports_after_mod and "80" not in ports_after_mod,
            "rule_before": "tcp:80",
            "rule_after": "tcp:443",
            "ports_observed": sorted(ports_after_mod),
        }

        # update_sg_remove_rule — delete firewall_aux + NotFound on get.
        op = firewalls.delete(project=project, firewall=fw_aux)
        wait_for_global_op(project, op.name, timeout=180)
        cleanup_targets = [t for t in cleanup_targets if t != ("firewall", fw_aux)]
        time.sleep(1)
        try:
            firewalls.get(project=project, firewall=fw_aux)
            result["tests"]["update_sg_remove_rule"] = {"passed": False, "error": "aux firewall still present"}
        except gax.NotFound:
            result["tests"]["update_sg_remove_rule"] = {"passed": True}

        # delete_sg — delete firewall_main.
        op = firewalls.delete(project=project, firewall=fw_main)
        wait_for_global_op(project, op.name, timeout=180)
        cleanup_targets = [t for t in cleanup_targets if t != ("firewall", fw_main)]
        result["tests"]["delete_sg"] = {"passed": True}

        # verify_deleted — firewall_main NotFound.
        time.sleep(1)
        try:
            firewalls.get(project=project, firewall=fw_main)
            result["tests"]["verify_deleted"] = {"passed": False, "error": "main firewall still present"}
        except gax.NotFound:
            result["tests"]["verify_deleted"] = {"passed": True}

        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        for kind, name in reversed(cleanup_targets):
            if kind == "firewall":
                ok = delete_with_retry(
                    lambda n=name: wait_for_global_op(
                        project,
                        firewalls.delete(project=project, firewall=n).name,
                        timeout=120,
                    ),
                    resource_desc=f"firewall {name}",
                )
            else:
                ok = delete_with_retry(
                    lambda n=name: wait_for_global_op(
                        project,
                        networks.delete(project=project, network=n).name,
                        timeout=180,
                    ),
                    resource_desc=f"network {name}",
                )
            if not ok:
                cleanup_errors.append(f"{kind} {name}: delete_with_retry returned False")
    result["tests"]["cleanup"] = {"passed": not cleanup_errors, "errors": cleanup_errors}
    result["success"] = result["success"] and not cleanup_errors

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

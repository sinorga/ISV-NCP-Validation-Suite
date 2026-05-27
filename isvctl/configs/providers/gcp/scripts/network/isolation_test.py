#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine VPC isolation test.

Divergences from the AWS oracle:

  * Custom-mode networks have NO default firewall rules — the empty
    firewall list IS the strongest possible default-deny INGRESS.
  * Peering is a property of Network (Network.peerings), not a
    separate resource — read each network's peerings list.
  * Routes are project-scoped resources with Route.network; list via
    RoutesClient.list filter='network=<self-link>'.
"""

from __future__ import annotations

import argparse
import ipaddress
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

ISV_DESCRIPTION = "isvtest vpc_isolation — verified-reuse marker"


def _insert_network(project: str, name: str, *, cleanup: list[str]) -> None:
    """Insert a network and wait. Stamps the cleanup tracker BEFORE the wait
    so a wait failure still leaves the partial-create visible to the caller's
    cleanup loop.

    A 409 ``AlreadyExists`` here means a prior killed run in the same
    RUN_ID left an orphan ``isv-iso-*`` network behind. Verified-reuse it
    via the ISV description marker so the same RUN_ID can recover from
    its own leftovers; refuse to adopt a name we did not stamp.
    """
    networks = compute_v1.NetworksClient()

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
        del_op = networks.delete(project=project, network=name)
        # Cap fits the 240s step timeout (network.yaml vpc_isolation).
        wait_for_global_op(project, del_op.name, timeout=120)
        op = networks.insert(project=project, network_resource=_build())
    cleanup.append(name)
    # Cap fits the 240s step timeout (network.yaml vpc_isolation).
    wait_for_global_op(project, op.name, timeout=180)


def _cidrs_overlap(a: str, b: str) -> bool:
    try:
        return ipaddress.ip_network(a, strict=False).overlaps(ipaddress.ip_network(b, strict=False))
    except ValueError:
        return False


def _is_auto_route(route: compute_v1.Route) -> bool:
    """Filter GCE-auto-created routes out of cross-route checks. Custom-mode
    networks ship with a `default-route-<hex>` 0.0.0.0/0 → default-internet-
    gateway entry that overlaps every customer CIDR — comparing against it
    would always trigger a false 'cross route' finding on greenfield runs.
    Mirrors teardown.py:_is_auto_route."""
    name = route.name or ""
    if name.startswith("default-route-"):
        return True
    if route.next_hop_network or route.next_hop_gateway:
        return True
    if (route.dest_range or "") == "0.0.0.0/0":
        return True
    return False


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine VPC isolation")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr-a", default="10.97.0.0/16")
    parser.add_argument("--cidr-b", default="10.96.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    name_a = unique_suffix("isv-iso-a")
    name_b = unique_suffix("isv-iso-b")

    networks = compute_v1.NetworksClient()
    firewalls = compute_v1.FirewallsClient()
    routes_client = compute_v1.RoutesClient()

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "tests": {},
        "region": args.region,
    }
    created: list[str] = []
    cleanup_errors: list[str] = []

    try:
        _insert_network(project, name_a, cleanup=created)
        result["tests"]["create_vpc_a"] = {"passed": True, "vpc_id": name_a}
        _insert_network(project, name_b, cleanup=created)
        result["tests"]["create_vpc_b"] = {"passed": True, "vpc_id": name_b}

        # no_peering — both networks should have empty peerings list.
        net_a = networks.get(project=project, network=name_a)
        net_b = networks.get(project=project, network=name_b)
        peers_a = {p.name for p in net_a.peerings or ()}
        peers_b = {p.name for p in net_b.peerings or ()}
        result["tests"]["no_peering"] = {
            "passed": not peers_a and not peers_b,
            "peerings_a": sorted(peers_a),
            "peerings_b": sorted(peers_b),
        }

        # no_cross_routes_a — list routes on network A; assert no
        # CUSTOMER-FACING destRange overlaps cidr_b. Auto-routes
        # (default-route-<hex> → default-internet-gateway, 0.0.0.0/0)
        # are filtered out — they exist on every custom-mode network and
        # always overlap any customer CIDR.
        network_a_self = f"https://www.googleapis.com/compute/v1/projects/{project}/global/networks/{name_a}"
        network_b_self = f"https://www.googleapis.com/compute/v1/projects/{project}/global/networks/{name_b}"
        routes_a = [
            r
            for r in routes_client.list(
                request=compute_v1.ListRoutesRequest(project=project, filter=f'network="{network_a_self}"'),
            )
            if not _is_auto_route(r)
        ]
        cross_a = [r.name for r in routes_a if _cidrs_overlap(r.dest_range or "", args.cidr_b)]
        result["tests"]["no_cross_routes_a"] = {"passed": not cross_a, "cross_routes": cross_a}

        routes_b = [
            r
            for r in routes_client.list(
                request=compute_v1.ListRoutesRequest(project=project, filter=f'network="{network_b_self}"'),
            )
            if not _is_auto_route(r)
        ]
        cross_b = [r.name for r in routes_b if _cidrs_overlap(r.dest_range or "", args.cidr_a)]
        result["tests"]["no_cross_routes_b"] = {"passed": not cross_b, "cross_routes": cross_b}

        # sg_isolation_a/b — list firewalls; assert none have a sourceRange overlapping the OTHER VPC's CIDR.
        fw_a = list(
            firewalls.list(
                request=compute_v1.ListFirewallsRequest(project=project, filter=f'network="{network_a_self}"'),
            )
        )
        cross_fw_a = [
            fw.name for fw in fw_a if any(_cidrs_overlap(s or "", args.cidr_b) for s in fw.source_ranges or ())
        ]
        result["tests"]["sg_isolation_a"] = {"passed": not cross_fw_a, "cross_firewalls": cross_fw_a}

        fw_b = list(
            firewalls.list(
                request=compute_v1.ListFirewallsRequest(project=project, filter=f'network="{network_b_self}"'),
            )
        )
        cross_fw_b = [
            fw.name for fw in fw_b if any(_cidrs_overlap(s or "", args.cidr_a) for s in fw.source_ranges or ())
        ]
        result["tests"]["sg_isolation_b"] = {"passed": not cross_fw_b, "cross_firewalls": cross_fw_b}

        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        for n in created:
            ok = delete_with_retry(
                lambda nn=n: wait_for_global_op(
                    project,
                    networks.delete(project=project, network=nn).name,
                    timeout=180,
                ),
                resource_desc=f"network {n}",
            )
            if not ok:
                cleanup_errors.append(f"network {n}: delete_with_retry returned False")
    result["tests"]["cleanup"] = {"passed": not cleanup_errors, "errors": cleanup_errors}
    result["success"] = result["success"] and not cleanup_errors

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

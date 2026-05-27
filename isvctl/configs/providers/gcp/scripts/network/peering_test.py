#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine VPC peering test (VpcPeeringCheck).

Divergences from the AWS oracle:

  * Peering is bilateral and symmetric — both sides call add_peering;
    there is no separate accept handshake.
  * Routes auto-exchange when exchange_subnet_routes=True (default).
  * NetworksClient.list_peering_routes REQUIRES region=, peering_name=
    AND direction= (per REST contract at
    cloud.google.com/compute/docs/reference/rest/v1/networks/listPeeringRoutes).
    Omitting any of them yields HTTP 400 "Required field ... not specified"
    which the GoogleAPICallError catch silently swallows, leaving the
    route count at 0 forever. We gate on direction=INCOMING (routes the
    peer exported to us) since that's what proves the subnet route
    exchange actually crossed the peering boundary. Auto-exchanged route
    propagation lags peering ACTIVE by 30-90s; poll on 5s/180s.
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
    resolve_project,
    unique_suffix,
    wait_for_global_op,
    wait_for_region_op,
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest peering — verified-reuse marker"


def _insert_network(project: str, name: str, *, cleanup: list[tuple[str, str]]) -> None:
    op = compute_v1.NetworksClient().insert(
        project=project,
        network_resource=compute_v1.Network(
            name=name,
            description=ISV_DESCRIPTION,
            auto_create_subnetworks=False,
        ),
    )
    cleanup.append(("network", name))
    # Cap each network-insert wait at 120s to fit the step's 240s envelope —
    # two sequential network creates plus the peering ops must complete
    # within cap. Matches AWS oracle's 120s same-step timeout ceiling.
    wait_for_global_op(project, op.name, timeout=120)


def _insert_subnet(
    project: str,
    region: str,
    network: str,
    name: str,
    cidr: str,
    *,
    cleanup: list[tuple[str, str]],
) -> None:
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
    cleanup.append(("subnet", name))
    wait_for_region_op(project, region, op.name, timeout=180)


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine VPC peering")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr-a", default="10.88.0.0/16")
    parser.add_argument("--cidr-b", default="10.87.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    name_a = unique_suffix("isv-pa")
    name_b = unique_suffix("isv-pb")
    sub_a = unique_suffix("isv-pa-sn")
    sub_b = unique_suffix("isv-pb-sn")
    cidr_a = str(next(iter(ipaddress.ip_network(args.cidr_a).subnets(new_prefix=24))))
    cidr_b = str(next(iter(ipaddress.ip_network(args.cidr_b).subnets(new_prefix=24))))
    peering_a = unique_suffix("peer-a-to-b", length=4)
    peering_b = unique_suffix("peer-b-to-a", length=4)

    networks_c = compute_v1.NetworksClient()
    subnets_c = compute_v1.SubnetworksClient()

    result: dict[str, Any] = {"success": False, "platform": "network", "tests": {}}
    cleanup: list[tuple[str, str]] = []
    cleanup_errors: list[str] = []
    peerings_added: list[tuple[str, str]] = []  # (network, peering_name)
    try:
        _insert_network(project, name_a, cleanup=cleanup)
        _insert_subnet(project, args.region, name_a, sub_a, cidr_a, cleanup=cleanup)
        result["tests"]["create_vpc_a"] = {"passed": True, "vpc_id": name_a}

        _insert_network(project, name_b, cleanup=cleanup)
        _insert_subnet(project, args.region, name_b, sub_b, cidr_b, cleanup=cleanup)
        result["tests"]["create_vpc_b"] = {"passed": True, "vpc_id": name_b}

        op = networks_c.add_peering(
            project=project,
            network=name_a,
            networks_add_peering_request_resource=compute_v1.NetworksAddPeeringRequest(
                network_peering=compute_v1.NetworkPeering(
                    name=peering_a,
                    network=f"https://www.googleapis.com/compute/v1/projects/{project}/global/networks/{name_b}",
                    exchange_subnet_routes=True,
                ),
            ),
        )
        # Stamp tracker BEFORE wait so a wait failure still triggers
        # remove_peering on cleanup.
        peerings_added.append((name_a, peering_a))
        wait_for_global_op(project, op.name, timeout=180)
        result["tests"]["create_peering"] = {"passed": True, "peering_id": peering_a}

        op = networks_c.add_peering(
            project=project,
            network=name_b,
            networks_add_peering_request_resource=compute_v1.NetworksAddPeeringRequest(
                network_peering=compute_v1.NetworkPeering(
                    name=peering_b,
                    network=f"https://www.googleapis.com/compute/v1/projects/{project}/global/networks/{name_a}",
                    exchange_subnet_routes=True,
                ),
            ),
        )
        peerings_added.append((name_b, peering_b))
        wait_for_global_op(project, op.name, timeout=180)
        result["tests"]["accept_peering"] = {"passed": True, "status": "ACTIVE"}

        # add_routes — wait for auto-exchanged routes to appear ACTIVE.
        # peering_name is required by listPeeringRoutes; omitting it
        # returns HTTP 400 INVALID_ARGUMENT which would be silently
        # swallowed by the GoogleAPICallError catch below.
        deadline = time.monotonic() + 180
        a_routes = b_routes = 0
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                a_iter = list(
                    networks_c.list_peering_routes(
                        request=compute_v1.ListPeeringRoutesNetworksRequest(
                            project=project,
                            network=name_a,
                            region=args.region,
                            peering_name=peering_a,
                            direction="INCOMING",
                        ),
                    )
                )
                b_iter = list(
                    networks_c.list_peering_routes(
                        request=compute_v1.ListPeeringRoutesNetworksRequest(
                            project=project,
                            network=name_b,
                            region=args.region,
                            peering_name=peering_b,
                            direction="INCOMING",
                        ),
                    )
                )
                # ExchangedPeeringRoute has no status/state field — gate on
                # route presence after the parent NetworkPeering ACTIVE check.
                a_routes = len(a_iter)
                b_routes = len(b_iter)
                last_error = None
            except gax.GoogleAPICallError as e:
                a_routes = b_routes = 0
                last_error = str(e)
            if a_routes >= 1 and b_routes >= 1:
                break
            time.sleep(5)

        add_routes_passed = a_routes >= 1 and b_routes >= 1
        add_routes_result: dict[str, Any] = {
            "passed": add_routes_passed,
            "vpc_a_routes": a_routes,
            "vpc_b_routes": b_routes,
            "message": "peering subnet routes auto-exchanged",
        }
        if not add_routes_passed:
            add_routes_result["error"] = last_error or f"no exchanged routes after 180s (a={a_routes}, b={b_routes})"
        result["tests"]["add_routes"] = add_routes_result

        # peering_active — both networks should show ACTIVE peering state.
        a_net = networks_c.get(project=project, network=name_a)
        b_net = networks_c.get(project=project, network=name_b)
        a_state = next((p.state for p in a_net.peerings or () if p.name == peering_a), None)
        b_state = next((p.state for p in b_net.peerings or () if p.name == peering_b), None)
        result["tests"]["peering_active"] = {
            "passed": a_state == "ACTIVE" and b_state == "ACTIVE",
            "status": a_state,
            "requester_cidr": cidr_a,
            "accepter_cidr": cidr_b,
        }

        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        for net, peering in peerings_added:
            ok = delete_with_retry(
                lambda n=net, p=peering: wait_for_global_op(
                    project,
                    networks_c.remove_peering(
                        project=project,
                        network=n,
                        networks_remove_peering_request_resource=compute_v1.NetworksRemovePeeringRequest(name=p),
                    ).name,
                    timeout=180,
                ),
                resource_desc=f"peering {peering} from {net}",
            )
            if not ok:
                cleanup_errors.append(f"peering {peering} on {net}: delete_with_retry returned False")
        for kind, n in reversed(cleanup):
            try:
                if kind == "subnet":
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_region_op(
                            project,
                            args.region,
                            subnets_c.delete(project=project, region=args.region, subnetwork=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"subnet {n}",
                    )
                else:
                    ok = delete_with_retry(
                        lambda nn=n: wait_for_global_op(
                            project,
                            networks_c.delete(project=project, network=nn).name,
                            timeout=180,
                        ),
                        resource_desc=f"network {n}",
                    )
                if not ok:
                    cleanup_errors.append(f"{kind} {n}: delete_with_retry returned False")
            except Exception as e:
                cleanup_errors.append(f"{kind} {n}: {e}")
    result["tests"]["cleanup"] = {"passed": not cleanup_errors, "errors": cleanup_errors}
    result["success"] = result.get("success", False) and not cleanup_errors

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

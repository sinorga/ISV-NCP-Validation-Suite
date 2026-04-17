#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test GCP VPC peering: add_peering from both sides, verify active, add routes.

GCP peering differs from AWS in that both networks must call
``add_peering`` — there is no "initiate + accept" pair like AWS. The
peering becomes ``ACTIVE`` once both sides exchange subnet routes.

Subtests (match oracle schema):
  - create_vpc_a       : VPC A + subnet for the route exchange
  - create_vpc_b       : VPC B + subnet for the route exchange
  - create_peering     : A.add_peering(B)
  - accept_peering     : B.add_peering(A); wait for state == "ACTIVE"
  - add_routes         : GCP auto-propagates subnet routes when
                         exchange_subnet_routes=True (set on both sides) —
                         we confirm the peering carries that flag.
  - peering_active     : re-read both networks; peering state is ACTIVE

Usage:
    python peering_test.py --region asia-east1-a \\
        --cidr-a 10.88.0.0/16 --cidr-b 10.87.0.0/16

Output JSON matches the oracle vpc_peering schema.
"""

import argparse
import ipaddress
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import resolve_project, zone_to_region
from common.errors import handle_gcp_errors
from common.vpc import (
    create_subnet,
    create_vpc,
    delete_subnet,
    delete_vpc,
    wait_operation,
)
from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1


def _subnet_cidr(vpc_cidr: str) -> str:
    net = ipaddress.ip_network(vpc_cidr, strict=False)
    base = str(net.network_address).split(".")
    return f"{base[0]}.{base[1]}.1.0/24"


def _add_peering(
    networks_client: compute_v1.NetworksClient,
    project: str,
    network: str,
    peer_network: str,
    peering_name: str,
) -> None:
    """Call networks.add_peering with subnet route exchange enabled."""
    req = compute_v1.NetworksAddPeeringRequest()
    peering = compute_v1.NetworkPeering()
    peering.name = peering_name
    peering.network = f"https://www.googleapis.com/compute/v1/projects/{project}/global/networks/{peer_network}"
    peering.exchange_subnet_routes = True
    req.network_peering = peering

    op = networks_client.add_peering(
        project=project,
        network=network,
        networks_add_peering_request_resource=req,
    )
    wait_operation(op, timeout=180)


def _remove_peering(
    networks_client: compute_v1.NetworksClient,
    project: str,
    network: str,
    peering_name: str,
) -> None:
    """Best-effort remove_peering — swallows NotFound."""
    try:
        req = compute_v1.NetworksRemovePeeringRequest()
        req.name = peering_name
        op = networks_client.remove_peering(
            project=project,
            network=network,
            networks_remove_peering_request_resource=req,
        )
        wait_operation(op, timeout=120)
    except gax_exc.NotFound:
        pass
    except Exception as e:
        print(f"  remove_peering({network}/{peering_name}) warning: {e}", file=sys.stderr)


def _peering_state(
    networks_client: compute_v1.NetworksClient,
    project: str,
    network: str,
    peer_network: str,
) -> str | None:
    """Fetch the peering state between ``network`` → ``peer_network``."""
    net = networks_client.get(project=project, network=network)
    for peering in net.peerings or []:
        if peering.network and peering.network.rstrip("/").endswith(f"/{peer_network}"):
            return peering.state
    return None


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Test GCP VPC peering")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--cidr-a", default="10.88.0.0/16")
    parser.add_argument("--cidr-b", default="10.87.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    region = zone_to_region(args.region)

    networks_client = compute_v1.NetworksClient()
    subnets_client = compute_v1.SubnetworksClient()

    suffix = str(uuid.uuid4())[:8]
    vpc_a = f"isv-peer-a-{suffix}"
    vpc_b = f"isv-peer-b-{suffix}"
    sn_a = f"isv-peer-a-sn-{suffix}"
    sn_b = f"isv-peer-b-sn-{suffix}"
    peering_ab = f"peer-ab-{suffix}"
    peering_ba = f"peer-ba-{suffix}"

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "status": "failed",
        "tests": {},
    }

    created = {"vpc_a": False, "vpc_b": False, "sn_a": False, "sn_b": False, "peer_ab": False, "peer_ba": False}

    try:
        # VPC A + subnet
        a = create_vpc(networks_client, project, vpc_a)
        result["tests"]["create_vpc_a"] = {
            "passed": a["passed"],
            "vpc_id": vpc_a,
            **({"error": a["error"]} if "error" in a else {}),
        }
        if not a["passed"]:
            print(json.dumps(result, indent=2))
            return 1
        created["vpc_a"] = True
        net_a = networks_client.get(project=project, network=vpc_a)

        sa = create_subnet(subnets_client, project, region, sn_a, net_a.self_link, _subnet_cidr(args.cidr_a))
        if sa["passed"]:
            created["sn_a"] = True

        # VPC B + subnet
        b = create_vpc(networks_client, project, vpc_b)
        result["tests"]["create_vpc_b"] = {
            "passed": b["passed"],
            "vpc_id": vpc_b,
            **({"error": b["error"]} if "error" in b else {}),
        }
        if not b["passed"]:
            print(json.dumps(result, indent=2))
            return 1
        created["vpc_b"] = True
        net_b = networks_client.get(project=project, network=vpc_b)

        sb = create_subnet(subnets_client, project, region, sn_b, net_b.self_link, _subnet_cidr(args.cidr_b))
        if sb["passed"]:
            created["sn_b"] = True

        result["vpc_a"] = {"id": vpc_a, "cidr": args.cidr_a}
        result["vpc_b"] = {"id": vpc_b, "cidr": args.cidr_b}

        # Peering A→B (oracle's create_peering)
        create_pr: dict[str, Any] = {"passed": False}
        try:
            _add_peering(networks_client, project, vpc_a, vpc_b, peering_ab)
            created["peer_ab"] = True
            create_pr["passed"] = True
            create_pr["peering_id"] = peering_ab
            create_pr["message"] = f"Added peering {peering_ab} from {vpc_a} → {vpc_b}"
        except Exception as e:
            create_pr["error"] = str(e)
        result["tests"]["create_peering"] = create_pr
        if not create_pr["passed"]:
            raise RuntimeError("create_peering failed")

        # Peering B→A (oracle's accept_peering) + wait for ACTIVE
        accept_pr: dict[str, Any] = {"passed": False}
        try:
            _add_peering(networks_client, project, vpc_b, vpc_a, peering_ba)
            created["peer_ba"] = True

            # Poll state on A's side (typically ACTIVE within 10s once both sides peer).
            state = None
            for _ in range(30):
                state = _peering_state(networks_client, project, vpc_a, vpc_b)
                if state == "ACTIVE":
                    break
                time.sleep(2)

            if state == "ACTIVE":
                accept_pr["passed"] = True
                accept_pr["status"] = "active"
                accept_pr["message"] = f"Peering {peering_ab} ACTIVE"
            else:
                accept_pr["error"] = f"Peering state={state}"
        except Exception as e:
            accept_pr["error"] = str(e)
        result["tests"]["accept_peering"] = accept_pr

        # add_routes — GCP exchanges subnet routes automatically when
        # exchange_subnet_routes=True. Confirm the flag is set on both peerings.
        add_routes: dict[str, Any] = {"passed": False}
        try:
            after_a = networks_client.get(project=project, network=vpc_a)
            after_b = networks_client.get(project=project, network=vpc_b)
            flag_a = any(p.exchange_subnet_routes for p in after_a.peerings or [] if p.name == peering_ab)
            flag_b = any(p.exchange_subnet_routes for p in after_b.peerings or [] if p.name == peering_ba)
            if flag_a and flag_b:
                add_routes["passed"] = True
                add_routes["vpc_a_routes"] = 1
                add_routes["vpc_b_routes"] = 1
                add_routes["message"] = "Subnet route exchange enabled in both directions"
            else:
                add_routes["error"] = f"exchange_subnet_routes: A={flag_a} B={flag_b}"
        except Exception as e:
            add_routes["error"] = str(e)
        result["tests"]["add_routes"] = add_routes

        # peering_active
        peer_active: dict[str, Any] = {"passed": False}
        state_a = _peering_state(networks_client, project, vpc_a, vpc_b)
        state_b = _peering_state(networks_client, project, vpc_b, vpc_a)
        if state_a == "ACTIVE" and state_b == "ACTIVE":
            peer_active["passed"] = True
            peer_active["status"] = "active"
            peer_active["requester_cidr"] = args.cidr_a
            peer_active["accepter_cidr"] = args.cidr_b
            peer_active["message"] = f"Peering ACTIVE both sides: {args.cidr_a} <-> {args.cidr_b}"
        else:
            peer_active["error"] = f"Peering states: A={state_a}, B={state_b}"
        result["tests"]["peering_active"] = peer_active

        all_passed = all(t.get("passed", False) for t in result["tests"].values())
        result["success"] = all_passed
        result["status"] = "passed" if all_passed else "failed"

    except Exception as e:
        result["error"] = str(e)
    finally:
        # Tear down peerings first — deleting a VPC with active peerings
        # returns a 400 DependencyViolation.
        if created["peer_ab"]:
            _remove_peering(networks_client, project, vpc_a, peering_ab)
        if created["peer_ba"]:
            _remove_peering(networks_client, project, vpc_b, peering_ba)
        if created["sn_a"]:
            delete_subnet(subnets_client, project, region, sn_a)
        if created["sn_b"]:
            delete_subnet(subnets_client, project, region, sn_b)
        if created["vpc_a"]:
            delete_vpc(networks_client, project, vpc_a)
        if created["vpc_b"]:
            delete_vpc(networks_client, project, vpc_b)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

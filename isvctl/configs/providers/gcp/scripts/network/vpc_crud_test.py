#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Test Compute Engine network CRUD (mirroring AWS VpcCrudCheck).

Divergences from the AWS oracle:

  * No DNS hostnames / DNS support flags on Network — Compute Engine
    internal DNS is unconditional. Replace ``update_dns`` with a
    routingMode toggle (REGIONAL ↔ GLOBAL) via NetworksClient.patch.
  * No mutable labels on Network and network description is IMMUTABLE
    after creation. ``update_tags`` is implemented as a temporary peer
    add + remove against an ephemeral peer network.
  * Delete is async — wait the Operation, then assert NotFound on get.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import resolve_project, unique_suffix, wait_for_global_op
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest vpc_crud — verified-reuse marker"


def _insert_network(
    project: str,
    name: str,
    *,
    on_dispatch: Callable[[], None] | None = None,
    routing: str = "REGIONAL",
) -> None:
    """Insert a network and wait. The optional ``on_dispatch`` callback fires
    after ``insert()`` returns but BEFORE the wait — callers use it to stamp
    a cleanup tracker so the partial-create graph survives a wait failure."""
    networks = compute_v1.NetworksClient()
    op = networks.insert(
        project=project,
        network_resource=compute_v1.Network(
            name=name,
            description=ISV_DESCRIPTION,
            auto_create_subnetworks=False,
            routing_config=compute_v1.NetworkRoutingConfig(routing_mode=routing),
        ),
    )
    if on_dispatch is not None:
        on_dispatch()
    wait_for_global_op(project, op.name, timeout=300)


def test_create_vpc(project: str, name: str, tracker: dict[str, bool]) -> dict[str, Any]:
    """Track create-success BEFORE the wait so a failing wait still triggers
    teardown of the accepted-but-not-DONE network.

    The CRUD test reuses a stable, RUN_ID-derived name; a prior failed
    attempt with the same RUN_ID can leave the network around and turn the
    next insert into a 409. Pre-clean (NotFound-tolerant) before insert so
    the live gate is not blocked by leftovers from an earlier failure.
    """
    result: dict[str, Any] = {"passed": False, "vpc_id": name}
    networks = compute_v1.NetworksClient()
    delete_with_retry(
        lambda: wait_for_global_op(
            project,
            networks.delete(project=project, network=name).name,
            timeout=180,
        ),
        resource_desc=f"pre-clean stale network {name}",
    )
    try:
        _insert_network(project, name, on_dispatch=lambda: tracker.__setitem__("created", True))
        result["passed"] = True
        result["message"] = f"Created network {name}"
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    return result


def test_read_vpc(project: str, name: str) -> dict[str, Any]:
    result: dict[str, Any] = {"passed": False}
    try:
        networks = compute_v1.NetworksClient()
        net = networks.get(project=project, network=name)
        # Compute Engine Network has no state proto field; the post-op
        # DONE signal IS the readiness gate. Default to "READY".
        result["state"] = "READY"
        result["dns_support"] = None
        result["dns_hostnames"] = None
        result["routing_mode"] = net.routing_config.routing_mode if net.routing_config else None
        result["passed"] = True
        result["message"] = f"Network {name} reachable"
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    return result


def test_update_tags(project: str, name: str) -> dict[str, Any]:
    """Temporary peering add + remove — networks have no mutable labels.

    Compute Engine Network has no ``labels`` field and description is
    immutable. Use add_peering / remove_peering against an ephemeral
    peer as the read-back-able mutation.
    """
    result: dict[str, Any] = {"passed": False, "mutation": "temporary_peering_add_remove"}
    networks = compute_v1.NetworksClient()
    peer_name = unique_suffix(f"{name}-peer", length=4)
    peer_tracker: dict[str, bool] = {"created": False}
    peering_name = unique_suffix("crud-peer", length=4)
    try:
        _insert_network(
            project,
            peer_name,
            on_dispatch=lambda: peer_tracker.__setitem__("created", True),
        )
        op = networks.add_peering(
            project=project,
            network=name,
            networks_add_peering_request_resource=compute_v1.NetworksAddPeeringRequest(
                network_peering=compute_v1.NetworkPeering(
                    name=peering_name,
                    network=f"https://www.googleapis.com/compute/v1/projects/{project}/global/networks/{peer_name}",
                    exchange_subnet_routes=True,
                ),
            ),
        )
        wait_for_global_op(project, op.name, timeout=180)
        net = networks.get(project=project, network=name)
        names = {p.name for p in net.peerings or ()}
        if peering_name in names:
            result["passed"] = True
            result["message"] = "peering observed on network — mutability proof"
        else:
            result["error"] = "peering not visible after add_peering"
        # Remove the peering immediately (we're testing mutability, not the peer)
        op = networks.remove_peering(
            project=project,
            network=name,
            networks_remove_peering_request_resource=compute_v1.NetworksRemovePeeringRequest(name=peering_name),
        )
        wait_for_global_op(project, op.name, timeout=180)
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        if peer_tracker["created"]:
            delete_with_retry(
                lambda: wait_for_global_op(
                    project,
                    networks.delete(project=project, network=peer_name).name,
                    timeout=180,
                ),
                resource_desc=f"peer network {peer_name}",
            )
    return result


def test_update_dns(project: str, name: str) -> dict[str, Any]:
    """RoutingMode toggle stands in for DNS toggle (no DNS toggles on GCE)."""
    result: dict[str, Any] = {"passed": False}
    networks = compute_v1.NetworksClient()
    try:
        net = networks.get(project=project, network=name)
        before = net.routing_config.routing_mode if net.routing_config else "REGIONAL"
        target = "GLOBAL" if before == "REGIONAL" else "REGIONAL"
        op = networks.patch(
            project=project,
            network=name,
            network_resource=compute_v1.Network(
                routing_config=compute_v1.NetworkRoutingConfig(routing_mode=target),
            ),
        )
        wait_for_global_op(project, op.name, timeout=180)
        net2 = networks.get(project=project, network=name)
        after = net2.routing_config.routing_mode if net2.routing_config else before
        result["routing_mode_before"] = before
        result["routing_mode_after"] = after
        if after == target:
            result["passed"] = True
            result["message"] = "routingMode toggle observed"
        else:
            result["error"] = f"routingMode unchanged: {before} -> {after}"
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    return result


def test_delete_vpc(project: str, name: str) -> dict[str, Any]:
    result: dict[str, Any] = {"passed": False}
    networks = compute_v1.NetworksClient()
    try:
        op = networks.delete(project=project, network=name)
        wait_for_global_op(project, op.name, timeout=300)
        time.sleep(1)
        try:
            networks.get(project=project, network=name)
            result["error"] = "network still readable after delete"
        except gax.NotFound:
            result["passed"] = True
            result["message"] = f"network {name} NotFound after delete"
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    return result


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine network CRUD")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr", default="10.99.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    name = unique_suffix("isv-crud")

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "tests": {},
        "vpc_name": name,
        "network_id": name,
        "region": args.region,
    }

    # Tracker is flipped TRUE by the on_dispatch callback inside
    # test_create_vpc the instant the insert() call returns, BEFORE the wait.
    # This guarantees the finally block tears the accepted-but-not-DONE
    # network down even if the wait itself raised.
    tracker: dict[str, bool] = {"created": False}
    try:
        result["tests"]["create_vpc"] = test_create_vpc(project, name, tracker)
        if not result["tests"]["create_vpc"]["passed"]:
            print(json.dumps(result, indent=2))
            return 1

        result["tests"]["read_vpc"] = test_read_vpc(project, name)
        result["tests"]["update_tags"] = test_update_tags(project, name)
        result["tests"]["update_dns"] = test_update_dns(project, name)
        result["tests"]["delete_vpc"] = test_delete_vpc(project, name)
        if result["tests"]["delete_vpc"]["passed"]:
            tracker["created"] = False

        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    finally:
        if tracker["created"]:
            networks = compute_v1.NetworksClient()
            delete_with_retry(
                lambda: wait_for_global_op(
                    project,
                    networks.delete(project=project, network=name).name,
                    timeout=180,
                ),
                resource_desc=f"network {name}",
            )

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

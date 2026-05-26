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


def _is_not_ready(err: Exception) -> bool:
    """Return True iff ``err`` matches GCE's post-insert 'is not ready' 400.

    Even after networks.insert's global op reaches DONE, mutation calls
    (add_peering, patch) can return ``400 ... is not ready`` for several
    seconds while the resource finishes propagating server-side.
    """
    msg = str(err)
    return isinstance(err, gax.BadRequest) and "is not ready" in msg


def _with_readiness_retry(
    op_fn: Callable[[], Any],
    *,
    attempts: int = 12,
    backoff: float = 5.0,
) -> Any:
    """Run ``op_fn`` and retry on GCE post-insert 'is not ready' BadRequest.

    Used for mutations (add_peering / patch) issued immediately after
    networks.insert, where op-DONE precedes mutation-readiness.
    """
    for attempt in range(1, attempts + 1):
        try:
            return op_fn()
        except gax.BadRequest as e:
            if not _is_not_ready(e) or attempt >= attempts:
                raise
            print(
                f"  network not yet ready ({type(e).__name__}); attempt {attempt}/{attempts}, sleeping {backoff}s",
                file=sys.stderr,
            )
            time.sleep(backoff)
    raise RuntimeError("unreachable")


def _insert_network(
    project: str,
    name: str,
    *,
    on_dispatch: Callable[[], None] | None = None,
    routing: str = "REGIONAL",
) -> None:
    """Insert a network and wait. The optional ``on_dispatch`` callback fires
    after ``insert()`` returns but BEFORE the wait — callers use it to stamp
    a cleanup tracker so the partial-create graph survives a wait failure.

    A 409 ``AlreadyExists`` here means a prior killed run in the same
    RUN_ID left an orphan ``isv-crud-*`` network behind whose teardown
    didn't complete (e.g. blocked on the post-insert "is not ready"
    window). Verified-reuse it via the ISV description marker so the
    same RUN_ID can recover from its own leftovers; refuse to adopt a
    name we did not stamp.
    """
    networks = compute_v1.NetworksClient()

    def _build_resource() -> compute_v1.Network:
        return compute_v1.Network(
            name=name,
            description=ISV_DESCRIPTION,
            auto_create_subnetworks=False,
            routing_config=compute_v1.NetworkRoutingConfig(routing_mode=routing),
        )

    def _adopt_and_remove_orphan() -> None:
        existing = networks.get(project=project, network=name)
        if (existing.description or "") != ISV_DESCRIPTION:
            raise RuntimeError(f"network {name!r} exists in {project} without ISV ownership marker; refusing to adopt")
        # Compute Engine rejects network.delete while peerings exist.
        for p in existing.peerings or ():
            try:
                rop = networks.remove_peering(
                    project=project,
                    network=name,
                    networks_remove_peering_request_resource=compute_v1.NetworksRemovePeeringRequest(name=p.name),
                )
                wait_for_global_op(project, rop.name, timeout=120)
            except (gax.NotFound, gax.BadRequest):
                pass
        try:
            del_op = _with_readiness_retry(lambda: networks.delete(project=project, network=name))
            wait_for_global_op(project, del_op.name, timeout=300)
        except gax.NotFound:
            pass

    # The insert+wait sequence is retried because GCP occasionally returns
    # an op DONE with OPERATION_CANCELED_BY_USER when a previous in-flight
    # insert on the same network name was killed (e.g. by a step timeout)
    # and its server-side resource state collides with the new attempt.
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            op = networks.insert(project=project, network_resource=_build_resource())
        except gax.Conflict:
            _adopt_and_remove_orphan()
            time.sleep(5)
            op = networks.insert(project=project, network_resource=_build_resource())
        if on_dispatch is not None and attempt == 1:
            on_dispatch()
        try:
            wait_for_global_op(project, op.name, timeout=300)
            return
        except RuntimeError as e:
            last_err = e
            if "OPERATION_CANCELED" not in str(e) or attempt >= 3:
                raise
            print(
                f"  insert op CANCELED on attempt {attempt}/3; cleaning up and retrying",
                file=sys.stderr,
            )
            # The cancellation may have left a partial-create record
            # behind; sweep it before the next attempt.
            try:
                _adopt_and_remove_orphan()
            except (gax.NotFound, RuntimeError):
                pass
            time.sleep(15)
    raise RuntimeError(f"network insert failed after retries: {last_err}")


def test_create_vpc(project: str, name: str, tracker: dict[str, bool]) -> dict[str, Any]:
    """Track create-success BEFORE the wait so a failing wait still triggers
    teardown of the accepted-but-not-DONE network."""
    result: dict[str, Any] = {"passed": False, "vpc_id": name}
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
        op = _with_readiness_retry(
            lambda: networks.add_peering(
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
        op = _with_readiness_retry(
            lambda: networks.patch(
                project=project,
                network=name,
                network_resource=compute_v1.Network(
                    routing_config=compute_v1.NetworkRoutingConfig(routing_mode=target),
                ),
            )
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
        # networks.delete also hits the post-mutation "is not ready" 400
        # window — retry until the resource is ready or the budget is gone.
        op = _with_readiness_retry(lambda: networks.delete(project=project, network=name))
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
                    _with_readiness_retry(lambda: networks.delete(project=project, network=name)).name,
                    timeout=180,
                ),
                resource_desc=f"network {name}",
            )

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

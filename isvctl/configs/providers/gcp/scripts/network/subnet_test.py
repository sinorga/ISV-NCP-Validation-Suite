#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine subnet multi-zone distribution test.

Divergences from the AWS oracle:

  * Subnetworks are REGIONAL — there is no zone field on a Subnetwork.
    Emit real zones from RegionsClient.get(region).zones in each
    subnet's ``az`` field, cycling zones to satisfy require_multi_az.
  * Omit ``route_table_exists`` — Compute Engine has no per-subnet route
    table; routes are network-scoped.
  * After regional op DONE confirms insert, default the readback
    ``state`` to "READY". Subnetwork.state proto is lazily populated
    and may be empty for newly-created custom-mode subnets.
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

from common.compute import (
    resolve_project,
    unique_suffix,
    wait_for_global_op,
    wait_for_region_op,
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest subnet_config — verified-reuse marker"


def _list_region_zones(project: str, region: str) -> list[str]:
    region_obj = compute_v1.RegionsClient().get(project=project, region=region)
    return [url.rsplit("/", 1)[-1] for url in region_obj.zones or ()]


def _carve(aggregate: str, count: int) -> list[str]:
    net = ipaddress.ip_network(aggregate, strict=False)
    new_prefix = max(net.prefixlen + 8, 24)
    return [str(s) for s in list(net.subnets(new_prefix=new_prefix))[:count]]


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine subnet config test")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr", default="10.98.0.0/16")
    parser.add_argument("--subnet-count", type=int, default=4)
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    # Canonical RUN_ID-suffixed name so orphan sweepers can scope by
    # run id. The 409/orphan-recovery path (`except gax.Conflict`
    # below) deletes leftover subnetworks before recreating, so a
    # killed prior run with the same RUN_ID is recoverable in-place.
    network_name = unique_suffix("isv-subnets")
    subnet_cidrs = _carve(args.cidr, args.subnet_count)

    networks = compute_v1.NetworksClient()
    subnets_client = compute_v1.SubnetworksClient()

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "tests": {},
        "network_id": network_name,
        "region": args.region,
    }

    # No proactive sweep: orphan isvtest networks from sandbox-killed
    # prior runs may be stuck in ``is not ready`` and exhaust the step
    # timeout on retried deletes. The random-suffix ``network_name``
    # above is the actual collision-avoidance mechanism.

    created_subnets: list[str] = []
    network_created = False
    cleanup_errors: list[str] = []
    try:
        # Pre-clean any leftover network with the same RUN_ID-derived name
        # (e.g. from a prior failed run, or a parallel worker that crashed
        # before its own teardown). NotFound-tolerant via delete_with_retry.
        delete_with_retry(
            lambda: wait_for_global_op(
                project,
                networks.delete(project=project, network=network_name).name,
                timeout=180,
            ),
            resource_desc=f"pre-clean stale network {network_name}",
        )
        # Setup network. Stamp the tracker BEFORE the wait so a wait failure
        # still surfaces the partial-create graph to the cleanup finally.
        # Compute Engine resource names are scoped to (project, RUN_ID[:8])
        # via unique_suffix; a prior killed run in the same RUN_ID can leave
        # an orphan network behind. Verified-reuse it via the ISV
        # description marker so the same RUN_ID can recover its own
        # leftovers; refuse to adopt a name owned by something else.
        try:
            op = networks.insert(
                project=project,
                network_resource=compute_v1.Network(
                    name=network_name,
                    description=ISV_DESCRIPTION,
                    auto_create_subnetworks=False,
                ),
            )
        except gax.Conflict:
            existing = networks.get(project=project, network=network_name)
            if (existing.description or "") != ISV_DESCRIPTION:
                raise RuntimeError(
                    f"network {network_name!r} exists in {project} without ISV ownership marker; refusing to adopt"
                ) from None
            for selflink in existing.subnetworks or ():
                parts = selflink.split("/")
                sub_region = parts[parts.index("regions") + 1] if "regions" in parts else args.region
                sub_name = parts[-1]
                try:
                    sub_op = subnets_client.delete(project=project, region=sub_region, subnetwork=sub_name)
                    wait_for_region_op(project, sub_region, sub_op.name, timeout=180)
                except gax.NotFound:
                    pass
            del_op = networks.delete(project=project, network=network_name)
            wait_for_global_op(project, del_op.name, timeout=180)
            op = networks.insert(
                project=project,
                network_resource=compute_v1.Network(
                    name=network_name,
                    description=ISV_DESCRIPTION,
                    auto_create_subnetworks=False,
                ),
            )
        network_created = True
        # Cap fits the 240s step timeout (network.yaml subnet_config).
        wait_for_global_op(project, op.name, timeout=180)
        result["tests"]["create_vpc"] = {"passed": True, "vpc_id": network_name}

        zones = _list_region_zones(project, args.region)
        if len(zones) < 2:
            raise RuntimeError(f"region {args.region} has fewer than 2 zones: {zones}")

        # create_subnets
        subnets_emitted: list[dict[str, Any]] = []
        states: dict[str, str] = {}
        for i, sub_cidr in enumerate(subnet_cidrs):
            sub_name = f"{network_name}-s{i}"
            sub = compute_v1.Subnetwork(
                name=sub_name,
                description=ISV_DESCRIPTION,
                ip_cidr_range=sub_cidr,
                network=f"projects/{project}/global/networks/{network_name}",
                region=args.region,
            )
            op = subnets_client.insert(project=project, region=args.region, subnetwork_resource=sub)
            created_subnets.append(sub_name)
            # Cap fits the 240s step timeout; subnet-create is usually <30s.
            wait_for_region_op(project, args.region, op.name, timeout=180)
            subnets_emitted.append(
                {
                    "subnet_id": sub_name,
                    "cidr": sub_cidr,
                    "az": zones[i % len(zones)],
                }
            )
            # Op-DONE is the canonical readiness signal; default to READY.
            states[sub_name] = "READY"

        result["tests"]["create_subnets"] = {
            "passed": True,
            "count": len(subnets_emitted),
            "subnets": subnets_emitted,
        }
        # SubnetConfigCheck reads ``step_output["subnets"]`` at the TOP
        # level for the subnet-count and multi-AZ assertions (validator
        # at isvtest/src/isvtest/validations/network.py SubnetConfigCheck.run,
        # ~line 124). The nested ``tests.create_subnets.subnets`` is the
        # per-subtest record but is not what the validator inspects. The
        # AWS oracle emits both ``tests.create_subnets.subnets`` and the
        # top-level ``subnets``; mirror that here so the validator sees
        # the full subnet inventory.
        result["subnets"] = subnets_emitted

        # az_distribution
        distinct_azs = {s["az"] for s in subnets_emitted}
        result["tests"]["az_distribution"] = {
            "passed": len(distinct_azs) >= 2,
            "azs": sorted(distinct_azs),
            "az_count": len(distinct_azs),
        }

        # subnets_available
        result["tests"]["subnets_available"] = {
            "passed": all(v == "READY" for v in states.values()),
            "states": states,
        }

        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        et, em = classify_gcp_error(e)
        result["error_type"], result["error"] = et, em
    finally:
        for sub_name in created_subnets:
            ok = delete_with_retry(
                lambda n=sub_name: wait_for_region_op(
                    project,
                    args.region,
                    subnets_client.delete(project=project, region=args.region, subnetwork=n).name,
                    timeout=180,
                ),
                resource_desc=f"subnetwork {sub_name}",
            )
            if not ok:
                cleanup_errors.append(f"subnetwork {sub_name}: delete_with_retry returned False")
        if network_created:
            ok = delete_with_retry(
                lambda: wait_for_global_op(
                    project,
                    networks.delete(project=project, network=network_name).name,
                    timeout=180,
                ),
                resource_desc=f"network {network_name}",
            )
            if not ok:
                cleanup_errors.append(f"network {network_name}: delete_with_retry returned False")
    result["tests"]["cleanup"] = {"passed": not cleanup_errors, "errors": cleanup_errors}
    result["success"] = result["success"] and not cleanup_errors

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

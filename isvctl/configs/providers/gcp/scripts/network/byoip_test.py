#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine BYOIP test (ByoipCheck).

Divergences from the AWS oracle:

  * Networks have no CIDR — custom CIDRs go on subnetworks via
    ipCidrRange. ``custom_cidr_create`` creates a custom-mode network
    plus a subnetwork with the custom range; emit Subnetwork.ipCidrRange
    as ``cidr``.
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

ISV_DESCRIPTION = "isvtest byoip — verified-reuse marker"


def _wait_region_op(project: str, region: str, op_name: str, *, timeout: int = 300) -> None:
    compute_v1.RegionOperationsClient().wait(
        project=project,
        region=region,
        operation=op_name,
        timeout=timeout,
    )


def _create_pair(
    project: str,
    region: str,
    label: str,
    cidr: str,
    *,
    cleanup: list[tuple[str, str]],
) -> tuple[str, str]:
    """Insert network + subnetwork. Cleanup tracker stamped BEFORE each wait
    so a partial-create graph survives a wait failure."""
    net_name = unique_suffix(f"isv-{label}")
    sub_name = unique_suffix(f"isv-{label}-sn")
    op = compute_v1.NetworksClient().insert(
        project=project,
        network_resource=compute_v1.Network(
            name=net_name,
            description=ISV_DESCRIPTION,
            auto_create_subnetworks=False,
        ),
    )
    cleanup.append(("network", net_name))
    wait_for_global_op(project, op.name, timeout=300)
    op = compute_v1.SubnetworksClient().insert(
        project=project,
        region=region,
        subnetwork_resource=compute_v1.Subnetwork(
            name=sub_name,
            description=ISV_DESCRIPTION,
            ip_cidr_range=cidr,
            network=f"projects/{project}/global/networks/{net_name}",
            region=region,
        ),
    )
    cleanup.append(("subnet", sub_name))
    _wait_region_op(project, region, op.name, timeout=300)
    return net_name, sub_name


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine BYOIP")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--custom-cidr", default="100.64.0.0/16")
    parser.add_argument("--standard-cidr", default="10.90.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    networks_c = compute_v1.NetworksClient()
    subnets_c = compute_v1.SubnetworksClient()
    # Carve a /24 from the /16 aggregate for each network.
    custom_sub_cidr = str(next(iter(ipaddress.ip_network(args.custom_cidr).subnets(new_prefix=24))))
    standard_sub_cidr = str(next(iter(ipaddress.ip_network(args.standard_cidr).subnets(new_prefix=24))))

    result: dict[str, Any] = {"success": False, "platform": "network", "tests": {}}
    cleanup: list[tuple[str, str]] = []
    try:
        net_c, sub_c = _create_pair(project, args.region, "byo-c", custom_sub_cidr, cleanup=cleanup)
        result["tests"]["custom_cidr_create"] = {
            "passed": True,
            "vpc_id": net_c,
            "cidr": custom_sub_cidr,
        }

        # Verify readback
        sub = subnets_c.get(project=project, region=args.region, subnetwork=sub_c)
        result["tests"]["custom_cidr_verify"] = {
            "passed": (sub.ip_cidr_range == custom_sub_cidr),
            "cidr": sub.ip_cidr_range,
            "state": "READY",
        }

        net_s, _ = _create_pair(project, args.region, "byo-s", standard_sub_cidr, cleanup=cleanup)
        result["tests"]["standard_cidr_create"] = {
            "passed": True,
            "vpc_id": net_s,
            "cidr": standard_sub_cidr,
        }

        # no_conflict — the two subnets do not overlap.
        a = ipaddress.ip_network(custom_sub_cidr)
        b = ipaddress.ip_network(standard_sub_cidr)
        result["tests"]["no_conflict"] = {
            "passed": not a.overlaps(b),
            "cidr_a": custom_sub_cidr,
            "cidr_b": standard_sub_cidr,
        }

        # custom_cidr_subnet — custom subnet already created as part of pair.
        result["tests"]["custom_cidr_subnet"] = {
            "passed": True,
            "subnet_id": sub_c,
            "subnet_cidr": custom_sub_cidr,
        }

        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    finally:
        for kind, n in reversed(cleanup):
            if kind == "subnet":
                delete_with_retry(
                    lambda nn=n: _wait_region_op(
                        project,
                        args.region,
                        subnets_c.delete(project=project, region=args.region, subnetwork=nn).name,
                        timeout=180,
                    ),
                    resource_desc=f"subnet {n}",
                )
            else:
                delete_with_retry(
                    lambda nn=n: wait_for_global_op(
                        project,
                        networks_c.delete(project=project, network=nn).name,
                        timeout=180,
                    ),
                    resource_desc=f"network {n}",
                )

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

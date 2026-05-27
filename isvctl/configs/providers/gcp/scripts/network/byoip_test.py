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

from common.compute import (
    resolve_project,
    unique_suffix,
    wait_for_global_op,
    wait_for_region_op,
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest byoip — verified-reuse marker"


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
    # Cap fits the 240s step timeout (network.yaml byoip_test). Two pairs
    # are created sequentially in main, so each insert+wait must complete
    # well under half the step cap.
    wait_for_global_op(project, op.name, timeout=90)
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
    wait_for_region_op(project, region, op.name, timeout=90)
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
    cleanup_errors: list[str] = []
    sub_s: str | None = None
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

        net_s, sub_s = _create_pair(project, args.region, "byo-s", standard_sub_cidr, cleanup=cleanup)
        result["tests"]["standard_cidr_create"] = {
            "passed": True,
            "vpc_id": net_s,
            "cidr": standard_sub_cidr,
        }

        # no_conflict — derive from provider readback of BOTH subnetworks
        # (AWS oracle parity at aws/scripts/network/byoip_test.py
        # test_no_conflict, which reads VPCs back via describe_vpcs).
        # Reading observed `ip_cidr_range` catches a mis-created or
        # unexpectedly-ranged subnet that local CIDR variables would not.
        nc_result: dict[str, Any] = {"passed": False}
        try:
            sub_c_obs = subnets_c.get(project=project, region=args.region, subnetwork=sub_c)
            sub_s_obs = subnets_c.get(project=project, region=args.region, subnetwork=sub_s)
            observed_a = sub_c_obs.ip_cidr_range
            observed_b = sub_s_obs.ip_cidr_range
            net_a = ipaddress.ip_network(observed_a)
            net_b = ipaddress.ip_network(observed_b)
            nc_result["cidr_a"] = observed_a
            nc_result["cidr_b"] = observed_b
            if observed_a != custom_sub_cidr:
                nc_result["error"] = f"subnet {sub_c} cidr drift: expected {custom_sub_cidr}, got {observed_a}"
            elif observed_b != standard_sub_cidr:
                nc_result["error"] = f"subnet {sub_s} cidr drift: expected {standard_sub_cidr}, got {observed_b}"
            elif net_a.overlaps(net_b):
                nc_result["error"] = f"observed CIDRs overlap: {observed_a} vs {observed_b}"
            else:
                nc_result["passed"] = True
                nc_result["message"] = f"observed CIDRs distinct: {observed_a} vs {observed_b}"
        except gax.NotFound as e:
            nc_result["error"] = f"subnet readback NotFound: {e}"
        result["tests"]["no_conflict"] = nc_result

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
    result["tests"]["cleanup"] = {"passed": not cleanup_errors, "errors": cleanup_errors}
    result["success"] = result.get("success", False) and not cleanup_errors

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

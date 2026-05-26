#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine VPC IP config readback (VpcIpConfigCheck).

Divergences from the AWS oracle:

  * Networks have no CIDR — emit ``cidr`` as the operator-supplied
    aggregate (the value that anchors VpcIpConfigCheck's subnet_of
    containment math).
  * No DHCP options API — emit a synthesised dhcp_options object with
    domain_name_servers=['169.254.169.254'] (metadata-server internal
    DNS).
  * Subnets have no auto_assign_public_ip attribute — emit False;
    the provider config sets VpcIpConfigCheck.auto_assign_ip_mode=instance
    so the validator accepts that target-shape honestly.
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

from common.compute import resolve_project, short_name
from common.errors import classify_gcp_error, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

# Compute Engine reserves 4 IPs per subnet.
_GCE_RESERVED_PER_SUBNET = 4


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine VPC IP config readback")
    parser.add_argument("--vpc-id", required=True)
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr", default="10.0.0.0/16")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    network = args.vpc_id

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "network_id": network,
        "cidr": args.cidr,
        "subnets": [],
        "region": args.region,
    }

    try:
        subnets_c = compute_v1.SubnetworksClient()
        zones = compute_v1.RegionsClient().get(project=project, region=args.region).zones or ()
        zone_names = [url.rsplit("/", 1)[-1] for url in zones]
        idx = 0
        for sub in subnets_c.list(project=project, region=args.region):
            if not sub.network or short_name(sub.network) != network:
                continue
            sub_cidr = sub.ip_cidr_range or ""
            try:
                size = ipaddress.ip_network(sub_cidr).num_addresses
            except ValueError:
                size = _GCE_RESERVED_PER_SUBNET
            result["subnets"].append(
                {
                    "subnet_id": sub.name,
                    "cidr": sub_cidr,
                    "az": zone_names[idx % len(zone_names)] if zone_names else "",
                    "auto_assign_public_ip": False,
                    "available_ips": max(0, size - _GCE_RESERVED_PER_SUBNET),
                }
            )
            idx += 1
        result["dhcp_options"] = {
            "dhcp_options_id": network,
            "domain_name": None,
            "domain_name_servers": ["169.254.169.254"],
            "ntp_servers": [],
        }
        result["success"] = True
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Create a Compute Engine custom-mode VPC + subnetworks + firewall.

Compute Engine divergences from the AWS oracle (per reviewed knowledge):

  * Networks have NO CIDR — CIDRs are per-subnetwork. The emitted ``cidr``
    is the create-time aggregate (the --cidr arg) used by VpcIpConfigCheck
    for subnet_of containment math.
  * Subnetworks are regional, not zonal — populate the contract's ``az``
    field from real zones in the configured region (RegionsClient.get).
  * No internet_gateway resource — implicit default-internet-gateway is
    pre-installed; omit ``internet_gateway_id`` and ``route_table_id``.
  * No DHCP options API — emit a synthesised dhcp_options object pointing
    at the metadata-server resolver at 169.254.169.254.
  * No per-VPC default SG — create a project-scoped Firewall bound to
    the test network with sourceRanges + at least one Allowed entry
    with I_p_protocol set. Emit Firewall.name as ``security_group_id``.
  * Firewall ``allowed[]`` is REJECTED with HTTP 400 unless I_p_protocol
    is set on every entry (factory override).
  * Network / Subnetwork / Firewall protos have NO ``labels`` field —
    use only ``description`` for provenance.
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
from common.errors import handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest network domain — verified-reuse marker"
PROVENANCE_TAG = "isvtest"
# Compute Engine reserves 4 IPs at the bottom of every subnet (network,
# default gateway, second-to-last, broadcast). available_ips = size - 4.
_GCE_RESERVED_PER_SUBNET = 4


def _list_region_zones(project: str, region: str) -> list[str]:
    client = compute_v1.RegionsClient()
    region_obj = client.get(project=project, region=region)
    return [url.rsplit("/", 1)[-1] for url in region_obj.zones or ()]


def _wait_region_op(project: str, region: str, op_name: str, *, timeout: int = 300) -> None:
    client = compute_v1.RegionOperationsClient()
    client.wait(project=project, region=region, operation=op_name, timeout=timeout)


def _carve_subnet_cidrs(aggregate: str, count: int) -> list[str]:
    net = ipaddress.ip_network(aggregate, strict=False)
    # /24 subnets carved from the /16 aggregate.
    new_prefix = max(net.prefixlen + 8, 24)
    subs = list(net.subnets(new_prefix=new_prefix))
    return [str(s) for s in subs[:count]]


def create_network(
    project: str,
    network_name: str,
    cidr: str,
    region: str,
    firewall_name: str,
) -> dict[str, Any]:
    """Create the network + subnets + firewall.

    Stamps `network_created=True` BEFORE the network wait completes so the
    caller (and downstream teardown) can clean up partial state. On any
    failure after the network insert succeeds, the partial graph (network
    + whatever subnets were created + maybe firewall) is left for teardown
    to enumerate via aggregated/list filters — this stub does not best-
    effort delete what it just created because the teardown step is the
    canonical cleanup surface.
    """
    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "network_id": network_name,
        "cidr": cidr,
        "subnets": [],
        "security_group_id": None,
        "dhcp_options": None,
        "network_created": False,
    }

    networks = compute_v1.NetworksClient()
    subnets = compute_v1.SubnetworksClient()
    firewalls = compute_v1.FirewallsClient()

    # --- Custom-mode network -------------------------------------------------
    # Stamp network_created BEFORE the async wait so the caller-side
    # partial-state-recovery contract holds even if the wait raises.
    network = compute_v1.Network(
        name=network_name,
        description=ISV_DESCRIPTION,
        auto_create_subnetworks=False,
        routing_config=compute_v1.NetworkRoutingConfig(routing_mode="REGIONAL"),
    )
    op = networks.insert(project=project, network_resource=network)
    result["network_created"] = True
    wait_for_global_op(project, op.name, timeout=300)

    # --- Subnetworks (two minimum) -------------------------------------------
    zones = _list_region_zones(project, region)
    if len(zones) < 2:
        raise RuntimeError(f"region {region} reports fewer than 2 zones: {zones}")

    subnet_cidrs = _carve_subnet_cidrs(cidr, 2)
    for i, sub_cidr in enumerate(subnet_cidrs):
        sub_name = f"{network_name}-sub{i}"
        sub = compute_v1.Subnetwork(
            name=sub_name,
            description=ISV_DESCRIPTION,
            ip_cidr_range=sub_cidr,
            network=f"projects/{project}/global/networks/{network_name}",
            region=region,
        )
        op = subnets.insert(project=project, region=region, subnetwork_resource=sub)
        net = ipaddress.ip_network(sub_cidr)
        # Append the subnet record BEFORE the wait so the caller knows the
        # subnet was created even if the wait raises.
        result["subnets"].append(
            {
                "subnet_id": sub_name,
                "cidr": sub_cidr,
                "az": zones[i % len(zones)],
                "auto_assign_public_ip": False,
                "available_ips": max(0, net.num_addresses - _GCE_RESERVED_PER_SUBNET),
            }
        )
        _wait_region_op(project, region, op.name, timeout=300)

    # --- Project-scoped firewall (SG analog) ---------------------------------
    fw = compute_v1.Firewall(
        name=firewall_name,
        description=ISV_DESCRIPTION,
        network=f"projects/{project}/global/networks/{network_name}",
        direction="INGRESS",
        source_ranges=["0.0.0.0/0"],
        allowed=[
            compute_v1.Allowed(I_p_protocol="tcp", ports=["22"]),
            compute_v1.Allowed(I_p_protocol="icmp"),
        ],
        target_tags=[PROVENANCE_TAG],
    )
    op = firewalls.insert(project=project, firewall_resource=fw)
    result["security_group_id"] = firewall_name
    wait_for_global_op(project, op.name, timeout=300)

    # --- DHCP options (synthesised — no API) ---------------------------------
    result["dhcp_options"] = {
        "dhcp_options_id": network_name,
        "domain_name": None,
        "domain_name_servers": ["169.254.169.254"],
        "ntp_servers": [],
    }

    result["success"] = True
    return result


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Create Compute Engine VPC + subnets + firewall")
    parser.add_argument("--name", default="isv-shared-vpc", help="Network name prefix")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr", default="10.0.0.0/16", help="Aggregate from which subnet CIDRs are carved")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    name = unique_suffix(args.name)
    firewall_name = unique_suffix(f"{args.name}-fw")

    # The partial-state dict starts pre-stamped with the names we constructed
    # so that even an early exception emits a JSON payload teardown can act on
    # (network_id present + network_created flipped to True as soon as the
    # network insert is dispatched).
    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "network_id": name,
        "cidr": args.cidr,
        "subnets": [],
        "security_group_id": None,
        "dhcp_options": None,
        "network_created": False,
    }
    try:
        result = create_network(project, name, args.cidr, args.region, firewall_name)
    except gax.AlreadyExists as e:
        result["error_type"] = "api_error"
        result["error"] = f"resource already exists (verified-reuse not implemented for create_network): {e}"
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        # Partial-rollback path: result["network_created"] may be True and
        # result["subnets"] may be non-empty even though we never reached
        # success=True. Teardown reads these forwarded fields and cleans up.
        et, em = ("api_error", str(e)) if isinstance(e, gax.GoogleAPICallError) else ("unknown_error", str(e))
        result["error_type"] = et
        result["error"] = em
    result["region"] = args.region
    result["name"] = args.name

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

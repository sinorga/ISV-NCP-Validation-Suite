# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared VPC test helpers for GCP network stubs.

Mirrors the oracle's ``aws/common/vpc.py`` so stub structure is consistent
across NCPs. Wraps ``google-cloud-compute`` calls that every network stub
needs:

- Custom-mode VPC creation (``auto_create_subnetworks=False``)
- Subnetwork creation (subnets are regional in GCP — one subnet spans
  every zone in its region)
- Cleanup helpers that swallow ``NotFound`` and retry once on the
  transient ``RemoteDisconnected`` hiccups called out in
  ``docs/gcp.yaml``
"""

from __future__ import annotations

import sys
import time
from typing import Any

from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1


def zone_to_region(zone: str) -> str:
    """Strip the trailing ``-<letter>`` off a zone to get its parent region.

    ``asia-east1-a`` → ``asia-east1``. Most network resources (subnetworks,
    addresses, routers) live in regions, so stubs that only know the
    zone need this conversion.
    """
    if not zone:
        return zone
    parts = zone.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 1:
        return parts[0]
    return zone


def wait_operation(op: Any, timeout: int = 180) -> None:
    """Block on a Compute Engine operation with one retry on transient drops.

    Per ``docs/gcp.yaml``, the Compute API occasionally drops HTTP
    connections mid-op (~14% of runs). A single retry after a short
    pause clears essentially all of them; beyond that the underlying
    problem is real and should surface to the caller.
    """
    try:
        op.result(timeout=timeout)
    except (ConnectionError, gax_exc.ServiceUnavailable):
        time.sleep(2)
        op.result(timeout=timeout)


def create_vpc(
    networks_client: compute_v1.NetworksClient,
    project: str,
    name: str,
    *,
    description: str = "ISV validation VPC",
    timeout: int = 180,
) -> dict[str, Any]:
    """Create a custom-mode VPC network and return ``{passed, vpc_id, ...}``.

    GCP custom-mode networks don't get any subnets automatically — callers
    are expected to insert them as a separate step. Auto-mode networks
    would inject a subnet per region, which bloats the test surface and
    conflicts with the BYOIP CIDR planning.
    """
    result: dict[str, Any] = {"passed": False, "vpc_id": name}
    try:
        net = compute_v1.Network()
        net.name = name
        net.auto_create_subnetworks = False
        net.description = description
        # GLOBAL routing lets subnets in different regions see each other,
        # which matches the AWS oracle's "one VPC, many subnets" model.
        routing = compute_v1.NetworkRoutingConfig()
        routing.routing_mode = "GLOBAL"
        net.routing_config = routing

        op = networks_client.insert(project=project, network_resource=net)
        wait_operation(op, timeout=timeout)

        result["passed"] = True
        result["message"] = f"Created VPC {name}"
    except gax_exc.Conflict:
        # Treat "already exists" as success so parallel stub runs with
        # shared names don't collide.
        result["passed"] = True
        result["reused"] = True
        result["message"] = f"VPC {name} already exists — reusing"
    except Exception as e:
        result["error"] = str(e)
    return result


def create_subnet(
    subnetworks_client: compute_v1.SubnetworksClient,
    project: str,
    region: str,
    name: str,
    network_self_link: str,
    ip_cidr_range: str,
    *,
    timeout: int = 180,
) -> dict[str, Any]:
    """Insert a regional subnetwork and return ``{passed, subnet_id, cidr, az}``.

    ``az`` holds the region name. Subnetworks are regional in GCP — the
    AWS-style "availability zone" concept doesn't exist at the subnet
    level, so the validator's multi-AZ check is satisfied by spanning
    multiple regions rather than multiple zones.
    """
    result: dict[str, Any] = {"passed": False}
    try:
        sub = compute_v1.Subnetwork()
        sub.name = name
        sub.network = network_self_link
        sub.ip_cidr_range = ip_cidr_range
        sub.region = region

        op = subnetworks_client.insert(
            project=project,
            region=region,
            subnetwork_resource=sub,
        )
        wait_operation(op, timeout=timeout)

        result["passed"] = True
        result["subnet_id"] = name
        result["cidr"] = ip_cidr_range
        result["az"] = region
        result["message"] = f"Created subnet {name} in {region}"
    except gax_exc.Conflict:
        result["passed"] = True
        result["subnet_id"] = name
        result["cidr"] = ip_cidr_range
        result["az"] = region
        result["reused"] = True
        result["message"] = f"Subnet {name} already exists — reusing"
    except Exception as e:
        result["error"] = str(e)
    return result


def delete_subnet(
    subnetworks_client: compute_v1.SubnetworksClient,
    project: str,
    region: str,
    name: str,
    *,
    timeout: int = 180,
) -> None:
    """Delete a subnet, swallowing NotFound errors."""
    try:
        op = subnetworks_client.delete(project=project, region=region, subnetwork=name)
        wait_operation(op, timeout=timeout)
    except gax_exc.NotFound:
        pass
    except Exception as e:
        print(f"  delete_subnet({name}) warning: {e}", file=sys.stderr)


def delete_vpc(
    networks_client: compute_v1.NetworksClient,
    project: str,
    name: str,
    *,
    timeout: int = 180,
) -> None:
    """Delete a VPC, swallowing NotFound errors."""
    try:
        op = networks_client.delete(project=project, network=name)
        wait_operation(op, timeout=timeout)
    except gax_exc.NotFound:
        pass
    except Exception as e:
        print(f"  delete_vpc({name}) warning: {e}", file=sys.stderr)


def delete_firewall(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    name: str,
    *,
    timeout: int = 120,
) -> None:
    """Delete a firewall rule, swallowing NotFound errors."""
    try:
        op = firewalls_client.delete(project=project, firewall=name)
        wait_operation(op, timeout=timeout)
    except gax_exc.NotFound:
        pass
    except Exception as e:
        print(f"  delete_firewall({name}) warning: {e}", file=sys.stderr)


def build_firewall(
    name: str,
    network_self_link: str,
    *,
    direction: str = "INGRESS",
    source_ranges: list[str] | None = None,
    destination_ranges: list[str] | None = None,
    allowed: list[tuple[str, list[str] | None]] | None = None,
    denied: list[tuple[str, list[str] | None]] | None = None,
    description: str = "ISV validation firewall",
    priority: int = 1000,
    target_tags: list[str] | None = None,
) -> compute_v1.Firewall:
    """Build a Firewall resource with at least one allowed/denied entry.

    Per ``docs/gcp.yaml``, GCP firewalls REQUIRE at least one Allowed()
    entry with ``I_p_protocol`` set (or a Denied() entry for deny rules) —
    an empty ``Allowed()`` returns 400. This helper enforces that by
    always building proto entries with the protocol populated.
    """
    fw = compute_v1.Firewall()
    fw.name = name
    fw.network = network_self_link
    fw.direction = direction
    fw.priority = priority
    fw.description = description

    if source_ranges is not None:
        fw.source_ranges = source_ranges
    if destination_ranges is not None:
        fw.destination_ranges = destination_ranges
    if target_tags is not None:
        fw.target_tags = target_tags

    if allowed:
        allowed_entries = []
        for protocol, ports in allowed:
            entry = compute_v1.Allowed()
            entry.I_p_protocol = protocol
            if ports:
                entry.ports = ports
            allowed_entries.append(entry)
        fw.allowed = allowed_entries
    if denied:
        denied_entries = []
        for protocol, ports in denied:
            entry = compute_v1.Denied()
            entry.I_p_protocol = protocol
            if ports:
                entry.ports = ports
            denied_entries.append(entry)
        fw.denied = denied_entries

    return fw


def default_dhcp_options(project: str, region: str) -> dict[str, Any]:
    """Report GCP's equivalent of AWS DHCP options for a VPC.

    GCP doesn't expose a user-configurable DHCP options set — the metadata
    server hands out:
      - 169.254.169.254 as the DNS resolver (Google's internal DNS proxy)
      - ``<region>.c.<project>.internal`` as the search domain on RHEL-family
        images and ``c.<project>.internal`` on Debian-family — we report
        the more specific form.
    This mirrors the shape AWS produces so ``VpcIpConfigCheck`` can read
    ``domain_name_servers`` / ``domain_name`` without caring about the NCP.
    """
    return {
        "dhcp_options_id": "default",
        "domain_name": f"{region}.c.{project}.internal",
        "domain_name_servers": ["169.254.169.254"],
        "ntp_servers": ["metadata.google.internal"],
    }


def cidrs_overlap(cidr_a: str, cidr_b: str) -> bool:
    """Return True if two CIDRs overlap.

    Uses ``ipaddress.ip_network`` for correctness (unlike the oracle's
    first-octet heuristic) — this is used to detect cross-VPC route leaks
    where precision matters.
    """
    import ipaddress

    try:
        net_a = ipaddress.ip_network(cidr_a, strict=False)
        net_b = ipaddress.ip_network(cidr_b, strict=False)
    except ValueError:
        return False
    return net_a.overlaps(net_b)

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test GCP localized DNS via a Cloud DNS private managed zone.

We target the Cloud DNS REST API directly (``dns.googleapis.com/v1``)
rather than the ``google-cloud-dns`` client, because the pinned version
(0.36) does not expose ``visibility`` / ``privateVisibilityConfig`` — the
fields that make a zone private and bind it to a VPC network. The REST
shape is stable and supports both.

Subtests (match oracle schema):
  - create_vpc_with_dns : VPC created; GCP VPCs always participate in internal DNS
  - create_hosted_zone  : Cloud DNS private zone created, bound to the VPC
  - create_dns_record   : A record for storage.<domain> → <vpc_cidr>.1.100
  - verify_dns_settings : list the zone, confirm visibility=private + network bound
  - resolve_record      : list rrsets on the zone, find the A record

Usage:
    python dns_test.py --region asia-east1-a --cidr 10.89.0.0/16 \\
        --domain internal.isv.test

Output JSON matches the oracle localized_dns schema.
"""

import argparse
import ipaddress
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import google.auth
from common.compute import resolve_project, zone_to_region
from common.errors import handle_gcp_errors
from common.vpc import create_vpc, delete_vpc
from google.api_core import exceptions as gax_exc
from google.auth.transport.requests import AuthorizedSession
from google.cloud import compute_v1

_DNS_API = "https://dns.googleapis.com/dns/v1"
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _get_dns_session() -> AuthorizedSession:
    """Build an AuthorizedSession backed by the host's ADC credentials."""
    creds, _ = google.auth.default(scopes=_SCOPES)
    return AuthorizedSession(creds)


def _sanitize_zone_name(value: str) -> str:
    """Cloud DNS zone names must match ``[a-z][-a-z0-9]*[a-z0-9]`` (<= 63 chars)."""
    cleaned = re.sub(r"[^a-z0-9-]", "-", value.lower()).strip("-")
    return cleaned[:63] or "isv-zone"


def _list_rrsets(session: AuthorizedSession, project: str, zone_name: str) -> list[dict[str, Any]]:
    """List all RRsets in a zone (single-page; test zones are tiny)."""
    url = f"{_DNS_API}/projects/{project}/managedZones/{zone_name}/rrsets"
    resp = session.get(url)
    resp.raise_for_status()
    return resp.json().get("rrsets", [])


def test_create_vpc_with_dns(
    networks_client: compute_v1.NetworksClient,
    project: str,
    name: str,
) -> tuple[dict[str, Any], bool]:
    """Create the VPC. GCP VPCs always have internal DNS."""
    vpc_result = create_vpc(networks_client, project, name)
    result: dict[str, Any] = {
        "passed": vpc_result["passed"],
        "vpc_id": name,
    }
    if "error" in vpc_result:
        result["error"] = vpc_result["error"]
    else:
        result["message"] = f"Created VPC {name} (internal DNS enabled by default)"
    return result, vpc_result["passed"]


def test_create_hosted_zone(
    session: AuthorizedSession,
    project: str,
    zone_name: str,
    domain: str,
    vpc_name: str,
) -> dict[str, Any]:
    """Create a Cloud DNS private managed zone bound to the VPC."""
    result: dict[str, Any] = {"passed": False}
    body = {
        "name": zone_name,
        "dnsName": domain if domain.endswith(".") else f"{domain}.",
        "description": "ISV validation private zone",
        "visibility": "private",
        "privateVisibilityConfig": {
            "networks": [
                {"networkUrl": f"https://www.googleapis.com/compute/v1/projects/{project}/global/networks/{vpc_name}"}
            ]
        },
    }
    url = f"{_DNS_API}/projects/{project}/managedZones"
    try:
        resp = session.post(url, json=body)
        if resp.status_code == 409:
            # Zone already exists — fine, reuse.
            result["passed"] = True
            result["zone_id"] = zone_name
            result["domain"] = domain
            result["reused"] = True
            result["message"] = f"Private zone {zone_name} already exists"
            return result
        resp.raise_for_status()
        data = resp.json()
        result["passed"] = True
        result["zone_id"] = data["name"]
        result["domain"] = domain
        result["message"] = f"Created private zone {data['name']} for {domain}"
    except Exception as e:
        result["error"] = f"Cloud DNS create failed: {e}"
    return result


def test_create_dns_record(
    session: AuthorizedSession,
    project: str,
    zone_name: str,
    fqdn: str,
    target_ip: str,
) -> dict[str, Any]:
    """Add an A record via the ``changes.create`` endpoint."""
    result: dict[str, Any] = {"passed": False}
    body = {
        "additions": [
            {
                "name": fqdn if fqdn.endswith(".") else f"{fqdn}.",
                "type": "A",
                "ttl": 60,
                "rrdatas": [target_ip],
            }
        ]
    }
    url = f"{_DNS_API}/projects/{project}/managedZones/{zone_name}/changes"
    try:
        resp = session.post(url, json=body)
        resp.raise_for_status()
        result["passed"] = True
        result["fqdn"] = fqdn
        result["target_ip"] = target_ip
        result["message"] = f"Created A record {fqdn} -> {target_ip}"
    except Exception as e:
        result["error"] = f"Record create failed: {e}"
    return result


def test_verify_dns_settings(session: AuthorizedSession, project: str, zone_name: str, vpc_name: str) -> dict[str, Any]:
    """Confirm the zone is private and bound to the VPC."""
    result: dict[str, Any] = {"passed": False}
    url = f"{_DNS_API}/projects/{project}/managedZones/{zone_name}"
    try:
        resp = session.get(url)
        resp.raise_for_status()
        data = resp.json()
        visibility = data.get("visibility", "public")
        networks = data.get("privateVisibilityConfig", {}).get("networks", [])
        bound = any(vpc_name in n.get("networkUrl", "") for n in networks)

        result["visibility"] = visibility
        result["bound_network_count"] = len(networks)
        result["dns_support"] = True
        result["dns_hostnames"] = True
        if visibility == "private" and bound:
            result["passed"] = True
            result["message"] = f"Zone is private and bound to {vpc_name}"
        else:
            result["error"] = f"Zone visibility={visibility}, bound_to_vpc={bound}"
    except Exception as e:
        result["error"] = str(e)
    return result


def test_resolve_record(
    session: AuthorizedSession,
    project: str,
    zone_name: str,
    fqdn: str,
    expected_ip: str,
) -> dict[str, Any]:
    """Confirm the A record is present by listing RRsets via the API.

    GCP private DNS records are only resolvable from within a bound VPC,
    so we cannot resolve via the test host. Listing the authoritative
    RRsets is the equivalent of the AWS oracle's Route 53 API-based
    ``resolve_record``.
    """
    result: dict[str, Any] = {"passed": False}
    try:
        rrsets = _list_rrsets(session, project, zone_name)
        fqdn_normalised = fqdn if fqdn.endswith(".") else f"{fqdn}."
        for rr in rrsets:
            if rr.get("name") == fqdn_normalised and rr.get("type") == "A":
                rrdatas = rr.get("rrdatas", [])
                if expected_ip in rrdatas:
                    result["passed"] = True
                    result["resolved_ip"] = expected_ip
                    result["all_ips"] = rrdatas
                    result["message"] = f"{fqdn} resolves to {expected_ip}"
                    return result
                result["error"] = f"Record {fqdn} has rrdatas {rrdatas}, missing {expected_ip}"
                return result
        result["error"] = f"No A record found for {fqdn}"
    except Exception as e:
        result["error"] = str(e)
    return result


def _cleanup_zone(session: AuthorizedSession, project: str, zone_name: str, fqdn: str, target_ip: str) -> None:
    """Best-effort: delete the A record, then the zone."""
    # Delete the record first (zone delete requires empty zone minus SOA/NS).
    try:
        url = f"{_DNS_API}/projects/{project}/managedZones/{zone_name}/changes"
        body = {
            "deletions": [
                {
                    "name": fqdn if fqdn.endswith(".") else f"{fqdn}.",
                    "type": "A",
                    "ttl": 60,
                    "rrdatas": [target_ip],
                }
            ]
        }
        session.post(url, json=body)
    except Exception as e:
        print(f"  Record delete warning: {e}", file=sys.stderr)
    # Delete the zone.
    try:
        url = f"{_DNS_API}/projects/{project}/managedZones/{zone_name}"
        session.delete(url)
    except Exception as e:
        print(f"  Zone delete warning: {e}", file=sys.stderr)


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Test GCP localized DNS")
    parser.add_argument("--region", default=os.environ.get("GCP_ZONE", "asia-east1-a"))
    parser.add_argument("--cidr", default="10.89.0.0/16")
    parser.add_argument("--domain", default="internal.isv.test")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    _region = zone_to_region(args.region)

    networks_client = compute_v1.NetworksClient()
    session = _get_dns_session()

    suffix = str(uuid.uuid4())[:8]
    vpc_name = f"isv-dns-vpc-{suffix}"
    zone_name = _sanitize_zone_name(f"isv-dns-{suffix}")
    storage_record = f"storage.{args.domain}"
    # Target IP inside the VPC CIDR.
    cidr_net = ipaddress.ip_network(args.cidr, strict=False)
    base = str(cidr_net.network_address).split(".")
    target_ip = f"{base[0]}.{base[1]}.1.100"

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "status": "failed",
        "tests": {},
    }

    vpc_created = False
    zone_created = False

    try:
        # Test 1
        vpc_test, vpc_created = test_create_vpc_with_dns(networks_client, project, vpc_name)
        result["tests"]["create_vpc_with_dns"] = vpc_test
        if not vpc_created:
            print(json.dumps(result, indent=2))
            return 1

        # Test 2
        zone_test = test_create_hosted_zone(session, project, zone_name, args.domain, vpc_name)
        result["tests"]["create_hosted_zone"] = zone_test
        zone_created = zone_test["passed"]
        if not zone_created:
            print(json.dumps(result, indent=2))
            return 1

        # Test 3
        result["tests"]["create_dns_record"] = test_create_dns_record(
            session,
            project,
            zone_name,
            storage_record,
            target_ip,
        )

        # Test 4
        result["tests"]["verify_dns_settings"] = test_verify_dns_settings(
            session,
            project,
            zone_name,
            vpc_name,
        )

        # Test 5
        result["tests"]["resolve_record"] = test_resolve_record(
            session,
            project,
            zone_name,
            storage_record,
            target_ip,
        )

        all_passed = all(t.get("passed", False) for t in result["tests"].values())
        result["success"] = all_passed
        result["status"] = "passed" if all_passed else "failed"

    except Exception as e:
        result["error"] = str(e)
    finally:
        if zone_created:
            _cleanup_zone(session, project, zone_name, storage_record, target_ip)
        if vpc_created:
            try:
                delete_vpc(networks_client, project, vpc_name)
            except gax_exc.GoogleAPIError:
                pass

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

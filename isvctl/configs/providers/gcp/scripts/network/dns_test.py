#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine localized DNS test (LocalizedDnsCheck).

Divergences from the AWS oracle:

  * No DNS toggles on Network — internal DNS is unconditional. Emit
    dns_support=True / dns_hostnames=True with a platform-difference
    message.
  * Private hosted zone is a Cloud DNS ManagedZone with visibility=private
    + privateVisibilityConfig.networks=[<network>]. The high-level
    `ManagedZone.visibility` / `_properties` set-property pattern is
    SILENTLY DROPPED by google-cloud-dns 0.36.x's
    `ManagedZone._build_resource()` — POST the explicit body via the
    lower-level connection (vendor-API override).
"""

from __future__ import annotations

import argparse
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
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1

ISV_DESCRIPTION = "isvtest dns_test — verified-reuse marker"


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine localized DNS")
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--cidr", default="10.89.0.0/16")
    parser.add_argument("--domain", default="internal.isv.test")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    project = resolve_project(args.project)
    network = unique_suffix("isv-dns")
    domain = args.domain if args.domain.endswith(".") else args.domain + "."
    zone_name = unique_suffix("isv-dnszn")
    record_fqdn = f"probe.{domain}"
    target_ip = "10.89.0.10"

    networks_c = compute_v1.NetworksClient()

    result: dict[str, Any] = {"success": False, "platform": "network", "tests": {}}
    cleanup_zone = False
    network_created = False
    try:
        op = networks_c.insert(
            project=project,
            network_resource=compute_v1.Network(
                name=network,
                description=ISV_DESCRIPTION,
                auto_create_subnetworks=False,
            ),
        )
        network_created = True
        wait_for_global_op(project, op.name, timeout=300)
        result["tests"]["create_vpc_with_dns"] = {"passed": True, "vpc_id": network}

        # Cloud DNS — bypass the silently-dropped set-property pattern.
        try:
            from google.cloud import dns as gdns  # type: ignore[attr-defined]
        except ImportError:
            result["error"] = "google-cloud-dns not installed"
            print(json.dumps(result, indent=2))
            return 1

        client = gdns.Client(project=project)
        body = {
            "name": zone_name,
            "dnsName": domain,
            "description": ISV_DESCRIPTION,
            "visibility": "private",
            "privateVisibilityConfig": {
                "networks": [
                    {
                        "networkUrl": f"https://www.googleapis.com/compute/v1/projects/{project}/global/networks/{network}"
                    }
                ]
            },
            "labels": {"createdby": "isvtest"},
        }
        client._connection.api_request(
            method="POST",
            path=f"/projects/{project}/managedZones",
            data=body,
        )
        cleanup_zone = True

        zone = client.zone(zone_name)
        zone.reload()
        is_private = zone._properties.get("visibility") == "private"
        result["tests"]["create_hosted_zone"] = {
            "passed": is_private,
            "zone_id": zone_name,
            "domain": domain,
        }

        # Insert A record.
        changes = zone.changes()
        rrs = zone.resource_record_set(record_fqdn, "A", 300, [target_ip])
        changes.add_record_set(rrs)
        changes.create()
        result["tests"]["create_dns_record"] = {
            "passed": True,
            "fqdn": record_fqdn,
            "target_ip": target_ip,
        }

        result["tests"]["verify_dns_settings"] = {
            "passed": True,
            "dns_support": True,
            "dns_hostnames": True,
            "message": "Compute Engine internal DNS is always on",
        }

        # resolve_record — read back via SDK.
        records = list(zone.list_resource_record_sets())
        match = [r for r in records if r.name == record_fqdn and r.record_type == "A"]
        resolved = match[0].rrdatas[0] if match else None
        result["tests"]["resolve_record"] = {
            "passed": resolved == target_ip,
            "resolved_ip": resolved,
            "all_ips": match[0].rrdatas if match else [],
        }

        result["success"] = all(t.get("passed", False) for t in result["tests"].values())
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)
    except Exception as e:
        result["error"] = str(e)
    finally:
        if cleanup_zone:
            try:
                from google.cloud import dns as gdns  # type: ignore[attr-defined]

                client = gdns.Client(project=project)
                z = client.zone(zone_name)
                z.reload()
                # Delete record sets first.
                ch = z.changes()
                for r in z.list_resource_record_sets():
                    if r.record_type in ("NS", "SOA"):
                        continue
                    ch.delete_record_set(r)
                if list(ch.deletions):
                    ch.create()
                z.delete()
            except Exception:
                pass
        if network_created:
            delete_with_retry(
                lambda: wait_for_global_op(
                    project,
                    networks_c.delete(project=project, network=network).name,
                    timeout=180,
                ),
                resource_desc=f"network {network}",
            )

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

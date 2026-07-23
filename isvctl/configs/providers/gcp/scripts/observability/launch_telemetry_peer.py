# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch one internal-only peer for the east-west telemetry fixture."""

from __future__ import annotations

import argparse
import ipaddress
import json
import secrets
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import (
    bounded_unique_name,
    first_internal_ip,
    get_instance,
    poll_instance_state,
    resolve_image,
    resolve_project,
)
from common.errors import classify_gcp_error, delete_with_retry
from common.network import (
    build_firewall,
    build_probe_instance,
    delete_firewall,
    delete_instance,
    insert_firewall,
    insert_instance,
    make_allowed,
)


def _source_range(value: str) -> str:
    """Normalize the primary VM address to one exact IPv4 /32."""
    return f"{ipaddress.IPv4Address(value)}/32"


def launch_peer(
    project: str,
    *,
    name: str,
    firewall_name: str,
    zone: str,
    network_id: str,
    subnet_id: str,
    source_private_ip: str,
    machine_type: str,
    image: str,
    image_project: str,
    skip_destroy: bool = False,
) -> dict[str, Any]:
    """Create the exact peer instance and ICMP rule, with ownership evidence."""
    disc = secrets.token_hex(2)
    peer_name = bounded_unique_name(name, disc)
    peer_firewall = bounded_unique_name(firewall_name, disc)
    source_range = _source_range(source_private_ip)
    result: dict[str, Any] = {
        "success": False,
        "platform": "observability",
        "test_name": "launch_telemetry_peer",
        "instance_id": peer_name,
        "zone": zone,
        "private_ip": "",
        "instance_created": False,
        "firewall_name": peer_firewall,
        "firewall_created": False,
        "source_range": source_range,
    }

    try:
        firewall = build_firewall(
            peer_firewall,
            network_id,
            project,
            allowed=[make_allowed("icmp")],
            source_ranges=[source_range],
            target_tags=[peer_name],
        )

        def _own_firewall() -> None:
            result["firewall_created"] = True

        insert_firewall(project, firewall, on_accepted=_own_firewall)

        source_image = resolve_image(image_project, image).self_link
        peer = build_probe_instance(
            project=project,
            zone=zone,
            name=peer_name,
            network_name=network_id,
            subnet_name=subnet_id,
            machine_type=machine_type,
            source_image=source_image,
            external_ip=False,
            network_tags=[peer_name],
        )

        def _own_instance() -> None:
            result["instance_created"] = True

        insert_instance(project, zone, peer, on_accepted=_own_instance)
        poll_instance_state(
            project,
            zone,
            peer_name,
            target_canonical="running",
            timeout=300,
        )
        result["private_ip"] = first_internal_ip(get_instance(project, zone, peer_name)) or ""
        if not result["private_ip"]:
            raise RuntimeError("Telemetry peer reached RUNNING without an internal IP")
        result["success"] = True
    except Exception as exc:
        error_type, error = classify_gcp_error(exc)
        result["error_type"] = error_type
        result["error"] = error
        if skip_destroy:
            # Preservation mode: SUPPRESS the compensating deletion so an operator
            # can inspect the partially launched peer. The exact run-owned
            # identifiers are already emitted (instance_id/instance_created,
            # firewall_name/firewall_created); surface them explicitly.
            # teardown_telemetry_peer (also --skip-destroy) leaves them be.
            result["skip_destroy"] = True
            result["preserved_on_failure"] = {
                "instance_id": peer_name if result["instance_created"] else "",
                "zone": zone if result["instance_created"] else "",
                "firewall_name": peer_firewall if result["firewall_created"] else "",
            }
            print(
                "Skip-destroy set: preserving run-owned peer resources on setup failure "
                f"(instance_created={result['instance_created']} instance={peer_name!r}@{zone!r}, "
                f"firewall_created={result['firewall_created']} firewall={peer_firewall!r})",
                file=sys.stderr,
            )
        else:
            if result["instance_created"]:
                delete_with_retry(
                    delete_instance,
                    project,
                    zone,
                    peer_name,
                    resource_desc=f"instance {peer_name}@{zone}",
                )
            if result["firewall_created"]:
                delete_with_retry(
                    delete_firewall,
                    project,
                    peer_firewall,
                    resource_desc=f"firewall {peer_firewall}",
                )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch an internal telemetry peer")
    parser.add_argument("--name", default="isv-observability-peer")
    parser.add_argument("--firewall-name", default="isv-observability-peer-icmp")
    parser.add_argument("--zone", required=True)
    parser.add_argument("--network-id", required=True)
    parser.add_argument("--subnet-id", required=True)
    parser.add_argument("--source-private-ip", required=True)
    parser.add_argument("--instance-type", default="e2-small")
    parser.add_argument("--image", default="ubuntu-2204-lts")
    parser.add_argument("--image-project", default="ubuntu-os-cloud")
    parser.add_argument("--project", default=None)
    parser.add_argument(
        "--skip-destroy",
        action="store_true",
        help=(
            "Preserve run-owned resources on setup failure instead of running the "
            "compensating deletion (GCP_OBSERVABILITY_SKIP_TEARDOWN passthrough)."
        ),
    )
    args = parser.parse_args()
    result = launch_peer(
        resolve_project(args.project),
        name=args.name,
        firewall_name=args.firewall_name,
        zone=args.zone,
        network_id=args.network_id,
        subnet_id=args.subnet_id,
        source_private_ip=args.source_private_ip,
        machine_type=args.instance_type,
        image=args.image,
        image_project=args.image_project,
        skip_destroy=args.skip_destroy,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compute Engine network teardown.

Verified-reuse cleanup contract: gate every delete on the
``network_created`` ownership bit forwarded from create_network.
Delete order: instances → firewalls → subnets → routes → peerings →
addresses → network. NotFound counts as success. Local SSH key files
(forwarded via --key-file / --key-name / --key-created) are deleted
even when the cloud read returns NotFound (vendor-API contract).

Auto-routes (Route.name starts with "default-route-" with
next_hop_network set) cannot be deleted via routes.delete (HTTP 400
"The local route cannot be deleted"); they are reaped automatically
on subnet deletion. Filter them out before any delete.
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

from common.compute import (
    resolve_project,
    short_name,
    wait_for_global_op,
    wait_for_region_op,
    wait_for_zonal_op,
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from google.api_core import exceptions as gax
from google.cloud import compute_v1
from requests import exceptions as rqx


def _is_auto_route(route: compute_v1.Route) -> bool:
    name = route.name or ""
    return name.startswith("default-route-") or bool(route.next_hop_network)


def _is_not_ready(e: Exception) -> bool:
    """GCE returns 400 'is not ready' on subnet/network deletes while the
    backend resource is still settling from a recent op. This is a
    transient state that the common ``delete_with_retry`` treats as
    non-transient (BadRequest is in the catch-all branch). Mirrors the
    pattern vpc_crud_test introduced for post-insert mutations.
    """
    if not isinstance(e, gax.BadRequest):
        return False
    msg = str(e)
    return "is not ready" in msg or "resourceNotReady" in msg


# Connection-level transients raised by requests/urllib3 underneath the
# compute_v1 REST transport during the wait_for_*_op poll loops. These
# are network hiccups, not GCE-state errors — the operation itself is
# still progressing server-side, so retrying the SAME wait_for_*_op call
# is safe (op_name is stable, a polled DONE returns immediately).
_TRANSIENT_CONNECTION_EXCEPTIONS = (
    rqx.SSLError,
    rqx.ConnectionError,
    rqx.Timeout,
    rqx.ChunkedEncodingError,
)


def _with_readiness_retry(
    op_fn: Callable[[], Any],
    *,
    attempts: int = 12,
    backoff: float = 5.0,
    desc: str = "resource",
) -> Any:
    """Retry ``op_fn`` on GCE 'is not ready' BadRequest plus transient
    connection errors (SSL EOF / ConnectionReset) from the underlying
    requests transport.
    """
    for attempt in range(1, attempts + 1):
        try:
            return op_fn()
        except gax.BadRequest as e:
            if not _is_not_ready(e) or attempt >= attempts:
                raise
            print(
                f"  {desc} not yet ready ({type(e).__name__}); attempt {attempt}/{attempts}, sleeping {backoff}s",
                file=sys.stderr,
            )
            time.sleep(backoff)
        except _TRANSIENT_CONNECTION_EXCEPTIONS as e:
            if attempt >= attempts:
                raise
            print(
                f"  {desc} transient connection ({type(e).__name__}); attempt {attempt}/{attempts}, sleeping {backoff}s",
                file=sys.stderr,
            )
            time.sleep(backoff)
    raise RuntimeError(f"unreachable: readiness retry loop exited without return for {desc}")


def _delete_network_resources(
    project: str,
    network: str,
    region: str,
) -> dict[str, Any]:
    deleted = {
        "instances": [],
        "firewalls": [],
        "subnets": [],
        "routes": [],
        "peerings": [],
        "addresses": [],
        "vpc": None,
        "internet_gateways": [],
    }
    successes: list[bool] = []
    # Enumeration failures (list/get) MUST be recorded — a failed list can
    # skip an entire resource class and silently leak; AWS oracle treats
    # describe errors as hard failures via the outer ClientError catch in
    # main(). NotFound on the network-level get (peerings) IS success
    # because the network has been deleted already.
    enum_errors: list[str] = []
    network_self = f"https://www.googleapis.com/compute/v1/projects/{project}/global/networks/{network}"

    instances_c = compute_v1.InstancesClient()
    firewalls_c = compute_v1.FirewallsClient()
    subnets_c = compute_v1.SubnetworksClient()
    routes_c = compute_v1.RoutesClient()
    networks_c = compute_v1.NetworksClient()

    # Instances first — aggregated list across all zones.
    try:
        agg = instances_c.aggregated_list(project=project)
        for zone_name, scoped in agg:
            if not scoped.instances:
                continue
            zone_short = zone_name.replace("zones/", "")
            for inst in scoped.instances:
                in_network = any(short_name(ni.network) == network for ni in inst.network_interfaces or ())
                if not in_network:
                    continue
                ok = delete_with_retry(
                    lambda nn=inst.name, zz=zone_short: wait_for_zonal_op(
                        project,
                        zz,
                        instances_c.delete(project=project, zone=zz, instance=nn).name,
                        timeout=300,
                    ),
                    resource_desc=f"instance {inst.name}",
                )
                successes.append(ok)
                deleted["instances"].append(inst.name)
    except gax.NotFound as e:
        # Project-level aggregated_list returning NotFound means the project
        # itself is gone — treat as no work to do, do not record as failure.
        print(f"WARN: instance aggregated_list NotFound: {e}", file=sys.stderr)
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        msg = f"instance enumeration failed: {e}"
        print(f"WARN: {msg}", file=sys.stderr)
        enum_errors.append(msg)

    # Firewalls.
    try:
        for fw in firewalls_c.list(
            request=compute_v1.ListFirewallsRequest(project=project, filter=f'network="{network_self}"'),
        ):
            ok = delete_with_retry(
                lambda nn=fw.name: wait_for_global_op(
                    project,
                    firewalls_c.delete(project=project, firewall=nn).name,
                    timeout=120,
                ),
                resource_desc=f"firewall {fw.name}",
            )
            successes.append(ok)
            deleted["firewalls"].append(fw.name)
    except gax.NotFound as e:
        print(f"WARN: firewall list NotFound: {e}", file=sys.stderr)
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        msg = f"firewall enumeration failed: {e}"
        print(f"WARN: {msg}", file=sys.stderr)
        enum_errors.append(msg)

    # Routes (skip auto-routes).
    try:
        for r in routes_c.list(
            request=compute_v1.ListRoutesRequest(project=project, filter=f'network="{network_self}"'),
        ):
            if _is_auto_route(r):
                continue
            ok = delete_with_retry(
                lambda nn=r.name: wait_for_global_op(
                    project,
                    routes_c.delete(project=project, route=nn).name,
                    timeout=120,
                ),
                resource_desc=f"route {r.name}",
            )
            successes.append(ok)
            deleted["routes"].append(r.name)
    except gax.NotFound as e:
        print(f"WARN: route list NotFound: {e}", file=sys.stderr)
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        msg = f"route enumeration failed: {e}"
        print(f"WARN: {msg}", file=sys.stderr)
        enum_errors.append(msg)

    # Peerings — remove on this network's side. NotFound on the network
    # get means the network is already gone (peerings reaped with it);
    # any other API error during the read MUST count as enumeration
    # failure so the step does not exit success while leaving peerings.
    try:
        net_obj = networks_c.get(project=project, network=network)
        for p in net_obj.peerings or ():
            ok = delete_with_retry(
                lambda pp=p.name: wait_for_global_op(
                    project,
                    networks_c.remove_peering(
                        project=project,
                        network=network,
                        networks_remove_peering_request_resource=compute_v1.NetworksRemovePeeringRequest(name=pp),
                    ).name,
                    timeout=180,
                ),
                resource_desc=f"peering {p.name}",
            )
            successes.append(ok)
            deleted["peerings"].append(p.name)
    except gax.NotFound:
        pass
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        msg = f"peering enumeration failed: {e}"
        print(f"WARN: {msg}", file=sys.stderr)
        enum_errors.append(msg)

    # Subnets (regional). Subnets can sit in a 'not ready' window for
    # tens of seconds after their parent network's recent ops; wrap the
    # delete in a readiness retry so transient 400s don't propagate as
    # non-transient failures.
    try:
        for sub in subnets_c.list(project=project, region=region):
            if short_name(sub.network) == network:
                ok = delete_with_retry(
                    lambda nn=sub.name: _with_readiness_retry(
                        lambda: wait_for_region_op(
                            project,
                            region,
                            subnets_c.delete(project=project, region=region, subnetwork=nn).name,
                            timeout=180,
                        ),
                        desc=f"subnet {nn}",
                    ),
                    resource_desc=f"subnet {sub.name}",
                )
                successes.append(ok)
                deleted["subnets"].append(sub.name)
    except gax.NotFound as e:
        print(f"WARN: subnet list NotFound: {e}", file=sys.stderr)
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        msg = f"subnet enumeration failed: {e}"
        print(f"WARN: {msg}", file=sys.stderr)
        enum_errors.append(msg)

    # Addresses are intentionally NOT swept here. Regional addresses are
    # not attached to a specific network — `addresses.list` returns every
    # address in the region, and a same-project parallel run (or an
    # unrelated address minted with a `createdby=isvtest` label by a
    # different ISV check) would be deleted by this teardown. The
    # individual stubs that mint addresses (`floating_ip_test`,
    # `stable_ip_test`, `byoip_test`) clean their own addresses in their
    # own `finally` blocks (cleanup-bool propagated), so the shared
    # network teardown only deletes network-scoped resources (instances,
    # firewalls, routes, peerings, subnets, network).
    # AWS oracle parity: aws/scripts/network/teardown.py scopes deletes
    # to the input VPC.

    # Network last. Same readiness retry as subnets — a network can sit
    # in 'is not ready' state immediately after firewall/peering/route
    # cleanup ops complete, before its own delete becomes acceptable.
    try:
        ok = delete_with_retry(
            lambda: _with_readiness_retry(
                lambda: wait_for_global_op(
                    project,
                    networks_c.delete(project=project, network=network).name,
                    timeout=300,
                ),
                desc=f"network {network}",
            ),
            resource_desc=f"network {network}",
        )
        successes.append(ok)
        deleted["vpc"] = network
    except gax.NotFound:
        deleted["vpc"] = network

    # all_ok requires both: every delete returned True AND no enumeration
    # failure was downgraded to WARN. Without the enum_errors gate, a list
    # failure can skip an entire resource class and exit success.
    all_ok = (all(successes) if successes else True) and not enum_errors
    return {"deleted": deleted, "all_ok": all_ok, "enum_errors": enum_errors}


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Engine network teardown")
    parser.add_argument("--vpc-id", required=True)
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--network-created", default="false")
    parser.add_argument("--key-file", default=None)
    parser.add_argument("--key-name", default=None)
    parser.add_argument("--key-created", default="false")
    parser.add_argument("--project", default=None)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "success": False,
        "platform": "network",
        "resources_destroyed": False,
        "network_id": args.vpc_id,
        "deleted": {
            "instances": [],
            "firewalls": [],
            "subnets": [],
            "routes": [],
            "peerings": [],
            "addresses": [],
            "vpc": None,
            "internet_gateways": [],
        },
        "message": "",
    }

    def _cleanup_local_keys() -> None:
        # Runs regardless of cloud result (vendor-API contract).
        if args.key_created.lower() == "true" and args.key_file and args.key_file != "none":
            for p in (args.key_file, args.key_file + ".pub"):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass

    # No-op short-circuit BEFORE the auth-resolving project lookup so an
    # un-credentialed environment can still complete the local-key-file
    # cleanup phase of teardown.
    if args.vpc_id == "none" or args.network_created.lower() != "true":
        result["message"] = (
            f"skipping network resource teardown (network_created={args.network_created}, "
            f"vpc-id={args.vpc_id}) — verified-reuse cleanup contract"
        )
        result["success"] = True
        result["resources_destroyed"] = True
        _cleanup_local_keys()
        print(json.dumps(result, indent=2))
        return 0

    project = resolve_project(args.project)
    try:
        sub = _delete_network_resources(project, args.vpc_id, args.region)
        result["deleted"] = sub["deleted"]
        result["resources_destroyed"] = sub["all_ok"]
        result["success"] = sub["all_ok"]
        if sub.get("enum_errors"):
            result["enum_errors"] = sub["enum_errors"]
            result["message"] = (
                "network teardown completed with enumeration failures; some resource classes may have been skipped"
            )
        else:
            result["message"] = "network teardown complete"
    except (gax.GoogleAPICallError, RuntimeError, TimeoutError) as e:
        result["error_type"], result["error"] = classify_gcp_error(e)

    _cleanup_local_keys()

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

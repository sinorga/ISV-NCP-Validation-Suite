#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Start a stopped Compute Engine VM and gate success on a stable guest.

Divergences from the AWS oracle:
  * Compute Engine reports raw ``TERMINATED`` for the canonical stopped
    state; use ``canonical_state(...)``.
  * Ephemeral external IPs are RELEASED on stop. ``--public-ip`` may be
    forwarded for diagnostics, but every post-start emission MUST come
    from a fresh ``instances.get`` / ``wait_for_public_ip`` read —
    public IP is NOT preserved across stop/start on Compute Engine.
  * First-SSH-success is not enough: the guest agent may rewrite
    authorized_keys mid-cloud-init replay. Gate on (1) first SSH
    connectivity, THEN (2) cloud-init completion, THEN (3) N consecutive
    successful SSH probes — in that order, because a stability streak
    collected before the replay finishes observes an sshd that cloud-init
    is about to restart. `ssh_ready` and `success` come from the final
    post-cloud-init gate; post-lifecycle steps gate on stability, not on
    first SSH success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

from common.compute import (
    CAPACITY_REACQUIRE_ATTEMPTS,
    CAPACITY_REACQUIRE_BACKOFFS,
    CAPACITY_REACQUIRE_BUDGET,
    canonical_state,
    first_external_ip,
    first_internal_ip,
    get_instance,
    narrow_region_to_zone,
    poll_instance_state,
    resolve_project,
    retry_zonal_lifecycle_op,
    wait_for_public_ip,
)
from common.errors import handle_gcp_errors
from common.ssh_utils import wait_for_cloud_init, wait_for_ssh, wait_for_ssh_stable
from common.step_budget import StepBudget
from google.cloud import compute_v1

# Self-imposed wall-clock budget for the step, sister to the one
# `launch_instance.py` enforces. Same hazard shape: the capacity ladder, the
# 'running' poll, the public-IP poll, the SSH-reachability probe, the cloud-init
# wait and the post-cloud-init SSH-stability gate run in SEQUENCE, each with its
# own independent timeout, and the orchestrator kills an over-cap step with
# `subprocess.run(timeout=...)` — SIGKILL, no signal, so the result payload
# printed at the end never lands and the step reports nothing at all about the
# VM it was starting.
#
# This step creates no resources, so there is nothing here to leak; what the
# bound buys is a truthful `success=false` payload (with `start_initiated`,
# `state`, and the fresh public IP) instead of an empty kill. Every wait after
# the ladder is derived from what is left of the budget and keeps its full
# window while the budget can fund it.
#
# Enforced bound: the ladder self-bounds at 660s (CAPACITY_REACQUIRE_BUDGET +
# the op-wait floor) and the clock it burns is charged against everything after
# it, so the worst case is 1080 + the floors below (30 state poll + 15 IP poll
# + 25 SSH reachability probe + 30 cloud-init + 25 stability probe) = 1205s.
# The provider config's 1500s cap clears that by ~295s. Each SSH floor is one
# probe at `interval + SSH_PROBE_TIMEOUT_S` = 25s.
_STEP_WALL_BUDGET = 1080.0
_MIN_STATE_POLL_S = 30
_MIN_IP_POLL_S = 15
_MIN_CLOUD_INIT_S = 30
_MIN_SSH_ATTEMPTS = 1


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Start a stopped Compute Engine VM")
    parser.add_argument("--instance-id", required=True, help="Instance name")
    parser.add_argument("--region", required=True, help="GCP region or zone")
    parser.add_argument("--zone", default=None, help="GCP zone (overrides region)")
    parser.add_argument("--project", default=None, help="GCP project ID (ADC fallback)")
    parser.add_argument("--key-file", required=True, help="SSH private key path")
    parser.add_argument(
        "--public-ip",
        default=None,
        help="Pre-stop public IP (informational; re-read after start)",
    )
    parser.add_argument("--ssh-user", default="ubuntu", help="SSH username")
    args = parser.parse_args()

    # Start of the step's self-imposed wall clock (see `_STEP_WALL_BUDGET`),
    # stamped before any cloud call so every derived wait accounts for the work
    # already done.
    budget = StepBudget(_STEP_WALL_BUDGET)

    project = resolve_project(args.project)
    zone = args.zone or narrow_region_to_zone(args.region)

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "instance_id": args.instance_id,
        "region": args.region,
        "zone": zone,
        "project": project,
        "key_file": args.key_file,
        "ssh_user": args.ssh_user,
        "start_initiated": False,
        "ssh_ready": False,
    }

    try:
        # 1. Pre-check current state. The canonical stopped state is
        # required before issuing start; a previously-running VM or a
        # skipped/no-op stop step must not produce a green start,
        # otherwise the lifecycle test no longer proves stop→start
        # behavior.
        print("Verifying instance is stopped before start...", file=sys.stderr)
        inst = get_instance(project, zone, args.instance_id)
        cstate = canonical_state(inst.status)

        if cstate != "stopped":
            result["state"] = cstate
            result["error"] = f"Instance is {cstate!r}, expected stopped"
            print(json.dumps(result, indent=2, default=str))
            return 1

        # 2. Start; wait on zonal op then poll for canonical 'running'.
        # Lifecycle ops are zone-bound (cannot walk on STOCKOUT, and a
        # stopped instance's disk pins the zone anyway) — wrap the
        # sync+wait pair in the in-zone retry-with-backoff envelope
        # (zone_capacity_handling). This step uses the long
        # CAPACITY_REACQUIRE_* ladder, not the short default one, because
        # `stop` RELEASED this machine's zonal capacity and `start` has to
        # take it back from a pool other tenants are drawing on: the
        # default 3-attempt / 180s ladder was observed exhausting against a
        # real g2-standard-8 STOCKOUT in us-central1-c.
        # The long ladder gives the pool ~9 minutes to churn instead of ~3;
        # it cannot rescue a stockout that outlasts the budget, but it does
        # stop the step from surrendering while capacity is still moving.
        # The budget bounds the whole envelope so the extra patience stays
        # inside this step's configured cap. The post-API stamp keeps
        # start_initiated tied to a real API acknowledgement rather than
        # firing speculatively.
        print(f"Starting instance {args.instance_id}...", file=sys.stderr)

        def _stamp_start_initiated() -> None:
            result["start_initiated"] = True

        client = compute_v1.InstancesClient()
        retry_zonal_lifecycle_op(
            lambda: client.start(project=project, zone=zone, instance=args.instance_id),
            project,
            zone,
            resource_desc=f"start {args.instance_id}",
            on_sync_success=_stamp_start_initiated,
            attempts=CAPACITY_REACQUIRE_ATTEMPTS,
            backoffs=CAPACITY_REACQUIRE_BACKOFFS,
            budget=CAPACITY_REACQUIRE_BUDGET,
        )

        print("Waiting for canonical 'running' state...", file=sys.stderr)
        result["state"] = poll_instance_state(
            project,
            zone,
            args.instance_id,
            target_canonical="running",
            timeout=budget.wait_timeout(300, floor=_MIN_STATE_POLL_S),
        )

        # 3. Re-read details from live state — public IP is the critical
        # one because Compute Engine releases the ephemeral on stop and
        # assigns a fresh one on start.
        inst = get_instance(project, zone, args.instance_id)
        result["private_ip"] = first_internal_ip(inst)
        fresh_ip = first_external_ip(inst) or wait_for_public_ip(
            project,
            zone,
            args.instance_id,
            timeout=budget.wait_timeout(120, floor=_MIN_IP_POLL_S),
        )
        if not fresh_ip:
            result["error"] = "Instance has no external IP after start (timed out polling)"
            print(json.dumps(result, indent=2, default=str))
            return 1
        result["public_ip"] = fresh_ip

        # 4. Readiness gate, in the ONLY order that proves a settled guest:
        # first SSH connectivity, THEN cloud-init completion, THEN the
        # consecutive-success stability gate.
        #
        # Order is load-bearing, not stylistic. A start replays cloud-init and
        # the guest agent restarts sshd and rewrites authorized_keys while that
        # replay runs, so three consecutive SSH successes collected BEFORE
        # cloud-init finishes describe the pre-replay sshd — they say nothing
        # about the one downstream validators will connect to. Running the
        # stability gate last is what makes `ssh_ready` a claim about the guest
        # this step hands over. Mirrors the create path in `launch_instance.py`
        # (SSH -> cloud-init -> stability), so both readiness gates read the
        # same way.
        print("Waiting for first SSH connectivity after start...", file=sys.stderr)
        ssh_reachable = wait_for_ssh(
            host=fresh_ip,
            user=args.ssh_user,
            key_file=args.key_file,
            max_attempts=budget.probe_attempts(20, interval=10, floor=_MIN_SSH_ATTEMPTS),
            interval=10,
        )
        if not ssh_reachable:
            result["error"] = "SSH never accepted a connection after start"
            print(json.dumps(result, indent=2, default=str))
            return 1

        cloud_init_ok = wait_for_cloud_init(
            host=fresh_ip,
            user=args.ssh_user,
            key_file=args.key_file,
            timeout_seconds=budget.wait_timeout(600, floor=_MIN_CLOUD_INIT_S),
        )
        result["cloud_init_ok"] = cloud_init_ok
        if not cloud_init_ok:
            result["error"] = "cloud-init did not complete after start (rc != 0/2)"
            print(json.dumps(result, indent=2, default=str))
            return 1

        # `ssh_ready` is derived from THIS gate — the post-cloud-init one — and
        # never from the earlier reachability probe, so it is never a literal
        # True and never an observation of the sshd cloud-init replaced.
        print("Waiting for post-cloud-init SSH to stabilize...", file=sys.stderr)
        ssh_ok = wait_for_ssh_stable(
            host=fresh_ip,
            user=args.ssh_user,
            key_file=args.key_file,
            consecutive=3,
            interval=10,
            max_attempts=budget.probe_attempts(36, interval=10, floor=_MIN_SSH_ATTEMPTS),
        )
        result["ssh_ready"] = ssh_ok
        if not ssh_ok:
            result["error"] = "SSH did not stabilize after cloud-init completed"
            print(json.dumps(result, indent=2, default=str))
            return 1

        result["success"] = True
        print("Start completed", file=sys.stderr)

    except Exception as e:
        result["error"] = str(e)
        print(f"ERROR: {e}", file=sys.stderr)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

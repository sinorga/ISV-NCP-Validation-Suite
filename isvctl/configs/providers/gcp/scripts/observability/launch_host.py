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

"""Launch the observability syslog-probe host on the run subnetwork (setup phase).

Translates the AWS oracle's bare-metal ``launch_instance`` (managed key pair +
security group + run_instances) onto Compute Engine. Documented divergences:

  * GCP has NO managed key-pair store — the SSH public key is injected via
    instance ``ssh-keys`` metadata and the surviving artifact is the local PEM
    pair (``key_file``, gated on ``key_created`` for teardown).
  * GCP has NO security-group resource — host SSH ingress is a project-scoped VPC
    firewall rule targeted by an instance network tag. Its ``sourceRanges`` come
    ONLY from the operator-trusted env var ``NETWORK_FIREWALL_TRUST_IP`` — there
    is no open-internet fallback (emitted as ``firewall_name``, gated on
    ``firewall_created``).
  * Compute Engine instances are ZONAL and the subnetwork is regional, so the
    launch walks only the OPERATOR REGION's zones (never cross-region — the
    regional subnet would not exist elsewhere) and emits the successful ``zone``
    for teardown. Partial async-insert records left by a stockout-class failure
    are reclaimed before advancing; any that cannot be inline-confirmed are
    recorded in ``leaked_zones`` for teardown to reclaim exactly.

This host is CPU-only (default ``e2-standard-2``); the syslog probe needs neither
Docker nor a GPU. A MANDATORY guest-readiness gate runs so the downstream
``host_syslogs`` SSH probe is handed a settled sshd: launch success REQUIRES the
instance to reach RUNNING with an external IP AND cloud-init completion AND stable
SSH readiness (consecutive SSH successes past the post-cloud-init sshd bounce),
matching the staged launch contract (``launch_host`` ``required_outputs.success``
source "running instance read-back plus stable SSH readiness"). The serial-port
console (a Compute Engine API channel with no tcp/22 dependency) is read as
DIAGNOSTIC evidence only — it can show the guest started booting but cannot prove
the SSH prerequisites the contract and operator guide promise, so it is never a
success path. An unreachable SSH endpoint FAILS the launch with an actionable
diagnostic distinguishing a trusted-source-CIDR firewall block (the harness egress
IP is outside ``NETWORK_FIREWALL_TRUST_IP``) from a guest-side sshd fault.

Emits:
    {
        "success":          bool,
        "platform":         "observability",
        "instance_id":      str,        # Instance.name
        "zone":             str,        # effective successful zone
        "leaked_zones":     [str, ...], # partial-insert phantoms for teardown
        "public_ip":        str,        # accessConfigs[].natIP
        "private_ip":       str,        # networkInterfaces[0].networkIP
        "key_file":         str,        # local SSH private-key path
        "ssh_user":         str,
        "instance_created": bool,
        "firewall_name":    str,
        "firewall_created": bool,
        "key_created":      bool,
        ...
    }

AWS reference implementation:
    ../../aws/scripts/bare_metal/launch_instance.py (launch_host reuses the bm stub)
"""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

from common.compute import (
    ISV_NETWORK_TAG,
    bounded_unique_name,
    delete_failed_zonal_instance,
    delete_local_keypair,
    first_external_ip,
    first_internal_ip,
    generate_ssh_keypair,
    get_instance,
    insert_ssh_firewall,
    is_zone_unavailable,
    poll_instance_state,
    read_ssh_pubkey,
    resolve_image,
    resolve_project,
    resolve_trusted_ssh_source_ranges,
    wait_for_global_op,
    wait_for_public_ip,
    wait_for_zonal_op,
)
from common.errors import classify_gcp_error, delete_with_retry, handle_gcp_errors
from common.network import region_zones, subnetwork_url
from common.ownership import (
    CREATED_BY_LABEL,
    CREATED_BY_VALUE,
    has_invocation_label,
    labels_with_invocation,
    new_invocation_id,
    submit_owned_create,
)
from common.ssh_utils import wait_for_cloud_init, wait_for_ssh, wait_for_ssh_stable
from google.cloud import compute_v1

DEFAULT_SSH_USER = "ubuntu"
DEFAULT_IMAGE = "ubuntu-2204-lts"
DEFAULT_IMAGE_PROJECT = "ubuntu-os-cloud"

# Per-attempt cleanup-on-failure op waits — bounded so a 3-attempt
# delete_with_retry does not multiply into the enclosing step budget.
_CLEANUP_INSTANCE_WAIT_S = 180
_CLEANUP_FIREWALL_WAIT_S = 120
# Happy-path op / readiness waits (all deadline-bounded; the step timeout is
# headroom over their worst-case sum, not their product).
_INSERT_OP_WAIT_S = 300
_FIREWALL_OP_WAIT_S = 120
_RUNNING_POLL_S = 300
_PUBLIC_IP_POLL_S = 120
_SSH_INTERVAL = 10
_CLOUD_INIT_WAIT_S = 300
# Launch-side SSH readiness is MANDATORY (the launch contract requires stable
# SSH), so a reachable guest MUST accept SSH within this bound. A reachable guest
# accepts SSH well within it; when tcp/22 ingress is dropped from this orchestrator
# it can never succeed, so this bound fails fast (with the trust-CIDR diagnostic)
# instead of burning the full retry budget against a firewall block.
_LAUNCH_SSH_PROBE_ATTEMPTS = 10


def _build_instance_resource(
    *,
    project: str,
    zone: str,
    region: str,
    name: str,
    machine_type: str,
    source_image: str,
    network_name: str,
    subnet_name: str,
    ssh_user: str,
    ssh_pubkey: str,
) -> compute_v1.Instance:
    """Build the observability host Instance bound to the run subnetwork."""
    instance = compute_v1.Instance()
    instance.name = name
    instance.machine_type = f"zones/{zone}/machineTypes/{machine_type}"
    instance.description = "ISV observability syslog-probe host (createdby=isvtest)"

    boot = compute_v1.AttachedDisk()
    boot.boot = True
    boot.auto_delete = True
    init = compute_v1.AttachedDiskInitializeParams()
    init.source_image = source_image
    init.disk_size_gb = 20
    boot.initialize_params = init
    instance.disks = [boot]

    nic = compute_v1.NetworkInterface()
    nic.network = f"projects/{project}/global/networks/{network_name}"
    nic.subnetwork = subnetwork_url(project, region, subnet_name)
    nat = compute_v1.AccessConfig()
    nat.type_ = "ONE_TO_ONE_NAT"
    nat.name = "External NAT"
    nic.access_configs = [nat]
    instance.network_interfaces = [nic]

    # Network tag must match the SSH firewall's target tag so the rule selects
    # this instance.
    instance.tags = compute_v1.Tags(items=[ISV_NETWORK_TAG])

    ssh_item = compute_v1.Items()
    ssh_item.key = "ssh-keys"
    ssh_item.value = f"{ssh_user}:{ssh_pubkey}"
    instance.metadata = compute_v1.Metadata(items=[ssh_item])

    return instance


def _diagnose_ssh_unreachable(host: str, source_ranges: list[str]) -> str:
    """Classify WHY initial SSH never came up, for the surfaced error message.

    ``wait_for_ssh`` collapses every failure mode (dropped SYN, refused port,
    failed auth) to a single ``False``, which hides whether the cause is a
    firewall/source-range block or a guest-side sshd/key problem. This runs ONE
    raw TCP connect to ``host:22`` from the SAME vantage as the SSH readiness
    probe and maps the socket outcome to an actionable cause. It mutates no cloud
    state; it only sharpens the diagnostic so an operator can tell an ingress
    block (the harness public egress IP is outside ``NETWORK_FIREWALL_TRUST_IP``,
    so the firewall drops the probe) apart from a real guest-side failure.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        sock.connect((host, 22))
        return (
            "tcp/22 reachable but the SSH session did not authenticate — guest-side "
            "sshd/host-key/injected-key issue, NOT a firewall block"
        )
    except TimeoutError:
        return (
            f"tcp/22 SYN dropped (connect timeout) — the SSH firewall sourceRanges={source_ranges} "
            "do not include this harness's public egress IP, so the ingress rule drops the readiness "
            "probe. Set NETWORK_FIREWALL_TRUST_IP to a CIDR that covers the harness egress IP"
        )
    except ConnectionRefusedError:
        return "tcp/22 connection refused — reached the host but sshd was not yet listening"
    except OSError as exc:
        return f"tcp/22 connect error: {exc}"
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _read_boot_console_signal(
    project: str,
    zone: str,
    instance_name: str,
    *,
    attempts: int = 3,
    interval: int = 5,
) -> bool:
    """Return True iff the guest has written boot output to serial port 1.

    Reads ``getSerialPortOutput`` (the same Compute Engine API surface as
    ``vm/serial_console.py``) for the boot console. Non-empty contents show the
    guest OS started booting WITHOUT any tcp/22 ingress dependency. This is
    DIAGNOSTIC evidence ONLY — it helps an operator tell "guest never booted" from
    "guest booted but SSH ingress blocked" — and never contributes to launch
    success, which requires cloud-init completion plus stable SSH readiness. Any
    read error (permission, transient, not-found) is swallowed to False, since it
    is best-effort evidence and never a gate. Mutates no cloud state.
    """
    for attempt in range(1, attempts + 1):
        try:
            client = compute_v1.InstancesClient()
            request = compute_v1.GetSerialPortOutputInstanceRequest(
                project=project,
                zone=zone,
                instance=instance_name,
                port=1,
            )
            response = client.get_serial_port_output(request=request)
            if (response.contents or "").strip():
                return True
        except Exception as exc:  # best-effort signal; never fatal
            print(
                f"  serial-console boot signal read failed (attempt {attempt}/{attempts}): {exc}",
                file=sys.stderr,
            )
        if attempt < attempts:
            time.sleep(interval)
    return False


def _delete_instance_op(project: str, zone: str, name: str) -> None:
    """Submit instances.delete and wait on the zonal op (NotFound idempotent)."""
    from google.api_core import exceptions as gax

    try:
        op = compute_v1.InstancesClient().delete(project=project, zone=zone, instance=name)
    except gax.NotFound:
        return
    op_name = getattr(op, "name", "")
    if op_name:
        wait_for_zonal_op(project, zone, op_name, timeout=_CLEANUP_INSTANCE_WAIT_S)


def _delete_firewall_op(project: str, name: str) -> None:
    """Submit firewalls.delete and wait on the global op (NotFound idempotent)."""
    from google.api_core import exceptions as gax

    try:
        op = compute_v1.FirewallsClient().delete(project=project, firewall=name)
    except gax.NotFound:
        return
    op_name = getattr(op, "name", "")
    if op_name:
        wait_for_global_op(project, op_name, timeout=_CLEANUP_FIREWALL_WAIT_S)


@handle_gcp_errors
def main() -> int:
    """Launch the observability host and emit structured JSON."""
    parser = argparse.ArgumentParser(description="Launch the GCP observability syslog-probe host")
    parser.add_argument("--name", default="isv-observability-host", help="Instance name prefix (run-id suffixed)")
    parser.add_argument("--instance-type", default="e2-standard-2", help="Compute Engine machine type (CPU-only)")
    parser.add_argument("--region", required=True, help="GCP region (subnet region + in-region zone walk scope)")
    parser.add_argument("--zone", default=None, help="GCP zone pin (single-zone; disables the in-region walk)")
    parser.add_argument("--vpc-id", required=True, help="Compute Engine network name for the NIC")
    parser.add_argument("--subnet-id", required=True, help="Regional subnetwork name for the NIC")
    parser.add_argument("--ami-id", default=DEFAULT_IMAGE, help="Image short-name or family (resolved under project)")
    parser.add_argument("--image-project", default=DEFAULT_IMAGE_PROJECT, help="Project owning the image")
    parser.add_argument("--firewall-name", default="isv-observability-ssh", help="SSH firewall name prefix")
    parser.add_argument("--key-name", default="isv-observability-host-key", help="Local SSH key pair name prefix")
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER, help="Guest SSH login user")
    parser.add_argument("--project", default=None, help="GCP project ID (ADC fallback)")
    parser.add_argument(
        "--skip-destroy",
        action="store_true",
        help=(
            "Preserve run-owned resources on setup failure instead of running the "
            "compensating deletion (GCP_OBSERVABILITY_SKIP_TEARDOWN passthrough)."
        ),
    )
    args = parser.parse_args()

    project = resolve_project(args.project)
    # A run-id-only suffix is NOT enough: parallel step-isolation jobs share a
    # single RUN_ID, so run-id-only instance / firewall / key names are identical
    # across those sibling jobs and collide on AlreadyExists. Fold a
    # per-invocation discriminator (4 hex chars) BETWEEN each base and the run-id
    # suffix so every invocation gets fresh names; the run id stays TERMINAL so
    # the run-id-scoped orphan sweep (which matches names ending in the run id)
    # still recognizes them. The full names are emitted and forwarded verbatim to
    # teardown_host / teardown_network, which never reconstruct them.
    # bounded_unique_name keeps the discriminator + terminal run id intact and
    # truncates ONLY the (possibly step-isolation-lengthened) name prefix so the
    # composed instance/firewall/key name never exceeds the 63-char Compute
    # Engine limit — an over-long instance name is otherwise rejected with 400.
    disc = secrets.token_hex(2)  # 4 hex chars, fresh per invocation
    instance_name = bounded_unique_name(args.name, disc)
    # key_name is a LOCAL key-file label (/tmp/<key_name>.pem) with no Compute
    # Engine 63-char cap, but it flows through bounded_unique_name like the GCE
    # names for consistency; for a base that already fits the output is
    # byte-identical to the plain run-id suffix, so nothing changes in practice.
    key_name = bounded_unique_name(args.key_name, disc)
    fw_name = bounded_unique_name(args.firewall_name, disc)
    image_arg = args.ami_id if args.ami_id and args.ami_id.lower() != "none" else DEFAULT_IMAGE
    image_project = (
        args.image_project if args.image_project and args.image_project.lower() != "none" else DEFAULT_IMAGE_PROJECT
    )

    result: dict[str, Any] = {
        "success": False,
        "platform": "observability",
        "test_name": "launch_host",
        "instance_id": "",
        "zone": "",
        "leaked_zones": [],
        "public_ip": "",
        "private_ip": "",
        "key_file": "",
        "ssh_user": args.ssh_user,
        "firewall_name": "",
        "firewall_created": False,
        "key_created": False,
        "instance_created": False,
    }

    # Tracker state for cleanup-on-failure.
    instance_created = False
    firewall_created = False
    key_created = False
    key_priv = ""
    zone = ""
    # Per-invocation ownership marker. A Compute Engine create can commit
    # server-side then lose its response to a transport disconnect / 5xx; this
    # discriminator lets exact readback prove THIS invocation owns a resource
    # found after an ambiguous acknowledgement, so cleanup/teardown never skips a
    # committed-but-unacknowledged instance or firewall.
    invocation_id = new_invocation_id()

    try:
        # 1. Trusted SSH ingress — fail closed when NETWORK_FIREWALL_TRUST_IP is
        # unset / invalid / open-internet (there is no fallback range).
        ssh_source_ranges = resolve_trusted_ssh_source_ranges()

        # 2. Resolve the boot image (operator project first, then family alias).
        image = resolve_image(image_project, image_arg)
        source_image = image.self_link

        # 3. Local SSH key pair (verified-reuse; created flag gates teardown).
        key_priv, key_created = generate_ssh_keypair(key_name)
        result["key_file"] = key_priv
        result["key_created"] = key_created
        ssh_pubkey = read_ssh_pubkey(key_priv)

        # 4. SSH firewall rule on the run network (verified-reuse). insert returns
        # op=None on adoption of an already-matching rule. The create is submitted
        # through the ownership-reconciliation helper: on a clean accept OR on an
        # ambiguous transport/5xx ack that exact-name readback confirms carries
        # this invocation's marker, `_stamp_firewall_created` fires so
        # firewall_created is truthful BEFORE the op wait. firewall_name is
        # recorded up front so teardown_host / teardown_network have the exact
        # delete target even when the create acknowledgement is lost after a
        # server-side commit.
        result["firewall_name"] = fw_name

        def _stamp_firewall_created() -> None:
            nonlocal firewall_created
            firewall_created = True
            result["firewall_created"] = True

        fw_name, fw_op = insert_ssh_firewall(
            project=project,
            name=fw_name,
            network_short=args.vpc_id,
            source_ranges=ssh_source_ranges,
            on_accepted=_stamp_firewall_created,
        )
        if fw_op is not None:
            wait_for_global_op(project, fw_op.name, timeout=_FIREWALL_OP_WAIT_S)

        # 5. Insert with an in-region zone walk. The subnetwork is regional, so
        # candidates are the operator region's zones ONLY — never cross-region.
        if args.zone and args.zone.lower() != "none":
            candidate_zones = [args.zone]
        else:
            candidate_zones = region_zones(project, args.region)
        if not candidate_zones:
            raise RuntimeError(f"region {args.region!r} reports no zones for the host launch")

        instances_client = compute_v1.InstancesClient()
        last_error: Exception | None = None
        for idx, candidate_zone in enumerate(candidate_zones, start=1):
            print(
                f"Inserting host {instance_name} in {project}/{candidate_zone} [{idx}/{len(candidate_zones)}]...",
                file=sys.stderr,
            )
            instance_resource = _build_instance_resource(
                project=project,
                zone=candidate_zone,
                region=args.region,
                name=instance_name,
                machine_type=args.instance_type,
                source_image=source_image,
                network_name=args.vpc_id,
                subnet_name=args.subnet_id,
                ssh_user=args.ssh_user,
                ssh_pubkey=ssh_pubkey,
            )
            # Mark the create with this invocation's ownership label so an
            # ambiguous-ack readback can prove THIS invocation created the instance
            # (Instances carry a labels field, unlike firewall rules).
            instance_resource.labels = labels_with_invocation({CREATED_BY_LABEL: CREATED_BY_VALUE}, invocation_id)

            def _stamp_instance_created(_zone: str = candidate_zone) -> None:
                # Stamp ownership + identifiers on a clean accept OR a reconciled
                # ambiguous ack, BEFORE the op wait, so a wait-side failure (or a
                # lost create response that still committed the instance) still
                # hands cleanup/teardown a truthful (zone, name) delete target.
                nonlocal instance_created, zone
                instance_created = True
                zone = _zone
                result["instance_id"] = instance_name
                result["instance_created"] = True
                result["zone"] = _zone

            def _submit_insert(_zone: str = candidate_zone, _res: Any = instance_resource) -> Any:
                return instances_client.insert(project=project, zone=_zone, instance_resource=_res)

            def _read_instance(_zone: str = candidate_zone) -> Any:
                return get_instance(project, _zone, instance_name)

            def _owns_instance(inst: Any) -> bool:
                return has_invocation_label(inst, invocation_id)

            op = None
            try:
                # Submit through the ownership-reconciliation helper: a clean insert
                # ack stamps ownership; an ambiguous transport/5xx ack that
                # exact (project, zone, name) readback confirms carries this
                # invocation's marker ALSO stamps ownership (then re-raises so the
                # normal failure path cleans up). Only a definitive "nothing
                # committed" outcome leaves instance_created False.
                op = submit_owned_create(
                    _submit_insert,
                    _read_instance,
                    _owns_instance,
                    on_accepted=_stamp_instance_created,
                )
                op_name = getattr(op, "name", "")
                if op_name:
                    wait_for_zonal_op(project, candidate_zone, op_name, timeout=_INSERT_OP_WAIT_S)
                break
            except Exception as exc:
                if not is_zone_unavailable(exc, op=op):
                    raise
                last_error = exc
                # Stockout-class shape. instance_created is stamped ONLY when the
                # insert was accepted (clean ack) or an ambiguous ack was reconciled
                # to this invocation; if nothing was accepted in this zone there is
                # no record to delete and NOTHING is leaked — recording a leaked
                # zone there would send teardown chasing an instance that was never
                # created. Add the zone to leaked_zones ONLY when an accepted
                # insert's reclaim delete could not be confirmed.
                if instance_created:
                    cleaned = delete_failed_zonal_instance(project, candidate_zone, instance_name)
                    if not cleaned:
                        result["leaked_zones"].append(candidate_zone)
                    instance_created = False
                    result["instance_created"] = False
                print(f"  walking past {candidate_zone} (stockout-class)", file=sys.stderr)
                continue
        else:
            raise RuntimeError(f"Zone-walk exhausted ({len(candidate_zones)} candidates); last error: {last_error}")

        # 6. Poll canonical RUNNING.
        print("Waiting for RUNNING status...", file=sys.stderr)
        poll_instance_state(project, zone, instance_name, target_canonical="running", timeout=_RUNNING_POLL_S)

        # 7. Re-read for the external IP (ephemeral IP is only populated once running).
        inst = get_instance(project, zone, instance_name)
        result["private_ip"] = first_internal_ip(inst) or ""
        if not result["private_ip"]:
            raise RuntimeError("Host reached RUNNING but no internal IP became observable")
        result["public_ip"] = first_external_ip(inst) or wait_for_public_ip(
            project, zone, instance_name, timeout=_PUBLIC_IP_POLL_S
        )
        if not result["public_ip"]:
            raise RuntimeError("Host reached RUNNING but no external IP became observable")

        # 8. Guest-readiness gate — MANDATORY. The staged launch contract
        # (launch_host required_outputs.success source "running instance read-back
        # plus stable SSH readiness") and the operator guide define a successful
        # launch as RUNNING + external IP + cloud-init completion + STABLE SSH
        # readiness. The downstream host_syslogs probe and every SSH-gated consumer
        # depend on a settled, reachable sshd, so all three readiness stages below
        # are REQUIRED — a failure in any of them fails the launch (cleanup-on-
        # failure runs and the step exits non-zero). Serial-console output is read
        # as DIAGNOSTIC evidence ONLY; it can show the guest started booting but
        # cannot prove the SSH prerequisites the contract promises, so it is never
        # a success path.
        cloud_init_ok = False
        ssh_stable_ok = False

        # Diagnostic-only boot signal — reads the guest serial console through the
        # Compute Engine API (no tcp/22 dependency). Recorded as evidence so an
        # operator can tell "guest never booted" from "guest booted but SSH ingress
        # blocked"; it does NOT contribute to launch success.
        boot_console_observed = _read_boot_console_signal(project, zone, instance_name)
        result["boot_console_observed"] = boot_console_observed

        # Stage 1 (REQUIRED): SSH must become reachable. An unreachable endpoint
        # fails the launch with the sharpened trust-CIDR-vs-guest diagnostic so a
        # wrong NETWORK_FIREWALL_TRUST_IP or an unsettled guest surfaces HERE at
        # setup, not later when host_syslogs fails.
        ssh_ok = wait_for_ssh(
            host=result["public_ip"],
            user=args.ssh_user,
            key_file=key_priv,
            max_attempts=_LAUNCH_SSH_PROBE_ATTEMPTS,
            interval=_SSH_INTERVAL,
        )
        result["ssh_ready"] = ssh_ok
        if not ssh_ok:
            diag = _diagnose_ssh_unreachable(result["public_ip"], ssh_source_ranges)
            result["ssh_unreachable_diagnostic"] = diag
            raise RuntimeError(
                "Host reached RUNNING with an external IP but SSH never became reachable; "
                "the launch contract requires stable SSH readiness"
                + (f"; diagnostic: {diag}" if diag else "")
                + f" (serial-console boot output observed={boot_console_observed}, diagnostic only)"
            )

        # Stage 2 (REQUIRED): cloud-init must complete. An SSH session that predates
        # cloud-init can be dropped when the guest agent refreshes host keys /
        # authorized_keys, so a settled guest is a launch prerequisite.
        cloud_init_ok = wait_for_cloud_init(
            host=result["public_ip"],
            user=args.ssh_user,
            key_file=key_priv,
            timeout_seconds=_CLOUD_INIT_WAIT_S,
        )
        result["cloud_init_ok"] = cloud_init_ok
        if not cloud_init_ok:
            raise RuntimeError(
                "Host is SSH-reachable but cloud-init did not complete within "
                f"{_CLOUD_INIT_WAIT_S}s; the launch contract requires a settled guest"
            )

        # Stage 3 (REQUIRED): SSH must be STABLE. Compute Engine's guest agent
        # restarts sshd shortly after cloud-init completes (refreshes
        # authorized_keys / host keys); require consecutive successes so the
        # host_syslogs probe inherits a settled sshd rather than racing that bounce.
        ssh_stable_ok = wait_for_ssh_stable(
            host=result["public_ip"],
            user=args.ssh_user,
            key_file=key_priv,
            consecutive=3,
            interval=_SSH_INTERVAL,
            max_attempts=18,
        )
        result["ssh_stable"] = ssh_stable_ok
        if not ssh_stable_ok:
            raise RuntimeError(
                "Host cloud-init completed but SSH did not stabilize (3 consecutive probes) "
                "after the post-cloud-init sshd bounce; the launch contract requires stable SSH readiness"
            )

        result["success"] = True
        print(
            f"Launch succeeded: {instance_name} @ {result['public_ip']} ({zone}) "
            f"[ssh_ready={ssh_ok}, cloud_init_ok={cloud_init_ok}, ssh_stable={ssh_stable_ok}, "
            f"boot_console_observed={boot_console_observed}]",
            file=sys.stderr,
        )
    except Exception as e:
        error_type, error_msg = classify_gcp_error(e)
        result.setdefault("error_type", error_type)
        result["error"] = error_msg
        result["success"] = False
        if args.skip_destroy:
            # Preservation mode: SUPPRESS the compensating deletion so an operator
            # can inspect the partially launched run. The exact run-owned identifiers
            # are already emitted (instance_id/zone/instance_created,
            # firewall_name/firewall_created, key_file/key_created plus any
            # leaked_zones); surface them explicitly. teardown_host / teardown_network
            # (also --skip-destroy) leave them be.
            result["skip_destroy"] = True
            result["preserved_on_failure"] = {
                "instance_id": instance_name if instance_created else "",
                "zone": zone if instance_created else "",
                "leaked_zones": list(result["leaked_zones"]),
                "firewall_name": fw_name if firewall_created else "",
                "key_file": key_priv if key_created else "",
            }
            print(
                "Skip-destroy set: preserving run-owned resources on setup failure "
                f"(instance_created={instance_created} instance={instance_name!r}@{zone!r}, "
                f"firewall_created={firewall_created} firewall={fw_name!r}, "
                f"key_created={key_created}, leaked_zones={result['leaked_zones']})",
                file=sys.stderr,
            )
        else:
            # Cleanup-on-failure — gate each delete on its ownership tracker so a
            # verified-reuse-adopted firewall / pre-existing key is never destroyed.
            try:
                if instance_created and zone:
                    delete_with_retry(
                        _delete_instance_op, project, zone, instance_name, resource_desc=f"instance {instance_name}"
                    )
                # Reclaim leaked-zone phantoms INDEPENDENT of instance_created: the
                # exhausted-zone-walk stockout path resets instance_created=false yet
                # leaves a populated leaked_zones with the retained deterministic name.
                # Gating this on instance_created (like the landed-zone delete above)
                # would orphan the billable phantom; the run-scoped teardown steps are
                # the load-bearing net but this local pass shrinks the leak window.
                for leak_zone in result["leaked_zones"]:
                    if instance_created and leak_zone == zone:
                        continue
                    delete_with_retry(
                        _delete_instance_op,
                        project,
                        leak_zone,
                        instance_name,
                        resource_desc=f"instance {instance_name}@{leak_zone}",
                    )
                if firewall_created:
                    delete_with_retry(_delete_firewall_op, project, fw_name, resource_desc=f"firewall {fw_name}")
                if key_created and key_priv:
                    delete_local_keypair(key_priv)
            except Exception as cleanup_exc:
                print(f"Cleanup-on-failure error: {cleanup_exc}", file=sys.stderr)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

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

"""Launch a GPU VM on Compute Engine for VM-domain validation.

Translates the AWS oracle's ``launch_instance`` workflow to Compute
Engine. Documented divergences:

  * No managed key-pair store — generate a local PEM/.pub pair and
    attach the public key via instance metadata.
  * Firewall rules are project-global and bound by network tag, not
    attached per-instance — create / verified-reuse a TCP/22 INGRESS
    rule on the launch network and assign the matching network tag.
    Because the rule is not attached to the instance, the
    existing-instance reuse branch cannot infer ingress from the VM
    record: it must look the covering rules up and prove them against
    the same trusted-ingress policy, or fail closed.
  * GPU-bearing machine types reject ``onHostMaintenance=MIGRATE``
    (HTTP 400); force ``TERMINATE`` + ``automatic_restart=true``.
  * ``instances.insert`` returns DONE before the guest is reachable —
    poll RUNNING, then run a best-effort SSH-or-cloud-init readiness gate.
  * Public IP is assigned only when an ``accessConfigs`` entry of type
    ``ONE_TO_ONE_NAT`` is requested on the NIC.
  * Compute Engine label keys must be lowercase. Project canonical
    mixed-case ``Name`` / ``CreatedBy`` keys to api-valid labels on
    create and back on read so ``InstanceTagCheck.required_keys`` does
    not change per provider.
  * Emit the effective ``zone``, ``firewall_created``, ``key_created``
    so every downstream zonal step + teardown can read them via
    ``{{steps.launch_instance.X}}`` (verified-reuse cleanup contract).

``--skip-destroy`` carries the operator's preservation decision — the same
one the terminal teardown step receives. Preservation is not a teardown-only
concern: the setup-failure path here deletes the instance, firewall rule, and
local key this step created, and it runs long before teardown is reached. With
the flag set that compensating deletion is suppressed WHOLE (never partially,
which would leave a fixture set that no longer reproduces the failure) and the
retained identifiers are reported in ``preserved_on_failure`` while every
ownership flag stays truthful for a later
``teardown.py --from-launch-output`` reclamation. The capacity-walk phantom
reclamation stays ungated — an abandoned zone record is a billable phantom,
not a debugging fixture.

Operator-supplied image identifiers are resolved against
``args.image_project`` FIRST — short-name identifiers MUST be resolved
against the operator's chosen project/account/region scope first.
A vendor-default fallback is allowed only as an explicit second attempt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # providers/gcp/scripts/

from common.compute import (
    CAPACITY_REACQUIRE_ATTEMPTS,
    CAPACITY_REACQUIRE_BACKOFFS,
    CAPACITY_REACQUIRE_BUDGET,
    ISV_NETWORK_TAG,
    canonical_state,
    canonical_tags_to_labels,
    delete_failed_zonal_instance,
    delete_local_keypair,
    first_external_ip,
    first_internal_ip,
    generate_ssh_keypair,
    get_instance,
    insert_ssh_firewall,
    instance_has_pubkey,
    is_gpu_machine_type,
    is_zone_unavailable,
    narrow_region_to_zone,
    poll_instance_state,
    pubkey_from_private_key,
    read_ssh_pubkey,
    resolve_image,
    resolve_project,
    resolve_trusted_ssh_source_ranges,
    retry_zonal_lifecycle_op,
    select_zones,
    short_name,
    unique_suffix,
    verify_trusted_ssh_firewall_for_instance,
    wait_for_global_op,
    wait_for_public_ip,
    wait_for_zonal_op,
)
from common.errors import delete_with_retry, handle_gcp_errors
from common.ownership import (
    UnreconciledCandidate,
    description_with_invocation,
    has_invocation_description,
    new_invocation_id,
    submit_owned_create,
)
from common.ssh_utils import wait_for_cloud_init, wait_for_ssh, wait_for_ssh_stable
from common.step_budget import StepBudget
from google.api_core import exceptions as gax
from google.cloud import compute_v1

# GCP Deep Learning VM Image — the closest public equivalent to AWS's
# Deep Learning AMIs. Ships with the NVIDIA driver + CUDA toolkit
# preinstalled. Does NOT ship Docker; for tests that require a container
# runtime (e.g. the NIM deploy step) operators must either supply a
# custom image via --image-project / --image-family / --ami-id, or
# install Docker out-of-band before invoking the suite. The image lives
# in a public GCP-published project so no operator-specific entitlement
# is needed.
DEFAULT_IMAGE_FAMILY = "common-cu129-ubuntu-2204-nvidia-580"
DEFAULT_IMAGE_PROJECT = "deeplearning-platform-release"
DEFAULT_NETWORK = "default"
DEFAULT_FIREWALL_NAME = "isv-test-vm-ssh"
DEFAULT_KEY_NAME = "isv-test-key"
DEFAULT_SSH_USER = "ubuntu"

# Self-imposed wall-clock budget for the provisioning path (entry through the
# readiness gate), plus the tail reserved for the failure path.
#
# This step's waits run in SEQUENCE and each carries its own independent
# timeout: the firewall op wait, the zone-walk op wait (once per candidate
# zone), the 'running' poll, the public-IP poll, and then the SSH /
# cloud-init / post-cloud-init SSH-stability readiness gate. Summed at their
# individual worst cases they run far past any cap a provider config can
# reasonably carry, and the orchestrator kills an over-cap step with
# `subprocess.run(timeout=...)` — SIGKILL, no signal, so the compensating
# cleanup in the `except` below never runs AND the result payload (printed
# once, at the very end) is never emitted. Teardown then has no `instance_id`
# / `instance_created` / `leaked_zones` provenance and, by its own ownership
# gate, refuses to delete a VM it cannot prove this run created: a billable
# GPU VM, its firewall rule, and the local key leak with no reclamation path.
#
# So the step bounds its own wall clock (`common.step_budget.StepBudget`)
# rather than trusting arithmetic in a config comment, the same way
# `retry_zonal_lifecycle_op(budget=...)` already bounds the lifecycle ladder.
# Every wait after entry runs for `min(its own timeout, what is left)`, floored
# so a healthy wait is never truncated to nothing, and `_CLEANUP_RESERVE` is a
# SEPARATE budget stamped when the failure path starts so the deletes always
# have a window of their own. Waits that still fit are handed their full value
# unchanged — the observed create path finishes in under two minutes, so the
# clamp only ever engages on the tail where the alternative is the SIGKILL.
#
# The floors are the only way either budget can be exceeded, which makes the
# enforced bound computable — and the provider config sizes `timeout:` above
# it (see the launch_instance comment in config/vm.yaml):
#
#   provisioning  <= 1020 + 185 (30 firewall op + 30 zonal op + 30 state poll
#                                + 15 IP poll + 25 SSH probe + 30 cloud-init
#                                + 25 stability probe)                 = 1205s
#   cleanup       <= 300 + 300 (5 floored delete attempts each for the
#                               instance and the firewall) + 115 (the
#                               delete_with_retry backoff ladders)       = 715s
#   step total                                                           1920s
#
# The capacity ladder on the reuse path is NOT clamped: it is the first wait
# on that branch and bounds itself at 660s (CAPACITY_REACQUIRE_BUDGET + the
# op-wait floor), well inside the budget, and the clock it consumes is already
# charged against every wait that follows it.
_STEP_WALL_BUDGET = 1020.0
_CLEANUP_RESERVE = 300.0

# Floors granted to each derived wait once the budget is spent.
_MIN_OP_WAIT_S = 30
_MIN_STATE_POLL_S = 30
_MIN_IP_POLL_S = 15
_MIN_CLOUD_INIT_S = 30
_MIN_SSH_ATTEMPTS = 1
# Least the budget must still hold for another zone candidate to be worth
# starting (submit + a floor-length op wait). Below this the walk stops
# instead of opening a create it cannot wait out.
_MIN_ZONE_ATTEMPT_S = 90

# Bound the per-attempt wait so the (default 5-attempt) delete_with_retry
# ladder does not multiply 600s zonal-op + 120s global-op budgets into the
# enclosing step timeout. Cleanup-on-failure runs from inside the
# launch_instance step; delete waits beyond 180s instance / 120s firewall are
# diminishing returns under transient control-plane errors. Each attempt is
# clamped again against `_CLEANUP_RESERVE`, so a retry ladder that keeps
# hitting a slow control plane cannot outrun the step cap either.
_CLEANUP_INSTANCE_WAIT_S = 180
_CLEANUP_FIREWALL_WAIT_S = 120

# Base description stamped on the launched instance. The per-invocation
# ownership marker is appended to it so an ambiguous insert acknowledgement
# can be reconciled by readback (see the instance create below).
_ISV_INSTANCE_DESCRIPTION = "ISV validation GPU VM (createdby=isvtest)"


def _build_instance_resource(
    *,
    project: str,
    zone: str,
    name: str,
    machine_type: str,
    source_image: str,
    network_name: str,
    subnet_name: str | None,
    ssh_user: str,
    ssh_pubkey: str,
    labels: dict[str, str],
) -> compute_v1.Instance:
    """Build a Compute Engine ``Instance`` resource for ``instances.insert``.

    Every property here serializes via proto-plus (so it survives the
    REST encode); ad-hoc ``obj._properties[...] = ...`` mutations would
    be silently dropped.

    Subnetwork (when supplied) MUST be the regional URL; ``machine_type``
    MUST be the zonal URL — bare tokens are rejected by the proto wire
    layer.
    """
    instance = compute_v1.Instance()
    instance.name = name
    instance.machine_type = f"zones/{zone}/machineTypes/{machine_type}"

    boot = compute_v1.AttachedDisk()
    boot.boot = True
    boot.auto_delete = True
    init = compute_v1.AttachedDiskInitializeParams()
    init.source_image = source_image
    init.disk_size_gb = 100
    boot.initialize_params = init
    instance.disks = [boot]

    nic = compute_v1.NetworkInterface()
    nic.network = f"projects/{project}/global/networks/{network_name}"
    if subnet_name:
        region = zone.rsplit("-", 1)[0]
        nic.subnetwork = f"projects/{project}/regions/{region}/subnetworks/{subnet_name}"
    nat = compute_v1.AccessConfig()
    nat.type_ = "ONE_TO_ONE_NAT"
    nat.name = "External NAT"
    nic.access_configs = [nat]
    instance.network_interfaces = [nic]

    instance.tags = compute_v1.Tags(items=[ISV_NETWORK_TAG])

    # GPU machine types REJECT the default `MIGRATE` and require
    # `TERMINATE` + `automatic_restart`. This override is GPU-only —
    # non-GPU types must keep the API default (MIGRATE) to preserve
    # live-migrate behavior.
    if is_gpu_machine_type(machine_type):
        sched = compute_v1.Scheduling()
        sched.on_host_maintenance = "TERMINATE"
        sched.automatic_restart = True
        instance.scheduling = sched

    instance.labels = labels

    ssh_item = compute_v1.Items()
    ssh_item.key = "ssh-keys"
    ssh_item.value = f"{ssh_user}:{ssh_pubkey}"
    instance.metadata = compute_v1.Metadata(items=[ssh_item])

    return instance


def _delete_instance_op(project: str, zone: str, name: str, *, wait_s: int = _CLEANUP_INSTANCE_WAIT_S) -> None:
    """Submit ``instances.delete`` and wait on the zonal op (NotFound is idempotent).

    ``wait_s`` lets the caller clamp the CONFIRMATION to what is left of the
    cleanup reserve. Truncating it is safe: the delete has already been
    accepted server-side, so a short wait loses the confirmation, never the
    deletion — whereas a wait that outruns the step cap loses the whole
    cleanup to the orchestrator's SIGKILL.
    """
    try:
        op = compute_v1.InstancesClient().delete(project=project, zone=zone, instance=name)
    except gax.NotFound:
        return
    op_name = getattr(op, "name", None) or getattr(op, "operation", "")
    if op_name:
        wait_for_zonal_op(project, zone, op_name, timeout=wait_s)


def _delete_firewall_op(project: str, name: str, *, wait_s: int = _CLEANUP_FIREWALL_WAIT_S) -> None:
    """Submit ``firewalls.delete`` and wait on the global op (NotFound is idempotent).

    ``wait_s`` carries the same clamped-confirmation contract as
    ``_delete_instance_op``.
    """
    try:
        op = compute_v1.FirewallsClient().delete(project=project, firewall=name)
    except gax.NotFound:
        return
    op_name = getattr(op, "name", None) or getattr(op, "operation", "")
    if op_name:
        wait_for_global_op(project, op_name, timeout=wait_s)


def _reuse_existing_instance(
    *,
    project: str,
    zone: str,
    instance_id: str,
    key_file: str,
    ssh_user: str,
    requested_key_name: str,
    ssh_source_ranges: list[str],
    budget: StepBudget,
) -> int:
    """Mirror the AWS oracle's ``AWS_VM_INSTANCE_ID``/``AWS_VM_KEY_FILE`` reuse path.

    GCP equivalents are ``GCP_VM_INSTANCE_ID`` / ``GCP_VM_KEY_FILE``. When
    both are set, the stub describes the existing instance (and starts it
    if it's canonically stopped) instead of provisioning a new one — the
    dev workflow for iterating against a long-lived VM.

    Verified-reuse semantics: ``firewall_created`` / ``key_created`` stay
    False so teardown's gates skip destruction of pre-existing resources.

    ``ssh_source_ranges`` is the operator-trusted ingress policy resolved from
    ``NETWORK_FIREWALL_TRUST_IP`` BEFORE this branch is taken. The adoption path
    creates no firewall, but it still has to PROVE the adopted VM's tcp/22
    ingress satisfies the same policy the create path enforces: this branch
    reports the VM as a verified fixture, and because ``firewall_created`` stays
    False the run also leaves whatever rule it depended on in place. Verification
    fails closed — an unverifiable or over-permissive SSH path ends the step with
    ``success=false`` instead of a passing run behind an open firewall.

    ``budget`` is the step's wall clock (see ``_STEP_WALL_BUDGET``). This
    branch is the same sequential-wait shape as the create path — capacity
    ladder, 'running' poll, public-IP poll, then the ordered readiness gate
    (SSH reachability, cloud-init, post-cloud-init SSH stability) — so it
    derives its waits from the same budget. It creates nothing, so there
    is no cleanup to protect here; what the bound buys is the honest result
    payload, which a SIGKILL would otherwise swallow whole.
    """
    print(f"Reusing existing instance {instance_id}", file=sys.stderr)

    # Reuse-branch must not fabricate keys it doesn't have evidence for.
    # Initialize fields as None and only fill them when live state
    # provides a value.
    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "instance_id": instance_id,
        "project": project,
        "zone": zone,
        "availability_zone": zone,
        "key_file": key_file,
        "key_created": False,
        "firewall_created": False,
        # Verified-reuse ownership: adoption path NEVER created the
        # instance, so teardown must skip both the primary and any
        # leaked-zone delete. False stays False — there is no in-stub
        # branch that could promote it on the reuse path.
        "instance_created": False,
        "firewall_name": None,
        "security_group_id": None,
        "key_name": None,
        # Specified-key contract (same as the create branch): requested label
        # plus the label confirmed against the adopted instance's ssh-keys
        # metadata. instance_key_name stays None until the readback proves the
        # operator key is present.
        "requested_key_name": requested_key_name,
        "instance_key_name": None,
        "ssh_user": ssh_user,
        "reused": True,
        "tags": {},
    }

    started_in_reuse = False
    try:
        inst = get_instance(project, zone, instance_id)
        cstate = canonical_state(inst.status)

        if cstate == "stopped":
            print(f"  {instance_id} is stopped — starting it", file=sys.stderr)
            # Sister-stub consistency: the dedicated `start_instance.py`
            # wraps the start sync+wait pair in the in-zone
            # retry-with-backoff envelope, and uses the long
            # CAPACITY_REACQUIRE_* ladder because `start` must take back
            # zonal capacity a prior `stop` released. The
            # reuse-from-stopped path runs the SAME lifecycle op against
            # the SAME zone-bound instance and MUST honor the same
            # recovery contract — operators stockout-flake here exactly as
            # they would on the canonical start step, so the short default
            # ladder would strand an adopted VM the longer wait recovers.
            client = compute_v1.InstancesClient()
            retry_zonal_lifecycle_op(
                lambda: client.start(project=project, zone=zone, instance=instance_id),
                project,
                zone,
                resource_desc=f"reuse-start {instance_id}",
                attempts=CAPACITY_REACQUIRE_ATTEMPTS,
                backoffs=CAPACITY_REACQUIRE_BACKOFFS,
                budget=CAPACITY_REACQUIRE_BUDGET,
            )
            poll_instance_state(
                project,
                zone,
                instance_id,
                target_canonical="running",
                timeout=budget.wait_timeout(300, floor=_MIN_STATE_POLL_S),
            )
            inst = get_instance(project, zone, instance_id)
            cstate = canonical_state(inst.status)
            started_in_reuse = True

        result["state"] = cstate
        result["instance_type"] = short_name(inst.machine_type)
        result["public_ip"] = first_external_ip(inst) or wait_for_public_ip(
            project,
            zone,
            instance_id,
            timeout=budget.wait_timeout(120, floor=_MIN_IP_POLL_S),
        )
        result["private_ip"] = first_internal_ip(inst)
        if inst.network_interfaces:
            result["vpc_id"] = short_name(inst.network_interfaces[0].network)
            if inst.network_interfaces[0].subnetwork:
                result["subnet_id"] = short_name(inst.network_interfaces[0].subnetwork)
        # Only emit canonical tag keys when their backing labels are
        # actually present on the live instance. Fabricating defaults
        # here would diverge from the AWS oracle reuse path, which
        # emits exactly what the API returned.
        actual_labels = dict(getattr(inst, "labels", {}) or {})
        derived_tags: dict[str, str] = {}
        if "isv_name" in actual_labels:
            derived_tags["Name"] = actual_labels["isv_name"]
        if "createdby" in actual_labels:
            derived_tags["CreatedBy"] = actual_labels["createdby"]
        result["tags"] = derived_tags

        # Trusted-ingress verification on the adoption path. This is a GATE,
        # not an informational lookup: the firewall handle is derived from live
        # state only after every enabled tcp/22 INGRESS rule that binds to the
        # adopted VM has been proven to match the same trusted shape the create
        # path builds (exact NETWORK_FIREWALL_TRUST_IP source ranges, no
        # source-tag / source-service-account selectors, target scope contained
        # in this instance's own identity, one tcp/22 allow entry). It runs
        # BEFORE the readiness gate so an over-permissive VM is rejected without
        # spending the SSH/cloud-init budget on it.
        #
        # Failing closed matters more here than on the create path: the run
        # never owns this rule (`firewall_created` stays False), so teardown
        # preserves it. Reporting such a VM as a verified fixture would present
        # unsafe reuse as verified AND leave the unsafe rule behind.
        try:
            verified_fw = verify_trusted_ssh_firewall_for_instance(project, inst, ssh_source_ranges)
        except ValueError as exc:
            result.setdefault("tests", {})["trusted_ssh_ingress"] = {
                "passed": False,
                "message": str(exc),
                "probes": ["firewalls_list", "instance_network_tags", "instance_service_accounts"],
            }
            result["error"] = str(exc)
            result["error_type"] = "untrusted_ssh_ingress"
            print(json.dumps(result, indent=2, default=str))
            return 1
        result["firewall_name"] = verified_fw
        result["security_group_id"] = verified_fw
        result.setdefault("tests", {})["trusted_ssh_ingress"] = {
            "passed": True,
            "message": (
                f"Adopted instance {instance_id} accepts tcp/22 only from the "
                f"NETWORK_FIREWALL_TRUST_IP ranges via firewall rule {verified_fw!r}"
            ),
            "probes": ["firewalls_list", "instance_network_tags", "instance_service_accounts"],
        }

        # Compute Engine has no managed key-pair store and no live
        # `KeyName` field on the instance record. There is no portable
        # counterpart to the AWS oracle signal, so `result["key_name"]`
        # stays None rather than synthesizing a basename / `<user>@reuse`
        # token. Local PEM identity flows through `key_file` alone.
        result["key_name"] = None

        # Specified-key contract on the reuse path. Derive the operator key's
        # public half from the supplied key_file and confirm it against the
        # adopted instance's ssh-keys metadata (readback — never assumed).
        # instance_key_name carries the requested label only when confirmed.
        reuse_pubkey = pubkey_from_private_key(key_file)
        key_confirmed = bool(reuse_pubkey) and instance_has_pubkey(inst, reuse_pubkey)
        result["instance_key_name"] = requested_key_name if key_confirmed else None
        result.setdefault("tests", {})["specified_key"] = {
            "passed": key_confirmed,
            "message": (
                f"Requested SSH key {requested_key_name!r} confirmed in instance ssh-keys metadata"
                if key_confirmed
                else f"Requested SSH key {requested_key_name!r} not found in instance ssh-keys metadata"
            ),
            "probes": ["instance_ssh_keys_metadata"],
        }

        if cstate != "running" or not result["public_ip"]:
            result["error"] = f"Instance {instance_id} is {cstate!r} or has no external IP"
            print(json.dumps(result, indent=2, default=str))
            return 1

        # The reuse branch must enforce the same readiness gate as the
        # create branch, in the same ORDER: first SSH connectivity, THEN
        # cloud-init completion, THEN the consecutive-success stability
        # gate. Consecutive-success (not first-SSH) is what keeps a reused
        # VM whose sshd transiently flakes from passing on one lucky probe
        # — but the streak only means something once cloud-init has
        # finished. A reuse-start replays cloud-init, and the guest agent
        # restarts sshd / rewrites authorized_keys during that replay, so a
        # streak collected first describes the sshd that is about to be
        # replaced rather than the one downstream validators will use.
        ssh_reachable = wait_for_ssh(
            host=result["public_ip"],
            user=ssh_user,
            key_file=key_file,
            max_attempts=budget.probe_attempts(20, interval=10, floor=_MIN_SSH_ATTEMPTS),
            interval=10,
        )
        cloud_init_ok = False
        ssh_ok = False
        if ssh_reachable:
            cloud_init_ok = wait_for_cloud_init(
                host=result["public_ip"],
                user=ssh_user,
                key_file=key_file,
                timeout_seconds=budget.wait_timeout(600, floor=_MIN_CLOUD_INIT_S),
            )
            # The stability gate runs after the cloud-init wait RETURNS,
            # whichever way it answered: either way the replay window is
            # over by then, so this is the gate that observes the sshd the
            # step hands over. Keeping it unconditional also preserves the
            # already-running adoption contract below, where SSH stability
            # alone is sufficient evidence.
            ssh_ok = wait_for_ssh_stable(
                host=result["public_ip"],
                user=ssh_user,
                key_file=key_file,
                consecutive=3,
                interval=10,
                max_attempts=budget.probe_attempts(36, interval=10, floor=_MIN_SSH_ATTEMPTS),
            )
        # `ssh_ready` is derived from the FINAL post-cloud-init gate, never
        # from the earlier reachability probe.
        result["ssh_ready"] = ssh_ok
        result["cloud_init_ok"] = cloud_init_ok
        # When the reuse branch just started the VM, both SSH stability
        # AND cloud-init completion are required to match the dedicated
        # start_instance step's success contract. Downstream validators
        # would otherwise race a guest whose cloud-init replay is still
        # rewriting authorized_keys / fstab. For a guest that was already
        # running when adopted, SSH stability alone is enough — cloud-init
        # only completes once per boot.
        if started_in_reuse:
            ready = ssh_ok and cloud_init_ok
        else:
            ready = ssh_ok or cloud_init_ok
        if ready:
            result["success"] = True
        else:
            result["error"] = (
                f"Instance {instance_id} is RUNNING but the reuse-path "
                "readiness gate (SSH reachability -> cloud-init completion -> "
                "post-cloud-init SSH stability) did not pass"
            )

    except Exception as e:
        result["error"] = str(e)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


@handle_gcp_errors
def main() -> int:
    parser = argparse.ArgumentParser(description="Launch a GPU VM on Compute Engine")
    parser.add_argument("--name", default="isv-test-gpu", help="Instance name")
    parser.add_argument(
        "--instance-type",
        required=True,
        help="Compute Engine machineType (e.g., g2-standard-8)",
    )
    parser.add_argument(
        "--region",
        required=True,
        help="GCP region or zone; if a region is given it's narrowed to <region>-a",
    )
    parser.add_argument("--zone", default=None, help="GCP zone (overrides region narrowing)")
    parser.add_argument("--project", default=None, help="GCP project ID (ADC fallback)")
    parser.add_argument("--vpc-id", default=DEFAULT_NETWORK, help="Network short name")
    parser.add_argument("--subnet-id", default=None, help="Subnetwork short name")
    parser.add_argument(
        "--image-family",
        default=DEFAULT_IMAGE_FAMILY,
        help="GCP image family (resolved to a concrete image at runtime)",
    )
    parser.add_argument(
        "--image-project",
        default=None,
        help=(
            "GCP project hosting the image. When omitted: short-name "
            "--ami-id resolves in the operator project first (parameter-"
            "surface parity with the AWS oracle, where AMI IDs are "
            "account-scoped); --image-family resolves in the default "
            "project (where the canonical GPU image lives). Pass "
            "explicitly to override either fallback."
        ),
    )
    parser.add_argument(
        "--ami-id",
        default=None,
        help=(
            "Parameter-surface parity with the AWS oracle's --ami-id; "
            "if set, overrides --image-family lookup with a literal image"
        ),
    )
    parser.add_argument("--key-name", default=DEFAULT_KEY_NAME, help="Local SSH key label")
    parser.add_argument(
        "--firewall-name",
        default=DEFAULT_FIREWALL_NAME,
        help="SSH firewall rule name",
    )
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER, help="SSH username")
    parser.add_argument(
        "--skip-destroy",
        action="store_true",
        help=(
            "Preserve run-owned resources on setup failure instead of running the "
            "compensating deletion (GCP_VM_SKIP_TEARDOWN passthrough). The same "
            "resolved preservation decision the terminal teardown step receives."
        ),
    )
    args = parser.parse_args()

    # The provider config wires --subnet-id / --ami-id from settings that
    # default to the literal "none" sentinel so the orchestrator does not
    # collapse the flag/value pair. Treat the sentinel as "operator did
    # not supply" — the default
    # subnet for the resolved zone and the canonical image family take
    # over.
    if args.subnet_id == "none":
        args.subnet_id = None
    if args.ami_id == "none":
        args.ami_id = None
    if args.image_project == "none":
        args.image_project = None

    # Start of the step's self-imposed wall clock (see `_STEP_WALL_BUDGET`).
    # Stamped before any cloud call so every wait derived from it accounts for
    # the work already done — including the API round trips that resolve the
    # project, the zone list, and the image.
    budget = StepBudget(_STEP_WALL_BUDGET)

    project = resolve_project(args.project)
    initial_zone = args.zone or narrow_region_to_zone(args.region)

    # Resolve the trusted SSH ingress source ranges from the operator
    # environment BEFORE the reuse branch below. NETWORK_FIREWALL_TRUST_IP is
    # the ONLY trusted source for tcp/22 ingress — there is no open-internet
    # fallback — and the policy binds BOTH branches: the create path builds the
    # rule from these ranges, and the adoption path proves the operator VM's
    # existing rule equals them. Resolving inside the create path alone would
    # let a reuse invocation return without the policy ever being read, so an
    # unset/empty/non-IPv4/0.0.0.0/0 value must fail the launch here, before
    # either branch and before any key, firewall, or instance is created.
    ssh_source_ranges = resolve_trusted_ssh_source_ranges()

    # Reuse-existing-instance branch (AWS oracle parity).
    reuse_instance = os.environ.get("GCP_VM_INSTANCE_ID")
    reuse_key = os.environ.get("GCP_VM_KEY_FILE")
    if reuse_instance and reuse_key:
        return _reuse_existing_instance(
            project=project,
            zone=initial_zone,
            instance_id=reuse_instance,
            key_file=reuse_key,
            ssh_user=args.ssh_user,
            requested_key_name=args.key_name,
            ssh_source_ranges=ssh_source_ranges,
            budget=budget,
        )

    # Apply the RUN_ID suffix. Compute Engine names ARE the API IDs —
    # without the suffix, parallel runs collide on AlreadyExists during
    # create and /tmp/<key>.pem clobbers across sessions (name-collision
    # risk). The suffix lives at runtime, NOT in provider config (only
    # the team-letter belongs in config).
    instance_name = unique_suffix(args.name)
    firewall_name_suffixed = unique_suffix(args.firewall_name)
    key_name_suffixed = unique_suffix(args.key_name)

    # Multi-zone walk candidates. select_zones honors a single-zone pin
    # (full ``us-central1-a`` form) and otherwise queries the operator-
    # supplied region's live zones via the GCP API before iterating
    # preferred-in-region → other-in-region → cross-region
    # (zone_capacity_handling). Passing the resolved project lets the
    # helper query the regions API so a
    # valid region missing from PREFERRED_ZONES still walks its OWN
    # zones first.
    candidate_zones = select_zones(args.zone or args.region, project=project)

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        # Stays None until instances.insert ack; emitted only when an
        # API-accepted name exists (AWS oracle parity).
        "instance_id": None,
        "instance_type": args.instance_type,
        "region": args.region,
        "zone": initial_zone,
        "availability_zone": initial_zone,
        "project": project,
        "vpc_id": args.vpc_id,
        "subnet_id": args.subnet_id,
        # Compute Engine has no managed key-pair store; local PEM
        # identity flows through `key_file` end-to-end. The CLI accepts
        # `--key-name` for AWS-oracle invocation parity but the emitted
        # `key_name` is unconditionally None — there is no server-side
        # key-pair identifier on Compute Engine.
        "key_name": None,
        # Specified-key contract: the operator-requested SSH-key label and
        # the label CONFIRMED present on the launched instance via an
        # ssh-keys metadata readback. `requested_key_name` is the honest
        # "requested" value; `instance_key_name` stays None until the
        # readback proves the injected public key is on the instance, so a
        # pre-readback failure fails InstanceSpecifiedKeyCheck honestly.
        "requested_key_name": args.key_name,
        "instance_key_name": None,
        # Producer-side sentinel defense: initialize to the canonical
        # sentinel so an exception BEFORE generate_ssh_keypair (e.g.,
        # resolve_image raising) still emits a non-empty value. The
        # consumer template in vm.yaml uses `default('none', true)`
        # (boolean mode) to collapse both undefined AND empty/sentinel
        # values to the same downstream arg; matching the producer side
        # keeps the contract tight under partial-failure JSON.
        "key_file": "none",
        "key_created": False,
        # Verified-reuse ownership for the instance itself. Stays False
        # until the instances.insert ack returns; teardown gates primary
        # and leaked-zone deletes on this so a pre-RUNNING failure
        # (e.g., image resolve) cannot make teardown destroy an
        # operator-supplied VM that this run never touched.
        "instance_created": False,
        "firewall_name": firewall_name_suffixed,
        "firewall_created": False,
        "security_group_id": firewall_name_suffixed,
        "ssh_user": args.ssh_user,
        "state": "",
        "public_ip": None,
        "private_ip": None,
        # Filled after resolve_image() — never echo the requested family.
        "ami_id": "",
        "tags": {},
        # leaked_zones flows into teardown's --leaked-zones arg so any
        # partial-create in a failed zone gets a second-chance delete
        # (zone_capacity_handling). Only zones where this run accepted a
        # create and could not confirm the reclaim delete belong here;
        # a zone that merely rejected the insert is NOT a leak.
        "leaked_zones": [],
        # Ambiguous-create candidates whose ownership could NOT be decided:
        # the create may have committed, but the reconciling readback was
        # denied or kept failing, so neither "ours" nor "not ours" is proven.
        # Nothing here is deleted on that unproven claim; each packed record
        # carries kind|name|project|zone|invocation so teardown (and a
        # `--from-launch-output` replay) can re-verify the marker itself and
        # delete ONLY what still echoes this invocation. Dropping this handoff
        # is how a committed-but-unacknowledged VM or firewall rule leaks.
        "unreconciled_resources": [],
    }

    def _retain_unreconciled(candidate: UnreconciledCandidate, error: Exception, detail: str) -> None:
        """Persist one candidate whose ownership the create path could not prove."""
        record = candidate.pack()
        if record not in result["unreconciled_resources"]:
            result["unreconciled_resources"].append(record)
        print(
            f"  {candidate.describe()} ownership unproven after ambiguous create "
            f"({error.__class__.__name__}: {detail}); retaining cleanup handoff for teardown",
            file=sys.stderr,
        )

    # Per-resource trackers for the cleanup-on-failure block.
    instance_created = False
    zone = initial_zone  # tracked separately so the walk can update it
    key_priv: str | None = None
    key_created = False
    firewall_created = False
    fw_name = firewall_name_suffixed

    try:
        # 0a. `ssh_source_ranges` was resolved above, ahead of the reuse branch,
        # so both branches are bound by the same operator ingress policy. It is
        # used verbatim as the created rule's sourceRanges and as the
        # verified-reuse match criterion.
        #
        # 0b. Resolve image. Three operator-supplied shapes are honored
        # (operator scope wins):
        #   * Full self-link (``https://...`` or ``projects/<P>/global/
        #     images/<N>``) — pass through verbatim; Compute Engine
        #     accepts it as ``sourceImage``.
        #   * Short name OR family alias under ``--image-project`` —
        #     route through ``resolve_image`` which tries
        #     ``images.get`` then ``images.get_from_family``. This is
        #     the AWS-oracle parameter-surface parity case: the
        #     operator passes ``--ami-id <short>`` (mirroring AWS) and
        #     the stub resolves it inside the operator's chosen
        #     project, NOT a hardcoded vendor default.
        # ``--ami-id`` is treated as a literal-or-short hint; the family
        # alias branch uses ``--image-family`` so operators can supply
        # either without ambiguity.
        if args.ami_id:
            literal_image = args.ami_id
            is_full_path = literal_image.startswith(("projects/", "https://"))
            if is_full_path:
                resolved_source_image = literal_image
                result["ami_id"] = short_name(literal_image)
                result["ami_self_link"] = literal_image
                result["ami_name"] = short_name(literal_image)
            else:
                # Short name — operator-scope parameter-surface parity:
                # AMI IDs are account-scoped on AWS, so the AWS-oracle
                # invocation pattern (`--ami-id <short>`) MUST resolve in
                # the operator project on GCP. Try operator project
                # first; on NotFound (e.g., operator following a tutorial
                # that names a vendor-default image) fall back to the
                # vendor default. An explicit `--image-project` wins
                # over both.
                explicit_project = args.image_project
                operator_scope = explicit_project or project
                try:
                    image = resolve_image(operator_scope, literal_image)
                except gax.NotFound:
                    if explicit_project:
                        # Operator explicitly named the scope — do not
                        # silently substitute. Surface the error.
                        raise RuntimeError(
                            f"Image {literal_image!r} not found in project {explicit_project!r}"
                        ) from None
                    if operator_scope == DEFAULT_IMAGE_PROJECT:
                        # Already searched the default project; nothing
                        # more to try.
                        raise RuntimeError(f"Image {literal_image!r} not found in project {operator_scope!r}") from None
                    print(
                        f"  image {literal_image!r} not in operator project "
                        f"{operator_scope!r}; falling back to default project "
                        f"{DEFAULT_IMAGE_PROJECT!r}",
                        file=sys.stderr,
                    )
                    try:
                        image = resolve_image(DEFAULT_IMAGE_PROJECT, literal_image)
                    except gax.NotFound as e:
                        raise RuntimeError(
                            f"Image {literal_image!r} not found in operator project "
                            f"{operator_scope!r} or default project "
                            f"{DEFAULT_IMAGE_PROJECT!r}: {e}"
                        ) from e
                resolved_source_image = image.self_link
                result["ami_id"] = short_name(image.self_link)
                result["ami_name"] = image.name
                result["ami_self_link"] = image.self_link
        else:
            # Image-family lookup — the canonical GPU image lives in
            # the default project, so the family-default route reads from
            # there unless the operator overrides --image-project.
            family_scope = args.image_project or DEFAULT_IMAGE_PROJECT
            try:
                image = resolve_image(family_scope, args.image_family)
                resolved_source_image = image.self_link
                result["ami_id"] = short_name(image.self_link)
                result["ami_name"] = image.name
                result["ami_self_link"] = image.self_link
            except gax.NotFound as e:
                raise RuntimeError(f"Image {args.image_family!r} in project {family_scope!r} not found: {e}") from e

        # 1. Local SSH key pair (verified-reuse). Use the run-id-suffixed
        # name so /tmp/<base>-<run_id>.pem can't collide between sessions.
        # The tuple-unpack shape matches the drift-guard contract.
        key_priv, key_created = generate_ssh_keypair(key_name_suffixed)
        ssh_pubkey = read_ssh_pubkey(key_priv)
        result["key_file"] = key_priv
        result["key_created"] = key_created

        # 2. SSH firewall on the target network (verified-reuse).
        # Stamp-on-accept pattern — ownership is transferred by the
        # helper's callback, never by inspecting its return value.
        # ``_stamp_firewall_created`` fires from INSIDE the helper in both
        # acknowledgement shapes:
        #   * clean ``firewalls.insert`` ack — before the op wait, so a
        #     wait-side failure still leaves a truthful flag for cleanup;
        #   * ambiguous ack (transport disconnect / 5xx) whose exact-name
        #     readback proves this invocation's ownership marker — the
        #     rule committed server-side but the response was lost, and
        #     the helper stamps ownership BEFORE re-raising so
        #     cleanup-on-failure below (and teardown, via the forwarded
        #     ``firewall_created``) delete the committed rule instead of
        #     orphaning it.
        #   * a 409 whose exact-name readback carries this invocation's
        #     marker — an internal retry of OUR OWN insert; the helper
        #     claims it instead of adopting it as pre-existing operator
        #     state, so teardown deletes the rule this run created.
        # A conflict on a rule carrying ANOTHER marker never fires the
        # callback, so verified-reuse adoption of a genuinely pre-existing
        # rule keeps ``firewall_created=False`` and teardown preserves it.
        # ``op is None`` on that adoption path (no wait required).
        # When the reconciling readback is denied or keeps failing, ownership
        # is neither claimed nor discarded: ``_retain_unreconciled`` records
        # the candidate so teardown re-verifies the marker before deleting.
        def _stamp_firewall_created() -> None:
            nonlocal firewall_created
            firewall_created = True
            result["firewall_created"] = True

        fw_name, fw_op = insert_ssh_firewall(
            project=project,
            name=firewall_name_suffixed,
            network_short=args.vpc_id,
            source_ranges=ssh_source_ranges,
            on_accepted=_stamp_firewall_created,
            on_unreconciled=_retain_unreconciled,
        )
        if fw_op is not None:
            wait_for_global_op(project, fw_op.name, timeout=budget.wait_timeout(120, floor=_MIN_OP_WAIT_S))
        result["firewall_name"] = fw_name
        result["security_group_id"] = fw_name

        # 3. Build / insert with multi-zone walk on STOCKOUT.
        # Canonical tag projection happens at the boundary; the emitted
        # ``tags`` dict comes from a live readback further below. The
        # ``Name`` tag carries the same suffixed instance_name so
        # ``gcloud compute instances list --filter "labels.name~$RUN_ID"``
        # works for cross-resource grouping.
        canonical_tags = {"Name": instance_name, "CreatedBy": "isvtest"}
        labels = canonical_tags_to_labels(canonical_tags)

        instances_client = compute_v1.InstancesClient()
        last_error: Exception | None = None
        op = None
        op_name = ""
        # Per-invocation ownership marker for the instance create, so an
        # ambiguous insert acknowledgement can be reconciled by exact
        # (project, zone, name) readback the same way the firewall create is.
        # The marker lives in ``description`` rather than in ``labels``
        # because verify_tags projects EVERY instance label into the suite's
        # canonical tag contract — an internal ownership discriminator must
        # not surface there as a user tag. Nothing in this domain reads the
        # instance description, so it is a contract-neutral marker channel.
        invocation_id = new_invocation_id()
        for candidate_idx, candidate_zone in enumerate(candidate_zones, start=1):
            # Per-zone op waits are already clamped to the step budget, but the
            # WALK itself is a loop: without this guard N candidates multiply
            # into N floor-length waits past the budget. Stop before opening a
            # create the budget cannot wait out, and raise so the failure path
            # (not the orchestrator's SIGKILL) reclaims whatever exists and
            # emits the ownership payload. The first candidate is always tried
            # — refusing to attempt anything would be a worse failure than a
            # truncated walk.
            if candidate_idx > 1 and not budget.can_afford(_MIN_ZONE_ATTEMPT_S):
                raise RuntimeError(
                    f"Step budget exhausted after {candidate_idx - 1} zone candidate(s) "
                    f"of {len(candidate_zones)}; last error: {last_error}"
                )
            print(
                f"Inserting instance {instance_name} in "
                f"{project}/{candidate_zone} [{candidate_idx}/{len(candidate_zones)}]...",
                file=sys.stderr,
            )
            instance_resource = _build_instance_resource(
                project=project,
                zone=candidate_zone,
                name=instance_name,
                machine_type=args.instance_type,
                source_image=resolved_source_image,
                network_name=args.vpc_id,
                subnet_name=args.subnet_id,
                ssh_user=args.ssh_user,
                ssh_pubkey=ssh_pubkey,
                labels=labels,
            )
            instance_resource.description = description_with_invocation(_ISV_INSTANCE_DESCRIPTION, invocation_id)

            def _stamp_instance_created(_zone: str = candidate_zone) -> None:
                # Stamp-on-accept: set the cleanup tracker AND
                # result['instance_id'] / result['zone'] IMMEDIATELY on the
                # insert acknowledgement, BEFORE the wait. A wait-side
                # failure then leaves the truthful identifier on disk
                # for teardown. result['instance_created'] is stamped
                # on the same tick so the teardown ownership flag
                # forwarded via vm.yaml stays in sync with the
                # in-process tracker driving cleanup-on-failure.
                nonlocal instance_created, zone
                instance_created = True
                zone = _zone
                result["instance_id"] = instance_name
                result["instance_created"] = True
                result["zone"] = _zone
                result["availability_zone"] = _zone

            def _submit_insert(_zone: str = candidate_zone, _resource: Any = instance_resource) -> Any:
                return instances_client.insert(
                    project=project,
                    zone=_zone,
                    instance_resource=_resource,
                )

            def _read_instance(_zone: str = candidate_zone) -> Any:
                return get_instance(project, _zone, instance_name)

            def _owns_instance(existing: Any) -> bool:
                return has_invocation_description(existing, invocation_id)

            def _retain_instance_candidate(
                error: Exception,
                detail: str,
                _zone: str = candidate_zone,
            ) -> None:
                # Ownership unproven: record the exact (project, zone, name,
                # invocation) so teardown can re-verify the marker later. A GPU
                # VM is the most expensive thing in this domain to leak, so the
                # handoff is retained rather than dropped.
                _retain_unreconciled(
                    UnreconciledCandidate(
                        resource_type="instance",
                        name=instance_name,
                        project=project,
                        zone=_zone,
                        invocation_id=invocation_id,
                    ),
                    error,
                    detail,
                )

            try:
                # Submit through the ownership-reconciliation helper so the
                # stamp fires on a clean ack OR on an ambiguous transport/5xx
                # ack whose exact readback proves this invocation created the
                # instance (the helper then re-raises, and the failure path
                # below deletes the committed VM instead of orphaning it).
                # A 409 is reconciled the same way: our marker means our own
                # retried insert committed and ownership transfers, another
                # run's marker stamps nothing so the fatal re-raise leaves a
                # pre-existing VM untouched. A denied/failing readback proves
                # neither, and lands in _retain_instance_candidate.
                op = submit_owned_create(
                    _submit_insert,
                    _read_instance,
                    _owns_instance,
                    on_accepted=_stamp_instance_created,
                    on_unreconciled=_retain_instance_candidate,
                )

                op_name = getattr(op, "name", None) or getattr(op, "operation", "")
                if op_name:
                    wait_for_zonal_op(
                        project,
                        candidate_zone,
                        op_name,
                        timeout=budget.wait_timeout(600, floor=_MIN_OP_WAIT_S),
                    )
                # Insert + DONE successful — break out of the walk.
                break
            except Exception as exc:
                # Async DONE-with-errors raises from wait_for_zonal_op.
                # Sync stockout raises from insert. is_zone_unavailable
                # covers all four shapes; treat non-zone errors as fatal.
                if not is_zone_unavailable(exc, op=op):
                    raise
                last_error = exc
                # leaked_zones is a CLEANUP-PROVENANCE list, not a
                # stockout log: it may name only zones where this
                # invocation actually accepted a create and could not
                # confirm the reclaim delete. instance_created is the
                # exact discriminator — it is stamped on a clean insert
                # ack and on an ambiguous ack whose exact-name readback
                # proved this invocation owns the record, so the
                # partial-state-possible shapes (async DONE-with-errors
                # and its RuntimeError polling fallback) always land in
                # the branch below. The shapes that raise out of
                # `instances.insert` itself return before any zonal
                # resource exists, so they have nothing to reclaim.
                if instance_created:
                    print(
                        f"  zone {candidate_zone} unavailable; cleaning partial create",
                        file=sys.stderr,
                    )
                    cleaned = delete_failed_zonal_instance(project, candidate_zone, instance_name)
                    if not cleaned:
                        result["leaked_zones"].append(candidate_zone)
                    # Reset the per-zone tracker so the next iteration's
                    # insert ack stamps it fresh. Do NOT null
                    # result['instance_id'] / result['zone'] — the
                    # instance_name is deterministic across walker
                    # attempts (suffix is rolled once before the loop),
                    # so the stamped value remains a valid teardown
                    # target whether the walker succeeds later or
                    # exhausts every candidate. Clearing it broke the
                    # cleanup-provenance chain on full-walk exhaustion —
                    # teardown then lost the deterministic
                    # name and the leaked instance in zone A survived.
                    instance_created = False
                else:
                    # Nothing was accepted in this zone, so there is no
                    # record to reclaim and NOTHING leaked. Recording the
                    # zone here would be a false allocation claim: once a
                    # later candidate succeeds, instance_created=true and
                    # teardown would issue a delete for this instance name
                    # in a zone this run never created in — an unnecessary
                    # API call at best, and at worst a delete aimed at an
                    # unrelated same-named VM. Stockout provenance stays on
                    # stderr (below) and in the exhaustion error.
                    print(
                        f"  zone {candidate_zone} rejected the insert; no partial state to reclaim",
                        file=sys.stderr,
                    )
                print(f"  walking past {candidate_zone} (stockout-class)", file=sys.stderr)
                op = None
                op_name = ""
                continue
        else:
            # Exhausted every candidate — raise the most recent
            # zone-unavailable error so the operator sees the actual
            # cause rather than a generic "no zones tried" message.
            raise RuntimeError(
                f"Zone-walk exhausted ({len(candidate_zones)} candidates); last error: {last_error}"
            ) from last_error

        # 5. Poll canonical 'running'.
        print("Waiting for RUNNING status...", file=sys.stderr)
        result["state"] = poll_instance_state(
            project,
            zone,
            instance_name,
            target_canonical="running",
            timeout=budget.wait_timeout(300, floor=_MIN_STATE_POLL_S),
        )

        # 6. Re-read instance for IPs + label round-trip.
        inst = get_instance(project, zone, instance_name)
        result["public_ip"] = first_external_ip(inst) or wait_for_public_ip(
            project,
            zone,
            instance_name,
            timeout=budget.wait_timeout(120, floor=_MIN_IP_POLL_S),
        )
        if not result["public_ip"]:
            raise RuntimeError("Instance has no external IP after RUNNING (timed out polling)")
        result["private_ip"] = first_internal_ip(inst)
        result["vpc_id"] = short_name(inst.network_interfaces[0].network)
        if inst.network_interfaces[0].subnetwork:
            result["subnet_id"] = short_name(inst.network_interfaces[0].subnetwork)
        # Only emit canonical tag keys when the backing label is actually
        # present on the live instance. Falling back to the REQUESTED
        # values would fabricate a vacuous readback round-trip and mask
        # any regression in `canonical_tags_to_labels` projection
        # (values must be derived from live state; mirrors the reuse-branch shape).
        actual_labels = dict(getattr(inst, "labels", {}) or {})
        derived_tags: dict[str, str] = {}
        if "isv_name" in actual_labels:
            derived_tags["Name"] = actual_labels["isv_name"]
        if "createdby" in actual_labels:
            derived_tags["CreatedBy"] = actual_labels["createdby"]
        result["tags"] = derived_tags

        # 6b. Specified-key contract. Confirm the public key this stub
        # injected is actually present in the instance's `ssh-keys` metadata
        # (readback — never pre-stamped). `key_name` stays None (no managed
        # key-pair store); `instance_key_name` carries the requested label
        # only when the readback proves the key, so
        # InstanceSpecifiedKeyCheck (requested == instance) passes honestly.
        key_confirmed = instance_has_pubkey(inst, ssh_pubkey)
        result["instance_key_name"] = args.key_name if key_confirmed else None
        result.setdefault("tests", {})["specified_key"] = {
            "passed": key_confirmed,
            "message": (
                f"Requested SSH key {args.key_name!r} confirmed in instance ssh-keys metadata"
                if key_confirmed
                else f"Requested SSH key {args.key_name!r} not found in instance ssh-keys metadata"
            ),
            "probes": ["instance_ssh_keys_metadata"],
        }

        # 7. Best-effort readiness gate — SSH OR cloud-init counts as
        # success. Failing BOTH is the only honest reason to call launch
        # failed at this point.
        #
        # These three waits are the long tail of the step: at their own worst
        # cases they run ~1700s BY THEMSELVES (each SSH attempt costs its
        # interval plus the probe's own 15s connect timeout), on top of every
        # wait above. They are the reason the step carries a wall-clock budget
        # instead of a summed cap — an over-cap kill here would strand the
        # RUNNING GPU VM that steps 3-6 just created. Each keeps its full
        # window while the budget can fund it; past that it degrades into an
        # honest `success=false` payload with truthful ownership, which
        # teardown can act on.
        ssh_ok = wait_for_ssh(
            host=result["public_ip"],
            user=args.ssh_user,
            key_file=key_priv,
            max_attempts=budget.probe_attempts(20, interval=10, floor=_MIN_SSH_ATTEMPTS),
            interval=10,
        )
        cloud_init_ok = False
        if ssh_ok:
            cloud_init_ok = wait_for_cloud_init(
                host=result["public_ip"],
                user=args.ssh_user,
                key_file=key_priv,
                timeout_seconds=budget.wait_timeout(600, floor=_MIN_CLOUD_INIT_S),
            )
            # Compute Engine's guest-agent restarts sshd shortly after
            # cloud-init completes (refreshes authorized_keys / host
            # keys). Downstream validators (e.g. CloudInitCheck) connect
            # via paramiko immediately after this step returns and race
            # that restart, surfacing as "Error reading SSH protocol
            # banner: Connection reset by peer". Require 3 consecutive
            # SSH successes here so the post-cloud-init bounce is washed
            # out before we hand control back. Mirrors the reuse-branch
            # readiness gate.
            if cloud_init_ok:
                ssh_stable_ok = wait_for_ssh_stable(
                    host=result["public_ip"],
                    user=args.ssh_user,
                    key_file=key_priv,
                    consecutive=3,
                    interval=10,
                    max_attempts=budget.probe_attempts(24, interval=10, floor=_MIN_SSH_ATTEMPTS),
                )
                if not ssh_stable_ok:
                    print(
                        "  SSH did not stabilize after cloud-init; continuing on best-effort",
                        file=sys.stderr,
                    )
                result["ssh_stable"] = ssh_stable_ok
        result["ssh_ready"] = ssh_ok
        result["cloud_init_ok"] = cloud_init_ok

        if not (ssh_ok or cloud_init_ok):
            raise RuntimeError(
                "Launch reached RUNNING but neither SSH nor cloud-init became observable within the step timeout"
            )

        result["success"] = True
        print("Launch succeeded", file=sys.stderr)

    except Exception as e:
        result.setdefault("error", str(e))
        result["success"] = False
        if args.skip_destroy:
            # Preservation mode: SUPPRESS the compensating deletion so the
            # operator keeps the exact failed fixture state they asked for.
            # Preservation is NOT a teardown-only concern — this block runs
            # long before the terminal teardown step consults the same flag,
            # so gating only there would destroy the very resources the
            # operator asked to keep.
            #
            # Never a PARTIAL clean: all three owned fixtures (instance,
            # firewall, local key) are retained together, because deleting one
            # of them leaves a fixture set that no longer reproduces the
            # failure. Preservation suppresses the delete, never the ownership
            # bookkeeping — instance_created / firewall_created / key_created,
            # the exact identities, and leaked_zones stay truthful in the
            # emitted payload so a later standalone teardown
            # (`teardown.py --from-launch-output`) reclaims exactly what this
            # run owns and nothing it merely adopted.
            #
            # The zone-walk phantom reclamation inside the walk loop stays
            # UNGATED by design: a phantom record in an abandoned capacity-walk
            # zone is not a debugging fixture, and retaining it would bill the
            # operator for an instance they cannot use.
            result["skip_destroy"] = True
            result["preserved_on_failure"] = {
                "instance_id": instance_name if instance_created else "",
                "zone": zone if instance_created else "",
                "leaked_zones": list(result["leaked_zones"]),
                "firewall_name": fw_name if firewall_created else "",
                "key_file": key_priv if (key_created and key_priv) else "",
                # Unproven candidates are preserved too — they are exactly the
                # resources whose existence the run could not settle, so the
                # replay needs them to re-verify the marker and reclaim.
                "unreconciled_resources": list(result["unreconciled_resources"]),
            }
            print(
                "Skip-destroy set: preserving run-owned resources on setup failure "
                f"(instance_created={instance_created} instance={instance_name!r}@{zone!r}, "
                f"firewall_created={firewall_created} firewall={fw_name!r}, "
                f"key_created={key_created}, leaked_zones={result['leaked_zones']})",
                file=sys.stderr,
            )
        else:
            # Cleanup-on-failure — gate on per-resource trackers so a failed
            # verified-reuse adoption doesn't take a pre-existing shared
            # resource with it.
            #
            # This block runs AFTER the provisioning budget may already be
            # spent, so it stamps a reserve of its own (`_CLEANUP_RESERVE`)
            # rather than inheriting an exhausted one — cleanup that is granted
            # no time is the leak the budget exists to prevent. Each retry
            # attempt re-derives its op-wait from what is left of the reserve,
            # which is what keeps a ladder of transient-error retries from
            # outrunning the step cap and losing the whole block (and the
            # payload below) to the SIGKILL.
            cleanup_budget = StepBudget(_CLEANUP_RESERVE)
            try:
                if instance_created:
                    print(
                        f"Cleanup-on-failure: deleting instance {instance_name}",
                        file=sys.stderr,
                    )
                    delete_with_retry(
                        lambda: _delete_instance_op(
                            project,
                            zone,
                            instance_name,
                            wait_s=cleanup_budget.wait_timeout(
                                _CLEANUP_INSTANCE_WAIT_S,
                                floor=_MIN_OP_WAIT_S,
                            ),
                        ),
                        resource_desc=f"instance {instance_name}",
                    )
                if firewall_created:
                    print(
                        f"Cleanup-on-failure: deleting firewall {fw_name}",
                        file=sys.stderr,
                    )
                    delete_with_retry(
                        lambda: _delete_firewall_op(
                            project,
                            fw_name,
                            wait_s=cleanup_budget.wait_timeout(
                                _CLEANUP_FIREWALL_WAIT_S,
                                floor=_MIN_OP_WAIT_S,
                            ),
                        ),
                        resource_desc=f"firewall {fw_name}",
                    )
                if key_created and key_priv:
                    delete_local_keypair(key_priv)
                if result["unreconciled_resources"]:
                    # Deliberately NOT deleted here: ownership is unproven, and
                    # deleting on an unproven claim can destroy another run's
                    # same-named resource. The handoff travels to teardown,
                    # which re-verifies the invocation marker first.
                    print(
                        f"Cleanup-on-failure: {len(result['unreconciled_resources'])} candidate(s) "
                        "left for teardown to marker-verify before any delete",
                        file=sys.stderr,
                    )
            except Exception as cleanup_exc:
                print(f"Cleanup-on-failure error: {cleanup_exc}", file=sys.stderr)

    print(json.dumps(result, indent=2, default=str))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

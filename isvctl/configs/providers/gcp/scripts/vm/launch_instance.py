#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

"""Launch a Compute Engine GPU instance for VM testing.

Multi-zone capacity walk classifies all 4 zone-unavailability wire shapes.
Emits the effective zone in ``result["zone"]`` and any zones where a
partial-state shape fired in ``result["leaked_zones"]`` so teardown can
reclaim phantom records.

Usage:
    python3 launch_instance.py --name isv-test-gpu --instance-type g2-standard-8 \\
        --region us-central1 [--ami-id <image-or-family>] [--image-project <project>]

Optional env-var reuse: when both GCP_VM_INSTANCE_ID and GCP_VM_KEY_FILE
are set, the stub describes the existing VM instead of creating one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import (
    DEFAULT_IMAGE_FAMILY,
    PREFERRED_ZONES,
    api_labels,
    canonical_state,
    canonical_tags,
    create_ssh_firewall_rule,
    delete_failed_zonal_instance,
    generate_ssh_keypair,
    is_sentinel,
    is_zone_unavailable,
    read_ssh_public_key,
    resolve_image,
    resolve_project,
    select_zones,
    unique_suffix,
    wait_for_public_ip,
    wait_for_zonal_op,
)
from common.ssh_utils import ssh_run, wait_for_cloud_init, wait_for_ssh

SSH_USER = "ubuntu"


def _zonal(zone: str, kind: str, value: str) -> str:
    """Build the zonal relative URL form Compute Engine accepts for proto fields."""
    return f"zones/{zone}/{kind}/{value}"


def _build_instance(
    *,
    name: str,
    machine_type: str,
    image_self_link: str,
    network: str,
    subnetwork: str | None,
    target_tag: str,
    labels: dict[str, str],
    ssh_user: str,
    public_key: str,
    zone: str,
) -> Any:
    """Construct the compute_v1.Instance proto used by ``instances.insert``."""
    from google.cloud import compute_v1

    init_params = compute_v1.AttachedDiskInitializeParams(
        source_image=image_self_link,
        disk_size_gb=100,
        disk_type=_zonal(zone, "diskTypes", "pd-balanced"),
    )
    boot_disk = compute_v1.AttachedDisk(
        auto_delete=True,
        boot=True,
        initialize_params=init_params,
    )
    access_config = compute_v1.AccessConfig(
        name="External NAT",
        type_="ONE_TO_ONE_NAT",
        network_tier="PREMIUM",
    )
    nic = compute_v1.NetworkInterface(access_configs=[access_config])
    nic.network = f"global/networks/{network}"
    if subnetwork and not is_sentinel(subnetwork):
        nic.subnetwork = subnetwork

    metadata = compute_v1.Metadata(
        items=[
            compute_v1.Items(key="ssh-keys", value=f"{ssh_user}:{public_key}"),
            compute_v1.Items(key="enable-oslogin", value="FALSE"),
        ]
    )
    # GPU machine types cannot live-migrate — GCE rejects MIGRATE on
    # ``instances.insert`` with HTTP 400 unless on_host_maintenance is
    # TERMINATE. See vendor docs:
    # https://cloud.google.com/compute/docs/instances/setting-instance-scheduling-options
    scheduling = compute_v1.Scheduling(on_host_maintenance="TERMINATE", automatic_restart=True)

    instance = compute_v1.Instance(
        name=name,
        machine_type=_zonal(zone, "machineTypes", machine_type),
        disks=[boot_disk],
        network_interfaces=[nic],
        labels=api_labels(labels),
        tags=compute_v1.Tags(items=[target_tag]),
        metadata=metadata,
        scheduling=scheduling,
    )
    return instance


def _wait_for_guest_signal(
    public_ip: str,
    user: str,
    key_file: str,
    *,
    ssh_attempts: int = 30,
    interval: int = 10,
) -> dict[str, bool]:
    """Try SSH first, then cloud-init wait, then SSH stability probe.

    Stability probe (``N=3`` consecutive successful SSH connections, ~10s
    apart) catches the case where first SSH succeeds during a transient
    sshd window that immediately drops as the guest re-applies metadata
    keys. The dict reports each probe independently so the caller can
    report partial signals in the JSON result.
    """
    signals = {"ssh": False, "cloud_init": False, "stable": False}
    if not public_ip or not key_file:
        return signals
    signals["ssh"] = wait_for_ssh(public_ip, user, key_file, max_attempts=ssh_attempts, interval=interval)
    if not signals["ssh"]:
        return signals
    # Use the shared retrying cloud-init helper. A one-shot ssh_run here
    # would treat a transient SSH transport failure (rc 255 — sshd
    # rebind, brief network hiccup) as a terminal cloud-init failure,
    # which is the wrong policy. The helper retries transport failures
    # within its deadline and only terminates on semantic failures
    # (rc 1 = cloud-init reports error).
    signals["cloud_init"] = wait_for_cloud_init(public_ip, user, key_file)
    # Consecutive-success SSH stability probe.
    stable = True
    for stability_attempt in range(3):
        rc2, _so, _se = ssh_run(public_ip, user, key_file, "exit 0", timeout=10, connect_timeout=5)
        if rc2 != 0:
            stable = False
            break
        if stability_attempt < 2:
            time.sleep(10)
    signals["stable"] = stable
    return signals


def _emit_failed(result: dict[str, Any], message: str) -> int:
    """Mark the launch as failed with ``message`` and print JSON."""
    result["error"] = message
    result["success"] = False
    print(json.dumps(result, indent=2, default=str))
    return 1


def _reuse_existing(args: argparse.Namespace, project: str) -> int:
    """Operator-owned reuse path triggered by GCP_VM_INSTANCE_ID / GCP_VM_KEY_FILE."""
    from google.cloud import compute_v1

    instance_id = os.environ["GCP_VM_INSTANCE_ID"]
    key_file = os.environ["GCP_VM_KEY_FILE"]
    zone = os.environ.get("GCP_VM_ZONE", "")
    if is_sentinel(zone):
        zone = select_zones(args.region)[0]

    instances = compute_v1.InstancesClient()
    inst = instances.get(project=project, zone=zone, instance=instance_id)
    state = canonical_state(getattr(inst, "status", None))

    # Start the instance if it is stopped — matches the AWS oracle's reuse
    # path, which uses `start_instances` + `instance_status_ok` before
    # reporting success.
    if state == "stopped":
        print(f"  reuse: starting stopped instance {instance_id}", file=sys.stderr)
        op = instances.start(project=project, zone=zone, instance=instance_id)
        wait_for_zonal_op(instances, project, zone, op, timeout=600)
        for _ in range(40):
            inst = instances.get(project=project, zone=zone, instance=instance_id)
            state = canonical_state(getattr(inst, "status", None))
            if state == "running":
                break
            time.sleep(5)

    nic = (inst.network_interfaces or [None])[0]
    access = (nic.access_configs or [None])[0] if nic else None
    public_ip = getattr(access, "nat_i_p", "") if access else ""
    private_ip = getattr(nic, "network_i_p", "") if nic else ""

    network_short = (nic.network.rsplit("/", 1)[-1]) if nic and nic.network else ""
    subnet_short = (nic.subnetwork.rsplit("/", 1)[-1]) if nic and nic.subnetwork else ""

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "instance_id": instance_id,
        "instance_type": (inst.machine_type or "").rsplit("/", 1)[-1],
        "region": args.region,
        "zone": zone,
        "availability_zone": zone,
        "state": state,
        "public_ip": public_ip,
        "private_ip": private_ip,
        "vpc_id": network_short,
        "subnet_id": subnet_short,
        "security_group_id": "",
        "firewall_name": os.environ.get("GCP_VM_FIREWALL_NAME", ""),
        "firewall_created": False,
        "key_file": key_file,
        "key_name": None,
        "key_created": False,
        "instance_created": False,
        "leaked_zones": [],
        "ami_id": (inst.disks[0].source if inst.disks and inst.disks[0].source else ""),
        "tags": canonical_tags(instance_id),
        "ssh_user": SSH_USER,
        "reused": True,
    }

    if state != "running":
        result["error"] = f"Instance {instance_id} is {state}, expected running"
        print(json.dumps(result, indent=2, default=str))
        return 1

    # Same readiness gates as the create path: a real external IP must be
    # observable and the guest must respond to SSH or cloud-init within the
    # step timeout. Without this, reuse can ship `success=true` for a
    # running-but-unreachable VM and downstream SSH validations fail
    # outside the launch contract.
    if not public_ip:
        public_ip = wait_for_public_ip(instances, project, zone, instance_id, timeout=180, interval=5) or ""
    result["public_ip"] = public_ip
    if not public_ip:
        result["error"] = "reuse instance has no external IP"
        print(json.dumps(result, indent=2, default=str))
        return 1

    signals = _wait_for_guest_signal(public_ip, SSH_USER, key_file, ssh_attempts=24)
    result["ssh_ready"] = signals["ssh"] and signals["stable"]
    result["cloud_init_ready"] = signals["cloud_init"]
    result["ssh_stable"] = signals["stable"]
    # Reuse must clear the same readiness bar as create: SSH success
    # alone is not enough; we require cloud-init done AND the consecutive
    # SSH-stability probe to pass.
    if not (signals["ssh"] and signals["cloud_init"] and signals["stable"]):
        result["error"] = "reuse guest readiness probes did not all pass (ssh+cloud_init+stable required)"
        print(json.dumps(result, indent=2, default=str))
        return 1

    result["success"] = True
    print(json.dumps(result, indent=2, default=str))
    return 0


def main() -> int:
    """Launch a Compute Engine GPU instance and emit structured JSON."""
    parser = argparse.ArgumentParser(description="Launch GCP Compute Engine GPU instance")
    parser.add_argument("--name", default="isv-test-gpu", help="Instance base name (run-id suffix appended)")
    parser.add_argument("--instance-type", default="g2-standard-8", help="GCE machine type")
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_VM_REGION", "us-central1"),
        help="GCP region or full zone; zone-pin is honored verbatim",
    )
    parser.add_argument("--vpc-id", default="default", help="Network short name (default: default)")
    parser.add_argument("--subnet-id", default="", help="Subnetwork short name or self-link")
    parser.add_argument(
        "--ami-id",
        default="",
        help="Source image short-name or family (default: public DLVM L4 family)",
    )
    parser.add_argument("--image-project", default="", help="Project hosting the source image")
    parser.add_argument("--key-name", default="isv-test-key", help="Local PEM basename only")
    parser.add_argument("--project", default="", help="GCP project (default: ADC)")
    args = parser.parse_args()

    project = resolve_project(args.project)

    if os.environ.get("GCP_VM_INSTANCE_ID") and os.environ.get("GCP_VM_KEY_FILE"):
        return _reuse_existing(args, project)

    try:
        from google.cloud import compute_v1
    except ImportError as exc:  # pragma: no cover
        print(
            json.dumps({"success": False, "platform": "vm", "error": f"google-cloud-compute missing: {exc}"}, indent=2)
        )
        return 1

    suffix_name = unique_suffix(args.name)
    suffix_key = unique_suffix(args.key_name)
    firewall_base = unique_suffix("isv-test-ssh")
    target_tag = unique_suffix("isv-test-vm")

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "instance_id": suffix_name,
        "instance_type": args.instance_type,
        "region": args.region,
        "zone": "",
        "availability_zone": "",
        "state": "",
        "public_ip": "",
        "private_ip": "",
        "vpc_id": args.vpc_id,
        "subnet_id": args.subnet_id if not is_sentinel(args.subnet_id) else "",
        "security_group_id": "",
        "firewall_name": "",
        "firewall_created": False,
        "key_file": "",
        "key_name": None,
        "key_created": False,
        "instance_created": False,
        "leaked_zones": [],
        "ami_id": "",
        "tags": canonical_tags(suffix_name),
        "ssh_user": SSH_USER,
    }

    instances_client = compute_v1.InstancesClient()
    firewalls_client = compute_v1.FirewallsClient()
    images_client = compute_v1.ImagesClient()

    key_file = ""
    key_created = False
    firewall_name: str | None = None
    firewall_created = False
    # Mutable tracker the firewall helper populates immediately after the
    # insert acks. The caller's finally: block reads it on wait-failure so
    # ownership of the accepted firewall never gets lost.
    firewall_tracker: dict[str, Any] = {}
    candidate_zones = select_zones(args.region, PREFERRED_ZONES)
    if not candidate_zones:
        return _emit_failed(result, "no candidate zones resolved from --region")

    try:
        # Local PEM generation (verified-reuse aware; tuple-unpack is the
        # documented cleanup-contract shape).
        key, key_created = generate_ssh_keypair(suffix_key)
        key_file = key
        public_key = read_ssh_public_key(key_file)
        result["key_file"] = key_file
        result["key_created"] = key_created

        # Resolve the source image (operator scope first; vendor DLVM second).
        image_arg = args.ami_id if not is_sentinel(args.ami_id) else DEFAULT_IMAGE_FAMILY
        # Honor an explicit image-project override by treating it as the
        # operator scope. The resolver already falls back to DLVM second.
        if not is_sentinel(args.image_project):
            scope_project = args.image_project
        else:
            scope_project = project
        image_self_link, image_name = resolve_image(images_client, scope_project, image_arg)
        result["ami_id"] = image_self_link
        result["ami_name"] = image_name

        # Verified-reuse firewall + per-VM target tag (tuple-unpack matches
        # the documented cleanup-contract shape). The tracker dict captures
        # ownership the moment the insert acks, so a transient wait error
        # never leaves the rule stranded.
        firewall, firewall_created = create_ssh_firewall_rule(
            firewalls_client,
            project,
            args.vpc_id,
            name=firewall_base,
            target_tag=target_tag,
            tracker=firewall_tracker,
        )
        firewall_name = firewall
        result["firewall_name"] = firewall_name
        result["firewall_created"] = firewall_created
        result["security_group_id"] = firewall_name

        instance_proto = _build_instance(
            name=suffix_name,
            machine_type=args.instance_type,
            image_self_link=image_self_link,
            network=args.vpc_id,
            subnetwork=args.subnet_id,
            target_tag=target_tag,
            labels=canonical_tags(suffix_name),
            ssh_user=SSH_USER,
            public_key=public_key,
            zone=candidate_zones[0],
        )

        # Multi-zone walk classifying all 4 capacity-error wire shapes.
        leaked: list[str] = []
        last_error: Exception | None = None
        effective_zone: str | None = None
        for zone in candidate_zones:
            # Re-stamp zonal proto fields (machineType / disk_type) for this zone.
            instance_proto.machine_type = _zonal(zone, "machineTypes", args.instance_type)
            for d in instance_proto.disks:
                if d.initialize_params and d.initialize_params.disk_type:
                    d.initialize_params.disk_type = _zonal(zone, "diskTypes", "pd-balanced")

            print(f"Attempting instances.insert in {zone}...", file=sys.stderr)
            try:
                op = instances_client.insert(project=project, zone=zone, instance_resource=instance_proto)
                result["instance_created"] = True
                # Stamp the attempted zone BEFORE the wait so a non-capacity
                # wait failure still surfaces a zone for teardown — without
                # it, the cleanup path has no zone to target the partial
                # record in.
                result["zone"] = zone
                result["availability_zone"] = zone
                wait_for_zonal_op(instances_client, project, zone, op, timeout=600)
                effective_zone = zone
                break
            except Exception as exc:
                last_error = exc
                if is_zone_unavailable(exc):
                    print(f"  zone {zone} unavailable: {exc}", file=sys.stderr)
                    # Shapes 2 / 4 may leave a partial record — reclaim it
                    # and accumulate the zone in leaked_zones either way so
                    # teardown can sweep anything the reclaim missed. Mirror
                    # the local list into result immediately so a later
                    # non-capacity failure inside this loop still surfaces
                    # the leaked-zone context to teardown.
                    cleaned = delete_failed_zonal_instance(instances_client, project, zone, suffix_name)
                    if not cleaned:
                        leaked.append(zone)
                    else:
                        leaked.append(zone)
                    result["leaked_zones"] = list(leaked)
                    continue
                # Non-capacity error: still re-stamp the accumulated leaked
                # zones into result before propagating so teardown has the
                # context it needs.
                result["leaked_zones"] = list(leaked)
                raise

        result["leaked_zones"] = leaked
        if effective_zone is None:
            return _emit_failed(
                result,
                f"no candidate zone accepted the instance (last error: {last_error})",
            )

        result["zone"] = effective_zone
        result["availability_zone"] = effective_zone

        # Poll for state + external IP.
        for _ in range(40):
            inst = instances_client.get(project=project, zone=effective_zone, instance=suffix_name)
            state = canonical_state(getattr(inst, "status", None))
            result["state"] = state
            if state == "running":
                break
            time.sleep(5)

        public_ip = wait_for_public_ip(instances_client, project, effective_zone, suffix_name, timeout=180, interval=5)
        if not public_ip:
            return _emit_failed(result, "instance has no external IP after launch")
        result["public_ip"] = public_ip

        # Refresh private IP + subnet / network shorts from the final state.
        inst = instances_client.get(project=project, zone=effective_zone, instance=suffix_name)
        nic = (inst.network_interfaces or [None])[0]
        if nic:
            result["private_ip"] = getattr(nic, "network_i_p", "")
            if nic.network:
                result["vpc_id"] = nic.network.rsplit("/", 1)[-1]
            if nic.subnetwork:
                result["subnet_id"] = nic.subnetwork.rsplit("/", 1)[-1]

        # Guest readiness gates: SSH first contact, cloud-init completion,
        # and consecutive-success SSH stability. All three must pass to
        # report success — first SSH success alone may be a transient
        # sshd that immediately drops as the guest re-applies metadata
        # keys.
        signals = _wait_for_guest_signal(public_ip, SSH_USER, key_file, ssh_attempts=24)
        result["ssh_ready"] = signals["ssh"] and signals["stable"]
        result["cloud_init_ready"] = signals["cloud_init"]
        result["ssh_stable"] = signals["stable"]

        if not (signals["ssh"] and signals["cloud_init"] and signals["stable"]):
            return _emit_failed(
                result,
                "guest readiness probes did not all pass (need ssh+cloud_init+stable)",
            )

        result["success"] = True
        print(json.dumps(result, indent=2, default=str))
        return 0

    except Exception as exc:
        result["error"] = str(exc)
        # Best-effort cleanup so a partial launch doesn't strand resources
        # in the operator's project. The teardown step will sweep any
        # leaked_zones entries the walker collected above.
        if result.get("instance_created") and result.get("zone"):
            delete_failed_zonal_instance(instances_client, project, result["zone"], suffix_name)
        # Pick up firewall ownership from the tracker — the helper sets
        # tracker["created"] immediately after insert acks, so a wait
        # failure inside the helper still surfaces the name here.
        if firewall_tracker.get("created"):
            firewall_name = firewall_tracker["name"]
            firewall_created = True
            result["firewall_name"] = firewall_name
            result["firewall_created"] = True
        if firewall_created and firewall_name:
            try:
                op = firewalls_client.delete(project=project, firewall=firewall_name)
                op.result(timeout=120)
            except Exception as cleanup_exc:
                print(f"Warning: firewall cleanup failed: {cleanup_exc}", file=sys.stderr)
        if key_created and key_file:
            for path in (Path(key_file), Path(key_file).with_suffix(".pub")):
                try:
                    if path.exists():
                        path.chmod(0o600)
                        path.unlink()
                except OSError:
                    pass
        print(json.dumps(result, indent=2, default=str))
        return 1


if __name__ == "__main__":
    sys.exit(main())

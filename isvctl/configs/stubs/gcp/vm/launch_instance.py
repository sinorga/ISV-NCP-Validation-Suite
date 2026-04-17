#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Launch a GCP Compute Engine GPU instance for VM testing.

Usage:
    python launch_instance.py --name isv-test-gpu --instance-type n1-standard-4 \
        --region asia-east1-a

Environment:
    GOOGLE_CLOUD_PROJECT / GCP_PROJECT  GCP project ID (required)
    GCP_VM_INSTANCE / GCP_VM_KEY_FILE   If set, reuse the existing instance
                                        (dev workflow — parallels AWS_VM_* flow)

Output JSON (per docs/existing-patterns/vm.md):
{
    "success": true,
    "platform": "vm",
    "instance_id": "<instance-name>",
    "public_ip": "34.x.x.x",
    "key_file": "/tmp/isv-test-key.pem",
    "vpc_id": "<network-name>",
    "instance_state": "running",
    "security_group_id": "<firewall-name>",
    "key_name": "isv-test-key",
    ...
}
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.compute import (
    canonical_state,
    create_ssh_key_pair,
    ensure_ssh_firewall,
    get_default_network,
    get_instance_external_ip,
    get_instance_internal_ip,
    read_public_key,
    requires_guest_accelerator,
    resolve_gpu_image,
    resolve_project,
    wait_for_ssh,
    zone_to_region,
)
from common.errors import handle_gcp_errors
from google.cloud import compute_v1


def build_instance(
    name: str,
    machine_type: str,
    zone: str,
    source_image: str,
    network_self_link: str,
    ssh_user: str,
    public_key: str,
    accelerator_type: str | None,
    accelerator_count: int,
    disk_size_gb: int = 100,
) -> compute_v1.Instance:
    """Build a compute_v1.Instance resource for the VM test fixture."""
    # Boot disk
    disk = compute_v1.AttachedDisk()
    disk.boot = True
    disk.auto_delete = True
    init_params = compute_v1.AttachedDiskInitializeParams()
    init_params.source_image = source_image
    init_params.disk_size_gb = disk_size_gb
    init_params.disk_type = f"zones/{zone}/diskTypes/pd-balanced"
    disk.initialize_params = init_params

    # Network interface with external IP (ONE_TO_ONE_NAT)
    access = compute_v1.AccessConfig()
    access.type_ = "ONE_TO_ONE_NAT"
    access.name = "External NAT"
    access.network_tier = "PREMIUM"
    nic = compute_v1.NetworkInterface()
    nic.network = network_self_link
    nic.access_configs = [access]

    # SSH public key via metadata (GCP's equivalent of AWS key pairs)
    metadata_ssh = compute_v1.Items()
    metadata_ssh.key = "ssh-keys"
    metadata_ssh.value = f"{ssh_user}:{public_key}"
    metadata = compute_v1.Metadata()
    metadata.items = [metadata_ssh]

    instance = compute_v1.Instance()
    instance.name = name
    instance.machine_type = f"zones/{zone}/machineTypes/{machine_type}"
    instance.disks = [disk]
    instance.network_interfaces = [nic]
    instance.metadata = metadata

    # Labels are GCP's equivalent of AWS tags; keys must be lowercase.
    # The tag validation in the provider config overrides required_keys to
    # match these lowercase names.
    instance.labels = {"name": name, "created-by": "isvtest"}
    # Network tags are separate from labels in GCP; not needed for SSH since
    # the firewall rule uses source_ranges.
    instance.tags = compute_v1.Tags(items=["isv-test"])

    # Attach GPU if the machine family requires an explicit guest_accelerator
    # (only n1-* in the supported matrix). GPU instances require
    # on_host_maintenance=TERMINATE.
    if accelerator_type and accelerator_count > 0:
        accel = compute_v1.AcceleratorConfig()
        accel.accelerator_count = accelerator_count
        accel.accelerator_type = f"zones/{zone}/acceleratorTypes/{accelerator_type}"
        instance.guest_accelerators = [accel]
        sched = compute_v1.Scheduling()
        sched.on_host_maintenance = "TERMINATE"
        sched.automatic_restart = True
        instance.scheduling = sched

    return instance


def reuse_existing_instance(project: str, zone: str) -> int:
    """Describe an existing instance rather than launching a new one.

    Used when GCP_VM_INSTANCE and GCP_VM_KEY_FILE are set (dev workflow).
    """
    instance_name = os.environ["GCP_VM_INSTANCE"]
    key_file = os.environ["GCP_VM_KEY_FILE"]

    print(f"Reusing existing instance {instance_name}", file=sys.stderr)

    client = compute_v1.InstancesClient()
    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "instance_id": instance_name,
        "region": zone,
        "key_file": key_file,
        "reused": True,
    }

    try:
        instance = client.get(project=project, zone=zone, instance=instance_name)
        gcp_status = instance.status

        # Start the instance if it's terminated (GCP's "stopped")
        if gcp_status == "TERMINATED":
            print(f"  Instance {instance_name} is TERMINATED — starting it...", file=sys.stderr)
            op = client.start(project=project, zone=zone, instance=instance_name)
            op.result(timeout=300)
            instance = client.get(project=project, zone=zone, instance=instance_name)
            gcp_status = instance.status

        state = canonical_state(gcp_status)
        result["state"] = state
        result["instance_state"] = state
        result["gcp_status"] = gcp_status
        result["instance_type"] = instance.machine_type.rsplit("/", 1)[-1]
        result["public_ip"] = get_instance_external_ip(instance)
        result["private_ip"] = get_instance_internal_ip(instance)
        network = instance.network_interfaces[0].network if instance.network_interfaces else ""
        result["vpc_id"] = network.rsplit("/", 1)[-1] if network else ""
        result["zone"] = zone
        result["key_name"] = Path(key_file).stem
        result["security_group_id"] = f"{instance_name}-ssh"
        result["success"] = state == "running"

        if not result["success"]:
            result["error"] = f"Instance {instance_name} is {gcp_status}, expected RUNNING"
    except Exception as e:
        result["error"] = str(e)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


@handle_gcp_errors
def main() -> int:
    """Launch a GPU-enabled Compute Engine instance for VM testing.

    If GCP_VM_INSTANCE and GCP_VM_KEY_FILE are set, describes the existing
    instance instead (dev workflow, parallels the oracle's AWS_VM_* flow).

    Otherwise, resolves the preferred GPU image, generates an SSH key pair,
    creates an SSH firewall rule on the default VPC, launches the instance
    with the key attached via instance metadata, and best-effort waits for
    sshd to accept the key.

    Returns:
        0 on success, 1 on failure
    """
    parser = argparse.ArgumentParser(description="Launch GPU Compute Engine instance")
    parser.add_argument("--name", default="isv-test-gpu", help="Instance name")
    parser.add_argument("--instance-type", default="n1-standard-4", help="GCP machine type")
    parser.add_argument(
        "--region",
        default=os.environ.get("GCP_ZONE", "asia-east1-a"),
        help="GCP zone (GCP has no region-level compute; the test config's 'region' is a zone)",
    )
    parser.add_argument("--project", help="GCP project ID (default: $GOOGLE_CLOUD_PROJECT)")
    parser.add_argument("--network", help="Network name (default: 'default')")
    parser.add_argument("--key-name", default="isv-test-key", help="SSH key basename")
    parser.add_argument("--ssh-user", default="ubuntu", help="SSH username baked into metadata")
    parser.add_argument("--accelerator-count", type=int, default=1, help="GPU count (0 disables)")
    parser.add_argument(
        "--accelerator-type",
        default=None,
        help="GCP accelerator type (e.g. nvidia-tesla-t4). Auto-selected by machine family if omitted.",
    )
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = args.region

    # Reuse existing instance if env vars are set
    if os.environ.get("GCP_VM_INSTANCE") and os.environ.get("GCP_VM_KEY_FILE"):
        return reuse_existing_instance(project, zone)

    instances_client = compute_v1.InstancesClient()
    networks_client = compute_v1.NetworksClient()
    firewalls_client = compute_v1.FirewallsClient()
    images_client = compute_v1.ImagesClient()

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "instance_id": args.name,
        "instance_type": args.instance_type,
        "region": zone,
        "zone": zone,
        "gcp_region": zone_to_region(zone),
        "project": project,
        "ssh_user": args.ssh_user,
    }

    # Track what we created so cleanup on failure deletes only our resources.
    created_instance = False
    created_firewall: str | None = None

    try:
        # ========= Step 1: Resolve image =========
        source_image, image_name = resolve_gpu_image(images_client)
        result["image_name"] = image_name

        # ========= Step 2: Network + firewall =========
        network_self_link = (
            f"projects/{project}/global/networks/{args.network}"
            if args.network
            else get_default_network(networks_client, project)
        )
        network_name = network_self_link.rsplit("/", 1)[-1]
        result["vpc_id"] = network_name

        firewall_name = f"{args.name}-ssh"
        ensure_ssh_firewall(firewalls_client, project, network_self_link, firewall_name)
        created_firewall = firewall_name
        result["security_group_id"] = firewall_name

        # ========= Step 3: SSH key pair =========
        priv_key, pub_key = create_ssh_key_pair(args.key_name)
        public_key = read_public_key(pub_key)
        result["key_name"] = args.key_name
        result["key_file"] = priv_key

        # ========= Step 4: Build + insert instance =========
        accelerator_type = args.accelerator_type or requires_guest_accelerator(args.instance_type)
        accelerator_count = args.accelerator_count if accelerator_type else 0
        result["accelerator_type"] = accelerator_type
        result["accelerator_count"] = accelerator_count

        instance_resource = build_instance(
            name=args.name,
            machine_type=args.instance_type,
            zone=zone,
            source_image=source_image,
            network_self_link=network_self_link,
            ssh_user=args.ssh_user,
            public_key=public_key,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
        )

        print(f"Launching instance {args.name} in {zone}...", file=sys.stderr)
        op = instances_client.insert(
            project=project,
            zone=zone,
            instance_resource=instance_resource,
        )
        op.result(timeout=600)
        created_instance = True
        print("  Instance operation completed", file=sys.stderr)

        # ========= Step 5: Describe and extract IPs =========
        instance = instances_client.get(project=project, zone=zone, instance=args.name)
        gcp_status = instance.status
        result["gcp_status"] = gcp_status
        result["state"] = canonical_state(gcp_status)
        result["instance_state"] = result["state"]
        result["public_ip"] = get_instance_external_ip(instance)
        result["private_ip"] = get_instance_internal_ip(instance)

        if not result["public_ip"]:
            raise RuntimeError(f"Instance {args.name} has no external IP after provisioning")

        # ========= Step 6: Best-effort SSH wait =========
        # GCP has no server-side "OS ready" waiter; poll SSH for up to ~3 min
        # so downstream SSH-based validations don't race cloud-init. A failed
        # wait does NOT fail the step (ConnectivityCheck reports real status).
        print("Waiting for SSH to accept the key...", file=sys.stderr)
        ssh_ready = wait_for_ssh(
            result["public_ip"],
            args.ssh_user,
            priv_key,
            max_attempts=20,
            interval=10,
        )
        result["ssh_ready"] = ssh_ready
        if not ssh_ready:
            print("  WARNING: SSH did not become ready; downstream checks will report", file=sys.stderr)

        result["success"] = result["state"] == "running"
        if not result["success"]:
            result["error"] = f"Instance state is {gcp_status}, expected RUNNING"

    except Exception as e:
        result["error"] = str(e)
        print(f"ERROR: {e}", file=sys.stderr)
        # Clean up resources this stub created; DON'T fall through to the
        # global teardown — stubs may run in isolation.
        if created_instance:
            try:
                print(f"  Cleanup: deleting instance {args.name}", file=sys.stderr)
                instances_client.delete(project=project, zone=zone, instance=args.name).result(timeout=300)
            except Exception as cleanup_err:
                print(f"  Cleanup warning (instance): {cleanup_err}", file=sys.stderr)
        if created_firewall:
            try:
                print(f"  Cleanup: deleting firewall {created_firewall}", file=sys.stderr)
                firewalls_client.delete(project=project, firewall=created_firewall).result(timeout=120)
            except Exception as cleanup_err:
                print(f"  Cleanup warning (firewall): {cleanup_err}", file=sys.stderr)

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

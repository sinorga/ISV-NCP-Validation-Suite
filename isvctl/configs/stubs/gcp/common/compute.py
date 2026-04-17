# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared Compute Engine helpers for GCP stubs.

Mirrors the oracle's ``aws/common/ec2.py`` so stub structure is consistent
across NCPs. Provides:

- Image resolution (preferring the custom docker+CUDA GPU image when present)
- SSH key-pair generation (GCP attaches the key via instance metadata)
- Firewall rule creation (GCP has no "security group" — stateful rules live
  on the VPC)
- Instance IP extraction
- Status canonicalisation (GCP's ``TERMINATED`` → the oracle's ``stopped``)
- Best-effort SSH wait
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from google.api_core import exceptions as gax_exc
from google.cloud import compute_v1

# Mapping from GCP instance.status to the canonical state names the
# validators already expect (oracle/aws uses lowercase AWS state names).
# See docs/gcp.yaml: "GCP stopped VMs have status TERMINATED".
_STATE_MAP = {
    "PROVISIONING": "pending",
    "STAGING": "pending",
    "RUNNING": "running",
    "STOPPING": "stopping",
    "STOPPED": "stopped",  # transient — GCP typically skips straight to TERMINATED
    "SUSPENDING": "stopping",
    "SUSPENDED": "stopped",
    "TERMINATED": "stopped",
    "REPAIRING": "pending",
}

# Preferred GPU image chain (from docs/gcp.yaml).
# The first entry has CUDA on the non-login PATH plus Docker + nvidia-container-toolkit
# preinstalled, which makes DriverCheck/ContainerRuntimeCheck pass without
# excluding them. The fallbacks require excluding those checks.
_GPU_IMAGE_CHAIN: list[tuple[str, str]] = [
    ("shoreline-eagle", "ncp-base-cu129-docker"),
    ("deeplearning-platform-release", "common-cu129-ubuntu-2204-nvidia-580"),
    ("deeplearning-platform-release", "common-cu128-ubuntu-2204-nvidia-570"),
    ("ubuntu-os-cloud", "ubuntu-2204-lts"),
]


def canonical_state(gcp_status: str) -> str:
    """Translate a GCP ``instance.status`` into a canonical state name.

    Validators and downstream stubs reference states like ``running`` and
    ``stopped``; GCP uses uppercase values like ``RUNNING`` and ``TERMINATED``.
    """
    if not gcp_status:
        return "unknown"
    return _STATE_MAP.get(gcp_status.upper(), gcp_status.lower())


def resolve_project(explicit: str | None = None) -> str:
    """Resolve the GCP project ID from args, env, or application-default creds.

    Resolution order:
      1. ``--project`` CLI arg
      2. ``GOOGLE_CLOUD_PROJECT`` env var (standard ADC convention)
      3. ``GCP_PROJECT`` env var (isvtest convention)
      4. ``google.auth.default()`` project — reads the gcloud ADC / service
         account JSON so stubs work in environments where the harness
         doesn't export the project env var (common when gcloud auth
         application-default login is the only configured credential source).
    """
    project = explicit or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if project:
        return project

    # Final fallback: ADC. google.auth.default returns (credentials, project_id).
    try:
        import google.auth

        _, adc_project = google.auth.default()
        if adc_project:
            return adc_project
    except Exception as e:
        print(f"  resolve_project: ADC lookup failed: {e}", file=sys.stderr)

    raise RuntimeError(
        "GCP project not set. Pass --project, export GOOGLE_CLOUD_PROJECT / GCP_PROJECT, "
        "or run `gcloud config set project <id>` so application-default credentials include it.",
    )


def zone_to_region(zone: str) -> str:
    """Strip the trailing ``-<letter>`` off a zone to get its parent region.

    ``asia-east1-a`` → ``asia-east1``.
    """
    if not zone:
        return zone
    parts = zone.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 1:
        return parts[0]
    return zone


def requires_guest_accelerator(instance_type: str) -> str | None:
    """Return the accelerator type to attach for ``instance_type``, or None.

    Only ``n1-*`` machines (and ``a2-highgpu-*``) need an explicit guest
    accelerator; ``g2-*`` instances ship with an integrated L4 and reject
    the ``guest_accelerators`` field. See docs/gcp.yaml.
    """
    if not instance_type:
        return None
    family = instance_type.split("-", 1)[0]
    if family == "n1":
        return "nvidia-tesla-t4"
    if instance_type.startswith("a2-highgpu"):
        return "nvidia-tesla-a100"
    return None


def _image_exists(images_client: compute_v1.ImagesClient, project: str, family: str) -> bool:
    try:
        images_client.get_from_family(project=project, family=family)
        return True
    except gax_exc.NotFound:
        return False
    except gax_exc.Forbidden:
        # The preferred custom image lives in another project we may lack
        # access to — skip it silently and try the next candidate.
        return False


def resolve_gpu_image(images_client: compute_v1.ImagesClient) -> tuple[str, str]:
    """Resolve a GPU-capable source image.

    Walks the preferred chain from ``docs/gcp.yaml`` and returns the
    ``(source_image_url, family_name)`` tuple for the first one available.
    """
    last_error: Exception | None = None
    for project, family in _GPU_IMAGE_CHAIN:
        try:
            image = images_client.get_from_family(project=project, family=family)
            source = f"projects/{project}/global/images/{image.name}"
            print(f"  Selected image: {image.name} (family={family}, project={project})", file=sys.stderr)
            return source, family
        except gax_exc.NotFound as e:
            last_error = e
            continue
        except gax_exc.Forbidden as e:
            last_error = e
            continue
    if last_error:
        raise RuntimeError(f"No usable GPU image from chain. Last error: {last_error}")
    raise RuntimeError("No usable GPU image from chain.")


def get_default_network(networks_client: compute_v1.NetworksClient, project: str) -> str:
    """Return the self-link for the project's ``default`` VPC network."""
    network = networks_client.get(project=project, network="default")
    return network.self_link


def ensure_ssh_firewall(
    firewalls_client: compute_v1.FirewallsClient,
    project: str,
    network_self_link: str,
    firewall_name: str,
) -> None:
    """Create an INGRESS TCP/22 firewall rule, or confirm an existing one.

    GCP firewall rules REQUIRE at least one ``allowed`` entry with
    ``I_p_protocol`` set (see docs/gcp.yaml) — an empty Allowed() returns 400.
    """
    try:
        firewalls_client.get(project=project, firewall=firewall_name)
        print(f"  Firewall rule already exists: {firewall_name}", file=sys.stderr)
        return
    except gax_exc.NotFound:
        pass

    allowed = compute_v1.Allowed()
    allowed.I_p_protocol = "tcp"
    allowed.ports = ["22"]

    fw = compute_v1.Firewall()
    fw.name = firewall_name
    fw.network = network_self_link
    fw.direction = "INGRESS"
    fw.source_ranges = ["0.0.0.0/0"]
    fw.allowed = [allowed]
    fw.description = "ISV validation SSH ingress"

    op = firewalls_client.insert(project=project, firewall_resource=fw)
    op.result(timeout=120)
    print(f"  Created firewall rule: {firewall_name}", file=sys.stderr)


def create_ssh_key_pair(key_name: str, key_dir: str | Path | None = None) -> tuple[str, str]:
    """Generate an ED25519 key pair and return ``(priv_path, pub_path)``.

    Reuses the pair if both files already exist. Uses ``ssh-keygen -N ''``
    so no passphrase is required (matches the oracle's behaviour of
    persisting EC2 key material with mode 0400).
    """
    if key_dir is None:
        key_dir = Path("/tmp")
    else:
        key_dir = Path(key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)

    priv = key_dir / f"{key_name}.pem"
    pub = key_dir / f"{key_name}.pub"

    if priv.exists() and pub.exists():
        priv.chmod(0o400)
        return str(priv), str(pub)

    # Regenerate from scratch. ssh-keygen refuses to overwrite without -f+Y prompting.
    if priv.exists():
        priv.chmod(0o600)
        priv.unlink()
    if pub.exists():
        pub.unlink()

    subprocess.run(
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-f",
            str(priv),
            "-N",
            "",
            "-C",
            key_name,
        ],
        check=True,
        capture_output=True,
    )
    priv.chmod(0o400)
    print(f"  Created SSH key: {priv}", file=sys.stderr)
    return str(priv), str(pub)


def read_public_key(pub_path: str) -> str:
    """Return the raw OpenSSH public key line (stripped)."""
    return Path(pub_path).read_text().strip()


def get_instance_external_ip(instance: compute_v1.Instance) -> str | None:
    """Return the instance's primary external IPv4, or None."""
    for nic in instance.network_interfaces or []:
        for ac in nic.access_configs or []:
            if ac.nat_i_p:
                return ac.nat_i_p
    return None


def get_instance_internal_ip(instance: compute_v1.Instance) -> str | None:
    """Return the instance's primary internal IPv4, or None."""
    for nic in instance.network_interfaces or []:
        if nic.network_i_p:
            return nic.network_i_p
    return None


def wait_for_ssh(
    host: str,
    user: str,
    key_file: str,
    max_attempts: int = 20,
    interval: int = 10,
) -> bool:
    """Poll the host for SSH readiness and return True on first success.

    Uses the SSH flags called out in ``docs/existing-patterns/vm.md``:
      - ``IdentitiesOnly=yes`` is CRITICAL so ssh doesn't burn through all
        agent keys before ours (causes "Too many authentication failures").
      - ``UserKnownHostsFile=/dev/null`` avoids stale host keys when cloud
        IPs get reused across runs.
      - ``PasswordAuthentication=no`` instead of ``BatchMode=yes`` — the
        latter can reject key auth on some sshd configs.
    """
    if not host or not key_file or not Path(key_file).exists():
        print(f"  wait_for_ssh: skipping (host={host!r}, key={key_file!r})", file=sys.stderr)
        return False

    ssh_cmd_base = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "ConnectTimeout=5",
        "-i",
        key_file,
        f"{user}@{host}",
        "exit 0",
    ]

    for attempt in range(1, max_attempts + 1):
        try:
            result = subprocess.run(ssh_cmd_base, capture_output=True, timeout=15)
            if result.returncode == 0:
                print(f"  SSH ready after attempt {attempt}", file=sys.stderr)
                return True
        except (subprocess.TimeoutExpired, OSError):
            pass

        print(f"  Waiting for SSH... (attempt {attempt}/{max_attempts})", file=sys.stderr)
        time.sleep(interval)

    return False


def ssh_exec(
    host: str,
    user: str,
    key_file: str,
    command: str,
    timeout: int = 30,
) -> tuple[int, str, str]:
    """Execute a single SSH command and return (rc, stdout, stderr)."""
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "ConnectTimeout=10",
            "-i",
            key_file,
            f"{user}@{host}",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def describe_instance(
    instances_client: compute_v1.InstancesClient,
    project: str,
    zone: str,
    name: str,
) -> Any:
    """Wrap ``InstancesClient.get`` with a single retry on transient disconnects.

    Per ``docs/gcp.yaml``, the Compute API occasionally drops HTTP connections
    mid-operation (~14% of runs); idempotent reads retry once.
    """
    try:
        return instances_client.get(project=project, zone=zone, instance=name)
    except gax_exc.ServiceUnavailable:
        time.sleep(2)
        return instances_client.get(project=project, zone=zone, instance=name)
    except ConnectionError:
        time.sleep(2)
        return instances_client.get(project=project, zone=zone, instance=name)

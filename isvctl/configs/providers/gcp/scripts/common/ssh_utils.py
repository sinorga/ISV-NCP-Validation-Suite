# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared SSH utilities for GCP stub scripts.

Mirrors the AWS helper. Compute Engine has no managed SSH transport
distinct from regular sshd, so the flags and rc handling are identical.
"""

from __future__ import annotations

import subprocess
import sys
import time

_SSH_OPTS = (
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "BatchMode=yes",
    "-o",
    "IdentitiesOnly=yes",
    "-o",
    "IdentityAgent=none",
    "-o",
    "PasswordAuthentication=no",
)


def ssh_run(
    host: str,
    user: str,
    key_file: str,
    command: str,
    *,
    timeout: int = 30,
    connect_timeout: int = 10,
) -> tuple[int, str, str]:
    """Run a single command over SSH. Returns (exit_code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            [
                "ssh",
                *_SSH_OPTS,
                "-o",
                f"ConnectTimeout={connect_timeout}",
                "-i",
                key_file,
                f"{user}@{host}",
                "--",
                command,
            ],
            capture_output=True,
            timeout=timeout,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        return 124, "", f"TimeoutExpired: {err}"
    except OSError as err:
        return 255, "", f"OSError: {err}"
    return proc.returncode, proc.stdout, proc.stderr


def wait_for_ssh(
    host: str,
    user: str,
    key_file: str,
    max_attempts: int = 30,
    interval: int = 10,
) -> bool:
    """Poll SSH until the host accepts a key-based connection or attempts run out."""
    for attempt in range(1, max_attempts + 1):
        rc, _stdout, _stderr = ssh_run(host, user, key_file, "exit 0", timeout=15, connect_timeout=5)
        if rc == 0:
            print(f"  SSH ready after attempt {attempt}", file=sys.stderr)
            return True
        print(f"  Waiting for SSH... (attempt {attempt}/{max_attempts})", file=sys.stderr)
        time.sleep(interval)
    return False


def wait_for_ssh_drop(
    host: str,
    user: str,
    key_file: str,
    *,
    max_attempts: int = 12,
    interval: int = 5,
) -> bool:
    """Poll SSH until a previously-reachable sshd stops accepting connections.

    Used after `instances.reset` to confirm the pre-reboot sshd dropped before
    we sample post-reboot uptime — otherwise we may probe the lingering pre-
    reboot sshd and falsely affirm the reboot.
    """
    for attempt in range(1, max_attempts + 1):
        rc, _stdout, _stderr = ssh_run(host, user, key_file, "exit 0", timeout=10, connect_timeout=3)
        if rc != 0:
            print(f"  SSH dropped after attempt {attempt}", file=sys.stderr)
            return True
        print(f"  Waiting for SSH to drop... (attempt {attempt}/{max_attempts})", file=sys.stderr)
        time.sleep(interval)
    return False


def wait_for_cloud_init(
    host: str,
    user: str,
    key_file: str,
    *,
    max_attempts: int = 30,
    interval: int = 10,
) -> bool:
    """Best-effort `cloud-init status --wait` over SSH.

    Treats rc 0 and 2 as terminal success (cloud-init's documented "done"
    codes). rc 255 = transport-level failure (sshd not yet bound); retry.
    rc 1 = semantic failure (cloud-init reports error); terminate.
    """
    deadline = max_attempts
    for attempt in range(1, deadline + 1):
        # No `|| true` — we need the real exit code. cloud-init rc 0/2 are
        # the documented "done" codes; rc 1 is a semantic failure
        # (terminate immediately, don't keep polling on a known-bad guest);
        # rc 255 is transport-level (sshd not yet bound, retry).
        rc, _stdout, _stderr = ssh_run(
            host,
            user,
            key_file,
            "cloud-init status --wait",
            timeout=60,
            connect_timeout=5,
        )
        if rc in (0, 2):
            return True
        if rc == 1:
            return False
        print(f"  Waiting for cloud-init... (attempt {attempt}/{deadline})", file=sys.stderr)
        time.sleep(interval)
    return False

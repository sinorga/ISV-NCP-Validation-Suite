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

"""Validate Compute Engine VM virtual-device hardening evidence.

Compute Engine exposes no customer-facing USB redirection or shared-clipboard
controls for tenant VMs; the only attached devices are persistent disks, NICs,
and local SSD when configured. Posture matches the EC2 oracle; provider-side
preamble names Compute Engine. Guest-side probes are identical to the AWS
oracle — Linux-level commands have no cloud-vendor divergence.

Usage:
    python3 virtual_device_hardening.py --instance-id <name> --region <zone> \\
        --public-ip <ip> --key-file <pem>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import is_sentinel
from common.ssh_utils import ssh_run

CLIPBOARD_PATTERNS = (
    "spice-vdagent",
    "vdagent",
    "xrdp-chansrv",
    "vncconfig",
    "vmtoolsd",
)
UNNECESSARY_DEVICE_PATTERNS = (
    "floppy",
    "cd-rom",
    "cdrom",
    "qxl",
    "spice",
    "open-vm-tools",
    "vgauth",
    "vmware",
    "virtualbox",
    "vbox",
    "tablet",
    "audio",
)
USB_DEVICE_PATTERNS = ("usb controller", "usb host")
REQUIRED_TESTS = (
    "usb_devices_disabled",
    "clipboard_disabled",
    "unnecessary_virtual_devices_absent",
)

PROBE_SENTINEL = "---ISVCTL-PROBE---"
PROBES: tuple[tuple[str, str], ...] = (
    (
        "usb_count",
        "if [ -d /sys/bus/usb/devices ]; then "
        "find /sys/bus/usb/devices -mindepth 1 -maxdepth 1 -type l -print 2>/dev/null | wc -l; "
        "else echo 0; fi",
    ),
    ("pci_devices", "command -v lspci >/dev/null 2>&1 && lspci || true"),
    ("processes", "ps -eo comm= 2>/dev/null || true"),
    (
        "services",
        "command -v systemctl >/dev/null 2>&1 && "
        "systemctl list-units --type=service --state=running --no-pager --output=json 2>/dev/null || true",
    ),
    (
        "device_paths",
        "find /dev -maxdepth 1 \\( -name fd0 -o -name sr0 -o -name cdrom -o -name dvd \\) -print 2>/dev/null || true",
    ),
)
SIGNAL_BINDINGS = (
    ("usb_signals", "usb_devices_disabled", "USB device/controller signals detected"),
    ("clipboard_signals", "clipboard_disabled", "Clipboard-sharing agent signals detected"),
    ("unnecessary_device_signals", "unnecessary_virtual_devices_absent", "Unnecessary virtual device signals detected"),
)


def _compact(text: str, max_length: int = 240) -> str:
    """Collapse whitespace and cap length for one-line diagnostics."""
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 3]}..."


def _matching(text: str, patterns: tuple[str, ...]) -> list[str]:
    """Return lines from ``text`` that contain any case-insensitive ``pattern``."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(p in lowered for p in patterns):
            out.append(stripped)
    return out


def _running_services(systemctl_output: str) -> list[str]:
    """Parse running .service unit names from systemctl JSON output.

    Raises ``ValueError`` on malformed JSON or non-list output. The main
    flow catches the exception and emits ``success=false`` so corrupted
    guest probe data cannot silently pass as "no risky service found".
    """
    text = systemctl_output.strip()
    if not text:
        return []
    try:
        units = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = "systemctl did not return valid JSON"
        raise ValueError(msg) from exc
    if not isinstance(units, list):
        msg = "systemctl JSON output must be a list"
        raise ValueError(msg)
    names: list[str] = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        norm = {str(k).lower(): v for k, v in unit.items()}
        name = str(norm.get("unit") or norm.get("name") or "")
        if not name.endswith(".service"):
            continue
        active = str(norm.get("active") or "").lower()
        sub = str(norm.get("sub") or "").lower()
        if active == "active" and sub == "running":
            names.append(name)
    return names


def _combined_script() -> str:
    """One shell script that runs every probe between sentinel markers."""
    parts: list[str] = []
    for name, cmd in PROBES:
        parts.append(f"echo '{PROBE_SENTINEL} {name}'")
        parts.append(f"({cmd})")
    return "\n".join(parts)


def _split_outputs(combined: str) -> dict[str, str]:
    """Slice the combined SSH output back into per-probe sections."""
    outs: dict[str, list[str]] = {n: [] for n, _ in PROBES}
    current: str | None = None
    for line in combined.splitlines():
        if line.startswith(PROBE_SENTINEL):
            current = line[len(PROBE_SENTINEL) :].strip()
            continue
        if current in outs:
            outs[current].append(line)
    return {n: "\n".join(lines) for n, lines in outs.items()}


def _base_tests() -> dict[str, dict[str, Any]]:
    return {
        "usb_devices_disabled": {
            "passed": True,
            "probes": ["compute_engine_no_customer_usb_redirection_api"],
            "message": "Compute Engine exposes no tenant-facing USB redirection or attach surface",
        },
        "clipboard_disabled": {
            "passed": True,
            "probes": ["compute_engine_no_shared_clipboard_api"],
            "message": "Compute Engine exposes no tenant-facing shared clipboard surface",
        },
        "unnecessary_virtual_devices_absent": {
            "passed": True,
            "probes": ["compute_engine_no_desktop_virtualization_peripheral_api"],
            "message": "Compute Engine exposes no customer-controlled desktop peripheral redirection surface",
        },
    }


def _collect_guest_probe(host: str, user: str, key_file: str, timeout: int) -> dict[str, Any]:
    if is_sentinel(host) or is_sentinel(key_file):
        return {"status": "skipped", "reason": "missing SSH details"}
    rc, stdout, stderr = ssh_run(
        host, user, key_file, _combined_script(), timeout=timeout, connect_timeout=min(timeout, 10)
    )
    if rc != 0:
        return {"status": "unavailable", "error": _compact(stderr or stdout or f"ssh rc={rc}")}
    outputs = _split_outputs(stdout)
    usb_count_raw = outputs.get("usb_count", "0").strip()
    if not usb_count_raw.isdigit():
        msg = f"USB count probe returned non-integer output: {usb_count_raw!r}"
        raise ValueError(msg)
    usb_count = int(usb_count_raw)
    usb_signals = [f"usb device entries present: {usb_count}"] if usb_count else []
    usb_signals.extend(_matching(outputs.get("pci_devices", ""), USB_DEVICE_PATTERNS))
    processes = outputs.get("processes", "")
    services_text = "\n".join(_running_services(outputs.get("services", "")))
    clipboard_signals = _matching(f"{processes}\n{services_text}", CLIPBOARD_PATTERNS)
    unnecessary = _matching(
        f"{outputs.get('pci_devices', '')}\n{processes}\n{services_text}",
        UNNECESSARY_DEVICE_PATTERNS,
    )
    unnecessary.extend(line.strip() for line in outputs.get("device_paths", "").splitlines() if line.strip())
    return {
        "status": "completed",
        "usb_device_count": usb_count,
        "usb_signals": usb_signals,
        "clipboard_signals": clipboard_signals,
        "unnecessary_device_signals": unnecessary,
    }


def _apply_guest(tests: dict[str, dict[str, Any]], probe: dict[str, Any]) -> None:
    if probe.get("status") != "completed":
        return
    for signal_key, test_name, msg in SIGNAL_BINDINGS:
        signals = list(probe.get(signal_key, []))
        if signals:
            tests[test_name].update({"passed": False, "error": f"{msg}: {_compact('; '.join(signals))}"})


def main() -> int:
    """Evaluate provider evidence + guest probes and emit canonical JSON."""
    parser = argparse.ArgumentParser(description="Validate Compute Engine VM virtual-device hardening")
    parser.add_argument("--instance-id", required=True, help="Instance name (echoed)")
    parser.add_argument("--region", default="", help="Effective zone (informational)")
    parser.add_argument("--public-ip", default="", help="Optional SSH host for guest probes")
    parser.add_argument("--key-file", default="", help="Optional SSH key path for guest probes")
    parser.add_argument("--ssh-user", default="ubuntu", help="SSH username")
    parser.add_argument("--ssh-timeout", type=int, default=60, help="Total seconds for the combined guest probe")
    args = parser.parse_args()

    tests = _base_tests()
    guest_error: str | None = None
    probe: dict[str, Any] = {"status": "unavailable"}
    try:
        probe = _collect_guest_probe(args.public_ip, args.ssh_user, args.key_file, args.ssh_timeout)
    except ValueError as exc:
        # Malformed probe output (systemctl JSON corruption, non-integer
        # usb count, etc.) — surface honestly rather than swallowing into
        # an artificially-clean signal list.
        guest_error = _compact(str(exc))
    if guest_error is None and probe.get("status") == "unavailable":
        guest_error = _compact(str(probe.get("error") or "guest probe unavailable"))
    _apply_guest(tests, probe)

    success = guest_error is None and all(tests[n]["passed"] is True for n in REQUIRED_TESTS)
    result: dict[str, Any] = {
        "success": success,
        "platform": "vm",
        "test_name": "virtual_device_hardening",
        "instance_id": args.instance_id,
        "provider_evidence": {
            "cloud": "compute-engine",
            "tenant_facing_usb_attach_api": False,
            "tenant_facing_shared_clipboard_api": False,
            "attached_device_categories": ["persistent_disks", "network_interfaces", "local_ssd_when_configured"],
        },
        "tests": tests,
    }
    if guest_error is not None:
        result["error"] = guest_error
    print(json.dumps(result, indent=2, default=str))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

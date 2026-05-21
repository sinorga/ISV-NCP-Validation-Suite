# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# pyright: reportMissingImports=false, reportAttributeAccessIssue=false

"""Shared Compute Engine helpers for GCP VM stubs.

Provides the small, NCP-specific surface every VM step needs:
  - project / ADC resolution
  - canonical state translation (RUNNING/TERMINATED/... -> AWS-like names)
  - per-run unique suffixing helper
  - zone selection with single-zone-pin detection
  - 4-shape zone-capacity error classifier
  - ExtendedOperation wait helper (uses .result(); joins op.error.code)
  - public-IP poller
  - local SSH keypair generator (verified-reuse, returns ``key_created`` bool)
  - firewall rule create/verified-reuse helper
  - source image resolver (operator scope first, vendor fallback second)
  - partial-instance cleanup helper for the multi-zone walk
"""

from __future__ import annotations

import http.client
import os
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

# Public GCP Deep Learning VM Image — the upstream-shipping default. Operator
# environments override via GCP_VM_IMAGE / GCP_VM_IMAGE_PROJECT (forwarded as
# ``--ami-id`` and ``--image-project``).
DEFAULT_IMAGE_FAMILY = "common-cu129-ubuntu-2204-nvidia-580"
DEFAULT_IMAGE_PROJECT = "deeplearning-platform-release"

# L4 GPU zones in priority walk-order (us-central1 first; cross-region tail).
PREFERRED_ZONES: tuple[str, ...] = (
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-east4-a",
    "us-east4-b",
    "us-east4-c",
    "us-east1-c",
    "us-east1-d",
    "us-west1-a",
    "us-west1-b",
    "us-west4-a",
    "us-west4-b",
    "europe-west4-a",
    "europe-west4-b",
    "europe-west1-b",
    "europe-west1-c",
    "asia-southeast1-a",
    "asia-southeast1-b",
    "asia-southeast1-c",
    "asia-northeast1-a",
    "asia-northeast1-c",
    "asia-east1-a",
    "asia-east1-b",
    "asia-east1-c",
)

# Sentinel tokens the provider config emits when a Jinja `default(...)` filler
# fires. The stub MUST treat these as "operator did not supply" and fall back
# to the documented vendor default — never pass the literal sentinel to the
# API.
_SENTINELS = frozenset({"", "none", "null", "false"})

# Compute Engine resource name pattern (RFC 1035): 1-63 chars,
# lowercase letters / digits / hyphens, must start with a letter.
_GCE_NAME_RE = re.compile(r"^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$")


def is_sentinel(value: str | None) -> bool:
    """Return True when ``value`` is one of the Jinja `default(...)` sentinels."""
    if value is None:
        return True
    return str(value).strip().lower() in _SENTINELS


def unique_suffix(base: str, *, length: int = 8) -> str:
    """Append the per-run id (or a random fallback) to ``base``."""
    sid = os.environ.get("RUN_ID") or os.environ.get("LS_RUN_ID") or ""
    suffix = sid[:length] if sid else uuid.uuid4().hex[:length]
    return f"{base}-{suffix}"


def resolve_project(arg_project: str | None = None) -> str:
    """Resolve the active GCP project id.

    The test harness does NOT forward GOOGLE_CLOUD_PROJECT / GCLOUD_PROJECT
    to spawned stubs, so we fall back to ADC discovery rather than failing
    at startup.
    """
    if arg_project and not is_sentinel(arg_project):
        return str(arg_project)
    for key in ("GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GCP_PROJECT"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        import google.auth  # noqa: PLC0415 — lazy so non-GCP linters don't need the dep
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "google-cloud-compute / google-auth not installed; install the GCP extra"
        ) from exc
    _credentials, project = google.auth.default()
    if not project:
        raise RuntimeError(
            "could not resolve a GCP project — pass --project, set GOOGLE_CLOUD_PROJECT, "
            "or run 'gcloud auth application-default set-quota-project <id>'"
        )
    return str(project)


def canonical_state(raw: str | None) -> str:
    """Translate a Compute Engine status name to the canonical lifecycle set.

    Emitted values match what the suite's InstanceStateCheck / lifecycle
    validators consume: ``pending`` / ``running`` / ``stopping`` / ``stopped``
    / ``terminated`` / ``unknown``. DEPROVISIONING maps to ``stopping`` so
    the delete-then-readback window doesn't dead-code branches that look
    for transient states.
    """
    mapping = {
        "RUNNING": "running",
        "PROVISIONING": "pending",
        "STAGING": "pending",
        "REPAIRING": "pending",
        "STOPPING": "stopping",
        "SUSPENDING": "stopping",
        "DEPROVISIONING": "stopping",
        "STOPPED": "stopped",
        "TERMINATED": "stopped",
        "SUSPENDED": "stopped",
    }
    if not raw:
        return "unknown"
    return mapping.get(str(raw).upper(), "unknown")


def select_zones(zone_or_region: str, preferred: tuple[str, ...] = PREFERRED_ZONES) -> list[str]:
    """Resolve the operator-supplied zone-or-region into a walk-ordered list.

    A full zone (`<region>-<azid>`, three dash tokens) is honored verbatim
    — no walk, no silent fallback to PREFERRED_ZONES[0]. A region prefix
    (one or two dash tokens) returns in-region preferred zones FIRST,
    then the rest of the preferred list as a documented cross-region
    stockout fallback for capacity walks. An operator-supplied region
    prefix that has zero in-region preferred-zone matches is treated as
    operator error rather than silently substituted with a different
    region's zones — pin a full zone or use a region in
    PREFERRED_ZONES.
    """
    value = (zone_or_region or "").strip()
    if not value:
        return list(preferred)
    parts = value.split("-")
    if len(parts) >= 3:
        return [value]
    in_region = [z for z in preferred if z.startswith(f"{value}-")]
    if not in_region:
        regions = sorted({z.rsplit("-", 1)[0] for z in preferred})
        raise ValueError(
            f"region {value!r} has no preferred zones — pin a full zone "
            f"(e.g. {value}-a) or use one of {regions}."
        )
    cross_region = [z for z in preferred if not z.startswith(f"{value}-")]
    return in_region + cross_region


def is_zone_unavailable(err: Exception, op: Any = None) -> bool:
    """Classify an error as a zone-capacity / unavailability shape (all 4)."""
    try:
        from google.api_core import exceptions as gax_exceptions  # noqa: PLC0415
    except ImportError:
        gax_exceptions = None  # type: ignore[assignment]
    if gax_exceptions is not None and isinstance(err, gax_exceptions.ResourceExhausted):
        return True
    msg = str(err) if err else ""
    if "does not exist in zone" in msg and "machineType" in msg:
        return True
    # Canonical capacity tokens can arrive on RuntimeError (polling-fallback
    # shape 4) OR on the underlying gax_exceptions class directly when the
    # sync insert returns HTTP 503 ServiceUnavailable carrying
    # ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS. Match by message regardless
    # of exception type — the tokens themselves only appear in capacity errors.
    if (
        "ZONE_RESOURCE" in msg
        or "STOCKOUT" in msg.upper()
        or "does not have enough resources" in msg
    ):
        return True
    if op is not None and getattr(op, "error", None):
        for e in op.error.errors:
            code = (e.code or "") if hasattr(e, "code") else ""
            if "ZONE_RESOURCE" in code or "STOCKOUT" in code:
                return True
    return False


def wait_for_zonal_op(client: Any, project: str, zone: str, op: Any, *, timeout: int = 600) -> None:
    """Block on a ``compute_v1`` ExtendedOperation until it completes.

    Calls ``op.result(timeout=...)`` — the documented blocking method on
    ExtendedOperation. ``.wait()`` is the older google.api_core.Operation
    API and raises AttributeError on the newer ExtendedOperation, then gets
    silently swallowed by retry-on-Exception wrappers.

    When the operation finishes with errors, raises a RuntimeError that
    joins both ``op.error.code`` and ``op.error.message`` — the canonical
    capacity token (``ZONE_RESOURCE_POOL_EXHAUSTED``) lives in ``.code``;
    ``.message`` carries only human wording (`state:STOCKOUT ...`).
    """
    del client, project, zone  # ExtendedOperation has the op handle baked in.
    op.result(timeout=timeout)
    err = getattr(op, "error", None)
    if err and getattr(err, "errors", None):
        parts: list[str] = []
        for e in err.errors:
            code = getattr(e, "code", "") or ""
            message = getattr(e, "message", "") or ""
            parts.append(f"{code}: {message}" if code else message)
        raise RuntimeError(f"Zonal op failed: {'; '.join(parts)}")


def wait_for_global_op(client: Any, project: str, op: Any, *, timeout: int = 300) -> None:
    """Same shape as ``wait_for_zonal_op`` for project-global ops (firewalls)."""
    del client, project
    op.result(timeout=timeout)
    err = getattr(op, "error", None)
    if err and getattr(err, "errors", None):
        parts: list[str] = []
        for e in err.errors:
            code = getattr(e, "code", "") or ""
            message = getattr(e, "message", "") or ""
            parts.append(f"{code}: {message}" if code else message)
        raise RuntimeError(f"Global op failed: {'; '.join(parts)}")


def wait_for_public_ip(
    client: Any,
    project: str,
    zone: str,
    instance_name: str,
    *,
    timeout: int = 120,
    interval: int = 5,
) -> str | None:
    """Poll ``instances.get`` until an external NAT IP shows up on NIC 0.

    Only retryable transient errors are swallowed: ServiceUnavailable (503),
    DeadlineExceeded, transport-level disconnects, and the
    occasionally-observed RemoteDisconnected. Terminal failures
    (NotFound, PermissionDenied, Unauthenticated, InvalidArgument)
    re-raise immediately so the launch step surfaces the real cause —
    matches the AWS oracle's narrow-transient catch.
    """
    try:
        from google.api_core import exceptions as gax_exceptions  # noqa: PLC0415
    except ImportError:
        gax_exceptions = None  # type: ignore[assignment]
    transient_gax: tuple[type[Exception], ...] = ()
    if gax_exceptions is not None:
        transient_gax = (
            gax_exceptions.ServiceUnavailable,
            gax_exceptions.DeadlineExceeded,
            gax_exceptions.InternalServerError,
        )
    transient_transport = (
        ConnectionError,
        TimeoutError,
        http.client.RemoteDisconnected,
    )
    deadline = time.monotonic() + timeout
    while True:
        try:
            inst = client.get(project=project, zone=zone, instance=instance_name)
            for nic in getattr(inst, "network_interfaces", []) or []:
                for ac in getattr(nic, "access_configs", []) or []:
                    if getattr(ac, "nat_i_p", None):
                        return str(ac.nat_i_p)
        except transient_gax as exc:
            print(f"Warning: instances.get transient API error: {exc}", file=sys.stderr)
        except transient_transport as exc:
            print(f"Warning: instances.get transport error: {exc}", file=sys.stderr)
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval)


def _safe_name(value: str) -> str:
    """Reject names that would escape the /tmp PEM directory or break the GCE name regex."""
    value = (value or "").strip()
    if not value or not _GCE_NAME_RE.fullmatch(value):
        raise ValueError(
            f"invalid GCE resource name {value!r}: must match {_GCE_NAME_RE.pattern} "
            "(1-63 chars, RFC 1035)."
        )
    return value


def generate_ssh_keypair(
    key_basename: str,
    *,
    key_dir: str | Path = "/tmp",
) -> tuple[str, bool]:
    """Generate (or verified-reuse) a local SSH keypair for instance metadata.

    Returns ``(private_key_path, key_created)``. ``key_created`` is True iff
    this call generated a fresh pair — teardown reads it to gate local
    PEM/.pub deletion (verified-reuse contract). Read the matching public
    key with :func:`read_ssh_public_key`.

    Compute Engine has no managed key-pair store: the public key is attached
    via instance metadata under ``ssh-keys`` at create time. There is no
    server-side resource to verified-reuse — the local PEM + .pub pair IS
    the resource.
    """
    name = _safe_name(key_basename)
    key_path = Path(key_dir) / f"{name}.pem"
    pub_path = key_path.with_suffix(".pub")

    if key_path.exists() and pub_path.exists() and key_path.stat().st_size > 0:
        try:
            if pub_path.read_text().strip():
                return str(key_path), False
        except OSError:
            pass

    # Stale files — wipe and regenerate.
    for path in (key_path, pub_path):
        if path.exists():
            try:
                path.chmod(0o600)
            except OSError:
                pass
            path.unlink()

    subprocess.run(
        ["ssh-keygen", "-t", "rsa", "-b", "2048", "-N", "", "-q", "-f", str(key_path), "-C", name],
        check=True,
        capture_output=True,
        timeout=30,
    )
    # ssh-keygen appends ".pub" to its -f argument, producing <name>.pem.pub.
    # The rest of the contract uses .with_suffix(".pub") which gives <name>.pub
    # (the .pem replaced, not appended), so move the file to that location.
    generated_pub = Path(f"{key_path}.pub")
    if generated_pub.exists():
        generated_pub.rename(pub_path)
    key_path.chmod(0o400)
    print(f"Generated SSH keypair at {key_path}", file=sys.stderr)
    return str(key_path), True


def read_ssh_public_key(key_file: str) -> str:
    """Read the ``.pub`` sibling of a ``.pem`` produced by :func:`generate_ssh_keypair`."""
    return Path(key_file).with_suffix(".pub").read_text().strip()


def _has_isv_description(description: str | None) -> bool:
    """Return True when a firewall description carries the suite's ownership marker."""
    return bool(description) and "created-by=isvtest" in description


def create_ssh_firewall_rule(
    firewalls_client: Any,
    project: str,
    network: str,
    *,
    name: str,
    target_tag: str,
    source_range: str = "0.0.0.0/0",
    tracker: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    """Create or verified-reuse a project-global firewall allowing TCP/22.

    Returns ``(firewall_name, firewall_created)``. Firewall labels are NOT
    supported on the GCE Firewall proto — using ``description`` as the
    ownership marker matches the documented pattern.

    ``tracker`` is an optional caller-owned dict the helper populates
    BEFORE waiting on the insert operation. When the wait raises, the
    caller's ``finally:`` can read ``tracker["name"]`` and
    ``tracker["created"]`` to clean up the accepted-but-not-confirmed
    firewall. Without this side channel, a wait failure after a
    successful insert ack would strand the rule because the helper
    return value never reaches the caller.
    """
    safe_name = _safe_name(name)
    try:
        existing = firewalls_client.get(project=project, firewall=safe_name)
    except Exception:  # NotFound on first run; fall through to create
        existing = None

    if existing is not None:
        # Exact-match the network short name parsed off the self-link, so
        # `default` cannot adopt a rule attached to `prod-default`.
        existing_network_short = (existing.network or "").rsplit("/", 1)[-1]
        net_match = existing_network_short == network
        tag_match = target_tag in (existing.target_tags or [])
        source_match = source_range in (existing.source_ranges or [])
        port_match = any(
            (a.I_p_protocol or "").lower() == "tcp" and "22" in (a.ports or [])
            for a in (existing.allowed or [])
        )
        if not _has_isv_description(existing.description):
            raise RuntimeError(
                f"firewall {safe_name!r} exists but description lacks 'created-by=isvtest' — "
                "refusing to adopt a resource this suite did not create."
            )
        if not (net_match and tag_match and source_match and port_match):
            raise RuntimeError(
                f"firewall {safe_name!r} exists but shape differs (network/target-tag/source-range/port)"
            )
        print(f"Reusing verified firewall: {safe_name}", file=sys.stderr)
        return safe_name, False

    try:
        from google.cloud import compute_v1  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("google-cloud-compute is not installed") from exc

    allowed = compute_v1.Allowed(I_p_protocol="tcp", ports=["22"])
    rule = compute_v1.Firewall(
        name=safe_name,
        network=f"projects/{project}/global/networks/{network}",
        direction="INGRESS",
        priority=1000,
        target_tags=[target_tag],
        source_ranges=[source_range],
        allowed=[allowed],
        description="created-by=isvtest; ssh-ingress",
    )
    op = firewalls_client.insert(project=project, firewall_resource=rule)
    # Ownership is real the moment insert acks — populate the tracker so
    # the caller's finally: clause can clean up if the wait raises.
    if tracker is not None:
        tracker["name"] = safe_name
        tracker["created"] = True
    wait_for_global_op(firewalls_client, project, op)
    print(f"Created firewall: {safe_name}", file=sys.stderr)
    return safe_name, True


def resolve_image(images_client: Any, project: str, image_arg: str | None) -> tuple[str, str]:
    """Resolve an operator-supplied image short-name / family / self-link.

    Search order — operator scope FIRST, vendor-default scope LAST:
      1. ``args.project`` exact image
      2. ``args.project`` image family
      3. ``DEFAULT_IMAGE_PROJECT`` (DLVM) exact image
      4. ``DEFAULT_IMAGE_PROJECT`` image family

    Returns ``(self_link, name)``. Without the operator-first attempt the
    helper would only find images the operator built in their own project
    when those happen to share the DLVM project name (i.e., never).
    """
    fallback_image = DEFAULT_IMAGE_FAMILY
    if image_arg and not is_sentinel(image_arg):
        candidate = str(image_arg)
    else:
        candidate = fallback_image

    if candidate.startswith("https://") or candidate.startswith("projects/"):
        return candidate, candidate.rsplit("/", 1)[-1]

    attempts: list[tuple[str, str, str]] = [
        (project, candidate, "get"),
        (project, candidate, "get_from_family"),
    ]
    if project != DEFAULT_IMAGE_PROJECT:
        attempts.extend(
            [
                (DEFAULT_IMAGE_PROJECT, candidate, "get"),
                (DEFAULT_IMAGE_PROJECT, candidate, "get_from_family"),
            ]
        )

    try:
        from google.api_core import exceptions as gax_exceptions  # noqa: PLC0415
    except ImportError:
        gax_exceptions = None  # type: ignore[assignment]
    not_found_types: tuple[type[Exception], ...] = ()
    if gax_exceptions is not None:
        not_found_types = (gax_exceptions.NotFound,)

    last_not_found: Exception | None = None
    for proj, name, method in attempts:
        try:
            if method == "get":
                image = images_client.get(project=proj, image=name)
            else:
                image = images_client.get_from_family(project=proj, family=name)
            return str(image.self_link), str(image.name)
        except not_found_types as exc:
            last_not_found = exc
            continue
        except Exception:
            # Permission, auth, transient, malformed-request — must surface
            # so the operator's explicit image scope is never silently
            # replaced by the vendor default.
            raise

    raise RuntimeError(
        f"could not resolve image {image_arg!r} in {project!r} or {DEFAULT_IMAGE_PROJECT!r}: "
        f"{last_not_found}"
    )


def delete_failed_zonal_instance(client: Any, project: str, zone: str, name: str) -> bool:
    """Best-effort delete of a partial-state instance after a STOCKOUT shape.

    Used in the launch-time zone-walk: shapes 2 and 4 from the GCP
    zone_capacity_error_shapes inventory may leave a phantom instance
    record in the failed zone. Reclaiming it prevents leaks across the
    walk. Returns True iff the delete operation acked successfully (or
    NotFound, which means the record was already absent).
    """
    try:
        op = client.delete(project=project, zone=zone, instance=name)
    except Exception as exc:
        if "404" in str(exc) or "notFound" in str(exc).lower():
            return True
        print(f"Warning: phantom delete failed in {zone}: {exc}", file=sys.stderr)
        return False

    try:
        wait_for_zonal_op(client, project, zone, op, timeout=180)
        return True
    except Exception as exc:
        if "404" in str(exc) or "notFound" in str(exc).lower():
            return True
        print(f"Warning: phantom delete wait failed in {zone}: {exc}", file=sys.stderr)
        return False


def canonical_tags(name: str, *, created_by: str = "isvtest") -> dict[str, str]:
    """Return canonical-cased suite tags (`Name`, `CreatedBy`).

    GCE label keys must be lowercase, so the API receives lower-cased copies
    via ``api_labels()`` — the canonical-cased dict here is what the stub
    emits in JSON output to satisfy ``InstanceTagCheck.required_keys``.
    """
    return {"Name": name, "CreatedBy": created_by}


def api_labels(tags: dict[str, str]) -> dict[str, str]:
    """Convert canonical-cased suite tags to GCE-API-valid lowercase labels."""
    return {k.lower(): v.lower() if isinstance(v, str) else v for k, v in tags.items()}


def fetch_adc_caller_email() -> str | None:
    """Resolve the current ADC caller's email via tokeninfo.

    Local user ADC (``gcloud auth application-default login``) often has
    ``service_account_email=None`` and ``account=""``, so the only stable
    way to learn who we are is to refresh the token and call tokeninfo.
    Returns None if anything along the path fails — the caller decides
    whether that's a hard error.
    """
    try:
        import google.auth  # noqa: PLC0415
        import google.auth.transport.requests  # noqa: PLC0415
    except ImportError:
        return None
    try:
        credentials, _ = google.auth.default()
        credentials.refresh(google.auth.transport.requests.Request())
        token = credentials.token
        if not token:
            return None
        url = f"https://oauth2.googleapis.com/tokeninfo?access_token={token}"
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 — fixed GCP endpoint
            import json  # noqa: PLC0415
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("email")
    except Exception:
        return None

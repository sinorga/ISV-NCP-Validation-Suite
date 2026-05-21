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

"""Validate Compute Engine serial-console access is RBAC-scoped.

GCP has no direct ``simulate_principal_policy`` equivalent for Compute
Engine serial console access — ``instances.testIamPermissions`` and
``instances.getSerialPortOutput`` evaluate the active caller. Real RBAC
evidence therefore requires real service-account principals.

Two paths are supported, in priority order:

1. **Operator-provided pre-provisioned probe principals.** When
   ``GCP_DENIED_PRINCIPAL_SA``, ``GCP_ALLOWED_PRINCIPAL_SA``, and
   ``GCP_OTHER_VM_NAME`` are set, the stub assumes the operator has set
   up the bindings out-of-band and only mints tokens / probes the
   serial-port surface.

2. **Self-provisioned temporary probe principals.** When the env vars
   above are not present, the stub creates two temporary service
   accounts, grants the calling identity ``roles/iam.serviceAccountTokenCreator``
   on each SA (so we can mint impersonation tokens), grants the allowed
   SA ``roles/compute.viewer`` on the target VM only (resource-scoped
   evidence), creates a small "other" VM for the scoping probe, runs the
   three RBAC probes, and cleans up the temporary SAs / bindings / VM
   before returning. Cleanup uses etag-aware
   ``getIamPolicy``/``setIamPolicy`` with retry on HTTP 409.

The self-provision path requires the caller to hold
``iam.serviceAccounts.create`` / ``actAs`` and the
``compute.instances.create`` / ``setIamPolicy`` permissions in the
target project. If any of these are missing, the stub fails the step
with failure-shaped diagnostics (``success=false``, ``failure_reason``,
``error``) rather than emitting a misleading ``skipped=true`` marker —
``skipped=true`` is reserved for true policy skips that pair with
``success=true`` / rc 0.

Usage:
    python3 console_rbac.py --instance-id <name> --region <zone>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.compute import (
    PREFERRED_ZONES,
    api_labels,
    canonical_tags,
    create_ssh_firewall_rule,
    fetch_adc_caller_email,
    generate_ssh_keypair,
    read_ssh_public_key,
    resolve_project,
    select_zones,
    unique_suffix,
)

REQUIRED_TESTS = (
    "denied_principal_cannot_access_console",
    "allowed_principal_can_access_console",
    "allowed_principal_is_resource_scoped",
)
SERIAL_PERMISSION = "compute.instances.getSerialPortOutput"
SCOPED_ROLE = "roles/compute.viewer"
TOKEN_CREATOR_ROLE = "roles/iam.serviceAccountTokenCreator"
GCP_BASE = "https://compute.googleapis.com/compute/v1"
IAM_BASE = "https://iam.googleapis.com/v1"
IAM_CREDS_BASE = "https://iamcredentials.googleapis.com/v1"


def _passing(passed: bool, **details: Any) -> dict[str, Any]:
    return {"passed": passed, **details}


def _failed(reason: str) -> dict[str, Any]:
    return {"passed": False, "failure_reason": reason}


def _caller_token() -> str | None:
    """Refresh the calling identity's ADC token. Returns None on failure."""
    try:
        import google.auth
        import google.auth.transport.requests

        creds, _ = google.auth.default()
        creds.refresh(google.auth.transport.requests.Request())
        return creds.token
    except Exception:
        return None


def _rest(
    method: str,
    url: str,
    token: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: int = 30,
) -> tuple[int, dict[str, Any] | str]:
    """One-shot REST call against a Google API."""
    payload: bytes | None = None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(text)
            except ValueError:
                return resp.status, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(text)
        except ValueError:
            return exc.code, text
    except urllib.error.URLError as exc:
        return 0, str(exc)


def _mint_token_for_sa(sa_email: str, caller_token: str) -> tuple[str | None, str]:
    """Mint a short-lived access token for ``sa_email`` via IAMCredentials REST."""
    url = f"{IAM_CREDS_BASE}/projects/-/serviceAccounts/{sa_email}:generateAccessToken"
    status, body = _rest(
        "POST",
        url,
        caller_token,
        body={"scope": ["https://www.googleapis.com/auth/cloud-platform"], "lifetime": "300s"},
    )
    if status == 200 and isinstance(body, dict) and "accessToken" in body:
        return body["accessToken"], ""
    return None, f"generateAccessToken status={status} body={body}"


# TokenCreator IAM bindings on a freshly created SA need up to ~180 seconds
# to become usable (eventual consistency). Self-provisioned RBAC probe must
# retry token mints within that budget so a normal propagation window does
# not pre-empt the actual RBAC evaluation.
_TOKEN_MINT_RETRY_BUDGET = 12
_TOKEN_MINT_RETRY_DELAY = 15


def _mint_token_with_retry(
    sa_email: str,
    caller_token: str,
    *,
    attempts: int = _TOKEN_MINT_RETRY_BUDGET,
    delay: int = _TOKEN_MINT_RETRY_DELAY,
) -> tuple[str | None, str]:
    """Retry :func:`_mint_token_for_sa` while the TokenCreator binding propagates.

    Returns on the first 200 OK. Permission errors (403 / 401) are treated
    as propagation-pending and retried; other failures still surface via
    the final return when the budget is exhausted.
    """
    last_err = ""
    for attempt in range(1, attempts + 1):
        token, err = _mint_token_for_sa(sa_email, caller_token)
        if token is not None:
            if attempt > 1:
                print(
                    f"token mint for {sa_email} succeeded on attempt {attempt}/{attempts}",
                    file=sys.stderr,
                )
            return token, ""
        last_err = err
        if attempt == attempts:
            break
        time.sleep(delay)
    return None, last_err


def _probe_serial(project: str, zone: str, instance: str, token: str) -> tuple[int, str]:
    """Call ``instances.getSerialPortOutput`` and return ``(status, body_snippet)``."""
    url = f"{GCP_BASE}/projects/{project}/zones/{zone}/instances/{instance}/serialPort?port=1"
    status, body = _rest("GET", url, token)
    snippet = json.dumps(body)[:200] if isinstance(body, dict) else str(body)[:200]
    return status, snippet


def _create_sa(project: str, account_id: str, caller_token: str) -> tuple[str | None, str]:
    """Create a temporary service account, returning ``(email, error)``."""
    url = f"{IAM_BASE}/projects/{project}/serviceAccounts"
    status, body = _rest(
        "POST",
        url,
        caller_token,
        body={
            "accountId": account_id,
            "serviceAccount": {"displayName": f"ISV console RBAC probe ({account_id})"},
        },
    )
    if status == 200 and isinstance(body, dict) and "email" in body:
        return body["email"], ""
    return None, f"create SA status={status} body={body}"


def _delete_sa(project: str, sa_email: str, caller_token: str) -> str:
    """Delete a temporary SA. Returns "" on success, error string otherwise.

    Callers append the error to ``cleanup_errors`` so cleanup failures
    flip the result's ``success`` to False — silent warnings would let a
    leaked SA ship as PASS.
    """
    url = f"{IAM_BASE}/projects/{project}/serviceAccounts/{sa_email}"
    status, body = _rest("DELETE", url, caller_token)
    if status in (200, 204, 404):
        return ""
    msg = f"delete SA {sa_email}: status={status} body={body}"
    print(f"Warning: {msg}", file=sys.stderr)
    return msg


def _is_retryable_iam_status(status: int) -> bool:
    """Return True for IAM-API codes that mean 'try again' rather than 'give up'.

    ``409`` = stale etag (refresh + retry). ``429`` = rate-limit. ``500/503/504``
    = transient backend failure. Anything else is a terminal status that
    must surface to the caller.
    """
    return status in (409, 429, 500, 502, 503, 504)


def _iam_policy_endpoints(resource_url: str) -> tuple[str, str, str]:
    """Return ``(get_method, get_url, set_url)`` for an IAM policy resource.

    Compute Engine instances expose ``GET .../instances/{name}/getIamPolicy``
    and ``POST .../instances/{name}/setIamPolicy``. IAM service accounts
    expose ``POST .../serviceAccounts/{email}:getIamPolicy`` and
    ``POST .../serviceAccounts/{email}:setIamPolicy``. The two endpoints
    diverge on both verb and separator, so we branch on the resource URL
    rather than assume one shape.
    """
    if resource_url.startswith(GCP_BASE):
        return "GET", f"{resource_url}/getIamPolicy", f"{resource_url}/setIamPolicy"
    return "POST", f"{resource_url}:getIamPolicy", f"{resource_url}:setIamPolicy"


def _modify_iam_policy(
    resource_url: str,
    caller_token: str,
    *,
    add_bindings: list[tuple[str, str]],
    max_retries: int = 5,
) -> str:
    """Read-modify-write an IAM policy with bounded retry-with-backoff.

    Retries on stale-etag conflicts (409) and on transient backend errors
    (429 / 5xx). Each retry re-reads the policy so the new etag is
    consistent with the next setIamPolicy.
    """
    get_method, get_url, set_url = _iam_policy_endpoints(resource_url)
    get_body: dict[str, Any] | None = None if get_method == "GET" else {}
    last_diag = "no attempt"
    for attempt in range(max_retries):
        get_status, policy = _rest(get_method, get_url, caller_token, body=get_body)
        if get_status != 200 or not isinstance(policy, dict):
            if _is_retryable_iam_status(get_status) and attempt < max_retries - 1:
                last_diag = f"getIamPolicy status={get_status} body={policy}"
                time.sleep(min(2**attempt, 8))
                continue
            return f"getIamPolicy status={get_status} body={policy}"
        bindings = list(policy.get("bindings") or [])
        for role, member in add_bindings:
            for binding in bindings:
                if binding.get("role") == role:
                    members = list(binding.get("members") or [])
                    if member not in members:
                        members.append(member)
                    binding["members"] = members
                    break
            else:
                bindings.append({"role": role, "members": [member]})
        policy["bindings"] = bindings
        set_status, set_body = _rest(
            "POST",
            set_url,
            caller_token,
            body={"policy": policy},
        )
        if set_status == 200:
            return ""
        if _is_retryable_iam_status(set_status) and attempt < max_retries - 1:
            last_diag = f"setIamPolicy status={set_status} body={set_body}"
            time.sleep(min(2**attempt, 8))
            continue
        return f"setIamPolicy status={set_status} body={set_body}"
    return f"setIamPolicy retries exhausted: {last_diag}"


def _remove_iam_bindings(
    resource_url: str,
    caller_token: str,
    *,
    remove_bindings: list[tuple[str, str]],
    max_retries: int = 5,
) -> str:
    """Inverse of :func:`_modify_iam_policy`; removes (role, member) bindings."""
    get_method, get_url, set_url = _iam_policy_endpoints(resource_url)
    get_body: dict[str, Any] | None = None if get_method == "GET" else {}
    last_diag = "no attempt"
    for attempt in range(max_retries):
        get_status, policy = _rest(get_method, get_url, caller_token, body=get_body)
        if get_status == 404:
            return ""
        if get_status != 200 or not isinstance(policy, dict):
            if _is_retryable_iam_status(get_status) and attempt < max_retries - 1:
                last_diag = f"getIamPolicy status={get_status} body={policy}"
                time.sleep(min(2**attempt, 8))
                continue
            return f"getIamPolicy status={get_status} body={policy}"
        bindings = list(policy.get("bindings") or [])
        for role, member in remove_bindings:
            for binding in bindings:
                if binding.get("role") == role:
                    members = [m for m in (binding.get("members") or []) if m != member]
                    binding["members"] = members
        # Drop any empty-binding entries so the policy doesn't accumulate
        # roles with no members.
        policy["bindings"] = [b for b in bindings if b.get("members")]
        set_status, set_body = _rest(
            "POST",
            set_url,
            caller_token,
            body={"policy": policy},
        )
        if set_status == 200:
            return ""
        if _is_retryable_iam_status(set_status) and attempt < max_retries - 1:
            last_diag = f"setIamPolicy(remove) status={set_status} body={set_body}"
            time.sleep(min(2**attempt, 8))
            continue
        return f"setIamPolicy(remove) status={set_status} body={set_body}"
    return f"setIamPolicy(remove) retries exhausted: {last_diag}"


def _create_other_vm(
    project: str,
    zone: str,
    name: str,
    caller_token: str,
    *,
    network: str,
    public_key: str,
    firewall_target_tag: str,
    tracker: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Create a minimal e2-micro VM for the resource-scoping probe.

    ``tracker`` is a caller-owned dict the helper populates immediately
    after the insert ack so a poll timeout / DONE-with-error still
    surfaces the VM name for cleanup.
    """
    image_status, image_body = _rest(
        "GET",
        f"{GCP_BASE}/projects/debian-cloud/global/images/family/debian-12",
        caller_token,
    )
    if image_status != 200 or not isinstance(image_body, dict):
        return False, f"image lookup status={image_status} body={image_body}"
    source_image = image_body["selfLink"]
    url = f"{GCP_BASE}/projects/{project}/zones/{zone}/instances"
    body = {
        "name": name,
        "machineType": f"zones/{zone}/machineTypes/e2-micro",
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "initializeParams": {
                    "sourceImage": source_image,
                    "diskSizeGb": 10,
                    "diskType": f"zones/{zone}/diskTypes/pd-standard",
                },
            }
        ],
        "networkInterfaces": [
            {
                "network": f"projects/{project}/global/networks/{network}",
                "accessConfigs": [{"name": "External NAT", "type": "ONE_TO_ONE_NAT"}],
            }
        ],
        "labels": api_labels(canonical_tags(name)),
        "tags": {"items": [firewall_target_tag]},
        "metadata": {"items": [{"key": "ssh-keys", "value": f"ubuntu:{public_key}"}]},
    }
    status, payload = _rest("POST", url, caller_token, body=body)
    if status not in (200, 201) or not isinstance(payload, dict):
        return False, f"insert other VM status={status} body={payload}"
    # Insert acked — record ownership now so a poll-side failure still
    # cleans up.
    if tracker is not None:
        tracker["name"] = name
        tracker["zone"] = zone
    op_name = payload.get("name")
    if not op_name:
        return False, f"insert other VM response missing op name: {payload}"
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        st, op_body = _rest(
            "GET",
            f"{GCP_BASE}/projects/{project}/zones/{zone}/operations/{op_name}",
            caller_token,
        )
        if st == 200 and isinstance(op_body, dict) and op_body.get("status") == "DONE":
            if op_body.get("error"):
                return False, f"insert other VM op error: {op_body['error']}"
            return True, ""
        time.sleep(5)
    return False, "insert other VM timed out"


def _delete_other_vm(project: str, zone: str, name: str, caller_token: str) -> str:
    """Delete the scoping VM and wait for the operation to complete.

    Returns "" on success, error string otherwise. Callers append the
    error to ``cleanup_errors`` so a leaked VM forces ``success=False``.
    Waiting on the operation is required — returning before the delete
    op completes can race the subsequent SA delete (the VM may still
    have a service-account binding when the SA is removed).
    """
    url = f"{GCP_BASE}/projects/{project}/zones/{zone}/instances/{name}"
    status, body = _rest("DELETE", url, caller_token)
    if status == 404:
        return ""
    if status not in (200, 202):
        msg = f"delete other VM submit: status={status} body={body}"
        print(f"Warning: {msg}", file=sys.stderr)
        return msg
    op_name = body.get("name") if isinstance(body, dict) else None
    if not op_name:
        return ""
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        st, op_body = _rest(
            "GET",
            f"{GCP_BASE}/projects/{project}/zones/{zone}/operations/{op_name}",
            caller_token,
        )
        if st == 200 and isinstance(op_body, dict) and op_body.get("status") == "DONE":
            if op_body.get("error"):
                msg = f"delete other VM op error: {op_body['error']}"
                print(f"Warning: {msg}", file=sys.stderr)
                return msg
            return ""
        time.sleep(5)
    msg = f"delete other VM {name} did not complete within 180s"
    print(f"Warning: {msg}", file=sys.stderr)
    return msg


def _run_three_probes(
    project: str,
    zone: str,
    instance_id: str,
    denied_sa: str,
    allowed_sa: str,
    other_vm: str,
    other_zone: str,
    denied_token: str,
    allowed_token: str,
    result: dict[str, Any],
) -> int:
    """Three subtests against ``getSerialPortOutput`` from each minted token."""
    denied_status, denied_body = _probe_serial(project, zone, instance_id, denied_token)
    result["tests"]["denied_principal_cannot_access_console"] = _passing(
        denied_status == 403, principal=denied_sa, status=denied_status, evidence=denied_body
    )
    allowed_status, allowed_body = _probe_serial(project, zone, instance_id, allowed_token)
    result["tests"]["allowed_principal_can_access_console"] = _passing(
        allowed_status == 200, principal=allowed_sa, status=allowed_status, evidence=allowed_body
    )
    other_status, other_body = _probe_serial(project, other_zone, other_vm, allowed_token)
    # 403 = correct scoping (denied on a different VM). 404 is not proof
    # of scoping — the resource may simply not exist.
    result["tests"]["allowed_principal_is_resource_scoped"] = _passing(
        other_status == 403, principal=allowed_sa, status=other_status, evidence=other_body
    )
    result["access_restricted"] = (
        result["tests"]["denied_principal_cannot_access_console"]["passed"]
        and result["tests"]["allowed_principal_is_resource_scoped"]["passed"]
    )
    result["success"] = all(result["tests"][n].get("passed") is True for n in REQUIRED_TESTS)
    return 0 if result["success"] else 1


def _run_pre_provisioned_path(
    project: str,
    zone: str,
    instance_id: str,
    caller_token: str,
    result: dict[str, Any],
) -> int:
    """Operator-pre-provisioned fallback (env-var SAs + env-var other VM)."""
    denied_sa = os.environ["GCP_DENIED_PRINCIPAL_SA"]
    allowed_sa = os.environ["GCP_ALLOWED_PRINCIPAL_SA"]
    other_vm = os.environ["GCP_OTHER_VM_NAME"]
    other_zone = os.environ.get("GCP_OTHER_VM_ZONE", zone)
    result["mode"] = "pre-provisioned"
    result["principals"]["denied"] = denied_sa
    result["principals"]["allowed"] = allowed_sa
    result["principals"]["other_vm"] = other_vm

    denied_token, derr = _mint_token_for_sa(denied_sa, caller_token)
    allowed_token, aerr = _mint_token_for_sa(allowed_sa, caller_token)
    if denied_token is None or allowed_token is None:
        return _fail_all(result, f"token minting failed (denied={derr} | allowed={aerr})")

    return _run_three_probes(
        project,
        zone,
        instance_id,
        denied_sa,
        allowed_sa,
        other_vm,
        other_zone,
        denied_token,
        allowed_token,
        result,
    )


def _fail_all(result: dict[str, Any], reason: str) -> int:
    """Mark every required subtest as failed with ``reason``.

    ``console_rbac`` is classified ``adapt`` — converting a setup failure
    into a success-shaped policy skip would let a missing IAM binding or
    a denied probe ship as a passing RBAC validation. This helper sets
    explicit failure state and rc 1 with failure-shaped evidence; it
    never emits ``skipped=True`` because the orchestrator reserves that
    marker for policy skips that pair with ``success=True`` / rc 0.
    ``main`` prints exactly one JSON document, so this helper must NOT
    print.
    """
    for name in REQUIRED_TESTS:
        result["tests"][name] = _failed(reason)
    result["success"] = False
    result["failure_reason"] = reason
    if "error" not in result:
        result["error"] = reason
    return 1


def _run_self_provision_path(
    project: str,
    zone: str,
    instance_id: str,
    caller_token: str,
    caller_email: str,
    result: dict[str, Any],
) -> int:
    """Default path: create temp SAs + scoping VM, probe, then clean up."""
    result["mode"] = "self-provisioned"

    # Per-process random discriminator so concurrent factory workers (each
    # in its own worktree but sharing the same GCP project + RUN_ID) don't
    # collide on the global SA / scoping-VM / firewall names. The script
    # records the generated names in ``created`` and the finally: cleanup
    # uses those same names, so the round-trip stays consistent.
    worker_tag = uuid.uuid4().hex[:6]
    suffix = unique_suffix("rbac")[-9:]
    denied_id = f"isv-d-{worker_tag}{suffix}"
    allowed_id = f"isv-a-{worker_tag}{suffix}"
    other_name = unique_suffix(f"isv-rbac-other-{worker_tag}")
    other_zone = zone
    other_firewall_tag = unique_suffix(f"isv-rbac-tag-{worker_tag}")
    other_firewall_name = unique_suffix(f"isv-rbac-fw-{worker_tag}")

    created: dict[str, str] = {}
    # Mutable ownership trackers for async helpers that may raise after
    # the resource is accepted server-side. The finally: clause reads
    # these so transient wait errors never strand resources.
    firewall_tracker: dict[str, Any] = {}
    other_vm_tracker: dict[str, Any] = {}
    iam_bindings_added: list[tuple[str, str, str]] = []  # (resource_url, role, member)
    cleanup_errors: list[str] = []
    key_created = False
    key_path = ""
    target_url = f"{GCP_BASE}/projects/{project}/zones/{zone}/instances/{instance_id}"
    rc = 1

    try:
        from google.cloud import compute_v1

        firewalls_client = compute_v1.FirewallsClient()

        denied_email, err = _create_sa(project, denied_id, caller_token)
        if denied_email is None:
            rc = _fail_all(result, f"create denied SA failed: {err}")
            return rc
        created["denied_sa"] = denied_email
        result["principals"]["denied"] = denied_email

        allowed_email, err = _create_sa(project, allowed_id, caller_token)
        if allowed_email is None:
            rc = _fail_all(result, f"create allowed SA failed: {err}")
            return rc
        created["allowed_sa"] = allowed_email
        result["principals"]["allowed"] = allowed_email

        # Caller member tag: SA emails are serviceAccount:; everything else
        # (user / group) defaults to user: as long as the address has '@'.
        if "iam.gserviceaccount.com" in caller_email:
            caller_member = f"serviceAccount:{caller_email}"
        elif "@" in caller_email:
            caller_member = f"user:{caller_email}"
        else:
            rc = _fail_all(result, f"unresolvable caller identity: {caller_email!r}")
            return rc

        for sa_email in (denied_email, allowed_email):
            sa_url = f"{IAM_BASE}/projects/{project}/serviceAccounts/{sa_email}"
            err = _modify_iam_policy(sa_url, caller_token, add_bindings=[(TOKEN_CREATOR_ROLE, caller_member)])
            if err:
                rc = _fail_all(result, f"tokenCreator on {sa_email} failed: {err}")
                return rc
            iam_bindings_added.append((sa_url, TOKEN_CREATOR_ROLE, caller_member))

        # Provision the scoping VM (small e2-micro). The firewall and key
        # are reused so the resource shape matches the main VM.
        key_path, key_created = generate_ssh_keypair(unique_suffix("isv-rbac-key"))
        created["key_file"] = key_path
        pub_key = read_ssh_public_key(key_path)
        firewall, firewall_created = create_ssh_firewall_rule(
            firewalls_client,
            project,
            "default",
            name=other_firewall_name,
            target_tag=other_firewall_tag,
            tracker=firewall_tracker,
        )
        if firewall_created:
            created["firewall"] = firewall

        ok, vm_err = _create_other_vm(
            project,
            other_zone,
            other_name,
            caller_token,
            network="default",
            public_key=pub_key,
            firewall_target_tag=other_firewall_tag,
            tracker=other_vm_tracker,
        )
        # Tracker is populated immediately after the insert ack, so even
        # if the helper later returns ok=False the cleanup-side branch
        # below knows there's a VM to delete.
        if "name" in other_vm_tracker:
            created["other_vm"] = other_vm_tracker["name"]
            result["principals"]["other_vm"] = other_vm_tracker["name"]
        if not ok:
            rc = _fail_all(result, f"create scoping VM failed: {vm_err}")
            return rc

        # Grant the allowed SA compute.viewer on the target VM only (the
        # scope binding lives on the instance, not on the project). The
        # denied SA never receives any role.
        binding_err = _modify_iam_policy(
            target_url,
            caller_token,
            add_bindings=[(SCOPED_ROLE, f"serviceAccount:{allowed_email}")],
        )
        if binding_err:
            rc = _fail_all(result, f"target-VM IAM bind failed: {binding_err}")
            return rc
        iam_bindings_added.append((target_url, SCOPED_ROLE, f"serviceAccount:{allowed_email}"))

        # IAM propagation for fresh TokenCreator bindings is eventually
        # consistent and can take ~180 seconds. The retry helper budgets
        # 12 attempts x 15s so a normal propagation window does not
        # surface as "token minting failed" before the RBAC probes run.
        denied_token, derr = _mint_token_with_retry(denied_email, caller_token)
        allowed_token, aerr = _mint_token_with_retry(allowed_email, caller_token)
        if denied_token is None or allowed_token is None:
            rc = _fail_all(result, f"token minting failed (denied={derr} | allowed={aerr})")
            return rc

        rc = _run_three_probes(
            project,
            zone,
            instance_id,
            denied_email,
            allowed_email,
            other_name,
            other_zone,
            denied_token,
            allowed_token,
            result,
        )

    except Exception as exc:
        # Real setup/runtime failure (SA creation, IAM binding, probe
        # execution). Surface it as a structured diagnostic — a bare
        # ``return rc`` in ``finally`` would suppress the exception and
        # leave the JSON with ``success=False`` but no ``error``.
        result["error"] = str(exc)
        result["success"] = False
        rc = 1
    finally:
        # Pick up firewall ownership from the tracker — the helper may
        # have raised mid-wait after the insert ack.
        if firewall_tracker.get("created") and "firewall" not in created:
            created["firewall"] = firewall_tracker["name"]

        # Cleanup order: scoping VM → IAM bindings (so we can still see
        # the SA emails) → SAs → firewall → local key. Every step records
        # errors into cleanup_errors so success flips false if anything
        # leaked.
        if "other_vm" in created:
            err = _delete_other_vm(project, other_zone, created["other_vm"], caller_token)
            if err:
                cleanup_errors.append(err)

        # Remove IAM bindings in reverse-add order. SAs are still present
        # at this point (the SA delete below also removes bindings keyed
        # on the SA, but explicit removal is the contract surface and
        # makes the cleanup auditable).
        for resource_url, role, member in reversed(iam_bindings_added):
            err = _remove_iam_bindings(resource_url, caller_token, remove_bindings=[(role, member)])
            if err:
                cleanup_errors.append(f"remove {role} from {resource_url}: {err}")

        for label in ("denied_sa", "allowed_sa"):
            if label in created:
                err = _delete_sa(project, created[label], caller_token)
                if err:
                    cleanup_errors.append(err)

        if "firewall" in created:
            try:
                from google.cloud import compute_v1

                firewalls_client = compute_v1.FirewallsClient()
                op = firewalls_client.delete(project=project, firewall=created["firewall"])
                op.result(timeout=120)
            except Exception as exc:
                cleanup_errors.append(f"scoping firewall cleanup: {exc}")

        if key_created and key_path:
            for path in (Path(key_path), Path(key_path).with_suffix(".pub")):
                try:
                    if path.exists():
                        path.chmod(0o600)
                        path.unlink()
                except OSError as exc:
                    cleanup_errors.append(f"local key cleanup {path}: {exc}")

        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
            existing_error = result.get("error", "")
            combined = "; ".join(cleanup_errors)
            result["error"] = f"{existing_error}; cleanup: {combined}" if existing_error else f"cleanup: {combined}"
            result["success"] = False
            rc = 1
    # Single source of truth: ``main`` prints the JSON once after this
    # function returns. The return is outside the finally so any
    # exception in the body still propagates to the except branch and
    # gets recorded as a structured error.
    return rc


def main() -> int:
    """Drive the three RBAC subtests from real ``getSerialPortOutput`` probes."""
    parser = argparse.ArgumentParser(description="Validate Compute Engine console RBAC")
    parser.add_argument("--instance-id", required=True, help="Target instance name")
    parser.add_argument("--region", required=True, help="Effective zone")
    parser.add_argument("--project", default="", help="GCP project (default: ADC)")
    args = parser.parse_args()

    project = resolve_project(args.project)
    zone = select_zones(args.region, PREFERRED_ZONES)[0]
    caller_email = fetch_adc_caller_email() or ""
    caller_token = _caller_token()

    result: dict[str, Any] = {
        "success": False,
        "platform": "vm",
        "test_name": "console_rbac",
        "instance_id": args.instance_id,
        "rbac_model": "gcp-iam-service-account-impersonation",
        "access_restricted": False,
        "restricted_actions": [SERIAL_PERMISSION],
        "tests": {},
        "principals": {"caller": caller_email or "unknown-caller", "denied": None, "allowed": None, "other_vm": None},
    }

    if caller_token is None:
        rc = _fail_all(result, "ADC token refresh failed; cannot run RBAC probes")
    else:
        denied_env = os.environ.get("GCP_DENIED_PRINCIPAL_SA", "")
        allowed_env = os.environ.get("GCP_ALLOWED_PRINCIPAL_SA", "")
        other_env = os.environ.get("GCP_OTHER_VM_NAME", "")
        if denied_env and allowed_env and other_env:
            rc = _run_pre_provisioned_path(project, zone, args.instance_id, caller_token, result)
        elif not caller_email:
            rc = _fail_all(result, "could not resolve calling identity; cannot self-provision RBAC probes")
        else:
            rc = _run_self_provision_path(project, zone, args.instance_id, caller_token, caller_email, result)

    # Print exactly one JSON document; mutations from probes / skip / cleanup
    # are all reflected here.
    print(json.dumps(result, indent=2, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# IAM Validation Guide (GCP)

Operator walkthrough for the GCP IAM domain. The NCP-level index — including the
full operator environment-variable contract — is
[`docs/references/gcp.md`](../../../../../../../docs/references/gcp.md); this guide
covers only how the IAM lifecycle executes and what an operator must have in
place for it.

The provider-agnostic contract is [`suites/iam.yaml`](../../../../../suites/iam.yaml);
the wiring that points it at these scripts is
[`providers/gcp/config/iam.yaml`](../../../config/iam.yaml). The
[AWS reference](../../../../aws/scripts/iam/docs/aws-iam.md) is the closest
implemented analog.

## Why service accounts

Google Cloud has no human IAM users, so this domain is an adaptation: the
closest managed, provider-owned identity primitive for the suite's `create_user`
→ `test_credentials` → `teardown` contract is a **service account**.

Because hardened organizations commonly block user-managed service-account keys
(`constraints/iam.disableServiceAccountKeyCreation`), the portable primary
credential path is **keyless**. No JSON key file is ever written, so that
constraint does **not** need to be relaxed for the suite to run.

The suite's field names stay AWS-shaped for contract compatibility; their GCP
meaning is:

| Contract field | GCP meaning |
|---|---|
| `username` | Service account email (`<id>@<project>.iam.gserviceaccount.com`) |
| `user_id` | `ServiceAccount.unique_id` |
| `access_key_id` | `ServiceAccount.unique_id` — **non-secret**; equals `tokeninfo.azp` |
| `secret_access_key` | Short-lived OAuth2 access token (600s, self-expiring) — sensitive, redacted from logs |
| `token_expiry` | RFC3339 `expire_time` returned by `generateAccessToken` |
| `account_id` | Project ID that owns the service account |

## Lifecycle

| Step | Phase | Script | What it does |
|---|---|---|---|
| `create_user` | setup | [`create_user.py`](../create_user.py) | Creates the run-scoped service account, grants the caller `roles/iam.serviceAccountTokenCreator` on it, and mints a 600s impersonation access token. |
| `test_credentials` | test | [`test_credentials.py`](../test_credentials.py) | Proves the minted token is live, belongs to the expected identity, and authenticates an IAM read. |
| `teardown` | teardown | [`delete_user.py`](../delete_user.py) | Deletes the service account; idempotent when it is already gone. |

### `create_user`

1. Resolves the project (see [project resolution](#authentication-and-project)).
2. Builds a service-account ID from `--username`. GCP caps the ID at 30
   characters, so the base name is truncated after reserving room for a
   per-invocation discriminator **and** the run suffix
   (`isv-test-user-<disc>-<run>`). The discriminator means a retry after a failed
   cleanup lands on a fresh name instead of colliding with the leftover.
3. Creates the account with a per-invocation marker in its description.
4. Grants `roles/iam.serviceAccountTokenCreator` to the calling principal on the
   **new account's own** IAM policy (read-modify-write with the etag, under
   bounded retry — never a blind overwrite).
5. Calls `IAMCredentials.generateAccessToken` with the `cloud-platform` scope and
   a 600s lifetime.

That last step is the slow one. A freshly-granted `tokenCreator` binding is
eventually consistent — convergence up to **~180s** has been observed on hardened
organizations, and until it converges the mint returns
`403 iam.serviceAccounts.getAccessToken denied` even though the binding API
already reported success. The mint therefore retries **12 × 15s = 180s**, and the
step timeout is **420s** (the propagation budget plus margin for the create,
bind, initial mint, and transient retries). A healthy run finishes in well under
two minutes; the budget only matters on the tail.

### `test_credentials`

Two probes, both using the **minted** credential — never the operator's ambient
credentials:

- **identity** — OAuth2 `tokeninfo` for the token must report
  `azp == create_user.access_key_id` (the service account `unique_id`) and
  `expires_in > 0`. This proves the token is live *and* is the expected identity.
- **access** — `IAMClient.get_service_account` for that account, built from
  `google.oauth2.credentials.Credentials(token=...)`. A successful read is
  authenticated access; `PermissionDenied` still proves the token authenticated
  and is reported as `permission_denied_expected`, because limited project
  permissions on a brand-new account are not a credential failure.
  `Unauthenticated`, an expired token, and transport errors are failures.

Both probes retry only transient / propagation shapes and draw from one
wall-clock budget stamped at entry, so the step always emits its JSON payload
rather than being killed mid-retry at the 120s step cap.

### `teardown`

Deletes only the service account this run created (its email arrives from
`create_user.username` — never a scan for accounts that "look like ours"). The
short-lived token needs no revocation: it self-expires, and no persistent
credential was created, which the payload states as
`credential_cleanup: not_required_short_lived_token`.

Two GCP-specific behaviors matter here:

- `NotFound` is the desired terminal state and reports success, so a re-run after
  a partial cleanup is safe.
- GCP returns `PermissionDenied` (403) both for a denied caller **and** for an
  absent account, so a 403 alone is never cleanup evidence. The step then polls
  the fully paginated project service-account inventory and reports success only
  once the exact email is absent. A still-present account or an unreadable
  inventory fails closed rather than reporting a green teardown over a leak.

The step also consumes `create_user`'s `unreconciled_resources` handoff (see
[ownership](#ownership-and-leak-safety)). Those candidates arrive only when
`create_user` failed — exactly when `username` is the `none` sentinel — and each
is deleted **only** after its recorded invocation marker is re-verified on that
exact account. A marker mismatch is preserved as another run's account; an
unverifiable candidate is preserved too and fails the step, because "might still
exist" is not a clean teardown.

## Ownership and leak safety

A create can commit server-side and still lose its response. Ownership is taken
from the **accepted** create (marker stamp plus exact-identity readback), not
from the presence of a well-formed response body, so an identity that was
committed before the response was lost is still inside the cleanup set.

If the bind or the mint fails after the account exists, `create_user` rolls that
account back and confirms its absence before returning the failure. Two honest
outcomes are reported instead of being swallowed:

- `cleanup_errors` — the rollback could not be confirmed. The payload keeps
  `username` / `service_account_name` so the identity can still be reclaimed.
- `unreconciled_resources` — the create may have committed but the reconciling
  readback was denied or kept failing, so ownership was proven neither way. The
  candidate is **not** deleted (it could belong to another run). Each entry is a
  packed record — `service_account|<email>|<project>||<invocation-id>` — that
  carries the marker itself, so nothing has to be reconstructed by hand. The
  config forwards these records to the `teardown` step, which re-verifies the
  marker on that exact account and deletes **only** on a match. The field is
  always present (empty on a healthy run).

To verify a candidate manually, compare its `<invocation-id>` against the
account's description, which reads `... (isv-invocation=<invocation-id>)`:

```bash
gcloud iam service-accounts describe <email> --format='value(description)'
```

`GCP_IAM_SKIP_TEARDOWN=true` reaches this rollback as well, not only the terminal
`teardown` step. A setup-failure compensating delete runs long before teardown
does, so preservation has to gate it too — otherwise the exact fixture the
operator asked to keep is gone before the flag is ever consulted. Under
preservation the delete is suppressed and the retained identity is reported
(`resources_preserved`, plus `username` / `service_account_name`); the cleanup is
never partial.

## Authentication and project

The scripts use Application Default Credentials:

```bash
gcloud auth application-default login          # user credentials
# or
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

Project resolution order: `--project` (threaded from the config setting) →
`GOOGLE_CLOUD_PROJECT` → `GCLOUD_PROJECT` → the project bundled with ADC. If
none resolve, the step fails fast with a structured credentials error.

### Operator environment variables (IAM scope)

| Variable | Required? | Default / fallback | Purpose |
|---|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | Optional | `GCLOUD_PROJECT`, then the ADC project | Project that owns the run-created service account. |
| `GCP_IAM_SKIP_TEARDOWN` | Optional | unset — teardown runs | `true` preserves the created service account for debugging. Forwarded as `--skip-destroy` to **both** fixture-owning steps: the `teardown` step and `create_user`'s setup-failure rollback. |

The complete cross-domain variable contract lives in
[`docs/references/gcp.md`](../../../../../../../docs/references/gcp.md).

### Required roles

The principal running this domain needs, on the project:

- `roles/iam.serviceAccountAdmin` — create and delete the test service account,
  and read the project service-account inventory used as the absence proof.
- `roles/iam.serviceAccountTokenCreator` — the suite grants this dynamically on
  the *new* service account, so the running principal needs the project-level
  binding that lets it set that policy and call `generateAccessToken`.

No additional API needs to be enabled beyond the IAM and IAM Credentials APIs,
and no Docker, GPU, or SSH access is involved — this domain creates no compute.

## Validations exercised

| Validation | Step | Checks |
|---|---|---|
| `setup_checks` | `create_user` | `FieldExistsCheck` (IAM01-01) requires `username` + `access_key_id`; `StepSuccessCheck`. |
| `credentials` | `test_credentials` | `IamCredentialAccessCheck` (IAM03-01) requires `tests.identity.passed` **and** `tests.access.passed`; `StepSuccessCheck`. |
| `teardown_checks` | `teardown` | `StepSuccessCheck`. |

## Running

```bash
# Prerequisites: ADC + a resolvable project.
uv run isvctl test run -f isvctl/configs/providers/gcp/config/iam.yaml
```

Expected wall clock is a couple of minutes on a healthy project; a run that hits
`tokenCreator` propagation can spend up to three additional minutes inside
`create_user`. No billable compute is created, so an interrupted run leaves at
most one free-tier service account behind.

## Cleanup

Set `GCP_IAM_SKIP_TEARDOWN=true` to keep the created service account after a run.
The account name carries a per-run random suffix, so a standalone
`isvctl test run --phase teardown` **cannot** clean it up: `create_user` did not
execute in that process, so the teardown step's `{{steps.create_user.username}}`
reference is unresolved. Copy the `username` value printed by the original
`create_user` step and delete the account directly (note that preservation must
be disabled — or simply absent — for any recovery path that goes back through the
suite):

```bash
uv run python3 isvctl/configs/providers/gcp/scripts/iam/delete_user.py \
  --username <username-from-create_user-output> \
  --project=<project>          # optional; the delete uses the projects/-/ wildcard
```

Two optional helper scripts ship alongside the wired steps and are **not**
invoked by the suite:

- [`setup.py`](../setup.py) — read-only inventory of the project's service
  accounts and the identity capabilities this domain relies on. Useful for
  confirming credentials and project access before a run.
- [`teardown.py`](../teardown.py) — reclaims the ambiguous-create candidates a
  run recorded in `create_user`'s `unreconciled_resources` and could not resolve
  itself. It deletes **only** those explicit records, and only after re-verifying
  each one's invocation marker on the exact account; there is no name-prefix
  sweep, because a name is not ownership and a sweep cannot tell a concurrent
  run's identity from a leftover. If the project or credentials do not resolve,
  the recorded candidates cannot be verified and the helper exits **nonzero**
  instead of reporting a cleanup it never performed:

  ```bash
  uv run python3 isvctl/configs/providers/gcp/scripts/iam/teardown.py \
    --project=<project> \
    --unreconciled-resources "$(jq -r '.unreconciled_resources | join(",")' create_user-output.json)"
  ```

## Org-policy considerations

- `constraints/iam.disableServiceAccountKeyCreation` — **does not** need to be
  disabled. The keyless path uses `IAMCredentials.generateAccessToken`; no key
  material is ever created or written to disk.
- Organizations that restrict `iam.serviceAccounts.create` project-wide will fail
  `create_user` with an access-denied error naming the missing permission; grant
  the roles above rather than working around the failure.

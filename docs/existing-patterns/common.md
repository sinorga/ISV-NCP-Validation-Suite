# Common Patterns — Config, Stubs, Test Harness, Validations

Shared patterns for all domains. For domain-specific patterns (JSON schemas, wait logic), see the domain file (e.g., `vm.md`).

For general script writing, config structure, and running validations, see the [External Validation Guide](../guides/external-validation-guide.md).

---

## 1. Config → Stub → Validation Flow

The config system has layers — all under `isvctl/configs/`:

```
isvctl/configs/tests/<domain>.yaml             → THE CONTRACT (NCP-agnostic, human-authored)
                                                 Defines: steps, validations, required JSON fields
                                                 Read-only — do not modify

isvctl/configs/providers/<target>/<domain>.yaml   → NCP IMPLEMENTATION (per-provider)
                                                 Imports tests/<domain>.yaml
                                                 Overrides: stub paths, settings (region, instance type)
                                                 Agent generates this for target NCPs

isvctl/configs/stubs/<domain>/*.py             → TEMPLATE STUBS (provider-agnostic)
                                                 Starting point with TODO blocks
                                                 Copy and implement for your NCP

isvctl/configs/stubs/<target>/<domain>/*.py       → NCP STUBS (per-provider)
                                                 Agent generates these for target NCPs
```

The full flow:

```
  ┌──────────────────────────┐
  │ isvctl/configs/tests/    │  Contract: steps, validations, required JSON fields
  │ (NCP-agnostic)           │  Human-authored, read-only
  └────────────┬─────────────┘
               │ imported by
               ▼
  ┌──────────────────────────┐
  │ isvctl/configs/providers/│  NCP-specific overrides: stub paths, settings
  │   <target>/<domain>.yaml │  ← Agent generates this
  └────────────┬─────────────┘
               │ defines steps, args, phases
               ▼
  ┌──────────────────────────┐
  │ isvctl (runner)          │  Resolves templates, invokes stubs
  └────────────┬─────────────┘
               │ invokes stub with args from config
               │ (templates resolved: {{steps.X.field}})
               ▼
  ┌──────────────────────────┐
  │ Stub Script              │  ← Agent generates this
  │ isvctl/configs/stubs/    │
  └────────────┬─────────────┘
               │ prints JSON to stdout
               ▼
                    ┌─────────────────┐
                    │  Validation     │
                    │  Classes        │
                    │  (NCP-agnostic) │
                    └─────────────────┘
                      reads step_output, calls set_passed/set_failed
```

### Provider Config Pattern

The test config defines steps with default stub paths. The provider config overrides them:

```yaml
# isvctl/configs/providers/<target>/<domain>.yaml
import:
  - ../../tests/<domain>.yaml   # imports the contract

commands:
  <domain>:
    steps:
      - name: <step_name>
        phase: setup
        command: "python3 ../../stubs/<target>/<domain>/<step_name>.py"
        args:
          - "--instance-type"
          - "{{instance_type}}"
          - "--region"
          - "{{region}}"

tests:
  settings:
    region: "<ncp-region>"
    instance_type: "<ncp-instance-type>"

  # Override validation parameters for NCP-specific behavior
  validations:
    cloud_init:
      checks:
        CloudInitCheck:
          metadata_url: "<ncp-metadata-url>"
          metadata_headers:
            <Header-Name>: "<header-value>"

  # Exclude validations that fail due to NCP/image limitations (not stub bugs).
  # Use the validation class name (e.g., DriverCheck), not the group name (e.g., driver_info).
  exclude:
    tests:
      - DriverCheck             # e.g., nvcc not on PATH
      - ContainerRuntimeCheck   # e.g., Docker not installed
    markers:
      - slow                    # e.g., skip long-running workload tests
```

### Error Classification Pattern

The oracle uses consistent error types. Target NCP stubs should follow the same pattern:

```python
try:
    # ... API calls ...
    result["success"] = True
except SomeNCPError as e:
    result["error_type"] = "access_denied"  # or "credentials_missing", "api_error", etc.
    result["error"] = str(e)
except Exception as e:
    result["error_type"] = "unknown_error"
    result["error"] = str(e)
```

Standard error types: `credentials_missing`, `credentials_expired`, `credentials_invalid`, `access_denied`, `api_error`, `unknown_error`.

---

## 2. Stub Implementation Pattern

Every stub must follow this pattern:

```python
#!/usr/bin/env python3
"""<Description of what this stub does>."""
import argparse
import json
import sys

def main() -> int:
    parser = argparse.ArgumentParser(description="<description>")
    parser.add_argument("--name", default="isv-test-resource")
    parser.add_argument("--region", default="<target-ncp-default-region>")
    # ... domain-specific args matching what the config passes ...
    args = parser.parse_args()

    result = {
        "success": False,
        "platform": "<domain>",  # e.g., "vm", "network", "iam"
        # ... all fields documented in the test config and domain guide ...
    }

    # Track resources for cleanup
    instance_name = None

    try:
        # Target NCP SDK calls go here
        # Populate result fields on success
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
    finally:
        # ALWAYS clean up resources created by this stub.
        # Do not rely on the shared teardown step.
        if instance_name:
            try:
                # delete instance, firewall rules, networks, etc.
                pass
            except Exception:
                pass

    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1

if __name__ == "__main__":
    sys.exit(main())
```

Key rules:
- The **test config** (`isvctl/configs/tests/<domain>.yaml`) documents the required JSON fields in comments for each step — output ALL of them
- The **domain guide** (`isvctl/configs/stubs/aws/<domain>/docs/`) has the exact JSON schema — use it as the definitive reference
- Use `argparse` to accept the same arguments the config passes
- `print(json.dumps(result, indent=2))` to stdout — this is the ONLY thing on stdout
- Progress/debug messages go to stderr: `print("...", file=sys.stderr)`
- Wrap main logic in try/except — errors must still produce valid JSON with `"success": false`
- **Every stub that creates infrastructure MUST have a `finally:` block that cleans up.** Do not rely on the shared teardown step — stubs may run in isolation (worktree testing, interrupted runs). If the stub creates an instance, firewall rule, or network, add `finally:` cleanup that deletes them.
- **Never use `time.sleep(N)` for readiness waits.** Hardcoded sleeps waste time on fast resources and timeout on slow ones. Instead, write or use a polling helper that checks the condition in a loop with short intervals and returns as soon as it's met (e.g., attempt SSH every 10s until connected, up to a max timeout). The static check will fail stubs with `time.sleep()` over 30 seconds.
- **Never hardcode output values to pass a validation.** Stub output must reflect actual target NCP behavior. If a validation expects a value that doesn't apply to your NCP (e.g., a subnet-level feature that only exists on AWS), do NOT hardcode the expected value to pass. Instead: (1) check if the validation can be excluded in the provider config `exclude.tests`, (2) if not excludable, report the real value and document the mismatch as an NCP gap in the commit message. Hardcoding fake values to pass validations is gaming the test — it hides real platform differences that need upstream clarification.

### Researching Target NCP SDK

When implementing a stub:
1. Read the domain guide to understand what the stub does
2. Look up the target NCP SDK (client class, method names, request/response format)
3. Implement the stub, mapping response fields to match the JSON schema in the test config

You may install additional Python packages needed for the target NCP SDK.

### Quality Principles

- **Match the oracle's behavior, not just its API calls.** If the oracle supports auto-approve, preflight checks, existing resource reuse, and configurable timeouts, your target NCP stubs should too.
- **Follow the oracle's code style and structure.** Same section organization, same level of comments, same error handling patterns.
- **Don't carry over oracle-specific names.** Use the target NCP's native service names, env var conventions, and terminology.
- **Check resource state names on the target NCP.** Every NCP uses different state enums for the same concepts. For example, AWS uses "stopped" but GCP uses "TERMINATED" for a stopped VM. Do NOT assume the oracle's state names apply — look up the target NCP's resource lifecycle states and use them in your polling/wait logic.
- **Maintainable code matters.** Write clean, well-documented code that passes ruff linting.

---

## 3. Test Harness

### Static Validation (Mode 1)

```bash
curl -X POST http://localhost:8080/test \
  -H "Content-Type: application/json" \
  -d '{"path": "<workspace-path>", "domain": "<domain>"}'
```

The response tells you the log path. Tail it and look for START/END markers.

The test harness validates 12 aspects:
1. Provider config exists and parses
2. Intent — resolved config has same validations as oracle
3. Syntax — Python/bash parseable
4. Lint — ruff passes
5. JSON contract — required fields present
6. Completeness — all oracle stubs have target equivalents
7. Coverage — all oracle test configs covered
8. SSH-readiness — setup stubs output public_ip + key_file
9. Connectivity — firewall/security group rules for SSH
10. IaC presence — K8s domains have Terraform
11. Remote execution — network domains have SSM/SSH
12. Command resolution — step command paths exist

When reading results, focus on ERROR lines and the SUMMARY table. Do not read full output into context.

### Live Execution (Mode 2)

```bash
curl -X POST http://localhost:8080/test \
  -H "Content-Type: application/json" \
  -d '{"path": "<workspace-path>", "mode": "live", "ncp": "<target-ncp>", "domain": "<domain>"}'
```

Mode 2 runs static checks first, then executes stubs against the real NCP via `isvctl test run`. Results appear in the same log format.

**Interpreting Mode 2 failures:**

- **Stub bug** (your code has an error) — fix and retry. Examples: wrong API call, missing parameter, auth error.
- **NCP gap** (the test correctly detected the NCP doesn't provide a feature) — do NOT fix. Document it. Examples: Docker not installed, CUDA not on PATH, API not supported.

Only retry for stub bugs. NCP gaps are valid test findings — that's why the test suite exists.

**If the test harness is unreachable** (curl returns connection refused): **STOP ALL WORK immediately.** Report in `current_tasks/` and wait. Do not attempt manual validation.

---

## 4. Validation Patterns

Validation class names are **provider-agnostic** (no `Ssh` prefix — the canonical config uses generic names like `ConnectivityCheck`, `GpuCheck`, etc.):

```yaml
tests:
  validations:
    setup_checks:
      step: launch_instance
      checks:
        InstanceStateCheck:
          expected_state: "running"

    ssh:
      step: launch_instance
      checks:
        ConnectivityCheck: {}          # NOT SshConnectivityCheck
        OsCheck:
          expected_os: "ubuntu"
```

**Key insight**: Validations ONLY check JSON field names and values. They never call NCP APIs directly. Your stub must output the same field names as the oracle — the values should be semantically correct for the target NCP.

---

## 5. Step-by-step Mode 2 testing

Mode 2 runs real cloud infrastructure — each full domain run can take 10-30 minutes and costs money. To iterate efficiently, **test one step at a time** using `skip: true` in the provider config.

### How it works

isvctl supports `skip: true` on any step. Skipped steps don't execute (no cloud resources created, no time wasted). Setup and teardown always run.

### Workflow

1. Set `skip: true` on ALL test-phase steps
2. Remove `skip: true` from exactly ONE step
3. Submit Mode 2 → only setup + that step + teardown runs
4. If it fails → fix the stub, retry (fast — only 1 step)
5. If it passes → add `skip: true` back to that step, remove it from the NEXT step
6. Repeat until all steps pass individually
7. Remove ALL `skip: true` and run the full suite as final validation

**Do NOT batch multiple test steps.** Even if steps look similar ("API-only" vs "instance-based"), test them one at a time. Batching defeats the purpose — a failure in one step blocks all subsequent steps and wastes the time spent on the others.

### Example: testing `dns_test` in network domain

```yaml
# isvctl/configs/providers/gcp/network.yaml
commands:
  network:
    steps:
      - name: create_network
        phase: setup
        command: "python3 ../../stubs/gcp/network/create_vpc.py"
        ...

      - name: vpc_crud
        phase: test
        skip: true              # ← already passing, skip it
        command: "python3 ../../stubs/gcp/network/vpc_crud_test.py"
        ...

      - name: subnet_config
        phase: test
        skip: true              # ← already passing, skip it
        command: "python3 ../../stubs/gcp/network/subnet_test.py"
        ...

      - name: dns_test
        phase: test
        # skip: true            # ← NOT skipped — this is the step we're testing
        command: "python3 ../../stubs/gcp/network/dns_test.py"
        ...

      - name: peering_test
        phase: test
        skip: true              # ← not yet tested, skip it
        command: "python3 ../../stubs/gcp/network/peering_test.py"
        ...

      - name: teardown
        phase: teardown
        command: "python3 ../../stubs/gcp/network/teardown.py"
        ...
```

This runs: `create_network` → `dns_test` → `teardown` (~2-3 min instead of ~25 min).

### Important

- **Never skip setup or teardown** — always let them run. Setup is cheap (one VPC create) and teardown ensures cleanup. Skipping them is error-prone.
- **Only skip test-phase steps** — use `skip: true` only on steps in `phase: test`.
- **Remove all `skip: true` for the final run** — the full suite must pass end-to-end before committing.
- Steps with `skip: true` show in results as "Step skipped" (not a failure).

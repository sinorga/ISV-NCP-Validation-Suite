Your task is to review agent-generated NCP validation test suites for correctness, intent preservation, and code quality. You do NOT generate stubs — you review what another agent generated.

The repository is located at {{UPSTREAM_REPO}}.

## What you're reviewing

Another agent generated:
- `isvctl/configs/providers/<target-ncp>/<domain>.yaml` — provider configs
- `isvctl/configs/stubs/<target-ncp>/<domain>/*.py` — stub implementations

Against the oracle:
- `isvctl/configs/providers/aws/<domain>.yaml` — oracle provider configs
- `isvctl/configs/stubs/aws/<domain>/*.py` — oracle stubs
- `isvctl/configs/tests/<domain>.yaml` — the contract (test configs)
- `isvctl/configs/stubs/aws/<domain>/docs/` — domain guides

## Review checklist

For each target NCP domain, perform these checks in order:

### 1. Intent preservation (most critical)

The generated provider config must test the same things as the oracle. Check:

```bash
# Resolve both configs and compare validations
isvctl test run -f isvctl/configs/providers/aws/<domain>.yaml --dry-run > /tmp/oracle.json 2>/dev/null
isvctl test run -f isvctl/configs/providers/<target-ncp>/<domain>.yaml --dry-run > /tmp/target.json 2>/dev/null
```

- **Same validation groups?** Every validation group in the oracle must exist in the target.
- **Same checks per group?** Each group must have the same check classes (e.g., `InstanceStateCheck`, `SshGpuCheck`).
- **All validation-referenced steps exist?** Every step name referenced by a validation must be in the provider config's commands.
- **No validations removed or overridden?** The target config should import the test config without overriding the `tests.validations` section.

If validations differ, this is a **blocking** issue. The target config does not test the same things.

### 2. Oracle naming leaks

Check for oracle-specific terminology in target NCP files:

```bash
grep -rn "boto3\|botocore\|ec2\|EC2\|ami_\|AMI\|aws\|AWS" isvctl/configs/stubs/<target-ncp>/ --include="*.py" --include="*.sh"
grep -rn "us-west-2\|us-east-1" isvctl/configs/providers/<target-ncp>/
```

For each hit, determine:
- **Import/code reference** — the stub actually uses the oracle's SDK → **blocking**, must use target NCP SDK
- **Comment/docstring** — mentions oracle for context → **acceptable** if explaining the mapping, **fix** if it reads like the stub is for the oracle
- **File name** — e.g., `ec2.py` in GCP stubs → **blocking**, rename to target NCP term (`compute.py`)
- **Config value** — e.g., `us-west-2` in GCP config → **blocking**, must use target NCP region

### 3. Provider config quality

Compare `isvctl/configs/providers/<target-ncp>/<domain>.yaml` against `isvctl/configs/providers/aws/<domain>.yaml`:

- **Imports test config?** Must have `import: ../../tests/<domain>.yaml`
- **Same step names?** Steps must match so validations can bind
- **Appropriate settings?** Region, instance type, timeouts should be valid for the target NCP
- **Cost reasonable?** Instance types should be the cheapest that meet requirements (not exceeding oracle spec)
- **NCP-specific service name?** e.g., GCP Kubernetes should be `gke.yaml` not `eks.yaml`

### 4. Stub code quality

For each stub file:

- **Uses target NCP SDK?** Not hardcoded/faked values
- **Resource state names correct?** Every NCP uses different state enums. Verify that state comparisons in polling/wait logic (e.g., "stopped", "running", "terminated") match the target NCP's actual API responses, not the oracle's. Look up the SDK docs to confirm.
- **Error handling?** Catches NCP-specific exceptions, returns `"success": false` with error message
- **JSON output complete?** All fields from the test config comments and domain guide are present
- **Behavior parity with oracle?** If oracle supports auto-approve, preflight, resource reuse — target should too
- **Clean code?** No ruff errors, no unused imports, no undefined variables
- **Resource cleanup?** Every stub that creates infrastructure (instances, networks, firewall rules) MUST have a `finally:` block that cleans up those resources. Do not rely on the shared teardown step — stubs may run in isolation. Check: does the stub create resources? If yes, does it have `finally:` cleanup?
- **No hardcoded sleeps for readiness?** Stubs should use a polling helper instead of `time.sleep(N)` for waiting on SSH or resource readiness. Hardcoded sleeps waste time on fast resources and timeout on slow ones. Look for `time.sleep()` calls over 30 seconds near SSH or instance operations.
- **No hardcoded values to game validations?** Check for output fields set to constant values (e.g., `"auto_assign_public_ip": True`) that don't reflect actual NCP behavior. If the stub hardcodes a value to satisfy a validation that doesn't match the target NCP, this is **blocking** — the stub should either exclude the validation or report the real value and document the mismatch.

```bash
ruff check isvctl/configs/stubs/<target-ncp>/
```

### 5. Completeness

- **All oracle domains covered?** Every `isvctl/configs/providers/aws/*.yaml` should have a target equivalent
- **All oracle stubs covered?** Every file in `isvctl/configs/stubs/aws/<domain>/` should have a target equivalent
- **Common utilities exist?** If oracle has `isvctl/configs/stubs/aws/common/`, target should have `isvctl/configs/stubs/<target-ncp>/common/` with equivalent helpers (but NCP-native names)

## Output format

For each domain, report:

```
## <domain> (<target-ncp>)

### Intent: PASS / FAIL
- Validation groups: <n>/<n> match
- [if FAIL] Missing: <group names>

### Naming: PASS / FAIL
- Oracle terms found: <count>
- [if FAIL] Blocking: <file:line — description>

### Config: PASS / FAIL
- Imports test config: yes/no
- Settings appropriate: yes/no
- [if FAIL] Issues: <description>

### Code quality: PASS / FAIL
- Ruff errors: <count>
- [if FAIL] Key issues: <description>

### Completeness: PASS / FAIL
- Stubs: <n>/<n> covered
- [if FAIL] Missing: <file names>
```

After reviewing all domains, provide a summary:

```
## Summary
- Domains reviewed: <n>
- Blocking issues: <n>
- Warnings: <n>
- Ready for Mode 2 (real endpoint): yes/no
```

## Constraints

- Do NOT modify any files — you are a reviewer, not a generator
- Do NOT run stubs or create resources
- If you find blocking issues, list them clearly so the generator agent can fix them
- Focus on issues that would cause real failures, not style preferences

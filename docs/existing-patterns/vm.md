# VM Domain Patterns

VM-specific patterns for generating stubs. Read `common.md` first.

---

## Wait/Readiness Patterns

The oracle's `launch_instance` does NOT use `wait_for_ssh` — it relies on the AWS API waiter (`instance_status_ok`) to confirm the OS is booted. Only `start_instance` and `reboot_instance` use `wait_for_ssh` (to confirm recovery after a state change).

If your NCP has no readiness API equivalent, you **must** add a best-effort SSH wait to `launch_instance`. Best-effort means: do NOT fail the step if SSH times out, because the validators will check SSH independently via `ConnectivityCheck`. Without this, every downstream step that SSHes into the instance will race cloud-init and flake.

### SSH flags for `wait_for_ssh`

Required flags: `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o PasswordAuthentication=no`

- `IdentitiesOnly=yes` — CRITICAL. Without this, the ssh client tries all keys from the SSH agent before the specified key file. If the agent has many keys loaded, the server disconnects with "Too many authentication failures" before your key is even tried.
- `UserKnownHostsFile=/dev/null` — cloud IPs get reused, prevents stale host key errors
- `PasswordAuthentication=no` instead of `BatchMode=yes` — BatchMode can reject key auth in some configurations

Keep attempts reasonable (20 × 10s = ~3 min). If SSH isn't ready in 3 minutes, something is wrong. The SSH check must use the same key file that the stub generated — verify the path exists before attempting.

---

## JSON Schemas

Every step must output a JSON object with `"success"` and `"platform"` fields, plus domain-specific fields. These are the required fields documented in the test config comments.

**launch_instance:**
```json
{
  "success": true,
  "platform": "vm",
  "instance_id": "<unique-id>",
  "public_ip": "<ip-address>",
  "key_file": "<path-to-ssh-key>",
  "vpc_id": "<network-id>",
  "instance_state": "running",
  "security_group_id": "<sg-id>",
  "key_name": "<key-pair-name>"
}
```

**list_instances:**
```json
{
  "success": true,
  "platform": "vm",
  "instances": [{"instance_id": "...", "state": "running"}],
  "total_count": 1
}
```

**verify_tags:**
```json
{
  "success": true,
  "platform": "vm",
  "instance_id": "<id>",
  "tags": {"Name": "...", "CreatedBy": "..."},
  "tag_count": 2
}
```

**stop_instance:**
```json
{
  "success": true,
  "platform": "vm",
  "instance_id": "...",
  "state": "stopped",
  "stop_initiated": true
}
```

**start_instance:**
```json
{
  "success": true,
  "platform": "vm",
  "instance_id": "...",
  "state": "running",
  "public_ip": "...",
  "key_file": "...",
  "start_initiated": true,
  "ssh_ready": true
}
```

**reboot_instance:**
```json
{
  "success": true,
  "platform": "vm",
  "instance_id": "...",
  "instance_state": "running",
  "public_ip": "...",
  "key_file": "...",
  "uptime_seconds": 45,
  "ssh_connectivity": true
}
```

**serial_console:**
```json
{
  "success": true,
  "platform": "vm",
  "instance_id": "<id>",
  "console_available": true,
  "serial_access_enabled": true,
  "output_length": 4096
}
```

**teardown:**
```json
{
  "success": true,
  "platform": "vm",
  "resources_deleted": ["..."],
  "message": "..."
}
```

---

## VM-Specific Validations

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
        ConnectivityCheck: {}
        OsCheck:
          expected_os: "ubuntu"

    gpu:
      step: launch_instance
      checks:
        GpuCheck:
          expected_gpus: 1

    tags:
      step: verify_tags
      checks:
        InstanceTagCheck:
          required_keys: ["Name", "CreatedBy"]

    serial_console:
      step: serial_console
      checks:
        SerialConsoleCheck: {}
```

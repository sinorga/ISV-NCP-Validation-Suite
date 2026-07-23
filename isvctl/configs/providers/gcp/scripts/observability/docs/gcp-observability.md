# GCP Observability Validation Guide

This guide is the operator walkthrough for validating GCP observability
capabilities (VPC Flow Logs, guest host syslogs, and the managed-BMC gap) with
the ISV validation framework. It is the GCP port of the AWS observability
provider ([`providers/aws/config/observability.yaml`](../../../../aws/config/observability.yaml)
+ [`providers/aws/scripts/observability/`](../../../../aws/scripts/observability/));
the [AWS reference](../../../../../../../docs/references/aws.md) is the canonical
contract and the GCP scripts under `providers/gcp/scripts/observability/`
translate it onto Compute Engine, Cloud Logging, and Resource Manager. For the
NCP-level operator contract (auth, project resolution, the required
`NETWORK_FIREWALL_TRUST_IP` firewall source, and the full operator env-var
table) see [`docs/references/gcp.md`](../../../../../../../docs/references/gcp.md).

## Overview

The GCP observability validation provisions an isolated network + host, turns on
VPC Flow Logs, and then probes four observability aspects:

1. **create_network** (setup) — Create a run-scoped custom-mode VPC and one
   regional subnetwork (a Compute Engine network owns no CIDR; the subnetwork
   owns the `/24` carved from `--cidr`).
2. **enable_vpc_flow_logs** (setup) — Patch VPC Flow Logs onto the run-created
   subnetwork(s): `flow_sampling=1.0`, `INCLUDE_ALL_METADATA`, no export filter.
3. **launch_host** (setup) — Launch a CPU host on the subnet, inject an SSH key
   via instance metadata, open a tcp/22 firewall (sourced only from
   `NETWORK_FIREWALL_TRUST_IP`), and wait for stable SSH.
4. **vpc_flow_logs** (test) — Read the live `Subnetwork.log_config` and run a
   project-scoped Cloud Logging query against `compute.googleapis.com/vpc_flows`.
5. **host_syslogs** (test) — SSH the host and sample `journalctl` (dmesg fallback).
6. **bmc_sel_logs** (test) — Provider-hidden (no customer BMC SEL API).
7. **bmc_gpu_telemetry** (test) — Provider-hidden (no customer BMC telemetry API).
8. **teardown_telemetry_peer** / **teardown_host** / **teardown_flow_logs** /
   **teardown_network** (teardown) — Delete only what this run created (the peer
   and its source-scoped ICMP firewall first, then the host, flow logs, and VPC).

### How each step maps onto GCP

| Step | AWS oracle | GCP translation |
|------|-----------|-----------------|
| `create_network` | `ec2.create_vpc` + subnet | `NetworksClient.insert` (custom-mode VPC, no CIDR) + `SubnetworksClient.insert` (regional, owns the CIDR). Emits `network_id`, `subnets`, the exact `created_subnets` allowlist, and `network_created`. |
| `enable_vpc_flow_logs` | standalone Flow Log + CloudWatch group + IAM role | `SubnetworksClient.patch` of `log_config` on ONLY the run-created subnets. No standalone flow-log / log-group / IAM-role resource exists, so none is emitted. Tracks `patched_flow_log_subnets` (the only list teardown disables) vs the observed `flow_log_subnets` diagnostic list. |
| `launch_host` | key pair + security group + `run_instances` | local SSH key (no managed key store) + VPC firewall rule (no security-group resource) + `InstancesClient.insert` on the run subnet. Region-scoped zone walk (the subnet is regional); emits `zone` + `leaked_zones`. |
| `vpc_flow_logs` | `describe_flow_logs` + CloudWatch group check + `TrafficType` | live `Subnetwork.log_config` read-back (Compute Engine has no native traffic-type field) + a Cloud Logging query for the `compute.googleapis.com/vpc_flows` log. `traffic_type=ALL` is projected only when every target subnet reads back enable=true, flow_sampling=1.0, empty filter. |
| `host_syslogs` | SSH + journalctl/dmesg | identical SSH guest boundary after metadata key injection. |
| `bmc_sel_logs` / `bmc_gpu_telemetry` | provider-hidden | provider-hidden after a real `ProjectsClient.get_project` identity probe (`bmc_endpoints_checked=0`). |
| `teardown_*` | terminate / delete FL / delete SG / delete VPC | idempotent per-resource deletes gated on the forwarded `*_created` ownership bits; subnets deleted only from the `created_subnets` allowlist; VPC deleted last. |

### Subtests exercised

| Validation group | Check | Step | Subtests |
|------------------|-------|------|----------|
| `network_logs` | `VpcFlowLogsCheck` | vpc_flow_logs | flow_log_endpoint_reachable, flow_logs_configured, traffic_type_all, log_destination_accessible |
| `host_logs` | `HostSyslogCheck` | host_syslogs | syslog_endpoint_reachable, host_log_source_present, entries_recent |
| `bmc_logs` | `BmcSelLogsCheck` | bmc_sel_logs | sel_log_endpoint_reachable, sel_log_source_present, sel_entries_queryable |
| `bmc_telemetry` | `BmcGpuTelemetryCheck` | bmc_gpu_telemetry | telemetry_endpoint_reachable, gpu_metrics_present, host_os_gap_identified, telemetry_samples_recent |

The four validations above are enabled. So are the sixteen v0.9.0 telemetry/log
validators — see the next section. No observability validator is excluded for
GCP.

### v0.9.0 telemetry/log validators (wired + enabled)

Observability suite v0.9.0 adds sixteen released telemetry and log validators.
GCP enables every one, so **all twenty** observability validators are enabled and
none is listed under `exclude.tests`. Each step is wired in `observability.yaml`
(mirroring the AWS oracle's stub set under
[`providers/aws/scripts/observability/`](../../../../aws/scripts/observability/)),
and each stub makes a **real GCP SDK call on its main path** and is honestly
gated (like the AWS oracle, success is the AND of every subtest — a plane that
produces no real evidence fails, it is never skipped). Ten of the sixteen are
provider-owned physical planes that pass the validator **provider-hidden** path
after a real `ProjectsClient.get_project` identity probe; the other six are
tenant-visible planes that first generate a bounded, run-scoped traffic / I-O
fixture on the launched host over the injected-key SSH boundary (VPC Flow Logs
and disk metrics record only real flows / I-O), then poll for the concrete
evidence — `provider_hidden` is never honest for them.

| Step(s) | Stub / real signal | Plane |
|---------|--------------------|-------|
| `telemetry_delivery_latency` | `telemetry_delivery_test.py` — bounded external-traffic fixture, then poll for the freshest queryable `vpc_flows` record (Cloud Logging) scope-bound to the run subnetworks; delivery latency = its real timestamp age | Tenant-visible — delivery latency from a real record timestamp. |
| `north_south_network_telemetry`, `east_west_network_telemetry`, `host_nic_network_telemetry` | `network_telemetry_test.py` — bounded external (north-south / host-NIC) or in-subnet (east-west) traffic fixture, then scope-bound `vpc_flows` poll (Cloud Logging) + real NIC inventory (Compute `get`) for host NIC | Tenant-visible — VPC Flow Log packet/byte telemetry; `provider_hidden` is never used. |
| `management_network_telemetry`, `nvswitch_fabric_telemetry` | `network_telemetry_test.py` — provider-hidden after `ProjectsClient.get_project` | Provider-owned physical plane — no tenant endpoint. |
| `storage_performance_telemetry` | `storage_telemetry_test.py` — bounded write+read I-O fixture (`dd`), then scope-bound Cloud Monitoring GCE persistent-disk time series (`monitoring projects.timeSeries.list` for read/write bytes+ops and `average_io_latency`, filtered to the run-owned `instance_id`) after a Compute disk enumeration (`InstancesClient.get`) | Tenant-visible — bandwidth/IOPS/latency from real disk metrics. |
| `storage_capacity_telemetry` | `storage_telemetry_test.py` — real Compute disk enumeration (`InstancesClient.get`) joined to guest filesystem used/free/total read over SSH (`df`) | Tenant-visible — observed capacity; provisioned disk size is never substituted. |
| `gpu_nvlink_telemetry`, `switch_nvlink_telemetry` | `nvlink_telemetry_test.py` — provider-hidden after `ProjectsClient.get_project` | Provider-owned physical plane — no customer NVLink switch-fabric API. |
| `fabric_manager_logs`, `ufm_event_logs`, `subnet_manager_logs`, `general_switch_logs`, `switch_syslogs`, `switch_kernel_logs` | `log_availability_test.py` — provider-hidden after `ProjectsClient.get_project` | Provider-owned physical plane — Compute Engine exposes no physical fabric / subnet-manager / switch log endpoint to tenants. |

Every telemetry/log step is `continue_on_failure`, so a probe that legitimately
finds no data never aborts the test phase or teardown.

### VPC Flow Logs "ALL" is a projection, not a native field

Compute Engine records VPC Flow Logs per subnetwork and has **no** traffic-type
field. `traffic_type_all` passes only after the live `Subnetwork.log_config`
read-back proves `enable=true`, `flow_sampling=1.0`, and an empty `filter_expr`
on every target subnetwork; the stub then projects the canonical
`traffic_type=ALL` probe. Here **ALL** means every GCP flow-log record remaining
after uncontrollable primary sampling is retained for inbound and outbound flows
— it is **not** packet-complete capture and **not** a native GCP field. The
Cloud Logging query independently proves the endpoint and destination are
accessible; a successful query with zero recent entries still passes those two
subtests (a real sample count is reported when present and never fabricated).

### BMC SEL / GPU telemetry is a platform gap

Compute Engine bare metal is fully managed by Google; the documented customer
monitoring surface is guest and Cloud Monitoring telemetry, not a BMC SEL or
Redfish GPU-telemetry API. So — exactly like the AWS oracle — `bmc_sel_logs` and
`bmc_gpu_telemetry` emit **provider-hidden** evidence (every subtest
`passed=true` + `provider_hidden=true`, `bmc_endpoints_checked=0`) after a real
Resource Manager `ProjectsClient.get_project` identity probe. Guest NVML / DCGM
data is never relabeled as BMC telemetry.

## Prerequisites

### Tools and APIs

```bash
# Python SDKs (installed via uv sync): google-cloud-compute,
# google-cloud-logging, google-cloud-resource-manager, and
# google-api-python-client (Cloud Monitoring v3 REST client used by the
# storage-performance probe).
uv run python -c "from google.cloud import compute_v1, logging_v2, resourcemanager_v3; import googleapiclient.discovery; print('OK')"
```

Enable these APIs on the project: **Compute Engine** (`compute.googleapis.com`),
**Cloud Logging** (`logging.googleapis.com`), **Cloud Resource Manager**
(`cloudresourcemanager.googleapis.com`), and **Cloud Monitoring**
(`monitoring.googleapis.com`) — the `storage_performance_telemetry` probe reads
GCE persistent-disk time series via Cloud Monitoring `projects.timeSeries.list`.

### IAM roles

The principal running the suite (user or service account) needs, on the project:

- `roles/compute.admin` — create / delete the network, subnetwork, firewall
  rule, and host instance, and patch subnetwork flow-log configuration.
- `roles/logging.viewer` — run the Cloud Logging query for the `vpc_flows` log.
- `roles/monitoring.viewer` (or the least-privilege `monitoring.timeSeries.list`
  permission) — read GCE persistent-disk time series for the
  `storage_performance_telemetry` probe's Cloud Monitoring
  `projects.timeSeries.list` call. Without it that step fails with an
  `access_denied` error and returns rc=1 (it does not silently skip).
- `roles/browser` (or `resourcemanager.projects.get`) — the BMC provider-hidden
  path's `ProjectsClient.get_project` identity probe.

### Authentication, project, and firewall source

Auth (ADC) and project resolution are the NCP-wide contract — see
[`docs/references/gcp.md`](../../../../../../../docs/references/gcp.md).
`launch_host` opens an SSH (tcp/22) firewall rule and there is **no
open-internet default**: it reads the **required** `NETWORK_FIREWALL_TRUST_IP`
operator env var (a bare IPv4 normalizes to `/32`; comma-separated IPv4 CIDRs
are allowed). When it is unset, empty, non-IPv4, or `0.0.0.0/0`, `launch_host`
fails fast with an operator error before creating any resource.

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id          # or rely on ADC's bundled project
export NETWORK_FIREWALL_TRUST_IP=203.0.113.10        # your workstation / CI egress IP
```

## Running

```bash
uv run isvctl test run -f isvctl/configs/providers/gcp/config/observability.yaml
```

### Skip teardown (debugging)

```bash
GCP_OBSERVABILITY_SKIP_TEARDOWN=true \
  uv run isvctl test run -f isvctl/configs/providers/gcp/config/observability.yaml
```

`GCP_OBSERVABILITY_SKIP_TEARDOWN=true` forwards `--skip-destroy` to all four
teardown steps, preserving the run-created resources (host instance, local SSH
key pair, SSH firewall rule, created subnetworks, VPC Flow Logs configuration,
VPC network) plus the internal telemetry peer instance and the
telemetry peer firewall rule (a source-scoped ICMP rule). The same flag is also
forwarded to the three setup producers — `create_network`, `launch_host`, and
`launch_telemetry_peer` — where `--skip-destroy` suppresses the compensating
deletion each step would otherwise run when its own setup **fails** partway, so a
partially created run (network + subnetworks, host + SSH firewall + local key, or
peer + ICMP firewall) is left in place for inspection and its exact ownership
identifiers are emitted under `preserved_on_failure`. Unset, teardown runs
normally and a setup failure compensates (deletes only its own accepted
resources) as before.

### Run notes

- **CPU host, no GPU/Docker.** The syslog probe host defaults to `e2-standard-2`
  running the public `ubuntu-2204-lts` image; the domain needs neither Docker nor
  a GPU. Override with `GCP_OBSERVABILITY_INSTANCE_TYPE` / `GCP_OBSERVABILITY_IMAGE`
  / `GCP_OBSERVABILITY_IMAGE_PROJECT` / `GCP_OBSERVABILITY_SSH_USER` if needed.
- **Region-scoped zone walk.** The subnetwork is regional, so `launch_host` walks
  ONLY the zones of `GCP_OBSERVABILITY_REGION` (default `us-central1`) on a
  stockout-class failure — never cross-region (the subnet would not exist
  elsewhere). Any partial-insert record is reclaimed before advancing, or
  recorded in `leaked_zones` for teardown to reclaim exactly.
- **Per-subnet flow-log ownership.** `enable_vpc_flow_logs` patches logging ONLY
  on the subnets this run created (the `created_subnets` allowlist) and forwards
  the exact `patched_flow_log_subnets` list to teardown, so operator-owned or
  other-run subnets already logging on an adopted VPC are never disabled.
- **Resource naming.** Every created resource (network, subnetwork, firewall,
  instance, SSH key) is suffixed with a unique per-run id (the `RUN_ID`
  environment variable) so parallel runs never collide on Compute Engine's
  name-as-id namespace and teardown owns only its own resources.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `NETWORK_FIREWALL_TRUST_IP is unset` (launch_host) | Export your operator IP/CIDR; there is no open-internet fallback for SSH ingress. |
| `traffic_type_all` fails | A target subnetwork read back with logging disabled, `flow_sampling` below 1.0, or a non-empty filter. Confirm `enable_vpc_flow_logs` ran and succeeded. |
| Host syslog (`host_logs`) fails to connect | Verify `NETWORK_FIREWALL_TRUST_IP` matches the host the suite runs from and that the image is an Ubuntu cloud image with cloud-init SSH-key injection. |
| Cloud Logging query permission denied | Grant `roles/logging.viewer` to the run principal. |
| `storage_performance_telemetry` fails with `access_denied` / Monitoring permission | Enable `monitoring.googleapis.com` and grant `roles/monitoring.viewer` (or `monitoring.timeSeries.list`) to the run principal. |
| `bmc_*` project probe fails | The run principal cannot `resourcemanager.projects.get`; grant `roles/browser`. |

## Related documentation

- [GCP reference (operator contract)](../../../../../../../docs/references/gcp.md)
- [Observability suite contract](../../../../../suites/observability.yaml)
- [AWS observability provider](../../../../aws/config/observability.yaml)

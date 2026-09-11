# Airline-shaped dependency-discovery testbed

This ARM64-compatible Compose testbed exercises live dependency discovery across a cloud-to-legacy integration boundary.

```text
                         cloud-tier
 reservations ─┐
 crew ─────────┼──HTTP──> boundary-gateway ──TCP──> legacy-core
 baggage ──────┘              (boundary)            (legacy-tier)
```

`reservations`, `crew`, and `baggage` are lightweight cloud-native HTTP services. Each forwards its domain request to `boundary-gateway`. The gateway is deliberately the only service attached to both Docker networks: it translates the cloud-side HTTP call into a TCP request to `legacy-core`. `legacy-core` has no published port and is attached only to the internal `legacy-tier`; cloud services are not members of that network and cannot resolve or bypass the gateway.

## Services and node types

| Service | Role | Node type |
| --- | --- | --- |
| `reservations` | Reservation domain HTTP API | `cloud-native` |
| `crew` | Crew domain HTTP API | `cloud-native` |
| `baggage` | Baggage domain HTTP API | `cloud-native` |
| `boundary-gateway` | HTTP-to-TCP cloud/legacy integration boundary | `boundary-gateway` |
| `legacy-core` | Deterministic mainframe-style TCP stub | `legacy` |

## Run traffic

```bash
docker compose -f testbed/docker-compose.yml up --build -d
curl --fail http://127.0.0.1:8081/reservation
curl --fail http://127.0.0.1:8082/crew
curl --fail http://127.0.0.1:8083/baggage
```

The gateway is also published at `http://127.0.0.1:8084/legacy?caller=manual` for boundary troubleshooting. The legacy core is intentionally not published.

## Discover dependencies

```bash
sudo python3 testbed/discovery/collect.py --duration 15 \
  --observations testbed/results/observations.jsonl \
  --graph testbed/results/dependency-graph.json \
  --node-type reservations=cloud-native \
  --node-type crew=cloud-native \
  --node-type baggage=cloud-native \
  --node-type boundary-gateway=boundary-gateway \
  --node-type legacy-core=legacy
```

While the collector runs, invoke the three cloud endpoints. It resolves ephemeral Docker IP addresses and host PIDs dynamically to Compose service names. The expected observed graph is:

```text
reservations -> boundary-gateway
crew -> boundary-gateway
baggage -> boundary-gateway
boundary-gateway -> legacy-core
```

Only observed TCP connections are written to JSONL and aggregated into the graph. DNS (port 53), NTP (port 123), and traffic without two container identities are filtered.

## Phase 5 integration validation

Run on the Linux host running Docker Engine, with Python 3.11+, `bpftrace`, kernel
BPF/kprobe support, and Docker Compose 2.24.4+ (`!override` support). Root access is
required for the real eBPF collector; Docker Desktop or a remote Docker daemon is
not sufficient because collection and container PIDs must share a host.

From the repository root:

```bash
sudo python3 -B testbed/validate_integration.py
```

The script builds and starts an isolated `phase5-<unique-id>` Compose project,
using temporary loopback ports and its own PostgreSQL volume. It leaves an existing
testbed running. It generates requests to all three cloud services while running
the existing eBPF collector, scopes observations to the validation project, and
reuses the discovery graph builder and database seed CLI. It then applies the
existing `connection-drop high` fault and clears it through the chaos CLI.

Baseline and recovery must return HTTP 200 with a legacy response; every request
during the fault must return 502. Each phase must discover the same five typed
nodes and four directed edges, including the failing gateway-to-legacy edge.
PostgreSQL edge counts must match the current graph snapshot. Telemetry must retain
one `tcp_connection_attempt` sample per distinct observation, including failed
attempts; it does not claim to measure connection success or latency. Re-importing
each snapshot must leave telemetry counts unchanged.

Exit status 0 means all checks and cleanup passed. Evidence is retained under
`testbed/results/phase5-<unique-id>/`: per-phase raw/scoped observations, graphs,
collector logs, HTTP results, plus `compose.log` and `summary.json`. These files
are root-owned when run with sudo. To select a new output directory and increase
the default 12-second collection windows:

```bash
sudo python3 -B testbed/validate_integration.py --duration 20 --output /tmp/phase5-evidence
```

The output directory must not already exist. Normal completion, errors, Ctrl-C,
and SIGTERM attempt to clear the fault and remove only the validation project's
containers, networks, and disposable volume. Built images and evidence are retained.
If the process is forcibly killed or Docker becomes unavailable, use the project
name printed in the evidence directory or recorded in `summary.json` to clean up:

```bash
docker compose -p phase5-REPLACE_WITH_RUN_ID -f testbed/docker-compose.yml down --volumes --remove-orphans
```

This deletes only that validation project's resources; do not substitute the name
of a project whose database you want to keep. No AI model or frontend is required.

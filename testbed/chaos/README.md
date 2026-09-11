# Phase 4 fault injection

Run commands from the repository root. Requires Python 3, Docker Engine with
Compose v2 or later, and permission to use Docker. Network faults require Linux
kernel netem/prio/u32 and iptables statistic support and `NET_ADMIN`; the Docker
VM supplies the Linux kernel on Docker Desktop. Rootless/restricted engines may
not support these operations. The helper image supports ARM64 and AMD64.

No Compose service, network, database, discovery code, or application is changed.
A temporary helper joins the existing gateway network namespace. It exits after
configuring the fault; kernel rules remain until cleared. Only IPv4 TCP traffic
from the gateway to the legacy core on port 9090 is selected.

| Fault | Low | Medium | High |
| --- | --- | --- | --- |
| `network-delay` | 100 ms | 500 ms | 1500 ms |
| `connection-drop` | Every 10th opening SYN (10%) | Every 2nd (50%) | Every opening SYN (100%) |
| `batch-job-stall` | Pause 1 s | Pause 5 s | Pause 15 s |

Profiles are defined in `profiles.json`. Delay is fixed per outgoing packet,
including handshake packets, with no jitter; request latency can exceed the
configured delay and high intensity can trigger the gateway's 2 s timeout.
Connection drop uses deterministic nth matching, starting with the first SYN,
and TCP reset rejection so the selected connection fails immediately. This is
connection rejection, not random packet loss or a silent network blackhole.
Counts are reproducible for an identical ordered traffic stream; other traffic
on this edge consumes the same counter. Existing connections are not terminated.

The Phase 2 core has no batch worker. Batch-job stall therefore emulates legacy
work stopping by pausing the entire `legacy-core` container. It also stops request
handling. Apply blocks for the configured duration, then automatically unpauses;
Ctrl-C/SIGTERM also unpauses. A separate clear can resume it early.

## Setup

If the Phase 2 services are not already running, start only the application
services (this does not start or change the database):

```bash
docker compose -f testbed/docker-compose.yml up --build -d reservations crew baggage boundary-gateway legacy-core
docker build -t airline-phase4-chaos:local testbed/chaos
curl --fail 'http://127.0.0.1:8084/legacy?caller=baseline'
```

## One CLI

Use `low`, `medium`, or `high` with any apply command. Clear does not need a level.
For an existing custom Compose project append `--project-name YOUR_PROJECT` to
every CLI and validation command; use the same project with Compose setup.

```bash
python3 testbed/chaos/chaos.py apply network-delay low
curl --max-time 10 -w '\n%{time_total}s\n' 'http://127.0.0.1:8084/legacy?caller=delay'
python3 testbed/chaos/chaos.py clear network-delay

python3 testbed/chaos/chaos.py apply connection-drop medium
for i in $(seq 1 10); do curl --max-time 10 -s -o /dev/null -w '%{http_code}\n' 'http://127.0.0.1:8084/legacy?caller=drop'; done
python3 testbed/chaos/chaos.py clear connection-drop

python3 testbed/chaos/chaos.py apply batch-job-stall high
# From a second terminal, to clear before the timer finishes:
python3 testbed/chaos/chaos.py clear batch-job-stall
```

Clear a fault before reapplying it or changing its level. Use one experiment at a
time, with no concurrent CLI invocations or service recreation. Different fault
types can coexist, but their effects combine. Delay refuses to replace an existing
root qdisc, and drop refuses to overwrite its existing chain. Stop background
traffic for reproducible drop counts. Do not start another stall until the earlier
apply process has exited, even after an early clear.

## Cleanup

These commands are idempotent while the target services are running:

```bash
python3 testbed/chaos/chaos.py clear network-delay
python3 testbed/chaos/chaos.py clear connection-drop
python3 testbed/chaos/chaos.py clear batch-job-stall
curl --fail 'http://127.0.0.1:8084/legacy?caller=recovered'
```

Cleanup removes only the reserved `4a00:` qdisc (and its children) or
`PHASE4_CHAOS` iptables chain and jump. Reserve these identifiers for this tool.
A killed CLI or failed helper can leave a fault active: run cleanup explicitly.
Batch clear unpauses the legacy container, so do not use it for a container paused
for some other purpose. If the helper cannot run, emergency recovery is:

```bash
docker compose -f testbed/docker-compose.yml unpause legacy-core
docker compose -f testbed/docker-compose.yml up -d --no-deps --force-recreate boundary-gateway
```

Gateway recreation removes its old network namespace and rules. It interrupts
in-flight gateway requests. No `down -v` or database cleanup is necessary.

## Validation

```bash
python3 -c "import ast,json,pathlib; p=pathlib.Path('testbed/chaos'); [ast.parse(f.read_text()) for f in p.glob('*.py')]; json.loads((p/'profiles.json').read_text())"
sh -n testbed/chaos/network.sh
docker compose -f testbed/docker-compose.yml config --quiet
python3 testbed/chaos/validate.py
```

The live validator requires an otherwise idle, healthy testbed with no active
faults. It applies all nine profiles, measures delay, checks exactly 1/5/10 rejected
requests per ten connections, observes each pause and automatic resume, and checks
HTTP recovery after every clear. It attempts all cleanup commands on failure.
Use `--url URL` if the gateway is published elsewhere.

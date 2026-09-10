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

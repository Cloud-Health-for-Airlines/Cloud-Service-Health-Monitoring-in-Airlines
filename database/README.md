# PostgreSQL persistence

Phase 3 stores Phase 2 discovery data only. PostgreSQL 16 runs on an isolated internal `data-tier` network with a named volume and no published port. The Compose credentials are local testbed defaults. Python uses only the standard library and invokes the container's `psql`; no host PostgreSQL client or Python driver is required.

## Schema

- `graph_nodes`: stable identity is `(compose_project, service_name)`; `node_id` is the foreign-key target. Stores the constrained generation type and latest observed container ID. Missing Compose project metadata is stored as an empty string. Multiple projects using the same service name in one input are rejected because the Phase 2 graph cannot distinguish them.
- `graph_edges`: directed source/destination node references, protocol, destination port, positive observation count, first/last timestamps, and JSON provenance. One row per source/destination/protocol/port. An import replaces that edge's count, time window and provenance with the supplied snapshot; it does not add counts. Edges absent from an import are retained. Import snapshots in chronological order if the latest snapshot should win.
- `telemetry`: timestamp, node reference, metric name/value, unit and source. Every observation produces a source-node `tcp_connection_attempt` sample with value `1`, unit `count`, and source from `provenance.collector`. These are connection attempts, not success or latency measurements. SHA-256 of canonical observation JSON is the primary key, so identical observations and repeated imports do not duplicate telemetry. No unobserved metrics are fabricated.

Imports are transactional: a failed insert rolls back the entire import. Container metadata is taken from the latest observation in the supplied file. Timestamps use PostgreSQL `timestamptz`. Schema application is re-runnable, but is not a migration system for future column changes.

## Local commands

Run from the repository root with Docker Compose and Python 3 available:

```bash
docker compose -f testbed/docker-compose.yml config --quiet
docker compose -f testbed/docker-compose.yml up -d --wait postgres
docker compose -f testbed/docker-compose.yml exec -T postgres pg_isready -U baccp -d baccp
```

The schema initializes automatically on first startup with an empty volume. To apply it explicitly to an existing database:

```bash
docker compose -f testbed/docker-compose.yml exec -T postgres psql -X -v ON_ERROR_STOP=1 -U baccp -d baccp < database/schema.sql
```

Use actual Phase 2 collector artifacts (see `testbed/README.md` for collection commands):

```bash
python3 -B database/seed.py \
  --graph testbed/results/dependency-graph.json \
  --observations testbed/results/observations.jsonl
```

Use `--database NAME` to target a separate validation database in the same container.

Both files must be nonempty and describe the same collection. Missing files fail explicitly. The loader reads these files without modifying them. Repeat the same command to verify idempotency, then inspect:

```bash
docker compose -f testbed/docker-compose.yml exec -T postgres psql -X -U baccp -d baccp -c 'SELECT * FROM graph_nodes ORDER BY compose_project, service_name;'
docker compose -f testbed/docker-compose.yml exec -T postgres psql -X -U baccp -d baccp -c 'SELECT s.service_name AS source, d.service_name AS destination, e.protocol, e.destination_port, e.observation_count, e.first_seen, e.last_seen FROM graph_edges e JOIN graph_nodes s ON s.node_id=e.source JOIN graph_nodes d ON d.node_id=e.destination ORDER BY 1, 2;'
docker compose -f testbed/docker-compose.yml exec -T postgres psql -X -U baccp -d baccp -c 'SELECT metric_name, count(*), sum(metric_value) FROM telemetry GROUP BY metric_name;'
python3 -B -c "import ast, pathlib; ast.parse(pathlib.Path('database/seed.py').read_text()); print('Python syntax OK')"
```

A complete Phase 2 capture should yield five nodes and four edges: each cloud service to `boundary-gateway`, then `boundary-gateway` to `legacy-core`. Telemetry row count equals the number of distinct observation records imported.

Stop PostgreSQL without deleting persisted data:

```bash
docker compose -f testbed/docker-compose.yml stop postgres
```

-- Re-runnable foundation; schema changes after initial creation require migration.
BEGIN;
CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    service_name text NOT NULL CHECK (service_name <> ''),
    node_type text NOT NULL CHECK (node_type IN ('legacy', 'boundary-gateway', 'cloud-native')),
    compose_project text NOT NULL DEFAULT '',
    container_id text,
    UNIQUE (compose_project, service_name)
);
CREATE TABLE IF NOT EXISTS graph_edges (
    source bigint NOT NULL REFERENCES graph_nodes(node_id),
    destination bigint NOT NULL REFERENCES graph_nodes(node_id),
    protocol text NOT NULL CHECK (protocol <> ''),
    destination_port integer NOT NULL CHECK (destination_port BETWEEN 1 AND 65535),
    observation_count bigint NOT NULL CHECK (observation_count > 0),
    first_seen timestamptz NOT NULL,
    last_seen timestamptz NOT NULL CHECK (last_seen >= first_seen),
    provenance jsonb NOT NULL CHECK (jsonb_typeof(provenance) = 'array'),
    PRIMARY KEY (source, destination, protocol, destination_port)
);
CREATE TABLE IF NOT EXISTS telemetry (
    observation_id text PRIMARY KEY,
    timestamp timestamptz NOT NULL,
    node_id bigint NOT NULL REFERENCES graph_nodes(node_id),
    metric_name text NOT NULL CHECK (metric_name <> ''),
    metric_value double precision NOT NULL,
    unit text NOT NULL CHECK (unit <> ''),
    source text NOT NULL CHECK (source <> '')
);
CREATE INDEX IF NOT EXISTS telemetry_node_timestamp_idx ON telemetry (node_id, timestamp);
COMMIT;

#!/usr/bin/env python3
"""Import one Phase 2 graph snapshot and its observations using Compose psql."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def literal(value):
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def seed_sql(graph, observations):
    nodes = {node['id']: node for node in graph['nodes']}
    if len(nodes) != len(graph['nodes']):
        raise ValueError('duplicate graph node identities')
    identities = {}
    # Most recent observed container supplies metadata; sorting makes input order irrelevant.
    for observation in sorted(observations, key=lambda o: (o['timestamp'], json.dumps(o, sort_keys=True))):
        for field in ('source_identity', 'destination_identity'):
            identity = observation[field]
            service = identity['service_name']
            if service not in nodes:
                raise ValueError(f'observed service missing from graph: {service}')
            project = identity.get('compose_project') or ''
            if service in identities and identities[service][0] != project:
                raise ValueError(f'graph conflates Compose projects for {service}')
            identities[service] = (project, identity.get('container_id'))
    if set(nodes) != set(identities):
        raise ValueError('each graph node requires observation metadata')

    def node_ref(service):
        project = identities[service][0]
        return ('(SELECT node_id FROM graph_nodes WHERE compose_project = '
                f'{literal(project)} AND service_name = {literal(service)})')

    statements = ['BEGIN;', 'SET LOCAL standard_conforming_strings = on;']
    for service, node in sorted(nodes.items()):
        project, container = identities[service]
        statements.append(
            'INSERT INTO graph_nodes (service_name, node_type, compose_project, container_id) '
            f'VALUES ({literal(service)}, {literal(node["node_type"])}, '
            f'{literal(project)}, {literal(container)}) '
            'ON CONFLICT (compose_project, service_name) DO UPDATE SET '
            'node_type = EXCLUDED.node_type, container_id = EXCLUDED.container_id;')
    for edge in graph['edges']:
        statements.append(
            'INSERT INTO graph_edges (source, destination, protocol, destination_port, '
            'observation_count, first_seen, last_seen, provenance) VALUES ('
            f'{node_ref(edge["source"])}, {node_ref(edge["destination"])}, '
            f'{literal(edge["protocol"])}, {literal(edge["destination_port"])}, '
            f'{literal(edge["observation_count"])}, {literal(edge["first_seen"])}, '
            f'{literal(edge["last_seen"])}, {literal(json.dumps(edge["provenance"], sort_keys=True))}::jsonb) '
            'ON CONFLICT (source, destination, protocol, destination_port) DO UPDATE SET '
            'observation_count = EXCLUDED.observation_count, first_seen = EXCLUDED.first_seen, '
            'last_seen = EXCLUDED.last_seen, provenance = EXCLUDED.provenance;')
    for observation in observations:
        canonical = json.dumps(observation, sort_keys=True, separators=(',', ':'), allow_nan=False)
        observation_id = hashlib.sha256(canonical.encode()).hexdigest()
        source = observation['provenance']['collector']
        if observation['protocol'] != 'tcp':
            raise ValueError('only Phase 2 TCP observations are supported')
        statements.append(
            'INSERT INTO telemetry (observation_id, timestamp, node_id, metric_name, metric_value, unit, source) '
            f'VALUES ({literal(observation_id)}, {literal(observation["timestamp"])}, '
            f'{node_ref(observation["source_identity"]["service_name"])}, '
            f"'tcp_connection_attempt', 1, 'count', {literal(source)}) "
            'ON CONFLICT (observation_id) DO NOTHING;')
    statements.append('COMMIT;')
    return '\n'.join(statements) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default='baccp', help='target database (default: baccp)')
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    args = parser.parse_args()
    try:
        graph = json.loads(args.graph.read_text())
        observations = [json.loads(line) for line in args.observations.read_text().splitlines() if line.strip()]
        if not graph['nodes'] or not graph['edges'] or not observations:
            raise ValueError('graph and observations must be nonempty')
        sql = seed_sql(graph, observations)
        subprocess.run(
            ['docker', 'compose', '-f', str(ROOT / 'testbed/docker-compose.yml'),
             'exec', '-T', 'postgres', 'psql', '-X', '-v', 'ON_ERROR_STOP=1', '-U', 'baccp', '-d', args.database],
            input=sql, text=True, check=True, stdout=subprocess.DEVNULL)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f'seed failed: {error}', file=sys.stderr)
        return 1
    print(f'Imported {len(graph["nodes"])} nodes, {len(graph["edges"])} edges, '
          f'{len(observations)} observations (existing telemetry deduplicated).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

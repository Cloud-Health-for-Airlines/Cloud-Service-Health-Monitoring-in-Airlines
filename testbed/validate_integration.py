#!/usr/bin/env python3
"""Validate discovery, persistence and chaos together in a disposable Compose project."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'testbed/discovery'))
from aggregate import build_graph

TYPES = {**dict.fromkeys(('reservations', 'crew', 'baggage'), 'cloud-native'),
         'boundary-gateway': 'boundary-gateway', 'legacy-core': 'legacy'}
EDGES = {(s, 'boundary-gateway', 'tcp', 8080) for s in ('reservations', 'crew', 'baggage')}
EDGES.add(('boundary-gateway', 'legacy-core', 'tcp', 9090))
PATHS = {'reservations': '/reservation', 'crew': '/crew', 'baggage': '/baggage'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=12,
                        help='seconds per discovery window (minimum 5; default 12)')
    parser.add_argument('--output', type=Path, help='new directory for evidence; must not exist')
    args = parser.parse_args()
    require(args.duration >= 5, '--duration must be at least 5 seconds')
    require(os.geteuid() == 0, 'Run with sudo on the Linux Docker host for eBPF access')
    for executable in ('docker', 'bpftrace'):
        require(shutil.which(executable), f'Missing prerequisite: {executable}')
    project = 'phase5-' + uuid.uuid4().hex[:12]
    output = (args.output or ROOT / 'testbed/results' / project).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, COMPOSE_PROJECT_NAME=project, PYTHONDONTWRITEBYTECODE='1')
    base = ['docker', 'compose', '-p', project, '-f', str(ROOT / 'testbed/docker-compose.yml')]
    fault_cli = [sys.executable, '-B', str(ROOT / 'testbed/chaos/chaos.py')]
    summary = {'project': project, 'phases': {}, 'success': False, 'cleanup': False}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Project: {project}; evidence: {output}', flush=True)

    def run(command, **kwargs):
        return subprocess.run(command, env=env, text=True, check=True, **kwargs)

    def capture(command):
        return run(command, stdout=subprocess.PIPE).stdout.strip()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    fault_attempted = False
    started = False
    with tempfile.TemporaryDirectory(prefix=project + '-') as temporary:
        override = Path(temporary) / 'ports.yml'
        override.write_text('services:\n' + ''.join(
            f'  {service}:\n    ports: !override\n      - "127.0.0.1::8080"\n'
            for service in (*PATHS, 'boundary-gateway')))
        compose = base + ['-f', str(override)]

        def sql(statement):
            return run(compose + ['exec', '-T', 'postgres', 'psql', '-X', '-A', '-t',
                                  '-v', 'ON_ERROR_STOP=1', '-U', 'baccp', '-d', 'baccp'],
                       input=statement, stdout=subprocess.PIPE).stdout.strip()

        def request(url):
            start = time.monotonic()
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    body = response.read().decode()
                    status = response.status
            except urllib.error.HTTPError as error:
                status, body = error.code, error.read().decode()
                error.close()
            return {'status': status, 'elapsed_seconds': time.monotonic() - start, 'body': body}

        try:
            run(compose + ['config', '--quiet'])
            started = True  # Also clean up a partially failed startup.
            run(compose + ['up', '--build', '-d', '--wait', '--wait-timeout', '90'])
            run(['docker', 'build', '-t', 'airline-phase4-chaos:local', str(ROOT / 'testbed/chaos')])
            urls = {service: 'http://' + capture(compose + ['port', service, '8080']) + path
                    for service, path in PATHS.items()}
            deadline = time.monotonic() + 30
            while True:
                try:
                    require(all(request(url)['status'] == 200 for url in urls.values()), 'HTTP not ready')
                    break
                except (OSError, RuntimeError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.25)

            unique_observations = set()

            def phase(name, expected_status):
                print(f'Validating {name} traffic, eBPF graph and PostgreSQL telemetry', flush=True)
                folder = output / name
                folder.mkdir()
                command = [sys.executable, '-B', str(ROOT / 'testbed/discovery/collect.py'),
                           '--duration', str(args.duration), '--observations', str(folder / 'raw.jsonl'),
                           '--graph', str(folder / 'raw-graph.json')]
                for service, node_type in TYPES.items():
                    command += ['--node-type', f'{service}={node_type}']
                traffic = []
                with (folder / 'collector.log').open('w') as log:
                    collector = subprocess.Popen(command, env=env, stdout=log, stderr=log,
                                                 start_new_session=True)
                    try:
                        # Continuous traffic covers probe attachment without guessing readiness.
                        deadline = time.monotonic() + args.duration + 15
                        while collector.poll() is None:
                            require(time.monotonic() < deadline, 'Collector timed out')
                            for service, url in urls.items():
                                result = {'service': service, 'timestamp_ns': time.time_ns(), **request(url)}
                                traffic.append(result)
                            time.sleep(0.1)
                        require(collector.returncode == 0, f'Collector failed: see {folder / "collector.log"}')
                    finally:
                        if collector.poll() is None:
                            os.killpg(collector.pid, signal.SIGINT)
                            try:
                                collector.wait(timeout=8)
                            except subprocess.TimeoutExpired:
                                os.killpg(collector.pid, signal.SIGKILL)
                                collector.wait()
                        (folder / 'traffic.json').write_text(json.dumps(traffic, indent=2) + '\n')
                require(traffic and all(r['status'] == expected_status for r in traffic),
                        f'{name}: expected only HTTP {expected_status}; see traffic.json')
                if expected_status == 200:
                    require(all('legacy-core: OK' in r['body'] for r in traffic), 'Missing legacy response')
                # Discovery is host-wide. Scope evidence to this project, then reuse its aggregator.
                observations = [json.loads(line) for line in (folder / 'raw.jsonl').read_text().splitlines()]
                observations = [o for o in observations if all(
                    o[field]['compose_project'] == project
                    for field in ('source_identity', 'destination_identity'))]
                graph = build_graph(observations, TYPES)
                require({(e['source'], e['destination'], e['protocol'], e['destination_port'])
                         for e in graph['edges']} == EDGES, f'{name}: expected four dependency edges')
                require({n['id']: n['node_type'] for n in graph['nodes']} == TYPES,
                        f'{name}: expected five correctly typed nodes')
                require(sum(e['observation_count'] for e in graph['edges']) == len(observations),
                        'Graph observation counts disagree')
                require(all(o['provenance']['collector'] == 'bpftrace' for o in observations),
                        'Expected real bpftrace provenance')
                (folder / 'observations.jsonl').write_text(''.join(json.dumps(o, sort_keys=True) + '\n' for o in observations))
                (folder / 'graph.json').write_text(json.dumps(graph, indent=2) + '\n')
                seed = [sys.executable, '-B', str(ROOT / 'database/seed.py'),
                        '--graph', str(folder / 'graph.json'), '--observations', str(folder / 'observations.jsonl')]
                run(seed)
                unique_observations.update(json.dumps(o, sort_keys=True, separators=(',', ':')) for o in observations)
                expected_sources = Counter(json.loads(o)['source_identity']['service_name'] for o in unique_observations)

                def verify_database():
                    require(sql('SELECT count(*) FROM graph_nodes;') == '5', 'Database node count mismatch')
                    actual_edges = sql('SELECT s.service_name,d.service_name,e.protocol,e.destination_port,e.observation_count '
                                       'FROM graph_edges e JOIN graph_nodes s ON s.node_id=e.source '
                                       'JOIN graph_nodes d ON d.node_id=e.destination;').splitlines()
                    expected_edges = {'|'.join(str(e[k]) for k in ('source', 'destination', 'protocol',
                                      'destination_port', 'observation_count')) for e in graph['edges']}
                    require(set(actual_edges) == expected_edges, 'Stored graph differs from snapshot')
                    counts = sql('SELECT n.service_name,count(*) FROM telemetry t JOIN graph_nodes n '
                                 'ON n.node_id=t.node_id GROUP BY n.service_name;').splitlines()
                    require(dict(line.split('|') for line in counts) ==
                            {s: str(n) for s, n in expected_sources.items()}, 'Telemetry count/source mismatch')
                    require(sql("SELECT count(*) FROM telemetry WHERE metric_name <> 'tcp_connection_attempt' "
                                "OR metric_value <> 1 OR unit <> 'count' OR source <> 'bpftrace';") == '0',
                            'Telemetry semantics mismatch')
                verify_database()
                run(seed)  # Same snapshot must not duplicate telemetry.
                verify_database()
                summary['phases'][name] = {'http_status': expected_status, 'requests': len(traffic),
                                          'observations': len(observations), 'nodes': 5, 'edges': 4,
                                          'cumulative_telemetry': len(unique_observations), 'seed_idempotent': True}

            phase('baseline', 200)
            fault_attempted = True
            run(fault_cli + ['apply', 'connection-drop', 'high', '--project-name', project])
            phase('fault', 502)
            run(fault_cli + ['clear', 'connection-drop', '--project-name', project])
            fault_attempted = False
            phase('recovery', 200)
            summary['success'] = True
        finally:
            cleanup_errors = []
            if fault_attempted:
                try:
                    run(fault_cli + ['clear', 'connection-drop', '--project-name', project])
                except subprocess.CalledProcessError as error:
                    cleanup_errors.append(str(error))
            if started:
                with (output / 'compose.log').open('w') as log:
                    subprocess.run(compose + ['logs', '--no-color'], env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    run(compose + ['down', '--volumes', '--remove-orphans', '--timeout', '5'])
                    for kind in ('container', 'network', 'volume'):
                        require(not capture(['docker', kind, 'ls', '-q', '--filter',
                                             f'label=com.docker.compose.project={project}']),
                                f'Cleanup left {kind} resources for {project}')
                except (subprocess.CalledProcessError, RuntimeError) as error:
                    cleanup_errors.append(str(error))
            summary['cleanup'] = not cleanup_errors
            summary['cleanup_errors'] = cleanup_errors
            (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(f'Evidence: {output}', flush=True)
            require(not cleanup_errors, f'Cleanup failed: {cleanup_errors}')
    print('PASS: baseline → fault → recovery; graph, telemetry, idempotency and cleanup verified')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Integration validation failed: {error}', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)

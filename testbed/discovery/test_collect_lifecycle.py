"""Collector subprocess regressions; controlled tools need neither root nor eBPF."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

COLLECT = Path(__file__).with_name('collect.py')


class CollectorLifecycleTests(unittest.TestCase):
    def collect(self, probe, duration=0.2, timeout=3):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docker = root / 'docker'
            docker.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
with Path(os.environ['DOCKER_CALLS']).open('a') as log:
    log.write(sys.argv[1] + '\\n')
if sys.argv[1] == 'ps':
    print('fake-container')
else:
    print(json.dumps([{'Id': 'fake-container', 'Config': {'Labels': {
        'com.docker.compose.service': 'legacy-core', 'com.docker.compose.project': 'test'}},
        'State': {'Pid': 12345}, 'NetworkSettings': {'Networks': {'test': {'IPAddress': '10.0.0.2'}}}}]))
''')
            tracer = root / 'bpftrace'
            tracer.write_text(f'#!{sys.executable}\n' + probe)
            docker.chmod(0o755)
            tracer.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       DOCKER_CALLS=str(root / 'docker-calls'))
            start = time.monotonic()
            result = subprocess.run([sys.executable, '-B', str(COLLECT), '--duration', str(duration),
                                     '--observations', str(root / 'events.jsonl'),
                                     '--graph', str(root / 'graph.json')],
                                    env=env, capture_output=True, text=True, timeout=timeout)
            elapsed = time.monotonic() - start
            graph = json.loads((root / 'graph.json').read_text()) if (root / 'graph.json').exists() else None
            calls = (root / 'docker-calls').read_text().splitlines()
            return result, elapsed, graph, calls

    def test_partial_line_and_shutdown_backlog_exit_normally(self):
        result, elapsed, graph, calls = self.collect('''import os, signal, time

def stop(signum, frame):
    # Finish the partial event, then exceed both pipe capacities during exit.
    os.write(1, b'0.2|9090\\n')
    os.write(2, b'diagnostic\\n' * 20000)
    for _ in range(2000):
        os.write(1, b'EVENT|1|99999999|10.0.0.99|9090\\n')
    raise SystemExit(0)

signal.signal(signal.SIGINT, stop)
os.write(1, b'EVENT|1|12345|10.0.')
while True:
    time.sleep(1)
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(elapsed, 3)
        self.assertEqual(graph['collection_summary']['events_seen'], 2001)
        self.assertEqual(graph['collection_summary']['observations_written'], 1)
        self.assertEqual(graph['collection_summary']['filtered'], {'unmapped': 2000})
        self.assertLessEqual(calls.count('ps'), 3, 'Unmapped events triggered repeated Docker refreshes')
        print(f'Normal collector exit: {elapsed:.3f}s, 2001 events, one mapped observation')

    def test_early_failure_is_reported_without_waiting_for_duration(self):
        result, elapsed, graph, _ = self.collect(
            'import sys\nprint("probe failed", file=sys.stderr)\nsys.exit(7)\n', duration=20)
        self.assertEqual(result.returncode, 7)
        self.assertIn('probe failed', result.stderr)
        self.assertIsNone(graph)
        self.assertLess(elapsed, 3)

    def test_unresponsive_child_is_killed_and_reaped(self):
        result, elapsed, graph, _ = self.collect('''import signal, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
while True:
    time.sleep(1)
''', timeout=8)
        self.assertEqual(result.returncode, 247)  # sys.exit(-SIGKILL) on POSIX
        self.assertIsNone(graph)
        self.assertLess(elapsed, 8)


if __name__ == '__main__':
    unittest.main()

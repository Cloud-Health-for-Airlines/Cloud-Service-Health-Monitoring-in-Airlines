#!/usr/bin/env python3
"""Live smoke test. Requires a running testbed and built chaos helper image."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-name")
parser.add_argument("--url", default="http://127.0.0.1:8084/legacy?caller=chaos-validation")
args = parser.parse_args()
cli = [sys.executable, str(HERE / "chaos.py")]
extra = ["--project-name", args.project_name] if args.project_name else []


def fault(action, name, level=None):
    subprocess.run(cli + [action, name] + ([level] if level else []) + extra,
                   check=True)


def request():
    start = time.monotonic()
    try:
        with urllib.request.urlopen(args.url, timeout=10) as response:
            response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
        error.close()
    return status, time.monotonic() - start


assert request()[0] == 200, "Baseline is unhealthy"
try:
    for level, ms in (("low", 100), ("medium", 500), ("high", 1500)):
        fault("apply", "network-delay", level)
        status, elapsed = request()
        assert elapsed >= ms / 1000 * 0.8, (level, status, elapsed)
        print(f"delay {level}: HTTP {status}, {elapsed:.3f}s", flush=True)
        fault("clear", "network-delay")
        assert request()[0] == 200
    for level, expected in (("low", 1), ("medium", 5), ("high", 10)):
        fault("apply", "connection-drop", level)
        statuses = [request()[0] for _ in range(10)]
        assert statuses.count(502) == expected, (level, statuses)
        assert all(status in (200, 502) for status in statuses), statuses
        print(f"drop {level}: {expected}/10 rejected", flush=True)
        fault("clear", "connection-drop")
        assert request()[0] == 200
    compose = ["docker", "compose", "-f", str(HERE.parent / "docker-compose.yml")] + extra
    legacy = subprocess.check_output(compose + ["ps", "-q", "legacy-core"], text=True).strip()
    for level, seconds in (("low", 1), ("medium", 5), ("high", 15)):
        process = subprocess.Popen(cli + ["apply", "batch-job-stall", level] + extra)
        try:
            deadline = time.monotonic() + 10
            while True:
                paused = subprocess.check_output(
                    ["docker", "inspect", "-f", "{{.State.Paused}}", legacy], text=True).strip()
                if paused == "true":
                    break
                assert process.poll() is None and time.monotonic() < deadline, "Stall did not apply"
                time.sleep(0.05)
            assert process.wait(timeout=seconds + 10) == 0
            assert subprocess.check_output(
                ["docker", "inspect", "-f", "{{.State.Paused}}", legacy], text=True).strip() == "false"
            assert request()[0] == 200
            print(f"stall {level}: pause and automatic clear verified", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
finally:
    # Clear is idempotent. Attempt all cleanup even if one command fails.
    results = [subprocess.run(cli + ["clear", name] + extra).returncode
               for name in ("network-delay", "connection-drop", "batch-job-stall")]
    if any(results):
        raise RuntimeError("Cleanup failed; run the README cleanup commands")
print("All nine profiles applied and cleared successfully")

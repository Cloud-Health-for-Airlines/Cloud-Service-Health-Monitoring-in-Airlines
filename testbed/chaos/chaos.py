#!/usr/bin/env python3
"""Apply/clear Phase 4 faults without changing Compose services or networks."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
IMAGE = "airline-phase4-chaos:local"
PROFILES = json.loads((HERE / "profiles.json").read_text())


def run(*args, capture=False):
    return subprocess.run(args, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["apply", "clear"])
    parser.add_argument("fault", choices=PROFILES)
    parser.add_argument("level", nargs="?", choices=["low", "medium", "high"])
    parser.add_argument("--project-name", help="Existing Compose project name")
    args = parser.parse_args()
    if args.action == "apply" and args.level is None:
        parser.error("apply requires low, medium, or high")
    compose = ["docker", "compose", "-f", str(HERE.parent / "docker-compose.yml")]
    if args.project_name:
        compose += ["--project-name", args.project_name]

    def container(service):
        ids = run(*compose, "ps", "-q", service, capture=True).split()
        if len(ids) != 1:
            raise RuntimeError(f"Expected exactly one running {service}; found {len(ids)}")
        return ids[0]

    legacy = container("legacy-core")
    value = PROFILES[args.fault][args.level or "low"]
    if args.fault == "batch-job-stall":
        paused = run("docker", "inspect", "-f", "{{.State.Paused}}", legacy,
                     capture=True).strip() == "true"
        if args.action == "clear":
            if paused:
                run("docker", "unpause", legacy)
            print("batch-job-stall cleared")
            return
        if paused:
            raise RuntimeError("legacy-core is already paused; clear it first")

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            run("docker", "pause", legacy)
            print(f"batch-job-stall active for {value}s", flush=True)
            time.sleep(value)
        finally:
            # A separate clear command may already have resumed it.
            if run("docker", "inspect", "-f", "{{.State.Paused}}", legacy,
                   capture=True).strip() == "true":
                run("docker", "unpause", legacy)
        print("batch-job-stall cleared")
        return

    gateway = container("boundary-gateway")
    legacy_info = json.loads(run("docker", "inspect", legacy, capture=True))[0]
    gateway_info = json.loads(run("docker", "inspect", gateway, capture=True))[0]
    networks = legacy_info["NetworkSettings"]["Networks"]
    shared = set(networks) & set(gateway_info["NetworkSettings"]["Networks"])
    if len(shared) != 1:
        raise RuntimeError("Expected exactly one shared gateway/legacy network")
    peer = networks[shared.pop()]["IPAddress"]
    run("docker", "run", "--rm", "--network", f"container:{gateway}",
        "--cap-drop", "ALL", "--cap-add", "NET_ADMIN", IMAGE,
        args.action, args.fault, peer, str(value))
    print(f"{args.fault} {args.action} complete")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"chaos: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)

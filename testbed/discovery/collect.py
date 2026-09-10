#!/usr/bin/env python3
"""Collect Docker TCP connections through bpftrace and emit observations + graph JSON."""

from __future__ import annotations

import argparse
import json
import selectors
import signal
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, monotonic_ns

from aggregate import build_graph
from identity import DockerIdentityResolver


INFRASTRUCTURE_PORTS = {53: "dns", 123: "ntp"}
ROOT = Path(__file__).resolve().parent
DEFAULT_PROBE = ROOT / "bpftrace" / "tcp_v4_connect.bt"


def timestamp_from_nsecs(nsecs: int, start_wall: datetime, start_mono_ns: int) -> str:
    seconds = start_wall.timestamp() + (nsecs - start_mono_ns) / 1_000_000_000
    return datetime.fromtimestamp(seconds, UTC).isoformat().replace("+00:00", "Z")


def parse_event(line: str) -> tuple[int, int, str, int] | None:
    fields = line.rstrip().split("|")
    if len(fields) != 5 or fields[0] != "EVENT":
        return None
    return int(fields[1]), int(fields[2]), fields[3], int(fields[4])


def normalized_observation(event, resolver, start_wall, start_mono_ns):
    nsecs, pid, destination_ip, destination_port = event
    if destination_port in INFRASTRUCTURE_PORTS:
        return None, INFRASTRUCTURE_PORTS[destination_port]
    source = resolver.source_for_pid(pid)
    destination = resolver.destination_for_ip(destination_ip)
    # A container can restart after the collection snapshot. Refresh only on a
    # miss so normal high-volume collection does not shell out per event.
    if source is None or destination is None:
        resolver.refresh()
        source = resolver.source_for_pid(pid)
        destination = resolver.destination_for_ip(destination_ip)
    if source is None or destination is None:
        return None, "unmapped"
    return {
        "timestamp": timestamp_from_nsecs(nsecs, start_wall, start_mono_ns),
        "source_identity": source.as_dict(),
        "destination_identity": destination.as_dict(),
        "destination_ip": destination_ip,
        "destination_port": destination_port,
        "protocol": "tcp",
        "source_pid": pid,
        "provenance": {"collector": "bpftrace", "probe": "kprobe:tcp_v4_connect", "probe_file": str(DEFAULT_PROBE)},
    }, None


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--node-type", action="append", default=[], metavar="SERVICE=TYPE")
    args = parser.parse_args()
    node_types = {}
    for item in args.node_type:
        service, separator, node_type = item.partition("=")
        if not separator:
            parser.error("--node-type must be SERVICE=TYPE")
        node_types[service] = node_type

    resolver = DockerIdentityResolver()
    start_wall, start_mono_ns = datetime.now(UTC), monotonic_ns()
    process = subprocess.Popen(["bpftrace", "-q", "-B", "line", str(args.probe)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    observations, filtered, seen = [], Counter(), 0
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    selector.register(process.stdout, selectors.EVENT_READ)

    def collect_line(line: str) -> None:
        nonlocal seen
        event = parse_event(line)
        if event is None:
            return
        seen += 1
        observation, reason = normalized_observation(event, resolver, start_wall, start_mono_ns)
        if observation:
            observations.append(observation)
        else:
            filtered[reason or "unknown"] += 1

    try:
        deadline = monotonic() + args.duration
        while monotonic() < deadline:
            for key, _ in selector.select(timeout=max(0, deadline - monotonic())):
                collect_line(key.fileobj.readline())
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        for line in process.stdout:
            collect_line(line)
        stderr = process.stderr.read() if process.stderr else ""
        selector.close()
    if process.returncode not in (0, -signal.SIGINT):
        sys.stderr.write(stderr)
        return process.returncode or 1
    args.observations.parent.mkdir(parents=True, exist_ok=True)
    with args.observations.open("w") as output:
        for observation in observations:
            output.write(json.dumps(observation, sort_keys=True) + "\n")
    graph = build_graph(observations, node_types)
    graph["collection_summary"] = {"events_seen": seen, "observations_written": len(observations), "filtered": dict(sorted(filtered.items()))}
    write_json(args.graph, graph)
    print(json.dumps(graph["collection_summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

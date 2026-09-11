"""Resolve ephemeral Docker network/process identities to Compose service names."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ContainerIdentity:
    container_id: str
    service_name: str
    compose_project: str | None
    pid: int
    ipv4_addresses: tuple[str, ...]

    def as_dict(self) -> dict[str, str | None]:
        return {
            "service_name": self.service_name,
            "compose_project": self.compose_project,
            "container_id": self.container_id,
        }


class DockerIdentityResolver:
    """Takes a fresh Docker metadata snapshot for every collection run."""

    def __init__(self) -> None:
        self.by_ip: dict[str, ContainerIdentity] = {}
        self.by_pid: dict[int, ContainerIdentity] = {}
        self.by_container_id: dict[str, ContainerIdentity] = {}
        self.refresh()

    @staticmethod
    def _docker_json(*args: str) -> object:
        result = subprocess.run(
            ["docker", *args], check=True, capture_output=True, text=True
        )
        return json.loads(result.stdout)

    def refresh(self) -> None:
        self.by_ip.clear()
        self.by_pid.clear()
        self.by_container_id.clear()
        container_ids = subprocess.run(
            ["docker", "ps", "-q"], check=True, capture_output=True, text=True
        ).stdout.split()
        if not container_ids:
            return
        inspected = self._docker_json("inspect", *container_ids)
        for item in inspected:
            labels = item.get("Config", {}).get("Labels", {}) or {}
            service = labels.get("com.docker.compose.service")
            if not service:
                continue
            addresses = tuple(
                network.get("IPAddress")
                for network in item.get("NetworkSettings", {}).get("Networks", {}).values()
                if network.get("IPAddress")
            )
            identity = ContainerIdentity(
                container_id=item["Id"],
                service_name=service,
                compose_project=labels.get("com.docker.compose.project"),
                pid=int(item["State"]["Pid"]),
                ipv4_addresses=addresses,
            )
            self.by_pid[identity.pid] = identity
            self.by_container_id[identity.container_id] = identity
            for address in addresses:
                self.by_ip[address] = identity

    def source_for_pid(self, pid: int) -> ContainerIdentity | None:
        """Resolve Docker's host PID first, then use cgroups for child processes."""
        identity = self.by_pid.get(pid)
        if identity is not None:
            return identity
        try:
            cgroups = Path(f"/proc/{pid}/cgroup").read_text()
        except FileNotFoundError:
            return None
        match = re.search(r"([0-9a-f]{64})", cgroups)
        if not match:
            return None
        return self.by_container_id.get(match.group(1))

    def destination_for_ip(self, address: str) -> ContainerIdentity | None:
        return self.by_ip.get(address)

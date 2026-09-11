"""Aggregate normalized observations into a generation-typed dependency graph."""

from __future__ import annotations

from collections import OrderedDict


NODE_TYPES = {"cloud-native", "boundary-gateway", "legacy"}


def build_graph(observations: list[dict], node_types: dict[str, str] | None = None) -> dict:
    node_types = node_types or {}
    invalid = set(node_types.values()) - NODE_TYPES
    if invalid:
        raise ValueError(f"unsupported node types: {sorted(invalid)}")

    nodes: dict[str, dict] = {}
    edges: OrderedDict[tuple[str, str, str, int], dict] = OrderedDict()
    for observation in observations:
        source = observation["source_identity"]["service_name"]
        destination = observation["destination_identity"]["service_name"]
        for service in (source, destination):
            nodes.setdefault(service, {"id": service, "node_type": node_types.get(service, "cloud-native")})
        key = (source, destination, observation["protocol"], observation["destination_port"])
        edge = edges.get(key)
        if edge is None:
            edge = {
                "source": source, "destination": destination,
                "protocol": observation["protocol"], "destination_port": observation["destination_port"],
                "observation_count": 0, "first_seen": observation["timestamp"],
                "last_seen": observation["timestamp"], "provenance": [],
            }
            edges[key] = edge
        edge["observation_count"] += 1
        edge["first_seen"] = min(edge["first_seen"], observation["timestamp"])
        edge["last_seen"] = max(edge["last_seen"], observation["timestamp"])
        if observation["provenance"] not in edge["provenance"]:
            edge["provenance"].append(observation["provenance"])
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}

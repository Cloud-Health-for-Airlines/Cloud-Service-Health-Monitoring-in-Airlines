"""Generation-typed heterogeneous graph schema and data structures for BACCP.

Represents distributed airline IT topologies containing:
- Node generations: legacy, boundary-gateway, cloud-native
- Generation-distinguished relation types:
    legacy <-> boundary-gateway
    boundary-gateway <-> cloud-native
    cloud-native <-> cloud-native
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import torch

NODE_TYPES: Set[str] = {"legacy", "boundary-gateway", "cloud-native"}

# Canonical relation triples: (source_node_type, relation_name, destination_node_type)
EDGE_RELATIONS: List[Tuple[str, str, str]] = [
    ("legacy", "sync_to", "boundary-gateway"),
    ("boundary-gateway", "sync_from", "legacy"),
    ("boundary-gateway", "route_to", "cloud-native"),
    ("cloud-native", "call_gateway", "boundary-gateway"),
    ("cloud-native", "inter_service", "cloud-native"),
]

# Standard 5-node canonical airline IT layout
CANONICAL_SERVICES: Dict[str, str] = {
    "legacy-core": "legacy",
    "boundary-gateway": "boundary-gateway",
    "reservations": "cloud-native",
    "crew": "cloud-native",
    "baggage": "cloud-native",
}


@dataclass
class HeteroGraphData:
    """Explicit PyTorch-compatible generation-typed heterogeneous graph snapshot.
    
    Contains node features partitioned by generation type, relation-typed edge indices,
    and generation-gap edge features. Compatible with both PyG HeteroData and custom GNN layers.
    """
    # x_dict: node_type -> Tensor of shape [num_nodes_of_type, node_feat_dim]
    x_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # edge_index_dict: (src_type, rel, dst_type) -> LongTensor of shape [2, num_edges]
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = field(default_factory=dict)
    
    # edge_attr_dict: (src_type, rel, dst_type) -> Tensor of shape [num_edges, edge_feat_dim]
    edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = field(default_factory=dict)
    
    # Node name to local typed index mapping: node_type -> {node_name: local_idx}
    node_maps: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Global flat node index mapping: {node_name: global_idx}
    global_node_map: Dict[str, int] = field(default_factory=dict)
    
    # Timestamp of snapshot
    timestamp: float = 0.0

    def to(self, device: torch.device) -> HeteroGraphData:
        return HeteroGraphData(
            x_dict={k: v.to(device) for k, v in self.x_dict.items()},
            edge_index_dict={k: v.to(device) for k, v in self.edge_index_dict.items()},
            edge_attr_dict={k: v.to(device) for k, v in self.edge_attr_dict.items()},
            node_maps=self.node_maps,
            global_node_map=self.global_node_map,
            timestamp=self.timestamp,
        )


class GenerationTypedGraph:
    """Builder and manager for generation-typed heterogeneous graphs."""

    def __init__(self, node_types: Optional[Dict[str, str]] = None):
        self.node_types = node_types or dict(CANONICAL_SERVICES)
        for ntype in self.node_types.values():
            if ntype not in NODE_TYPES:
                raise ValueError(f"Unsupported node type '{ntype}'. Allowed: {NODE_TYPES}")

        # Partition nodes by generation type
        self.typed_nodes: Dict[str, List[str]] = {nt: [] for nt in NODE_TYPES}
        for name, nt in sorted(self.node_types.items()):
            self.typed_nodes[nt].append(name)

        # Index maps
        self.node_maps: Dict[str, Dict[str, int]] = {}
        for nt, names in self.typed_nodes.items():
            self.node_maps[nt] = {name: idx for idx, name in enumerate(names)}

        self.global_node_map: Dict[str, int] = {
            name: idx for idx, name in enumerate(sorted(self.node_types.keys()))
        }

    def get_relation_type(self, src: str, dst: str) -> Optional[Tuple[str, str, str]]:
        """Map source and destination service names to their typed relation triple."""
        src_t = self.node_types.get(src)
        dst_t = self.node_types.get(dst)
        if not src_t or not dst_t:
            return None

        if src_t == "legacy" and dst_t == "boundary-gateway":
            return ("legacy", "sync_to", "boundary-gateway")
        elif src_t == "boundary-gateway" and dst_t == "legacy":
            return ("boundary-gateway", "sync_from", "legacy")
        elif src_t == "boundary-gateway" and dst_t == "cloud-native":
            return ("boundary-gateway", "route_to", "cloud-native")
        elif src_t == "cloud-native" and dst_t == "boundary-gateway":
            return ("cloud-native", "call_gateway", "boundary-gateway")
        elif src_t == "cloud-native" and dst_t == "cloud-native":
            return ("cloud-native", "inter_service", "cloud-native")
        return None

    def generation_gap_distance(self, src: str, dst: str) -> float:
        """Compute the architectural generation gap between two services.
        
        0.0 = same generation (cloud-cloud)
        0.5 = modern cloud to boundary gateway
        1.0 = boundary gateway to legacy mainframe
        1.5 = direct legacy to cloud cross-boundary
        """
        src_t = self.node_types.get(src, "cloud-native")
        dst_t = self.node_types.get(dst, "cloud-native")
        ranks = {"legacy": 0, "boundary-gateway": 1, "cloud-native": 2}
        return abs(ranks[src_t] - ranks[dst_t]) * 0.5

    def build_snapshot(
        self,
        node_metrics: Dict[str, Dict[str, float]],
        edges: List[Dict[str, Any]],
        timestamp: float = 0.0,
    ) -> HeteroGraphData:
        """Construct a HeteroGraphData snapshot from node metrics and edge observations.
        
        Node features (dim=6):
        [request_rate, error_rate, latency_ms_p95, queue_depth, cpu_load, sync_drift_score]
        
        Edge attributes (dim=3):
        [generation_gap, observation_count, protocol_is_tcp]
        """
        # Build node feature tensors per generation
        x_dict: Dict[str, torch.Tensor] = {}
        for nt, names in self.typed_nodes.items():
            if not names:
                x_dict[nt] = torch.zeros((0, 6), dtype=torch.float32)
                continue
            feats = []
            for name in names:
                m = node_metrics.get(name, {})
                f = [
                    float(m.get("request_rate", 10.0)),
                    float(m.get("error_rate", 0.0)),
                    float(m.get("latency_ms", 15.0)),
                    float(m.get("queue_depth", 0.0)),
                    float(m.get("cpu_load", 0.2)),
                    float(m.get("sync_drift_score", 12.0)),
                ]
                feats.append(f)
            x_dict[nt] = torch.tensor(feats, dtype=torch.float32)

        # Collect edges by typed relation
        edge_lists: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {
            rel: [] for rel in EDGE_RELATIONS
        }
        edge_attrs: Dict[Tuple[str, str, str], List[List[float]]] = {
            rel: [] for rel in EDGE_RELATIONS
        }

        for edge in edges:
            src = edge.get("source")
            dst = edge.get("destination") or edge.get("target")
            if not src or not dst:
                continue
            rel = self.get_relation_type(src, dst)
            if not rel:
                continue

            src_type, _, dst_type = rel
            src_idx = self.node_maps[src_type].get(src)
            dst_idx = self.node_maps[dst_type].get(dst)
            if src_idx is None or dst_idx is None:
                continue

            edge_lists[rel].append((src_idx, dst_idx))
            gap = self.generation_gap_distance(src, dst)
            obs_count = float(edge.get("observation_count", 1))
            is_tcp = 1.0 if edge.get("protocol", "tcp").lower() == "tcp" else 0.0
            edge_attrs[rel].append([gap, obs_count, is_tcp])

        # Convert to tensors
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}

        for rel, elist in edge_lists.items():
            if elist:
                edge_index_dict[rel] = torch.tensor(elist, dtype=torch.long).t().contiguous()
                edge_attr_dict[rel] = torch.tensor(edge_attrs[rel], dtype=torch.float32)
            else:
                edge_index_dict[rel] = torch.empty((2, 0), dtype=torch.long)
                edge_attr_dict[rel] = torch.empty((0, 3), dtype=torch.float32)

        return HeteroGraphData(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            node_maps=self.node_maps,
            global_node_map=self.global_node_map,
            timestamp=timestamp,
        )

    def to_flat_tensors(self, snapshot: HeteroGraphData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten heterogeneous graph into a single homogeneous (flat) graph for Tier 0 GAT.
        
        Returns:
            flat_x: [N, node_feat_dim] where N = total nodes
            flat_edge_index: [2, E] with global node indices
            flat_edge_attr: [E, edge_feat_dim] containing generation gap scalar
        """
        N = len(self.global_node_map)
        feat_dim = 6
        flat_x = torch.zeros((N, feat_dim), dtype=torch.float32)

        # Place typed node features into global matrix
        for nt, local_map in self.node_maps.items():
            x_t = snapshot.x_dict.get(nt)
            if x_t is not None and x_t.shape[0] > 0:
                for name, local_idx in local_map.items():
                    global_idx = self.global_node_map[name]
                    flat_x[global_idx] = x_t[local_idx]

        # Accumulate edges
        flat_edges: List[Tuple[int, int]] = []
        flat_attrs: List[List[float]] = []

        for (src_t, rel, dst_t), edge_idx in snapshot.edge_index_dict.items():
            if edge_idx.shape[1] == 0:
                continue
            attrs = snapshot.edge_attr_dict[(src_t, rel, dst_t)]
            src_rev_map = {idx: name for name, idx in self.node_maps[src_t].items()}
            dst_rev_map = {idx: name for name, idx in self.node_maps[dst_t].items()}

            for e_i in range(edge_idx.shape[1]):
                s_local = int(edge_idx[0, e_i].item())
                d_local = int(edge_idx[1, e_i].item())
                s_name = src_rev_map[s_local]
                d_name = dst_rev_map[d_local]
                s_global = self.global_node_map[s_name]
                d_global = self.global_node_map[d_name]
                flat_edges.append((s_global, d_global))
                flat_attrs.append(attrs[e_i].tolist())

        if flat_edges:
            flat_edge_index = torch.tensor(flat_edges, dtype=torch.long).t().contiguous()
            flat_edge_attr = torch.tensor(flat_attrs, dtype=torch.float32)
        else:
            flat_edge_index = torch.empty((2, 0), dtype=torch.long)
            flat_edge_attr = torch.empty((0, 3), dtype=torch.float32)

        return flat_x, flat_edge_index, flat_edge_attr

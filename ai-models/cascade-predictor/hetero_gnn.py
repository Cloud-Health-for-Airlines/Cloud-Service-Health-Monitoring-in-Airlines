"""Tier 1A: Generation-Typed Relational Graph Convolutional Network (RGCN).

Directly addresses the fundamental literature gap identified in RESEARCH_GAP.md:
Prior GNN failure predictors treat cloud-legacy architectures as homogeneous graphs
with shared message-passing matrices, washing out the distinct queueing and latency
physics of legacy mainframe links.

This model employs relation-specific parameter matrices W_r for:
1. legacy -> boundary-gateway (sync_to)
2. boundary-gateway -> legacy (sync_from)
3. boundary-gateway -> cloud-native (route_to)
4. cloud-native -> boundary-gateway (call_gateway)
5. cloud-native -> cloud-native (inter_service)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from importlib import import_module
import sys
from pathlib import Path
AI_MODELS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_MODELS_DIR))

schema_mod = import_module("graph-builder.schema")
EDGE_RELATIONS = schema_mod.EDGE_RELATIONS
HeteroGraphData = schema_mod.HeteroGraphData


class RelationalConvLayer(nn.Module):
    """Relational Graph Convolutional Network (RGCN) layer over heterogeneous relation triples."""

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 64,
        relations: Optional[List[Tuple[str, str, str]]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relations = relations or EDGE_RELATIONS

        # Relation-specific transformation matrices W_r
        # Store as nn.ModuleDict with sanitized string keys
        self.weight_dict = nn.ModuleDict({
            f"{s}__{r}__{d}": nn.Linear(in_dim, out_dim, bias=False)
            for (s, r, d) in self.relations
        })

        # Self-loop transformation W_0
        self.root_transform = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Forward relational message passing.
        
        Returns:
            out_x_dict: updated node features per generation type
            relation_energy: normalized message volume per relation type (for explainability)
        """
        out_dict: Dict[str, torch.Tensor] = {}
        relation_energy: Dict[str, float] = {}

        # 1. Initialize with self-loop transformations
        for node_type, x in x_dict.items():
            if x.shape[0] > 0:
                out_dict[node_type] = self.root_transform(x)
            else:
                out_dict[node_type] = torch.empty((0, self.out_dim), device=x.device)

        # 2. Aggregate relation-specific messages
        for (src_type, rel_name, dst_type) in self.relations:
            key = f"{src_type}__{rel_name}__{dst_type}"
            edge_index = edge_index_dict.get((src_type, rel_name, dst_type))

            if edge_index is None or edge_index.shape[1] == 0:
                relation_energy[key] = 0.0
                continue

            src_x = x_dict.get(src_type)
            if src_x is None or src_x.shape[0] == 0:
                relation_energy[key] = 0.0
                continue

            # Linear projection via relation-specific matrix W_r
            w_layer = self.weight_dict[key]
            transformed_src = w_layer(src_x)  # [N_src, out_dim]

            src_idx = edge_index[0]
            dst_idx = edge_index[1]

            # Gather messages
            messages = transformed_src[src_idx]  # [E, out_dim]

            # Degree normalization
            dst_count = out_dict[dst_type].shape[0]
            deg = torch.zeros(dst_count, device=messages.device).scatter_add_(
                0, dst_idx, torch.ones_like(dst_idx, dtype=torch.float32)
            ).clamp(min=1.0)
            norm_messages = messages / deg[dst_idx].unsqueeze(-1)

            # Scatter add to destination nodes
            msg_agg = torch.zeros((dst_count, self.out_dim), device=messages.device)
            msg_agg.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, self.out_dim), norm_messages)

            out_dict[dst_type] = out_dict[dst_type] + msg_agg
            relation_energy[key] = float(norm_messages.norm(p=2).item())

        # 3. Activation & Dropout
        for node_type in out_dict:
            out_dict[node_type] = self.dropout(F.relu(out_dict[node_type]))

        return out_dict, relation_energy


class HeteroCascadePredictor(nn.Module):
    """Tier 1A: Heterogeneous RGCN with GRU temporal sequence aggregator."""

    def __init__(
        self,
        node_in_dim: int = 6,
        hidden_dim: int = 64,
        gru_hidden_dim: int = 48,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_hidden_dim = gru_hidden_dim

        # Input projections per generation type to shared hidden_dim with LayerNorm
        self.input_projections = nn.ModuleDict({
            "legacy": nn.Sequential(nn.Linear(node_in_dim, hidden_dim), nn.LayerNorm(hidden_dim)),
            "boundary-gateway": nn.Sequential(nn.Linear(node_in_dim, hidden_dim), nn.LayerNorm(hidden_dim)),
            "cloud-native": nn.Sequential(nn.Linear(node_in_dim, hidden_dim), nn.LayerNorm(hidden_dim)),
        })

        # Relational convolution layers
        self.convs = nn.ModuleList([
            RelationalConvLayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Sequence modeling over temporal snapshots
        # Graph embedding = concatenate mean pool across each generation type (3 * hidden_dim)
        graph_embed_dim = 3 * hidden_dim
        self.gru = nn.GRU(
            input_size=graph_embed_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Output classifier for cascade probability
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def encode_hetero_snapshot(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Encode a single generation-typed heterogeneous snapshot."""
        # 1. Project inputs
        h_dict = {}
        for nt, x in x_dict.items():
            if x.shape[0] > 0:
                h_dict[nt] = self.input_projections[nt](x)
            else:
                h_dict[nt] = torch.empty((0, self.hidden_dim), device=x.device)

        # 2. Relational convolutions
        total_energy: Dict[str, float] = {}
        for conv in self.convs:
            h_dict, energy = conv(h_dict, edge_index_dict)
            for k, v in energy.items():
                total_energy[k] = total_energy.get(k, 0.0) + v

        # 3. Generation-partitioned readout
        pools = []
        for nt in ["legacy", "boundary-gateway", "cloud-native"]:
            if nt in h_dict and h_dict[nt].shape[0] > 0:
                pools.append(h_dict[nt].mean(dim=0, keepdim=True))
            else:
                pools.append(torch.zeros((1, self.hidden_dim), device=next(self.parameters()).device))

        graph_embedding = torch.cat(pools, dim=-1)  # [1, 3 * hidden_dim]
        return graph_embedding, total_energy

    def forward(
        self,
        snapshot_sequence: List[Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Process sequence of heterogeneous snapshots.
        
        Returns:
            prob: cascade probability in [0.0, 1.0]
            shared_representation: GRU hidden vector [1, gru_hidden_dim] (feeds Tier 1C multi-task head)
            relation_attributions: normalized message energies per relation type (Tier 1D explainability)
        """
        embeddings = []
        cumulative_energy: Dict[str, float] = {}

        for x_dict, edge_index_dict in snapshot_sequence:
            embed, energy = self.encode_hetero_snapshot(x_dict, edge_index_dict)
            embeddings.append(embed)
            for k, v in energy.items():
                cumulative_energy[k] = cumulative_energy.get(k, 0.0) + v

        seq_tensor = torch.stack(embeddings, dim=1)  # [1, T, graph_embed_dim]
        gru_out, _ = self.gru(seq_tensor)
        shared_repr = gru_out[:, -1, :]  # [1, gru_hidden_dim]

        logits = self.classifier(shared_repr)
        prob = torch.sigmoid(logits).squeeze()

        # Normalize energy into percentage distribution for explainability
        total = sum(cumulative_energy.values()) + 1e-8
        normalized_attributions = {k: round(v / total, 4) for k, v in cumulative_energy.items()}

        return prob, shared_repr, normalized_attributions

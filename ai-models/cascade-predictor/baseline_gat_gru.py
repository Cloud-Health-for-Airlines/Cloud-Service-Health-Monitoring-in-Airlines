"""Tier 0 Baseline Model: GAT with generation-gap edge features + GRU temporal encoder.

As specified in BACCP Tier 0:
- Graph Attention Network (GAT) layer processing snapshots
- Generation-gap treated naively as an edge feature
- GRU sequence model over sliding temporal snapshots
- Binary cascade probability prediction head
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeFeaturedGATLayer(nn.Module):
    """Graph Attention Network (GAT) layer that incorporates edge features.
    
    In Tier 0, the architectural generation gap between nodes is passed as an edge
    scalar feature into the attention coefficient calculation.
    """

    def __init__(
        self,
        in_node_feats: int = 6,
        in_edge_feats: int = 3,
        out_feats: int = 32,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.head_dim = out_feats // num_heads

        self.lin_node = nn.Linear(in_node_feats, out_feats, bias=False)
        self.lin_edge = nn.Linear(in_edge_feats, out_feats, bias=False)

        # Attention vector: [head_dim + head_dim + head_dim]
        self.attn_vec = nn.Parameter(torch.empty(num_heads, 3 * self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [N, in_node_feats]
        edge_index: [2, E] (src, dst)
        edge_attr: [E, in_edge_feats]
        
        Returns:
            out_x: [N, out_feats]
        """
        N = x.size(0)
        E = edge_index.size(1)

        # Node projection: [N, num_heads, head_dim]
        h_node = self.lin_node(x).view(N, self.num_heads, self.head_dim)

        if E == 0:
            # Degenerate case (isolated nodes)
            return h_node.view(N, self.out_feats)

        # Edge projection: [E, num_heads, head_dim]
        h_edge = self.lin_edge(edge_attr).view(E, self.num_heads, self.head_dim)

        src, dst = edge_index[0], edge_index[1]
        h_src = h_node[src]  # [E, num_heads, head_dim]
        h_dst = h_node[dst]  # [E, num_heads, head_dim]

        # Concatenate [h_src, h_dst, h_edge] -> [E, num_heads, 3 * head_dim]
        cat_features = torch.cat([h_src, h_dst, h_edge], dim=-1)

        # Dot product with attention vector: [E, num_heads]
        score = (cat_features * self.attn_vec.unsqueeze(0)).sum(dim=-1)
        alpha = self.leaky_relu(score)

        # Softmax over incoming edges for each target node
        # Compute numerically stable softmax per head
        alpha_exp = torch.exp(alpha - alpha.max(dim=0, keepdim=True)[0])
        denom = torch.zeros(N, self.num_heads, device=x.device).scatter_add_(
            0, dst.unsqueeze(-1).expand(-1, self.num_heads), alpha_exp
        ) + 1e-8
        weights = alpha_exp / denom[dst]
        weights = self.dropout(weights)

        # Message aggregation: [E, num_heads, head_dim] * [E, num_heads, 1]
        messages = h_src * weights.unsqueeze(-1)
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand(-1, self.num_heads, self.head_dim), messages)

        # Residual connection + Flatten heads
        out = out.view(N, self.out_feats) + self.lin_node(x)
        return F.elu(out)


class BaselineGATGRU(nn.Module):
    """Tier 0 Baseline: Temporal GAT + GRU cascade predictor.
    
    Processes a sequence of T graph snapshots to predict cascade probability.
    """

    def __init__(
        self,
        in_node_feats: int = 6,
        in_edge_feats: int = 3,
        gnn_hidden_dim: int = 64,
        gru_hidden_dim: int = 48,
        num_heads: int = 2,
        num_nodes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gru_hidden_dim = gru_hidden_dim

        # Snapshot GAT encoder
        self.gat1 = EdgeFeaturedGATLayer(
            in_node_feats=in_node_feats,
            in_edge_feats=in_edge_feats,
            out_feats=gnn_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.gat2 = EdgeFeaturedGATLayer(
            in_node_feats=gnn_hidden_dim,
            in_edge_feats=in_edge_feats,
            out_feats=gnn_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Graph pooling: mean + max pooling produces 2 * gnn_hidden_dim
        graph_embed_dim = 2 * gnn_hidden_dim

        # Temporal sequence modeling
        self.gru = nn.GRU(
            input_size=graph_embed_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Prediction head
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def encode_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a single graph snapshot into a vector embedding."""
        h = self.gat1(x, edge_index, edge_attr)
        h = self.gat2(h, edge_index, edge_attr)
        # Graph readout: concat mean and max pool across all nodes
        mean_pool = h.mean(dim=0, keepdim=True)
        max_pool = h.max(dim=0, keepdim=True)[0]
        return torch.cat([mean_pool, max_pool], dim=-1)  # [1, 2 * gnn_hidden_dim]

    def forward(
        self,
        snapshot_sequence: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Process temporal sequence of T graph snapshots.
        
        snapshot_sequence: list of T tuples (x, edge_index, edge_attr)
        
        Returns:
            prob: scalar tensor in [0.0, 1.0] representing cascade probability
        """
        embeddings = []
        for x, edge_index, edge_attr in snapshot_sequence:
            embed = self.encode_snapshot(x, edge_index, edge_attr)
            embeddings.append(embed)

        # Sequence tensor: [1, T, graph_embed_dim]
        seq_tensor = torch.stack(embeddings, dim=1)
        gru_out, _ = self.gru(seq_tensor)
        
        # Last time step hidden state
        last_hidden = gru_out[:, -1, :]  # [1, gru_hidden_dim]
        logits = self.classifier(last_hidden)
        prob = torch.sigmoid(logits).squeeze()
        return prob

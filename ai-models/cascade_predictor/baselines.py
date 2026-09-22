"""Baseline Cascade Prediction Models for Experimental Comparison.

Implements:
1. FlatGraphCascadePredictor (Baseline 1): Homogeneous GCN, no generation or relation types, no boundary sync drift.
2. DomainTypedCascadePredictor (Baseline 2): Domain-typed RGCN, typed strictly by application domain, without technology-generation boundary awareness and without sync drift score.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_DIR = ROOT / "ai-models/cascade_predictor"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PREDICTOR_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICTOR_DIR))

try:
    from model import CANONICAL_DST, CANONICAL_SRC, MultiTaskHead, TemporalAggregator
except (ImportError, ModuleNotFoundError):
    from .model import CANONICAL_DST, CANONICAL_SRC, MultiTaskHead, TemporalAggregator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baccp.baselines")


# =====================================================================
# Baseline 1: Flat / Homogeneous Graph Model
# =====================================================================
class FlatGCNLayer(nn.Module):
    """Homogeneous GCN Layer with single shared weight matrix and no relation types."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.self_weight = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: [B, num_nodes, in_features]"""
        batch_size, num_nodes, _ = x.shape
        out = self.self_weight(x)

        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg = (deg + 1.0).view(1, num_nodes, 1)

        msg = self.linear(x[:, src, :])  # [B, num_edges, out_features]
        for idx, d_node in enumerate(dst):
            out[:, d_node:d_node + 1, :] += msg[:, idx:idx + 1, :]

        out = out / deg
        out = self.norm(out)
        out = F.relu(out)
        return self.dropout(out)


class FlatGraphCascadePredictor(nn.Module):
    """Baseline 1: Homogeneous GCN + Temporal GRU (no relation/generation types, no sync drift)."""

    def __init__(self, node_in_dim: int = 4, hidden_dim: int = 64, gru_layers: int = 2):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim

        self.conv1 = FlatGCNLayer(node_in_dim, hidden_dim)
        self.conv2 = FlatGCNLayer(hidden_dim, hidden_dim)
        self.pool_gate = nn.Linear(hidden_dim, 1)

        self.temporal = TemporalAggregator(hidden_dim=hidden_dim, num_layers=gru_layers)
        self.heads = MultiTaskHead(hidden_dim=hidden_dim, num_nodes=5, num_severities=5)

        self.register_buffer("edge_index", torch.tensor([CANONICAL_SRC, CANONICAL_DST], dtype=torch.long))

    def forward_tensors(self, x_seq: torch.Tensor, drift_seq: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Strip generation one-hot: only use 4 telemetry features [latency, rate, error, cpu]
        x_flat_telemetry = x_seq[:, :, :, :self.node_in_dim]
        B, T, N, D = x_flat_telemetry.shape

        x_flat = x_flat_telemetry.view(B * T, N, D)
        h1 = self.conv1(x_flat, self.edge_index)
        h2 = self.conv2(h1, self.edge_index)

        weights = F.softmax(self.pool_gate(h2), dim=1)
        g_embeds = torch.sum(h2 * weights, dim=1).view(B, T, self.hidden_dim)

        h_temp = self.temporal(g_embeds)
        return self.heads(h_temp)

    def predict_inference(self, snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.eval()
        device = self.edge_index.device
        with torch.no_grad():
            T = len(snapshots)
            x_list = [snap["node_features"] for snap in snapshots]
            x_tensor = torch.tensor(x_list, dtype=torch.float, device=device).unsqueeze(0)

            out = self.forward_tensors(x_tensor)
            prob = float(out["cascade_prob"][0].cpu().item())
            lead_time = float(out["lead_time"][0].cpu().item())
            root_idx = int(torch.argmax(out["root_cause_logits"][0]).cpu().item())
            sev_idx = int(torch.argmax(out["severity_logits"][0]).cpu().item())

            node_names = ["reservations", "crew", "baggage", "boundary-gateway", "legacy-core"]
            sev_names = ["NOMINAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

            margin = 0.15
            return {
                "cascade_probability": round(prob, 4),
                "estimated_lead_time_seconds": round(lead_time, 1),
                "predicted_root_cause_node": node_names[root_idx],
                "severity": sev_names[sev_idx],
                "severity_index": sev_idx,
                "confidence_interval_90": [max(0.0, round(prob - margin, 4)), min(1.0, round(prob + margin, 4))],
                "model_architecture": "Baseline1-FlatGCN-GRU",
            }


# =====================================================================
# Baseline 2: Domain-Typed Graph Model (No generation types, no sync drift)
# =====================================================================
class DomainRGCNLayer(nn.Module):
    """RGCN Layer where relations are typed by operational domain only (no drift bias)."""

    def __init__(self, in_features: int, out_features: int, num_relations: int = 5, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations

        self.relation_weights = nn.Parameter(torch.empty(num_relations, in_features, out_features))
        self.self_weight = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_weights)
        self.self_weight.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        out = self.self_weight(x)

        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg = (deg + 1.0).view(1, num_nodes, 1)

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if not mask.any():
                continue
            r_src = src[mask]
            r_dst = dst[mask]

            msg = torch.matmul(x[:, r_src, :], self.relation_weights[r])
            for idx, d_node in enumerate(r_dst):
                out[:, d_node:d_node + 1, :] += msg[:, idx:idx + 1, :]

        out = out / deg
        out = self.norm(out)
        out = F.relu(out)
        return self.dropout(out)


class DomainTypedCascadePredictor(nn.Module):
    """Baseline 2: Domain-typed RGCN + Temporal GRU (types domains, but NO generation boundary & NO drift)."""

    def __init__(self, node_in_dim: int = 9, hidden_dim: int = 64, num_relations: int = 5, gru_layers: int = 2):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim

        self.conv1 = DomainRGCNLayer(node_in_dim, hidden_dim, num_relations)
        self.conv2 = DomainRGCNLayer(hidden_dim, hidden_dim, num_relations)
        self.pool_gate = nn.Linear(hidden_dim, 1)

        self.temporal = TemporalAggregator(hidden_dim=hidden_dim, num_layers=gru_layers)
        self.heads = MultiTaskHead(hidden_dim=hidden_dim, num_nodes=5, num_severities=5)

        self.register_buffer("edge_index", torch.tensor([CANONICAL_SRC, CANONICAL_DST], dtype=torch.long))
        # Domain relations: 0: res, 1: crew, 2: bag, 3: gateway, 4: mainframe
        domain_relations = [0, 3, 1, 3, 2, 3, 3, 4, 0, 1, 0, 2]
        self.register_buffer("edge_type", torch.tensor(domain_relations, dtype=torch.long))

    def _add_domain_features(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Replaces generation one-hot (3) with domain one-hot (5 nodes)."""
        B, T, N, _ = x_seq.shape
        telemetry = x_seq[:, :, :, :4]  # [B, T, 5, 4]
        # 5-node domain identity
        domain_eye = torch.eye(5, device=x_seq.device).view(1, 1, 5, 5).expand(B, T, 5, 5)
        return torch.cat([telemetry, domain_eye], dim=-1)  # [B, T, 5, 9]

    def forward_tensors(self, x_seq: torch.Tensor, drift_seq: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x_domain = self._add_domain_features(x_seq)
        B, T, N, D = x_domain.shape

        x_flat = x_domain.view(B * T, N, D)
        h1 = self.conv1(x_flat, self.edge_index, self.edge_type)
        h2 = self.conv2(h1, self.edge_index, self.edge_type)

        weights = F.softmax(self.pool_gate(h2), dim=1)
        g_embeds = torch.sum(h2 * weights, dim=1).view(B, T, self.hidden_dim)

        h_temp = self.temporal(g_embeds)
        return self.heads(h_temp)

    def predict_inference(self, snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.eval()
        device = self.edge_index.device
        with torch.no_grad():
            x_list = [snap["node_features"] for snap in snapshots]
            x_tensor = torch.tensor(x_list, dtype=torch.float, device=device).unsqueeze(0)

            out = self.forward_tensors(x_tensor)
            prob = float(out["cascade_prob"][0].cpu().item())
            lead_time = float(out["lead_time"][0].cpu().item())
            root_idx = int(torch.argmax(out["root_cause_logits"][0]).cpu().item())
            sev_idx = int(torch.argmax(out["severity_logits"][0]).cpu().item())

            node_names = ["reservations", "crew", "baggage", "boundary-gateway", "legacy-core"]
            sev_names = ["NOMINAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

            margin = 0.14
            return {
                "cascade_probability": round(prob, 4),
                "estimated_lead_time_seconds": round(lead_time, 1),
                "predicted_root_cause_node": node_names[root_idx],
                "severity": sev_names[sev_idx],
                "severity_index": sev_idx,
                "confidence_interval_90": [max(0.0, round(prob - margin, 4)), min(1.0, round(prob + margin, 4))],
                "model_architecture": "Baseline2-DomainTyped-GRU",
            }


# =====================================================================
# Training Routines for Baselines
# =====================================================================
def train_baseline(
    model: nn.Module,
    train_data: List[Any],
    val_data: List[Any],
    epochs: int = 35,
    lr: float = 0.001,
    patience: int = 8,
    checkpoint_path: str = "",
) -> nn.Module:
    device = torch.device("cpu")
    model.to(device)

    n_pos = sum(1 for s in train_data if s.cascade == 1.0)
    pos_weight = torch.tensor([(len(train_data) - n_pos) / max(1, n_pos)], dtype=torch.float, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    smooth_l1 = nn.SmoothL1Loss()
    ce = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_weights = None
    epochs_no_improve = 0

    batch_size = 32
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            b_x = torch.stack([s.x_seq for s in batch]).to(device)

            targets = {
                "cascade": torch.tensor([s.cascade for s in batch], dtype=torch.float, device=device),
                "lead_time": torch.tensor([s.lead_time for s in batch], dtype=torch.float, device=device),
                "root_cause": torch.tensor([s.root_cause for s in batch], dtype=torch.long, device=device),
                "severity": torch.tensor([s.severity for s in batch], dtype=torch.long, device=device),
            }

            optimizer.zero_grad()
            outputs = model.forward_tensors(b_x)

            l_casc = bce(outputs["cascade_logits"], targets["cascade"])
            l_lead = smooth_l1(outputs["lead_time_norm"], targets["lead_time"] / 600.0)
            l_root = ce(outputs["root_cause_logits"], targets["root_cause"])
            l_sev = ce(outputs["severity_logits"], targets["severity"])

            loss = l_casc + 0.5 * l_lead + 0.5 * l_root + 0.5 * l_sev
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        # Validation F1
        model.eval()
        with torch.no_grad():
            v_x = torch.stack([s.x_seq for s in val_data]).to(device)
            v_targets = [s.cascade for s in val_data]
            v_out = model.forward_tensors(v_x)
            v_preds = [1 if p >= 0.5 else 0 for p in v_out["cascade_prob"].cpu().tolist()]

            tp = sum(1 for p, t in zip(v_preds, v_targets) if p == 1 and t == 1)
            fp = sum(1 for p, t in zip(v_preds, v_targets) if p == 1 and t == 0)
            fn = sum(1 for p, t in zip(v_preds, v_targets) if p == 0 and t == 1)
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * (prec * rec) / max(1e-6, prec + rec)

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    if checkpoint_path:
        out_p = Path(checkpoint_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_p)
        logger.info(f"Saved baseline checkpoint: {out_p.name} (Val F1: {best_val_f1:.4f})")

    return model


def train_all_baselines(data_path: str = "ai-models/data/dataset.json", seed: int = 42) -> None:
    """Train both baselines on the same training set as Model 3."""
    sys.path.insert(0, str(PREDICTOR_DIR))
    from train import load_and_split_data, set_seed
    set_seed(seed)

    train_data, val_data, _, _ = load_and_split_data(Path(data_path), seed=seed)

    logger.info("Training Baseline 1: Flat/Homogeneous GCN...")
    flat_model = FlatGraphCascadePredictor()
    train_baseline(
        flat_model, train_data, val_data, epochs=30,
        checkpoint_path="ai-models/weights/baseline_flat_graph.pt"
    )

    logger.info("Training Baseline 2: Domain-Typed RGCN...")
    domain_model = DomainTypedCascadePredictor()
    train_baseline(
        domain_model, train_data, val_data, epochs=30,
        checkpoint_path="ai-models/weights/baseline_domain_typed.pt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline models for BACCP.")
    parser.add_argument("--data", type=str, default="ai-models/data/dataset.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_all_baselines(args.data, args.seed)

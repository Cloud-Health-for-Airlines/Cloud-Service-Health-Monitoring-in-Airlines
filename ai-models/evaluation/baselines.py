"""Three-way baseline comparison harness for BACCP.

Directly tests the central hypothesis formulated in RESEARCH_GAP.md and WORK_DISTRIBUTION.md:
Does technology-generation typing outperform domain-typed and flat homogeneous graph architectures?

Compares on the exact same held-out test split:
1. Flat Graph: Homogeneous graph without typing, shared message-passing weights.
2. Domain-Typed Graph: Nodes typed by service domain (reservation, crew, legacy) but without generation-pair relational modeling.
3. Generation-Typed Hetero-RGCN: BACCP relational model with generation-specific weights W_r.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

AI_MODELS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = AI_MODELS_DIR.parent
sys.path.insert(0, str(AI_MODELS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from importlib import import_module
hetero_mod = import_module("cascade-predictor.hetero_gnn")
HeteroCascadePredictor = hetero_mod.HeteroCascadePredictor

DATASET_PATH = AI_MODELS_DIR / "data" / "dataset" / "baccp_dataset.pt"


# -------------------------------------------------------------
# Baseline 1: Flat Homogeneous GNN
# -------------------------------------------------------------
class FlatHomogeneousGNN(nn.Module):
    """Homogeneous GNN ignoring all node/edge generation types."""

    def __init__(self, in_dim: int = 6, hidden_dim: int = 48, gru_hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim * 2, gru_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode_snapshot(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Standard degree-normalized message passing with shared matrix
        h = F.relu(self.conv1(x))
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            deg = torch.zeros(x.size(0), device=x.device).scatter_add_(
                0, dst, torch.ones_like(dst, dtype=torch.float32)
            ).clamp(min=1.0)
            msg = h[src] / deg[dst].unsqueeze(-1)
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, h.size(1)), msg)
            h = F.relu(self.conv2(h + agg))
        else:
            h = F.relu(self.conv2(h))

        return torch.cat([h.mean(dim=0, keepdim=True), h.max(dim=0, keepdim=True)[0]], dim=-1)

    def forward(self, snapshot_sequence: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        embeds = [self.encode_snapshot(x, edge_idx) for x, edge_idx in snapshot_sequence]
        seq = torch.stack(embeds, dim=1)
        gru_out, _ = self.gru(seq)
        logits = self.classifier(gru_out[:, -1, :])
        return torch.sigmoid(logits).squeeze()


# -------------------------------------------------------------
# Baseline 2: Domain-Typed Graph
# -------------------------------------------------------------
class DomainTypedGNN(nn.Module):
    """Domain-typed GNN: distinguishes functional domains (core vs edge) but NOT generation gaps."""

    def __init__(self, in_dim: int = 6, hidden_dim: int = 48, gru_hidden: int = 32):
        super().__init__()
        # Domain embeddings (3 domains: core-db, gateway, business-service)
        self.domain_embed = nn.Embedding(3, in_dim)
        self.conv1 = nn.Linear(in_dim * 2, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim * 2, gru_hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, snapshot_sequence: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        # Domain mapping: legacy=0, gateway=1, cloud-services=2
        domains = torch.tensor([0, 1, 2, 2, 2], dtype=torch.long)
        embeds = []
        for x, edge_idx in snapshot_sequence:
            d_vecs = self.domain_embed(domains)
            x_cat = torch.cat([x, d_vecs], dim=-1)
            h = F.relu(self.conv1(x_cat))
            if edge_idx.shape[1] > 0:
                src, dst = edge_idx[0], edge_idx[1]
                deg = torch.zeros(x.size(0), device=x.device).scatter_add_(
                    0, dst, torch.ones_like(dst, dtype=torch.float32)
                ).clamp(min=1.0)
                msg = h[src] / deg[dst].unsqueeze(-1)
                agg = torch.zeros_like(h)
                agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, h.size(1)), msg)
                h = F.relu(self.conv2(h + agg))
            embeds.append(torch.cat([h.mean(dim=0, keepdim=True), h.max(dim=0, keepdim=True)[0]], dim=-1))

        seq = torch.stack(embeds, dim=1)
        gru_out, _ = self.gru(seq)
        logits = self.classifier(gru_out[:, -1, :])
        return torch.sigmoid(logits).squeeze()


def run_baseline_comparison() -> Dict[str, Any]:
    payload = torch.load(DATASET_PATH, map_location="cpu")
    runs = payload["runs"]
    splits = payload["manifest"]["splits"]
    train_ids = splits["train_run_ids"]
    test_ids = splits["test_run_ids"]

    train_runs = [r for r in runs if r["run_id"] in train_ids]
    test_runs = [r for r in runs if r["run_id"] in test_ids]
    y_test = np.array([r["cascade_occurred"] for r in test_runs])

    # 1. Train Flat GNN
    print("[1/3] Training Flat Homogeneous GNN...")
    flat_model = FlatHomogeneousGNN()
    opt_flat = optim.Adam(flat_model.parameters(), lr=0.005)
    for epoch in range(12):
        flat_model.train()
        for r in train_runs:
            opt_flat.zero_grad()
            indices = np.linspace(0, len(r["snapshots"]) - 1, num=5, dtype=int)
            seq = [(r["snapshots"][i]["flat_x"], r["snapshots"][i]["flat_edge_index"]) for i in indices]
            p = flat_model(seq)
            loss = F.binary_cross_entropy(p, torch.tensor(float(r["cascade_occurred"])))
            loss.backward()
            opt_flat.step()

    # 2. Train Domain-Typed GNN
    print("[2/3] Training Domain-Typed GNN...")
    domain_model = DomainTypedGNN()
    opt_dom = optim.Adam(domain_model.parameters(), lr=0.005)
    for epoch in range(12):
        domain_model.train()
        for r in train_runs:
            opt_dom.zero_grad()
            indices = np.linspace(0, len(r["snapshots"]) - 1, num=5, dtype=int)
            seq = [(r["snapshots"][i]["flat_x"], r["snapshots"][i]["flat_edge_index"]) for i in indices]
            p = domain_model(seq)
            loss = F.binary_cross_entropy(p, torch.tensor(float(r["cascade_occurred"])))
            loss.backward()
            opt_dom.step()

    # 3. Load Trained Generation-Typed Hetero RGCN + MultiTaskHead
    print("[3/3] Evaluating Generation-Typed Hetero RGCN...")
    multi_mod = import_module("cascade-predictor.multi_task_head")
    MultiTaskHead = multi_mod.MultiTaskHead

    tier1_pkg = torch.load(AI_MODELS_DIR / "weights" / "cascade_predictor_tier1a.pt", map_location="cpu")
    hetero_model = HeteroCascadePredictor(node_in_dim=6, hidden_dim=48, gru_hidden_dim=32, num_layers=2)
    hetero_model.load_state_dict(tier1_pkg["encoder"])
    hetero_model.eval()

    head = MultiTaskHead(in_dim=32, num_nodes=5, num_severities=4, hidden_dim=32)
    head.load_state_dict(tier1_pkg["head"])
    head.eval()

    # Evaluate all three
    flat_model.eval()
    domain_model.eval()

    flat_probs, domain_probs, hetero_probs = [], [], []
    with torch.no_grad():
        for r in test_runs:
            indices = np.linspace(0, len(r["snapshots"]) - 1, num=5, dtype=int)
            seq_flat = [(r["snapshots"][i]["flat_x"], r["snapshots"][i]["flat_edge_index"]) for i in indices]
            seq_hetero = [(r["snapshots"][i]["x_dict"], r["snapshots"][i]["edge_index_dict"]) for i in indices]

            flat_probs.append(flat_model(seq_flat).item())
            domain_probs.append(domain_model(seq_flat).item())
            _, shared_repr, _ = hetero_model(seq_hetero)
            preds = head(shared_repr)
            hetero_probs.append(preds["cascade_probability"].item())

    flat_probs = np.array(flat_probs)
    domain_probs = np.array(domain_probs)
    hetero_probs = np.array(hetero_probs)

    def calc_metrics(probs: np.ndarray) -> Dict[str, float]:
        preds = (probs >= 0.5).astype(int)
        return {
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4) if len(np.unique(y_test)) > 1 else 1.0,
        }

    results = {
        "flat_graph": calc_metrics(flat_probs),
        "domain_typed_graph": calc_metrics(domain_probs),
        "generation_typed_hetero_rgcn": calc_metrics(hetero_probs),
    }

    print("\n" + "=" * 65)
    print(" THREE-WAY GRAPH ARCHITECTURE COMPARISON RESULTS")
    print("=" * 65)
    print(f"1. Flat Graph                 | F1: {results['flat_graph']['f1']:.3f} | AUC: {results['flat_graph']['roc_auc']:.3f} | Precision: {results['flat_graph']['precision']:.3f}")
    print(f"2. Domain-Typed Graph         | F1: {results['domain_typed_graph']['f1']:.3f} | AUC: {results['domain_typed_graph']['roc_auc']:.3f} | Precision: {results['domain_typed_graph']['precision']:.3f}")
    print(f"3. Generation-Typed (BACCP)   | F1: {results['generation_typed_hetero_rgcn']['f1']:.3f} | AUC: {results['generation_typed_hetero_rgcn']['roc_auc']:.3f} | Precision: {results['generation_typed_hetero_rgcn']['precision']:.3f}")

    return results


if __name__ == "__main__":
    run_baseline_comparison()

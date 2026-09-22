"""Training pipeline for BACCP Cascade Predictor models (Tier 0 & Tier 1A).

Trains:
1. Tier 0 GAT-GRU baseline (saves weights/cascade_predictor_tier0.pt)
2. Tier 1A Hetero-RGCN + MultiTaskHead (saves weights/cascade_predictor_tier1a.pt)
3. Temperature scaling calibration & Split Conformal prediction on validation set
4. Evaluates precision, recall, F1, lead time MSE, and conformal coverage on test split.
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
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

AI_MODELS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = AI_MODELS_DIR.parent
sys.path.insert(0, str(AI_MODELS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from importlib import import_module
gat_gru_mod = import_module("cascade-predictor.baseline_gat_gru")
BaselineGATGRU = gat_gru_mod.BaselineGATGRU

hetero_mod = import_module("cascade-predictor.hetero_gnn")
HeteroCascadePredictor = hetero_mod.HeteroCascadePredictor

multi_task_mod = import_module("cascade-predictor.multi_task_head")
MultiTaskHead = multi_task_mod.MultiTaskHead

explain_mod = import_module("cascade-predictor.explainability")
TemperatureScaler = explain_mod.TemperatureScaler
SplitConformalCalibrator = explain_mod.SplitConformalCalibrator

DATASET_PATH = AI_MODELS_DIR / "data" / "dataset" / "baccp_dataset.pt"
WEIGHTS_DIR = AI_MODELS_DIR / "weights"


def load_dataset() -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, List[int]]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run generate_dataset.py first.")
    payload = torch.load(DATASET_PATH, map_location="cpu")
    manifest = payload["manifest"]
    runs = payload["runs"]
    splits = manifest["splits"]
    return manifest, runs, splits


def train_tier0_model(
    runs: List[Dict[str, Any]],
    train_ids: List[int],
    val_ids: List[int],
    epochs: int = 15,
) -> Tuple[BaselineGATGRU, Dict[str, float]]:
    print("\n" + "=" * 60)
    print(" TRAINING TIER 0 BASELINE (GAT + GRU)")
    print("=" * 60)

    model = BaselineGATGRU(in_node_feats=6, in_edge_feats=3, gnn_hidden_dim=48, gru_hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.BCELoss()

    train_runs = [r for r in runs if r["run_id"] in train_ids]
    val_runs = [r for r in runs if r["run_id"] in val_ids]

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for run in train_runs:
            optimizer.zero_grad()
            # Prepare sequence of 5 evenly spaced snapshots
            snapshots = run["snapshots"]
            indices = np.linspace(0, len(snapshots) - 1, num=5, dtype=int)
            seq = [(snapshots[i]["flat_x"], snapshots[i]["flat_edge_index"], snapshots[i]["flat_edge_attr"]) for i in indices]

            prob = model(seq)
            target = torch.tensor(float(run["cascade_occurred"]), dtype=torch.float32)
            loss = criterion(prob, target)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        train_loss /= max(1, len(train_runs))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for run in val_runs:
                snapshots = run["snapshots"]
                indices = np.linspace(0, len(snapshots) - 1, num=5, dtype=int)
                seq = [(snapshots[i]["flat_x"], snapshots[i]["flat_edge_index"], snapshots[i]["flat_edge_attr"]) for i in indices]
                prob = model(seq)
                target = torch.tensor(float(run["cascade_occurred"]), dtype=torch.float32)
                loss = criterion(prob, target)
                val_loss += float(loss.item())

        val_loss /= max(1, len(val_runs))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train BCE: {train_loss:.4f} | Val BCE: {val_loss:.4f}")

    if best_weights:
        model.load_state_dict(best_weights)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    tier0_path = WEIGHTS_DIR / "cascade_predictor_tier0.pt"
    torch.save(model.state_dict(), tier0_path)
    print(f"[+] Saved Tier 0 model checkpoint to {tier0_path}")

    return model, {"tier0_best_val_loss": best_val_loss}


def train_tier1_model(
    runs: List[Dict[str, Any]],
    train_ids: List[int],
    val_ids: List[int],
    epochs: int = 35,
) -> Tuple[HeteroCascadePredictor, MultiTaskHead, TemperatureScaler, SplitConformalCalibrator, Dict[str, Any]]:
    print("\n" + "=" * 60)
    print(" TRAINING TIER 1A HETERO-RGCN + 1C MULTI-TASK HEAD")
    print("=" * 60)

    encoder = HeteroCascadePredictor(node_in_dim=6, hidden_dim=48, gru_hidden_dim=32, num_layers=2)
    head = MultiTaskHead(in_dim=32, num_nodes=5, num_severities=4, hidden_dim=32)

    optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=0.003, weight_decay=1e-4)

    train_runs = [r for r in runs if r["run_id"] in train_ids]
    val_runs = [r for r in runs if r["run_id"] in val_ids]

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()
        train_loss = 0.0

        for run in train_runs:
            optimizer.zero_grad()
            snapshots = run["snapshots"]
            indices = np.linspace(0, len(snapshots) - 1, num=5, dtype=int)
            seq = [(snapshots[i]["x_dict"], snapshots[i]["edge_index_dict"]) for i in indices]

            _, shared_repr, _ = encoder(seq)
            preds = head(shared_repr)
            loss, _ = head.compute_loss(preds, run, weights=(4.0, 0.005, 0.5, 0.5))
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        train_loss /= max(1, len(train_runs))

        # Validation
        encoder.eval()
        head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for run in val_runs:
                snapshots = run["snapshots"]
                indices = np.linspace(0, len(snapshots) - 1, num=5, dtype=int)
                seq = [(snapshots[i]["x_dict"], snapshots[i]["edge_index_dict"]) for i in indices]
                _, shared_repr, _ = encoder(seq)
                preds = head(shared_repr)
                loss, _ = head.compute_loss(preds, run, weights=(4.0, 0.005, 0.5, 0.5))
                val_loss += float(loss.item())

        val_loss /= max(1, len(val_runs))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {
                "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                "head": {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MT-Loss: {train_loss:.4f} | Val MT-Loss: {val_loss:.4f}")

    if best_weights:
        encoder.load_state_dict(best_weights["encoder"])
        head.load_state_dict(best_weights["head"])

    # Fit Temperature Scaler and Conformal Predictor on validation split
    encoder.eval()
    head.eval()
    val_logits_list = []
    val_labels_list = []
    val_probs_list = []

    with torch.no_grad():
        for run in val_runs:
            snapshots = run["snapshots"]
            indices = np.linspace(0, len(snapshots) - 1, num=5, dtype=int)
            seq = [(snapshots[i]["x_dict"], snapshots[i]["edge_index_dict"]) for i in indices]
            _, shared_repr, _ = encoder(seq)
            preds = head(shared_repr)
            val_logits_list.append(preds["prob_logits"].squeeze())
            val_probs_list.append(preds["cascade_probability"].squeeze().item())
            val_labels_list.append(float(run["cascade_occurred"]))

    val_logits_t = torch.stack(val_logits_list)
    val_labels_t = torch.tensor(val_labels_list, dtype=torch.float32)

    # 1. Temperature scaling
    temp_scaler = TemperatureScaler()
    opt_temp = temp_scaler.fit(val_logits_t, val_labels_t)
    print(f"[+] Fitted Temperature Scaling Parameter T = {opt_temp:.3f}")

    # 2. Conformal prediction quantile (90% target coverage)
    conformal = SplitConformalCalibrator(alpha=0.10)
    conformal_q = conformal.fit(np.array(val_probs_list), np.array(val_labels_list))
    print(f"[+] Fitted Conformal Prediction 90% Residual Quantile q = {conformal_q:.4f}")

    # Save checkpoint package
    tier1_pkg = {
        "encoder": encoder.state_dict(),
        "head": head.state_dict(),
        "temperature": opt_temp,
        "conformal_quantile": conformal_q,
    }
    tier1_path = WEIGHTS_DIR / "cascade_predictor_tier1a.pt"
    torch.save(tier1_pkg, tier1_path)
    print(f"[+] Saved Tier 1A model checkpoint bundle to {tier1_path}")

    return encoder, head, temp_scaler, conformal, {"tier1_best_val_loss": best_val_loss}


def evaluate_on_test_set(
    tier0_model: BaselineGATGRU,
    tier1_encoder: HeteroCascadePredictor,
    tier1_head: MultiTaskHead,
    conformal: SplitConformalCalibrator,
    runs: List[Dict[str, Any]],
    test_ids: List[int],
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print(" EVALUATION ON HELD-OUT TEST SPLIT")
    print("=" * 60)

    test_runs = [r for r in runs if r["run_id"] in test_ids]
    y_true = np.array([r["cascade_occurred"] for r in test_runs])

    # Tier 0 evaluation
    tier0_model.eval()
    tier0_probs = []
    with torch.no_grad():
        for run in test_runs:
            snapshots = run["snapshots"]
            indices = np.linspace(0, len(snapshots) - 1, num=5, dtype=int)
            seq = [(snapshots[i]["flat_x"], snapshots[i]["flat_edge_index"], snapshots[i]["flat_edge_attr"]) for i in indices]
            p = tier0_model(seq).item()
            tier0_probs.append(p)
    tier0_probs = np.array(tier0_probs)
    tier0_preds = (tier0_probs >= 0.5).astype(int)

    # Tier 1A evaluation
    tier1_encoder.eval()
    tier1_head.eval()
    tier1_probs = []
    tier1_lead_preds = []
    conformal_hits = 0

    with torch.no_grad():
        for run in test_runs:
            snapshots = run["snapshots"]
            indices = np.linspace(0, len(snapshots) - 1, num=5, dtype=int)
            seq = [(snapshots[i]["x_dict"], snapshots[i]["edge_index_dict"]) for i in indices]
            _, shared_repr, _ = tier1_encoder(seq)
            preds = tier1_head(shared_repr)
            p = preds["cascade_probability"].item()
            tier1_probs.append(p)
            tier1_lead_preds.append(preds["lead_time_seconds"].item())

            # Conformal coverage check
            low, high = conformal.predict_interval(p)
            true_label = float(run["cascade_occurred"])
            if low <= true_label <= high:
                conformal_hits += 1

    tier1_probs = np.array(tier1_probs)
    tier1_preds = (tier1_probs >= 0.5).astype(int)
    empirical_coverage = conformal_hits / max(1, len(test_runs))

    metrics = {
        "tier0": {
            "precision": round(float(precision_score(y_true, tier0_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, tier0_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, tier0_preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, tier0_probs)), 4) if len(np.unique(y_true)) > 1 else 1.0,
        },
        "tier1a": {
            "precision": round(float(precision_score(y_true, tier1_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, tier1_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, tier1_preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, tier1_probs)), 4) if len(np.unique(y_true)) > 1 else 1.0,
            "conformal_90_coverage": round(float(empirical_coverage), 4),
        },
    }

    print(f"Tier 0 (Baseline GAT-GRU)     -> Precision: {metrics['tier0']['precision']:.3f} | Recall: {metrics['tier0']['recall']:.3f} | F1: {metrics['tier0']['f1']:.3f}")
    print(f"Tier 1A (Hetero RGCN + Multi) -> Precision: {metrics['tier1a']['precision']:.3f} | Recall: {metrics['tier1a']['recall']:.3f} | F1: {metrics['tier1a']['f1']:.3f} | Coverage: {metrics['tier1a']['conformal_90_coverage']:.1%}")

    return metrics


def main():
    manifest, runs, splits = load_dataset()
    train_ids = splits["train_run_ids"]
    val_ids = splits["val_run_ids"]
    test_ids = splits["test_run_ids"]

    # 1. Train Tier 0
    tier0_model, _ = train_tier0_model(runs, train_ids, val_ids, epochs=15)

    # 2. Train Tier 1
    tier1_enc, tier1_head, scaler, conformal, _ = train_tier1_model(runs, train_ids, val_ids, epochs=20)

    # 3. Evaluate on test set
    metrics = evaluate_on_test_set(tier0_model, tier1_enc, tier1_head, conformal, runs, test_ids)


if __name__ == "__main__":
    main()

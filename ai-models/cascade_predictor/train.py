"""Training pipeline for BACCP Heterogeneous RGCN Cascade Predictor.

Features:
- Deterministic random seeding
- Strict 70/15/15 train/val/test split
- Vectorized batch training with pre-tensorized data
- Class-weighted multi-objective loss
- Early stopping with patience
- Model checkpointing & metadata generation
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PREDICTOR_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICTOR_DIR))

try:
    from model import HeteroCascadePredictor
except (ImportError, ModuleNotFoundError):
    from .model import HeteroCascadePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baccp.train")


class TensorizedSample(NamedTuple):
    x_seq: torch.Tensor       # [T, 5, 7]
    drift_seq: torch.Tensor   # [T]
    cascade: float
    lead_time: float
    root_cause: int
    severity: int


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def tensorize_sample(sample: Dict[str, Any]) -> TensorizedSample:
    snaps = sample["snapshots"]
    x_list = [s["node_features"] for s in snaps]
    drift_list = [s["sync_drift_score"] / 100.0 for s in snaps]

    return TensorizedSample(
        x_seq=torch.tensor(x_list, dtype=torch.float),
        drift_seq=torch.tensor(drift_list, dtype=torch.float),
        cascade=float(sample["cascade_label"]),
        lead_time=float(sample["lead_time_seconds"]),
        root_cause=int(sample["root_cause_node"]),
        severity=int(sample["severity_level"]),
    )


def load_and_split_data(
    data_path: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[TensorizedSample], List[TensorizedSample], List[TensorizedSample], Dict[str, Any]]:
    """Load dataset, tensorize in RAM, and perform strict train/val/test split with zero leakage."""
    raw_text = data_path.read_text()
    data = json.loads(raw_text)
    dataset_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]

    raw_samples = data["samples"]
    total = len(raw_samples)

    # Deterministic shuffle of raw samples
    rng = random.Random(seed)
    shuffled = list(raw_samples)
    rng.shuffle(shuffled)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    raw_train = shuffled[:n_train]
    raw_val = shuffled[n_train:n_train + n_val]
    raw_test = shuffled[n_train + n_val:]

    train_set = [tensorize_sample(s) for s in raw_train]
    val_set = [tensorize_sample(s) for s in raw_val]
    test_set = [tensorize_sample(s) for s in raw_test]

    metadata = {
        "dataset_hash": dataset_hash,
        "total_samples": total,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "feature_metadata": data.get("metadata", {}),
    }

    logger.info(
        f"Dataset split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test "
        f"(total: {total}, hash: {dataset_hash})"
    )
    return train_set, val_set, test_set, metadata


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch_targets: Dict[str, torch.Tensor],
    bce_loss_fn: nn.Module,
    smooth_l1: nn.Module,
    ce_loss: nn.Module,
    weights: Tuple[float, float, float, float] = (1.0, 0.5, 0.5, 0.5),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    w_casc, w_lead, w_root, w_sev = weights

    # 1. Cascade prediction loss
    l_casc = bce_loss_fn(outputs["cascade_logits"], batch_targets["cascade"])

    # 2. Lead-time regression loss (normalized by 600s)
    pred_lead_norm = outputs.get("lead_time_norm", outputs["lead_time"] / 600.0)
    true_lead_norm = batch_targets["lead_time"] / 600.0
    l_lead = smooth_l1(pred_lead_norm, true_lead_norm)

    # 3. Root-cause classification loss
    l_root = ce_loss(outputs["root_cause_logits"], batch_targets["root_cause"])

    # 4. Severity classification loss
    l_sev = ce_loss(outputs["severity_logits"], batch_targets["severity"])

    total_loss = w_casc * l_casc + w_lead * l_lead + w_root * l_root + w_sev * l_sev

    loss_dict = {
        "loss_total": float(total_loss.item()),
        "loss_cascade": float(l_casc.item()),
        "loss_lead_time": float(l_lead.item()),
        "loss_root_cause": float(l_root.item()),
        "loss_severity": float(l_sev.item()),
    }
    return total_loss, loss_dict


def evaluate_split(
    model: HeteroCascadePredictor,
    split_data: List[TensorizedSample],
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evaluate model on a dataset split using vectorized batches."""
    model.eval()

    all_preds_prob = []
    all_targets_casc = []
    lead_time_errors = []
    correct_root = 0
    correct_sev = 0
    total = len(split_data)

    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = split_data[i:i + batch_size]
            b_x = torch.stack([s.x_seq for s in batch]).to(device)
            b_drift = torch.stack([s.drift_seq for s in batch]).to(device)

            outputs = model.forward_tensors(b_x, b_drift)

            probs = outputs["cascade_prob"].cpu().tolist()
            lead_preds = outputs["lead_time"].cpu().tolist()
            root_preds = torch.argmax(outputs["root_cause_logits"], dim=-1).cpu().tolist()
            sev_preds = torch.argmax(outputs["severity_logits"], dim=-1).cpu().tolist()

            for j, s in enumerate(batch):
                all_preds_prob.append(probs[j])
                all_targets_casc.append(s.cascade)
                lead_time_errors.append(abs(lead_preds[j] - s.lead_time))

                if root_preds[j] == s.root_cause:
                    correct_root += 1
                if sev_preds[j] == s.severity:
                    correct_sev += 1

    # Compute binary classification metrics
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds_prob]
    tp = sum(1 for p, t in zip(binary_preds, all_targets_casc) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(binary_preds, all_targets_casc) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(binary_preds, all_targets_casc) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(binary_preds, all_targets_casc) if p == 0 and t == 1)

    acc = (tp + tn) / max(1, total)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * (prec * rec) / max(1e-6, prec + rec)
    mae_lead = sum(lead_time_errors) / max(1, len(lead_time_errors))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mae_lead_time_seconds": round(mae_lead, 2),
        "root_cause_accuracy": round(correct_root / max(1, total), 4),
        "severity_accuracy": round(correct_sev / max(1, total), 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def train_model(
    data_path: str,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
    patience: int = 10,
    device_name: str = "cpu",
    output_weight_path: str = "ai-models/weights/cascade_predictor_tier1a.pt",
    output_meta_path: str = "ai-models/weights/cascade_predictor_metadata.json",
) -> Tuple[HeteroCascadePredictor, Dict[str, Any]]:
    set_seed(seed)
    device = torch.device(device_name)
    logger.info(f"Training on device: {device}")

    # Load and pre-tensorize data
    train_data, val_data, test_data, split_meta = load_and_split_data(Path(data_path), seed=seed)

    # Class balance calculation
    n_pos = sum(1 for s in train_data if s.cascade == 1.0)
    n_neg = len(train_data) - n_pos
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float, device=device)
    logger.info(f"Train class balance: {n_pos} positive, {n_neg} negative (pos_weight: {pos_weight.item():.2f})")

    # Model architecture
    model = HeteroCascadePredictor(
        node_in_dim=7,
        hidden_dim=64,
        num_relations=5,
        num_nodes=5,
        num_severities=5,
        gru_layers=2,
        dropout=0.1,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    smooth_l1 = nn.SmoothL1Loss()
    ce_loss = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_weights = None
    epochs_no_improve = 0
    history = []

    logger.info(f"Beginning training for {epochs} epochs (batch_size={batch_size}, lr={lr})...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)

        train_losses = []
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            b_x = torch.stack([s.x_seq for s in batch]).to(device)
            b_drift = torch.stack([s.drift_seq for s in batch]).to(device)

            targets = {
                "cascade": torch.tensor([s.cascade for s in batch], dtype=torch.float, device=device),
                "lead_time": torch.tensor([s.lead_time for s in batch], dtype=torch.float, device=device),
                "root_cause": torch.tensor([s.root_cause for s in batch], dtype=torch.long, device=device),
                "severity": torch.tensor([s.severity for s in batch], dtype=torch.long, device=device),
            }

            optimizer.zero_grad()
            outputs = model.forward_tensors(b_x, b_drift)
            loss, loss_details = compute_loss(outputs, targets, bce_loss_fn, smooth_l1, ce_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_losses.append(loss_details["loss_total"])

        avg_train_loss = sum(train_losses) / max(1, len(train_losses))

        # Validate
        val_metrics = evaluate_split(model, val_data, device, batch_size=batch_size)
        scheduler.step(1.0 - val_metrics["f1"])

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_lead_time_mae": val_metrics["mae_lead_time_seconds"],
        })

        if epoch % 5 == 0 or epoch == 1 or val_metrics["f1"] > best_val_f1:
            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val Lead-Time MAE: {val_metrics['mae_lead_time_seconds']:.1f}s"
            )

        # Early stopping logic on val F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    training_duration = round(time.time() - start_time, 2)
    logger.info(f"Training completed in {training_duration}s. Restoring best model (Val F1: {best_val_f1:.4f})...")
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Evaluate on unseen Test Set
    test_metrics = evaluate_split(model, test_data, device, batch_size=batch_size)
    logger.info(
        f"Final Test Evaluation: Accuracy={test_metrics['accuracy']:.4f}, "
        f"Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}, "
        f"F1={test_metrics['f1']:.4f}, Lead-Time MAE={test_metrics['mae_lead_time_seconds']:.1f}s, "
        f"Root-Cause Acc={test_metrics['root_cause_accuracy']:.4f}, "
        f"Severity Acc={test_metrics['severity_accuracy']:.4f}"
    )

    # Save checkpoint
    out_weight_path = Path(output_weight_path).resolve()
    out_weight_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_weight_path)
    logger.info(f"Model checkpoint saved to: {out_weight_path}")

    # Build and save metadata
    metadata = {
        "model_architecture": "HeteroRGCN-GRU-MultiTask (Tier 1A)",
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_seed": seed,
        "training_duration_seconds": training_duration,
        "dataset_hash": split_meta["dataset_hash"],
        "dataset_samples": split_meta["total_samples"],
        "split_counts": {
            "train": split_meta["train_samples"],
            "validation": split_meta["val_samples"],
            "test": split_meta["test_samples"],
        },
        "hyperparameters": {
            "node_in_dim": 7,
            "hidden_dim": 64,
            "num_relations": 5,
            "num_nodes": 5,
            "num_severities": 5,
            "gru_layers": 2,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs_trained": len(history),
            "early_stopping_patience": patience,
        },
        "feature_schema": {
            "node_features": [
                "norm_latency", "norm_req_rate", "error_rate", "cpu_saturation",
                "is_cloud_native", "is_boundary_gateway", "is_legacy_core"
            ],
            "relation_types": {
                "0": "cloud_to_gateway",
                "1": "gateway_to_cloud",
                "2": "gateway_to_legacy",
                "3": "legacy_to_gateway",
                "4": "cloud_to_cloud",
            },
            "boundary_drift_feature": "sync_drift_score_normalized",
        },
        "test_metrics": test_metrics,
        "training_history": history,
    }

    out_meta_path = Path(output_meta_path).resolve()
    out_meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Model metadata saved to: {out_meta_path}")

    return model, metadata


def main():
    parser = argparse.ArgumentParser(description="Train BACCP Heterogeneous RGCN Cascade Predictor.")
    parser.add_argument("--data", type=str, default="ai-models/data/dataset.json", help="Path to dataset JSON")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or mps)")
    parser.add_argument("--output", type=str, default="ai-models/weights/cascade_predictor_tier1a.pt", help="Output weights path")
    parser.add_argument("--meta", type=str, default="ai-models/weights/cascade_predictor_metadata.json", help="Output metadata path")
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        device_name=args.device,
        output_weight_path=args.output,
        output_meta_path=args.meta,
    )


if __name__ == "__main__":
    main()

"""Train and evaluate the BACCP cascade predictor on the real-workload dataset.

Re-uses the team's batched trainer (ai-models/cascade_predictor/train.py)
unchanged: episodes are exported to its dataset.json format and its split
loader is swapped for one that honours our event split (train on one
airline/disruption, test on another).

Observation modes
-----------------
full   : 5 snapshots spread over the whole 60-minute episode, including
         snapshots *after* the cascade started (detection - what the
         synthetic pipeline measures).
early  : the 5 most recent 30-second readings strictly before
         ``manifest - horizon`` (last 2 minutes of history). Negatives
         are cut at cutoffs drawn from the positives' cutoff distribution so
         the window length carries no label information (genuine prediction).

Baselines on exactly the same inputs: the dashboard's static drift rule
(eps >= 45 %) and logistic regression on the flattened 5-snapshot features.

    python ai-models/real_data/train_on_trace.py --mode early --horizon-min 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

AI_MODELS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_MODELS_DIR.parent
sys.path.insert(0, str(AI_MODELS_DIR / "cascade_predictor"))

import train as cp_train  # noqa: E402  (ai-models/cascade_predictor/train.py)

DATASET = AI_MODELS_DIR / "data" / "dataset_trace" / "baccp_dataset.pt"
RESULTS = REPO_ROOT / "results" / "real_data"
SEVERITY = {"LOW": 0, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
GEN_ONE_HOT = [[1.0, 0.0, 0.0]] * 3 + [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _node_features(node_seq: List[List[float]]) -> List[List[float]]:
    """Same normalisation as backend/cloud/sagemaker.py so the weights stay serveable."""
    out = []
    for (lat, rate, err, cpu), oh in zip(node_seq, GEN_ONE_HOT):
        out.append([min(10.0, max(0.0, lat / 100.0)), min(5.0, max(0.0, rate / 1000.0)),
                    min(1.0, max(0.0, err)), min(1.0, max(0.0, cpu))] + oh)
    return out


def select_snapshots(run: Dict[str, Any], mode: str, cutoff: float) -> List[Dict[str, Any]]:
    """5 snapshots as {time, drift_score, node_seq}.

    early: the 5 most recent 30-second readings strictly before ``cutoff``
           (the last 2 minutes of history an operator would have).
    full : 5 readings spread evenly over the whole episode.
    """
    fine = run.get("fine")
    if fine is None:  # datasets built before the 30-second history existed
        snaps = run["snapshots"]
        if mode == "early":
            snaps = [s for s in snaps if s["time"] < cutoff] or snaps[:1]
        idx = np.linspace(0, len(snaps) - 1, num=5).round().astype(int)
        return [snaps[i] for i in idx]
    times = np.asarray(fine["time"])
    if mode == "early":
        avail = np.flatnonzero(times < cutoff)
        avail = avail if len(avail) else np.array([0])
        idx = avail[-5:]
        idx = np.concatenate([np.full(5 - len(idx), idx[0]), idx]).astype(int)
    else:
        idx = np.linspace(0, len(times) - 1, num=5).round().astype(int)
    return [{"time": float(times[i]), "drift_score": fine["drift"][i], "node_seq": fine["node_seq"][i]}
            for i in idx]


def export(runs: List[Dict[str, Any]], splits: Dict[str, List[int]], mode: str, horizon_min: float,
           seed: int) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    h = horizon_min * 60.0
    pos_cut = [r["manifest_time"] - h for r in runs if r["cascade_occurred"] and r["manifest_time"] - h > 0]
    split_of = {i: name for name, ids in [("train", splits["train_run_ids"]), ("val", splits["val_run_ids"]),
                                          ("test", splits["test_run_ids"])] for i in ids}
    samples = []
    for r in runs:
        cut = (r["manifest_time"] - h) if r["cascade_occurred"] else float(rng.choice(pos_cut or [600.0]))
        chosen = select_snapshots(r, mode, cut)
        samples.append({
            "sample_id": r["run_id"], "split": split_of[r["run_id"]], "source": r["source"],
            "fault_type": r["fault_type"], "fault_intensity": r["fault_level"],
            "cascade_label": int(r["cascade_occurred"]),
            "lead_time_seconds": float(r["lead_time_seconds"]),
            "root_cause_node": 3, "root_cause_name": "boundary-gateway",
            "severity_level": SEVERITY.get(r["severity"], 0) if r["cascade_occurred"] else 0,
            "snapshots": [{"t": t, "sync_drift_score": s["drift_score"],
                           "node_features": _node_features(s["node_seq"])} for t, s in enumerate(chosen)],
        })
    return {"metadata": {"version": "trace-1.0", "observation_mode": mode, "horizon_min": horizon_min,
                         "nodes": ["reservations", "crew", "baggage", "boundary-gateway", "legacy-core"],
                         "provenance": "real-workload trace-driven episodes (ai-models/real_data)"},
            "samples": samples}


def make_split_loader(data: Dict[str, Any]):
    def load_and_split_data(data_path: Path, seed: int = 42, **_) -> Tuple[list, list, list, Dict[str, Any]]:
        parts = {"train": [], "val": [], "test": []}
        for s in data["samples"]:
            parts[s["split"]].append(cp_train.tensorize_sample(s))
        raw = Path(data_path).read_text()
        import hashlib
        meta = {"dataset_hash": hashlib.sha256(raw.encode()).hexdigest()[:16], "total_samples": len(data["samples"]),
                "train_samples": len(parts["train"]), "val_samples": len(parts["val"]),
                "test_samples": len(parts["test"]), "feature_metadata": data["metadata"]}
        return parts["train"], parts["val"], parts["test"], meta
    return load_and_split_data


def _scores(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    pred = (p >= thr).astype(int)
    two = len(np.unique(y)) > 1
    return {"precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4) if two else None,
            "pr_auc": round(float(average_precision_score(y, p)), 4) if two else None}


def _flat(sample: Dict[str, Any]) -> np.ndarray:
    return np.array([v for s in sample["snapshots"] for row in s["node_features"] for v in row[:4]]
                    + [s["sync_drift_score"] / 100.0 for s in sample["snapshots"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "early"], default="early")
    ap.add_argument("--horizon-min", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    # weights_only=False: the bundle holds our own dicts/tensors produced by build_trace_dataset.py
    payload = torch.load(DATASET, map_location="cpu", weights_only=False)
    manifest, runs = payload["manifest"], payload["runs"]
    data = export(runs, manifest["splits"], a.mode, a.horizon_min, a.seed)
    json_path = AI_MODELS_DIR / "data" / "dataset_trace" / f"dataset_{a.mode}.json"
    json_path.write_text(json.dumps(data))

    wdir = AI_MODELS_DIR / "weights" / f"trace_{a.mode}"
    cp_train.load_and_split_data = make_split_loader(data)
    model, meta = cp_train.train_model(str(json_path), epochs=a.epochs, seed=a.seed, patience=15,
                                       output_weight_path=str(wdir / "cascade_predictor_tier1a.pt"),
                                       output_meta_path=str(wdir / "cascade_predictor_metadata.json"))

    test = [s for s in data["samples"] if s["split"] == "test"]
    train = [s for s in data["samples"] if s["split"] == "train"]
    y_te = np.array([s["cascade_label"] for s in test])
    model.eval()
    with torch.no_grad():
        ts = [cp_train.tensorize_sample(s) for s in test]
        out = model.forward_tensors(torch.stack([t.x_seq for t in ts]), torch.stack([t.drift_seq for t in ts]))
        p_model = out["cascade_prob"].numpy()
    drift_max = np.array([max(sn["sync_drift_score"] for sn in s["snapshots"]) for s in test])
    X_tr, y_tr = np.stack([_flat(s) for s in train]), np.array([s["cascade_label"] for s in train])
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    lr = LogisticRegression(max_iter=5000, C=0.5).fit((X_tr - mu) / sd, y_tr)
    p_lr = lr.predict_proba((np.stack([_flat(s) for s in test]) - mu) / sd)[:, 1]

    result = {
        "dataset_version": manifest["dataset_version"], "observation_mode": a.mode,
        "horizon_min": a.horizon_min if a.mode == "early" else None,
        "split_strategy": manifest["split_strategy"],
        "train_sources": sorted({s["source"] for s in train}), "test_sources": sorted({s["source"] for s in test}),
        "n": {k: sum(1 for s in data["samples"] if s["split"] == k) for k in ("train", "val", "test")},
        "test_positive_rate": round(float(y_te.mean()), 3),
        "models": {
            "baccp_hetero_rgcn (cascade_predictor)": {**_scores(y_te, p_model),
                                                    "lead_time_mae_s": meta["test_metrics"]["mae_lead_time_seconds"]},
            "drift_rule_eps_ge_45": {**_scores(y_te, (drift_max >= 45.0).astype(float)),
                                     "roc_auc": round(float(roc_auc_score(y_te, drift_max)), 4)},
            "logistic_regression_flat": _scores(y_te, p_lr),
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"cascade_metrics_{a.mode}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[+] wrote {out_path}")


if __name__ == "__main__":
    main()

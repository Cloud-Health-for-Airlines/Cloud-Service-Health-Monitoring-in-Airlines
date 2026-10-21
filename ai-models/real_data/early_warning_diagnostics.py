"""Why is early warning hard? Leakage, horizon and visibility diagnostics.

    python ai-models/real_data/early_warning_diagnostics.py

Uses two fast, transparent models (logistic regression, gradient boosting) on
the same 5-reading inputs as ``train_on_trace.py --mode early`` and reports:

1. Split protocol: a *random* split inside one airline vs a split by day vs
   the cross-airline event split. Overlapping windows of the same fault make
   random splits leak.
2. Horizon: AUC when warning 0.5 / 1 / 2 / 5 minutes ahead.
3. Visibility: how many cascades show any precursor before the cutoff at all
   (abrupt faults and very fast ramps cannot be predicted, only detected).

Writes results/real_data/early_warning_diagnostics.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

AI_MODELS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_MODELS_DIR))
sys.path.insert(0, str(AI_MODELS_DIR / "cascade_predictor"))

from real_data.train_on_trace import DATASET, RESULTS, _flat, export  # noqa: E402


def _auc(X, y, tr, te) -> dict:
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    lr = LogisticRegression(max_iter=5000, C=0.5).fit((X[tr] - mu) / sd, y[tr])
    gb = GradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
    return {"n_train": int(tr.sum()), "n_test": int(te.sum()),
            "logistic_auc": round(float(roc_auc_score(y[te], lr.predict_proba((X[te] - mu) / sd)[:, 1])), 3),
            "gbm_auc": round(float(roc_auc_score(y[te], gb.predict_proba(X[te])[:, 1])), 3)}


def main() -> None:
    # weights_only=False: the bundle holds our own dicts/tensors produced by build_trace_dataset.py
    payload = torch.load(DATASET, map_location="cpu", weights_only=False)
    runs, sp = payload["runs"], payload["manifest"]["splits"]
    sources = sorted({r["source"] for r in runs})
    train_src, test_src = sources[-1], sources[0]  # e.g. WN:202212 -> DL:202407
    for r in runs:
        if r["run_id"] in set(sp["test_run_ids"]):
            test_src = r["source"]
        if r["run_id"] in set(sp["train_run_ids"]):
            train_src = r["source"]
    day = {r["run_id"]: pd.Timestamp(r["real_window_start"]).day for r in runs}
    out = {"dataset_version": payload["manifest"]["dataset_version"], "train_source": train_src,
           "test_source": test_src}

    pos = [r for r in runs if r["cascade_occurred"]]
    out["visibility"] = {}
    for h in (0.5, 1, 2, 5):
        vis = sum(1 for r in pos if r["onset_time"] is not None and r["onset_time"] < r["manifest_time"] - h * 60)
        out["visibility"][f"{h}min"] = {"cascades": len(pos), "precursor_visible": vis,
                                         "share": round(vis / max(1, len(pos)), 3)}

    out["split_protocol_2min"] = {}
    out["horizon"] = {}
    for h in (0.5, 1, 2, 5):
        S = export(runs, sp, "early", h, 42)["samples"]
        X = np.stack([_flat(s) for s in S])
        y = np.array([s["cascade_label"] for s in S])
        src = np.array([s["source"] for s in S])
        dd = np.array([day[s["sample_id"]] for s in S])
        out["horizon"][f"{h}min"] = {
            "event_split": _auc(X, y, src == train_src, src == test_src),
            "day_split_both_airlines": _auc(X, y, dd % 2 == 0, dd % 2 == 1),
        }
        if h == 2:
            idx = np.flatnonzero(src == train_src)
            rng = np.random.RandomState(0)
            rng.shuffle(idx)
            tr_r = np.zeros(len(S), bool)
            te_r = np.zeros(len(S), bool)
            tr_r[idx[: len(idx) // 2]] = True
            te_r[idx[len(idx) // 2:]] = True
            out["split_protocol_2min"] = {
                f"random_split_within_{train_src} (LEAKS)": _auc(X, y, tr_r, te_r),
                f"day_split_within_{train_src}": _auc(X, y, (src == train_src) & (dd % 2 == 0),
                                                       (src == train_src) & (dd % 2 == 1)),
                f"event_split_{train_src}_to_{test_src}": _auc(X, y, src == train_src, src == test_src),
            }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "early_warning_diagnostics.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[+] wrote {path}")


if __name__ == "__main__":
    main()

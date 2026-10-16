"""Training pipeline for BACCP workload forecasting.

Version: baccp-train-v1.1.0

Trains per-service forecasters across horizons H in {1, 5, 15}:
- Fits Persistence, Seasonal Naive, Ridge Autoregression, and Gradient Boosting.
- Computes training workload domain bounds strictly on training data for out-of-distribution detection.
- Evaluates on Validation split for model selection.
- Calibrates nominal 90% conformal prediction intervals on Calibration split.
- Packages versioned feature schema, training bounds, and fallback policy into model bundle.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np

from .data_contract import CONFIGURED_SERVICE_CAPACITIES, ServiceIdentifier
from .features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    SUPPORTED_HORIZONS,
    build_forecasting_dataset_for_service,
)
from .models import (
    AutoregressiveForecaster,
    ConformalIntervalCalibrator,
    GradientBoostingForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "ai-models" / "data" / "workload_forecasting" / "workload_dataset.json"
WEIGHTS_DIR = REPO_ROOT / "ai-models" / "weights" / "workload_forecasting"


def load_dataset_by_splits(dataset_path: Path = DATA_PATH) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Loads dataset and partitions trajectories by incident ID according to split manifest."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run generate_workload_data.py first.")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = data["metadata"]
    splits_config = manifest["splits"]
    trajectories = data["trajectories"]

    traj_by_id = {t["incident_id"]: t for t in trajectories}

    partitioned_trajectories = {
        "train": [traj_by_id[inc] for inc in splits_config["train"] if inc in traj_by_id],
        "validation": [traj_by_id[inc] for inc in splits_config["validation"] if inc in traj_by_id],
        "calibration": [traj_by_id[inc] for inc in splits_config["calibration"] if inc in traj_by_id],
        "test_standard": [traj_by_id[inc] for inc in splits_config["test_standard"] if inc in traj_by_id],
        "test_unseen_surge_diagnostic": [
            traj_by_id[inc] for inc in splits_config.get("test_unseen_surge_diagnostic", []) if inc in traj_by_id
        ],
        "test_unseen_surge_locked": [
            traj_by_id[inc] for inc in splits_config.get("test_unseen_surge_locked", []) if inc in traj_by_id
        ],
    }

    return partitioned_trajectories, manifest


def train_workload_models(
    dataset_path: Path = DATA_PATH,
    output_weights_dir: Path = WEIGHTS_DIR,
) -> Dict[str, Any]:
    """Fits all models and calibrates prediction intervals across all services and horizons."""
    output_weights_dir.mkdir(parents=True, exist_ok=True)
    partitioned, manifest = load_dataset_by_splits(dataset_path)

    train_trajs = partitioned["train"]
    val_trajs = partitioned["validation"]
    cal_trajs = partitioned["calibration"]

    services = [s.value for s in ServiceIdentifier]
    horizons = SUPPORTED_HORIZONS

    print(f"[*] Commencing training across {len(services)} services and horizons {horizons}...")
    print(f"    Train incidents: {len(train_trajs)} | Val: {len(val_trajs)} | Cal: {len(cal_trajs)}")

    trained_bundle: Dict[str, Any] = {
        "metadata": {
            "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset_sha256": manifest.get("dataset_sha256"),
            "contract_version": manifest.get("contract_version"),
            "feature_schema": {
                "version": FEATURE_SCHEMA_VERSION,
                "feature_names": FEATURE_NAMES,
                "feature_count": len(FEATURE_NAMES),
            },
            "services": services,
            "horizons": list(horizons),
            "capacities": CONFIGURED_SERVICE_CAPACITIES,
            "training_domain_bounds": {},
            "fallback_policy": {
                "trigger_rule": "input_workload_rps > surge_threshold_rps",
                "default_model": "gradient_boosting",
                "fallback_model": "autoregressive_ridge",
                "rationale": "HistGradientBoostingRegressor saturates at leaf maximums during extreme surges; Ridge AR extrapolates linearly.",
            },
        },
        "models": {},
        "calibrators": {},
        "validation_metrics": {},
    }

    for svc in services:
        trained_bundle["models"][svc] = {}
        trained_bundle["calibrators"][svc] = {}
        trained_bundle["validation_metrics"][svc] = {}

        # 1. Build tabular matrices
        train_data = build_forecasting_dataset_for_service(train_trajs, svc, horizons)
        val_data = build_forecasting_dataset_for_service(val_trajs, svc, horizons)
        cal_data = build_forecasting_dataset_for_service(cal_trajs, svc, horizons)

        X_train = train_data["X"]
        X_val = val_data["X"]
        X_cal = cal_data["X"]

        # Compute training domain bounds strictly on training data
        min_workload = float(np.min(X_train[:, 0]))
        max_workload = float(np.max(X_train[:, 0]))
        surge_threshold = max_workload

        trained_bundle["metadata"]["training_domain_bounds"][svc] = {
            "min_rps": round(min_workload, 3),
            "max_rps": round(max_workload, 3),
            "surge_threshold_rps": round(surge_threshold, 3),
        }

        print(
            f"\n[-] Service '{svc}': {len(X_train)} train windows, {len(X_val)} val windows, {len(X_cal)} cal windows "
            f"[Train range: {min_workload:.1f} - {max_workload:.1f} RPS]"
        )

        for h in horizons:
            y_train_h = train_data["y"][h]
            y_val_h = val_data["y"][h]
            y_cal_h = cal_data["y"][h]

            trained_bundle["models"][svc][h] = {}
            trained_bundle["calibrators"][svc][h] = {}
            trained_bundle["validation_metrics"][svc][h] = {}

            # Instantiations
            candidate_models = {
                "persistence": PersistenceForecaster(horizon=h),
                "seasonal_naive": SeasonalNaiveForecaster(horizon=h, seasonal_period=15),
                "autoregressive_ridge": AutoregressiveForecaster(horizon=h, alpha=1.0),
                "gradient_boosting": GradientBoostingForecaster(horizon=h, random_state=42),
            }

            for model_name, model in candidate_models.items():
                # Fit model on training data strictly
                model.fit(X_train, y_train_h)
                trained_bundle["models"][svc][h][model_name] = model

                # Validation metrics
                val_preds = model.predict(X_val)
                val_mae = float(np.mean(np.abs(y_val_h - val_preds)))
                val_rmse = float(np.sqrt(np.mean((y_val_h - val_preds) ** 2)))
                denom = float(np.sum(y_val_h))
                if denom == 0.0:
                    val_wape = 0.0 if float(np.sum(np.abs(y_val_h - val_preds))) == 0.0 else None
                else:
                    val_wape = float(np.sum(np.abs(y_val_h - val_preds)) / denom * 100.0)

                trained_bundle["validation_metrics"][svc][h][model_name] = {
                    "mae": round(val_mae, 4),
                    "rmse": round(val_rmse, 4),
                    "wape_pct": round(val_wape, 2) if val_wape is not None else None,
                }

                # Calibrate prediction intervals on calibration split
                cal_preds = model.predict(X_cal)
                calibrator = ConformalIntervalCalibrator(target_coverage=0.90)
                calibrator.calibrate(y_cal_h, cal_preds)
                trained_bundle["calibrators"][svc][h][model_name] = calibrator

            gb_val = trained_bundle["validation_metrics"][svc][h]["gradient_boosting"]
            per_val = trained_bundle["validation_metrics"][svc][h]["persistence"]
            ridge_val = trained_bundle["validation_metrics"][svc][h]["autoregressive_ridge"]
            print(
                f"    H={h:2d} ({(h * 20):3d}s) | GB Val MAE: {gb_val['mae']:.3f} | Ridge MAE: {ridge_val['mae']:.3f} | Persist MAE: {per_val['mae']:.3f}"
            )

    # Serialize trained bundle to pickle
    bundle_path = output_weights_dir / "workload_models_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(trained_bundle, f)
    print(f"\n[+] Saved trained model bundle to {bundle_path} ({bundle_path.stat().st_size / 1024:.1f} KB)")

    # Serialize validation summary to JSON
    meta_json_path = output_weights_dir / "validation_summary.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": trained_bundle["metadata"],
            "validation_metrics": trained_bundle["validation_metrics"],
        }, f, indent=2)
    print(f"[+] Saved validation summary to {meta_json_path}")

    return trained_bundle


def main():
    parser = argparse.ArgumentParser(description="Train BACCP Workload Forecasters")
    parser.add_argument("--dataset", type=Path, default=DATA_PATH, help="Path to workload dataset JSON")
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR, help="Output directory for weights")
    args = parser.parse_args()

    train_workload_models(dataset_path=args.dataset, output_weights_dir=args.weights_dir)


if __name__ == "__main__":
    main()

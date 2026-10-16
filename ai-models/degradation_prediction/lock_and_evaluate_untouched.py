"""Lock Manifest and Untouched Final Evaluation Suite.

Version: baccp-lock-untouched-v1.0.0

Locks code, models, calibration, thresholds, and schemas into a cryptographic manifest,
then evaluates the frozen models on the untouched assessment dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .data_contract import (
        CONTRACT_VERSION,
        SAMPLING_INTERVAL_SECONDS,
        SERVICE_PRIORITY_WEIGHTS,
        ServiceIdentifier,
        SLO_THRESHOLDS,
        detect_confirmed_sustained_episodes,
    )
    from .features import (
        FEATURE_SCHEMA_VERSION,
        TELEMETRY_ONLY_FEATURE_NAMES,
        TELEMETRY_PLUS_FORECAST_FEATURE_NAMES,
        assert_feature_schema,
        get_feature_index,
    )
    from .models import (
        SimpleAnomalyBaseline,
        StaticThresholdBaseline,
    )
    from .train import extract_split_samples
    from .evaluate import (
        compute_binary_metrics,
        compute_early_warning_metrics,
        compute_priority_weighted_metrics,
    )
except ImportError:
    from degradation_prediction.data_contract import (
        CONTRACT_VERSION,
        SAMPLING_INTERVAL_SECONDS,
        SERVICE_PRIORITY_WEIGHTS,
        ServiceIdentifier,
        SLO_THRESHOLDS,
        detect_confirmed_sustained_episodes,
    )
    from degradation_prediction.features import (
        FEATURE_SCHEMA_VERSION,
        TELEMETRY_ONLY_FEATURE_NAMES,
        TELEMETRY_PLUS_FORECAST_FEATURE_NAMES,
        assert_feature_schema,
        get_feature_index,
    )
    from degradation_prediction.models import (
        SimpleAnomalyBaseline,
        StaticThresholdBaseline,
    )
    from degradation_prediction.train import extract_split_samples
    from degradation_prediction.evaluate import (
        compute_binary_metrics,
        compute_early_warning_metrics,
        compute_priority_weighted_metrics,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_MODELS_DIR = REPO_ROOT / "ai-models"
DATA_DIR = AI_MODELS_DIR / "data" / "degradation_prediction"
WEIGHTS_PATH = AI_MODELS_DIR / "weights" / "degradation_prediction" / "degradation_models_bundle.pkl"
UNTOUCHED_DATA_PATH = DATA_DIR / "untouched_final_test.json"
RESULTS_DIR = REPO_ROOT / "results" / "degradation_prediction"

from importlib import import_module
try:
    _wf_mod = import_module("workload_forecasting.inference")
    WorkloadForecaster = getattr(_wf_mod, "WorkloadForecaster")
except Exception:
    try:
        _wf_mod = import_module("ai-models.workload_forecasting.inference")
        WorkloadForecaster = getattr(_wf_mod, "WorkloadForecaster")
    except Exception:
        WorkloadForecaster = None


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    if not path.exists():
        return "FILE_NOT_FOUND"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_lock_manifest() -> Dict[str, Any]:
    """Builds cryptographic lock manifest for all pipeline components."""
    tracked_files = {
        "data_contract_py": AI_MODELS_DIR / "degradation_prediction" / "data_contract.py",
        "features_py": AI_MODELS_DIR / "degradation_prediction" / "features.py",
        "models_py": AI_MODELS_DIR / "degradation_prediction" / "models.py",
        "train_py": AI_MODELS_DIR / "degradation_prediction" / "train.py",
        "inference_py": AI_MODELS_DIR / "degradation_prediction" / "inference.py",
        "evaluate_py": AI_MODELS_DIR / "degradation_prediction" / "evaluate.py",
        "degradation_models_bundle_pkl": WEIGHTS_PATH,
        "degradation_dataset_json": DATA_DIR / "degradation_dataset.json",
        "untouched_final_test_json": UNTOUCHED_DATA_PATH,
        "workload_models_bundle_pkl": AI_MODELS_DIR / "weights" / "workload_forecasting" / "workload_models_bundle.pkl",
    }

    file_hashes = {name: compute_file_sha256(p) for name, p in tracked_files.items()}

    with open(WEIGHTS_PATH, "rb") as f:
        bundle = pickle.load(f)

    thresholds_lock = {}
    for h in [1, 5, 15]:
        m_a = bundle["models_telemetry_only"][h]
        m_b = bundle["models_telemetry_forecast"][h]
        m_c = bundle.get("models_temporal_comparator", {}).get(h)
        thresholds_lock[str(h)] = {
            "model_a_threshold": getattr(m_a, "alert_threshold", 0.50),
            "model_b_threshold": getattr(m_b, "alert_threshold", 0.50),
            "model_c_threshold": getattr(m_c, "alert_threshold", 0.35) if m_c else 0.35,
        }

    manifest = {
        "manifest_version": "baccp-lock-manifest-v1.0.0",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "contract_version": CONTRACT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "file_sha256_hashes": file_hashes,
        "tuned_alert_thresholds": thresholds_lock,
        "priority_weights": SERVICE_PRIORITY_WEIGHTS,
        "cooldown_policy_seconds": 100.0,
        "sustained_degradation_definition": ">= 3 consecutive SLO breaches",
    }

    manifest_path = RESULTS_DIR / "lock_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated lock manifest saved to {manifest_path}")
    return manifest


def evaluate_untouched_set(
    data_path: Path = UNTOUCHED_DATA_PATH,
    weights_path: Path = WEIGHTS_PATH,
    output_path: Path = RESULTS_DIR / "untouched_test_metrics.json",
) -> Dict[str, Any]:
    """Evaluates locked models on untouched assessment set."""
    if not data_path.exists():
        try:
            from .generate_untouched_test import generate_untouched_dataset
        except ImportError:
            from degradation_prediction.generate_untouched_test import generate_untouched_dataset
        generate_untouched_dataset(data_path, seed=9999)

    with open(data_path, "r") as f:
        data_payload = json.load(f)

    with open(weights_path, "rb") as f:
        bundle = pickle.load(f)

    forecaster = None
    if WorkloadForecaster is not None:
        try:
            forecaster = WorkloadForecaster.get_instance()
        except Exception:
            pass

    incidents = data_payload["incidents"]
    horizons = [1, 5, 15]

    eval_results: Dict[str, Any] = {
        "metadata": {
            "assessment_type": "FINAL_UNTOUCHED_EVALUATION",
            "contract_version": CONTRACT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "dataset_sha256": compute_file_sha256(data_path),
            "weights_sha256": compute_file_sha256(weights_path),
            "total_incidents": len(incidents),
            "horizons": horizons,
        },
        "system_wide_event_evaluation": {},
        "early_warning_episode_evaluation": {},
        "target2_time_to_onset_evaluation": {},
        "target3_type_and_severity_evaluation": {},
        "priority_weighted_evaluation": {},
    }

    for h in horizons:
        h_str = str(h)
        m_a = bundle["models_telemetry_only"][h]
        m_b = bundle["models_telemetry_forecast"][h]
        m_c = bundle.get("models_temporal_comparator", {}).get(h)
        m_time = bundle.get("models_time_to_onset", {}).get(h)
        m_typesev = bundle.get("models_type_severity", {}).get(h)
        m_syscal = bundle.get("models_system_calibrator", {}).get(h)

        # Extract samples
        X_a, X_temp, y_lat, y_q, y_dr, y_ev, y_tm, y_tp_str, y_sv_str, meta_samples = extract_split_samples(
            incidents, horizon=h, include_workload_forecasts=False, forecaster=forecaster
        )
        X_b, _, _, _, _, _, _, _, _, _ = extract_split_samples(
            incidents, horizon=h, include_workload_forecasts=True, forecaster=forecaster
        )

        assert_feature_schema(X_a, "telemetry")
        assert_feature_schema(X_b, "combined")

        prob_a_ev = m_a.predict_event_proba(X_a)
        prob_b_ev = m_b.predict_event_proba(X_b)
        prob_c_ev = m_c.predict_event_proba(X_temp) if m_c else prob_b_ev

        static_base = StaticThresholdBaseline(ratio=0.75)
        anomaly_base = SimpleAnomalyBaseline()

        prob_static_row = np.array([
            static_base.predict_event_prob(
                meta_samples[i]["service_id"],
                X_a[i, get_feature_index("curr_latency_ms", "telemetry")],
                X_a[i, get_feature_index("curr_backlog", "telemetry")],
                X_a[i, get_feature_index("curr_error_rate", "telemetry")],
                X_a[i, get_feature_index("gw_sync_drift", "telemetry")],
            )
            for i in range(len(X_a))
        ])
        prob_anom_row = np.array([
            anomaly_base.predict_event_prob(
                X_a[i, get_feature_index("latency_delta_1", "telemetry")],
                X_a[i, get_feature_index("backlog_delta_1", "telemetry")],
                X_a[i, get_feature_index("gw_drift_delta_1", "telemetry")],
            )
            for i in range(len(X_a))
        ])

        # Group by origin
        origin_groups: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for idx, m in enumerate(meta_samples):
            key = (m["incident_id"], m["origin_step"])
            if key not in origin_groups:
                origin_groups[key] = {
                    "incident_id": m["incident_id"],
                    "origin_step": m["origin_step"],
                    "is_system_window_degraded": m["is_system_window_degraded"],
                    "is_curr_system_healthy": m["is_curr_system_healthy"],
                    "prob_a_list": [],
                    "prob_b_list": [],
                    "prob_c_list": [],
                    "prob_static_list": [],
                    "prob_anom_list": [],
                }
            origin_groups[key]["prob_a_list"].append(prob_a_ev[idx])
            origin_groups[key]["prob_b_list"].append(prob_b_ev[idx])
            origin_groups[key]["prob_c_list"].append(prob_c_ev[idx])
            origin_groups[key]["prob_static_list"].append(prob_static_row[idx])
            origin_groups[key]["prob_anom_list"].append(prob_anom_row[idx])

        unique_origins = list(origin_groups.values())
        y_sys_true = np.array([o["is_system_window_degraded"] for o in unique_origins], dtype=int)
        is_curr_healthy = np.array([o["is_curr_system_healthy"] for o in unique_origins], dtype=int)

        comp_prob_a = np.array([max(o["prob_a_list"]) for o in unique_origins])
        comp_prob_b = np.array([max(o["prob_b_list"]) for o in unique_origins])
        comp_prob_c = np.array([max(o["prob_c_list"]) for o in unique_origins])

        cal_prob_b = m_syscal.predict_proba(comp_prob_b) if m_syscal else comp_prob_b

        pred_a = (comp_prob_a >= m_a.alert_threshold).astype(int)
        pred_b = (comp_prob_b >= m_b.alert_threshold).astype(int)
        thresh_c = getattr(m_c, "alert_threshold", 0.35) if m_c else 0.35
        pred_c = (comp_prob_c >= thresh_c).astype(int)

        prob_static = np.array([max(o["prob_static_list"]) for o in unique_origins])
        pred_static = prob_static.astype(int)

        prob_anom = np.array([max(o["prob_anom_list"]) for o in unique_origins])
        pred_anom = prob_anom.astype(int)

        candidate_evals = {
            "static_thresholding": (prob_static, pred_static),
            "simple_anomaly": (prob_anom, pred_anom),
            "model_a_telemetry_only": (comp_prob_a, pred_a),
            "model_b_telemetry_forecast": (comp_prob_b, pred_b),
            "model_b_system_calibrated": (cal_prob_b, pred_b),
            "model_c_temporal_comparator": (comp_prob_c, pred_c),
        }

        # Healthy origins only
        h_mask = (is_curr_healthy == 1)
        sub_y = y_sys_true[h_mask]
        cohort_res = {}
        for m_name, (m_prob, m_pred) in candidate_evals.items():
            cohort_res[m_name] = compute_binary_metrics(sub_y, m_prob[h_mask], m_pred[h_mask])
        eval_results["system_wide_event_evaluation"][h_str] = cohort_res

        # Early warning metrics
        ew_res = {}
        for m_name, (_, m_pred) in candidate_evals.items():
            orig_pred_map = {
                (unique_origins[i]["incident_id"], unique_origins[i]["origin_step"]): int(m_pred[i])
                for i in range(len(unique_origins))
            }
            ew_res[m_name] = compute_early_warning_metrics(incidents, orig_pred_map, horizon=h)
        eval_results["early_warning_episode_evaluation"][h_str] = ew_res

        # Target 2 Time to onset
        if m_time is not None:
            observed_mask = (y_ev == 1) & (y_tm >= 0)
            if np.sum(observed_mask) > 0:
                pred_times = m_time.predict_time_to_onset_sec(X_b[observed_mask], np.ones(np.sum(observed_mask), dtype=int))
                valid_preds = [pt for pt in pred_times if pt is not None]
                actual_times = y_tm[observed_mask] * SAMPLING_INTERVAL_SECONDS
                t_mae = float(np.mean(np.abs(np.array(valid_preds) - actual_times))) if valid_preds else None
            else:
                t_mae = None
            eval_results["target2_time_to_onset_evaluation"][h_str] = {
                "eligible_observed_events": int(np.sum(observed_mask)),
                "time_to_onset_mae_sec": round(t_mae, 2) if t_mae is not None else None,
            }

        # Target 3 Type and severity
        if m_typesev is not None:
            active_mask = (y_ev == 1)
            if np.sum(active_mask) > 0:
                pred_types, pred_sevs = m_typesev.predict(X_b[active_mask], np.ones(np.sum(active_mask), dtype=int))
                true_types = [y_tp_str[i] for i in range(len(y_ev)) if active_mask[i]]
                true_sevs = [y_sv_str[i] for i in range(len(y_ev)) if active_mask[i]]
                type_acc = float(np.mean([pred_types[i] == true_types[i] for i in range(len(true_types))]))
                sev_acc = float(np.mean([pred_sevs[i] == true_sevs[i] for i in range(len(true_sevs))]))
            else:
                type_acc, sev_acc = None, None
            eval_results["target3_type_and_severity_evaluation"][h_str] = {
                "type_accuracy": round(type_acc, 4) if type_acc is not None else None,
                "severity_accuracy": round(sev_acc, 4) if sev_acc is not None else None,
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"Untouched evaluation results saved to {output_path}")
    return eval_results


if __name__ == "__main__":
    generate_lock_manifest()
    evaluate_untouched_set()

"""BACCP Task 4A Final Comprehensive Evaluation, Comparisons, Robustness, and Limitations.

Preserves primary model weights, calibration, thresholds, and splits.
Operates strictly under <= 2 CPU threads (OMP_NUM_THREADS=2, torch.set_num_threads(2)).
Reconciles all tables, baseline inputs, timing units, and exposure accounting.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

# Enforce hardware compute constraints (max 2 CPU threads)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
torch.set_num_threads(2)

from degradation_prediction.data_contract import (
    BOUNDARY_SYNC_DRIFT_THRESHOLD,
    CONTRACT_VERSION,
    NOMINAL_BASELINES,
    SAMPLING_INTERVAL_SECONDS,
    SERVICE_PRIORITY_WEIGHTS,
    DegradationSeverity,
    DegradationType,
    ServiceIdentifier,
    SLO_THRESHOLDS,
    check_window_label_observability,
    classify_service_step_type_and_severity,
    detect_confirmed_sustained_episodes,
    get_operational_state_at_origin,
    is_boundary_step_degraded,
    is_service_step_degraded,
    is_system_step_degraded,
)
from degradation_prediction.features import (
    FEATURE_SCHEMA_VERSION,
    TELEMETRY_ONLY_FEATURE_NAMES,
    TELEMETRY_PLUS_FORECAST_FEATURE_NAMES,
    extract_features_at_origin,
    extract_temporal_sequence_tensor,
    get_feature_index,
)
from degradation_prediction.models import (
    CausalQueueRolloutBaseline,
    LearnedDegradationHorizonModel,
    LearnedTemporalHorizonModel,
    PersistenceDegradationBaseline,
    StaticThresholdBaseline,
    SystemLevelCalibrator,
)
from degradation_prediction.train import extract_split_samples
from degradation_prediction.inference import DegradationPredictor, MAX_TELEMETRY_STALENESS_SECONDS
from workload_forecasting.inference import WorkloadForecaster

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_MODELS_DIR = REPO_ROOT / "ai-models"
DATA_PATH = AI_MODELS_DIR / "data" / "degradation_prediction" / "degradation_dataset.json"
WEIGHTS_PATH = AI_MODELS_DIR / "weights" / "degradation_prediction" / "degradation_models_bundle.pkl"
RESULTS_DIR = REPO_ROOT / "results" / "degradation_prediction"
PLOTS_DIR = RESULTS_DIR / "plots"

SERVICES_3DOMAIN = ["reservations", "crew", "baggage"]
ALL_SERVICES = ["reservations", "crew", "baggage", "boundary-gateway", "legacy-core"]
DEPLOYED_THRESHOLDS = {1: 0.280, 5: 0.300, 15: 0.320}


def extract_baseline_inputs(
    X_mat: np.ndarray,
    metas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extracts required baseline inputs using named feature indices with strict validation.

    Reads current observations from named feature indices (curr_latency_ms, curr_backlog,
    curr_error_rate, gw_sync_drift). Required inputs must not silently default.

    Raises:
        ValueError: If input matrix is malformed, has insufficient width, or contains non-finite values.
        KeyError: If meta dictionary lacks required 'service_id'.
    """
    if not isinstance(X_mat, np.ndarray) or X_mat.ndim != 2:
        raise ValueError(f"Expected 2D numpy feature matrix, got {type(X_mat)} with ndim={getattr(X_mat, 'ndim', None)}")
    if X_mat.shape[0] != len(metas):
        raise ValueError(f"Row count mismatch: X has {X_mat.shape[0]} rows, metas has {len(metas)}")

    idx_lat = get_feature_index("curr_latency_ms", schema="telemetry")
    idx_q = get_feature_index("curr_backlog", schema="telemetry")
    idx_err = get_feature_index("curr_error_rate", schema="telemetry")
    idx_dr = get_feature_index("gw_sync_drift", schema="telemetry")

    max_idx = max(idx_lat, idx_q, idx_err, idx_dr)
    if X_mat.shape[1] <= max_idx:
        raise ValueError(f"Feature matrix width {X_mat.shape[1]} insufficient for required baseline indices (max needed: {max_idx})")

    inputs = []
    for i in range(len(metas)):
        m = metas[i]
        if "service_id" not in m:
            raise KeyError(f"Meta record {i} missing required 'service_id' field")

        lat = float(X_mat[i, idx_lat])
        q = float(X_mat[i, idx_q])
        err = float(X_mat[i, idx_err])
        dr = float(X_mat[i, idx_dr])

        if not (np.isfinite(lat) and np.isfinite(q) and np.isfinite(err) and np.isfinite(dr)):
            raise ValueError(f"Non-finite baseline observation detected at row {i}: lat={lat}, q={q}, err={err}, dr={dr}")

        inputs.append({
            "service_id": m["service_id"],
            "curr_latency": lat,
            "curr_backlog": q,
            "curr_error_rate": err,
            "curr_sync_drift": dr,
        })
    return inputs


class TrainingFittedZScoreBaseline:
    """Lightweight per-service anomaly baseline fitted strictly on training data."""

    def __init__(self):
        self.stats: Dict[str, Dict[str, float]] = {}
        self.operating_threshold: float = 2.0

    def fit(self, train_incidents: List[Dict[str, Any]]) -> None:
        """Computes mean and std of latency and backlog per service on training data."""
        svc_lats: Dict[str, List[float]] = {s: [] for s in ALL_SERVICES}
        svc_qs: Dict[str, List[float]] = {s: [] for s in ALL_SERVICES}

        for inc in train_incidents:
            for s_id in ALL_SERVICES:
                for step in inc["service_telemetry"].get(s_id, []):
                    svc_lats[s_id].append(float(step.get("latency_ms", 15.0)))
                    svc_qs[s_id].append(float(step.get("backlog_requests", 0.0)))

        for s_id in ALL_SERVICES:
            lats = np.array(svc_lats[s_id]) if svc_lats[s_id] else np.array([20.0])
            qs = np.array(svc_qs[s_id]) if svc_qs[s_id] else np.array([0.0])
            self.stats[s_id] = {
                "lat_mean": float(np.mean(lats)),
                "lat_std": float(np.std(lats)) if float(np.std(lats)) > 1e-3 else 5.0,
                "q_mean": float(np.mean(qs)),
                "q_std": float(np.std(qs)) if float(np.std(qs)) > 1e-3 else 5.0,
            }

    def compute_z_score(self, service_id: str, latency: float, backlog: float) -> float:
        st = self.stats.get(service_id, {"lat_mean": 20.0, "lat_std": 5.0, "q_mean": 0.0, "q_std": 5.0})
        z_lat = (latency - st["lat_mean"]) / st["lat_std"]
        z_q = (backlog - st["q_mean"]) / st["q_std"]
        return max(z_lat, z_q)

    def tune_operating_point(
        self,
        X_val: np.ndarray,
        metas_val: List[Dict[str, Any]],
        y_true: np.ndarray,
    ) -> float:
        """Selects operating threshold on validation split maximizing F1 using named feature indices."""
        inputs = extract_baseline_inputs(X_val, metas_val)
        candidate_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        best_thresh = 2.0
        best_f1 = -1.0

        z_scores = np.array([
            self.compute_z_score(item["service_id"], item["curr_latency"], item["curr_backlog"])
            for item in inputs
        ])

        for th in candidate_thresholds:
            preds = (z_scores >= th).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = th

        self.operating_threshold = best_thresh
        return best_thresh

    def predict(self, service_id: str, latency: float, backlog: float) -> int:
        return 1 if self.compute_z_score(service_id, latency, backlog) >= self.operating_threshold else 0


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: Optional[np.ndarray],
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Computes precision, recall, F1, PR-AUC, ROC-AUC, Brier score, 10-bin ECE, and confusion counts."""
    n = len(y_true)
    if n == 0:
        return {}

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    prevalence = float(np.mean(y_true))

    if y_prob is not None:
        try:
            pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else prevalence
        except Exception:
            pr_auc = prevalence

        try:
            roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.50
        except Exception:
            roc_auc = 0.50

        brier = float(brier_score_loss(y_true, y_prob))

        # 10-bin Expected Calibration Error (ECE)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            low, high = bin_edges[b], bin_edges[b + 1]
            mask = (y_prob >= low) & (y_prob <= high) if b == n_bins - 1 else (y_prob >= low) & (y_prob < high)
            count = int(np.sum(mask))
            if count > 0:
                bin_acc = float(np.mean(y_true[mask]))
                bin_conf = float(np.mean(y_prob[mask]))
                ece += (count / n) * abs(bin_acc - bin_conf)
    else:
        pr_auc = None
        roc_auc = None
        brier = None
        ece = None

    return {
        "sample_count": n,
        "prevalence": round(prevalence, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "brier_score": round(brier, 4) if brier is not None else None,
        "ece": round(ece, 4) if ece is not None else None,
    }


def compute_early_warning_metrics(
    incidents: List[Dict[str, Any]],
    origin_predictions: Dict[Tuple[str, str, int], int],
    cached_records_by_key: Dict[Tuple[str, str, int], Dict[str, Any]],
    horizon: int,
    cooldown_steps: int = 5,
    min_response_time_sec: float = 20.0,
) -> Dict[str, Any]:
    """Evaluates strict episode early warning, lead-time distribution, and false alarms/day.

    Separates current-state detection from future early warning:
    - Current-state detection: alerts triggered while service is already degraded at origin.
    - Future-onset early warning: strictly evaluated when service is nominal at origin and onset occurs in (t, t+H].
    An alert at or after onset does NOT count as an early warning.
    """
    dt = SAMPLING_INTERVAL_SECONDS
    h_sec = float(horizon * dt)

    all_episodes = []
    for inc in incidents:
        inc_id = inc["incident_id"]
        for s_id in ALL_SERVICES:
            eps = detect_confirmed_sustained_episodes(
                inc["service_telemetry"], inc["boundary_telemetry"], target_service_id=s_id, min_consecutive=3
            )
            for ep in eps:
                ep["incident_id"] = inc_id
                ep["service_id"] = s_id
                all_episodes.append(ep)

    evaluable_episodes = [ep for ep in all_episodes if not ep["is_born_degraded"] and ep["first_breach_step"] > 3]

    recalled_evaluable = 0
    recalled_actionable = 0
    lead_times_sec: List[float] = []
    per_event_records = []

    for ep in evaluable_episodes:
        inc_id = ep["incident_id"]
        s_id = ep["service_id"]
        onset_step = ep["first_breach_step"]
        onset_time = ep["first_breach_time_sec"]

        earliest_origin = max(3, onset_step - horizon)
        latest_origin = onset_step - 1

        first_alert_time = None
        first_alert_step = None
        for t in range(earliest_origin, latest_origin + 1):
            key = (inc_id, s_id, t)
            if key not in origin_predictions:
                continue  # Do not count absent/ineligible prediction rows as evaluated negative decisions
            if origin_predictions[key] == 1:
                first_alert_time = float(t * dt)
                first_alert_step = t
                break

        lead = (onset_time - first_alert_time) if first_alert_time is not None else None
        is_rec = bool(first_alert_time is not None and 0.0 < lead <= h_sec)
        is_act = bool(is_rec and lead >= min_response_time_sec)
        is_miss = not is_rec

        if is_rec:
            recalled_evaluable += 1
            lead_times_sec.append(lead)
            if is_act:
                recalled_actionable += 1

        per_event_records.append({
            "incident_id": inc_id,
            "service_id": s_id,
            "onset_step": onset_step,
            "onset_time_sec": onset_time,
            "eligible_warning_window_steps": [earliest_origin, latest_origin],
            "first_alert_step": first_alert_step,
            "first_alert_time_sec": first_alert_time,
            "lead_time_sec": lead,
            "is_recalled": is_rec,
            "is_actionable": is_act,
            "is_missed": is_miss,
        })

    # False alarms and healthy exposure accounting
    # Evaluated on eligible healthy origins strictly within the horizon's predictions
    healthy_exposure_origins = 0
    raw_false_alerts = 0
    dedup_false_episodes = 0

    for inc in incidents:
        inc_id = inc["incident_id"]
        step_count = inc["step_count"]

        for s_id in ALL_SERVICES:
            in_cooldown = 0
            for t in range(3, step_count - horizon):
                key = (inc_id, s_id, t)
                if key not in cached_records_by_key:
                    continue  # Eligible origin windows only
                row = cached_records_by_key[key]

                # Sustained operational state check: if already degraded, not healthy exposure
                if row.get("is_curr_service_degraded", 0) == 1:
                    continue

                healthy_exposure_origins += 1
                is_alert = (origin_predictions.get(key, 0) == 1)
                has_imminent_onset = (row.get("y_event", 0) == 1)

                if is_alert and not has_imminent_onset:
                    raw_false_alerts += 1
                    if in_cooldown == 0:
                        dedup_false_episodes += 1
                        in_cooldown = cooldown_steps
                    else:
                        in_cooldown -= 1
                else:
                    if in_cooldown > 0:
                        in_cooldown -= 1

    exposure_sec = healthy_exposure_origins * dt
    exposure_days = round(exposure_sec / 86400.0, 4)
    false_alarms_per_day = round((dedup_false_episodes / exposure_days), 2) if exposure_days > 0 else 0.0

    tot_eval = len(evaluable_episodes)
    lead_stats = {
        "mean_sec": round(float(np.mean(lead_times_sec)), 2) if lead_times_sec else None,
        "median_sec": round(float(np.median(lead_times_sec)), 2) if lead_times_sec else None,
        "p10_sec": round(float(np.percentile(lead_times_sec, 10)), 2) if lead_times_sec else None,
        "min_sec": round(float(np.min(lead_times_sec)), 2) if lead_times_sec else None,
        "max_sec": round(float(np.max(lead_times_sec)), 2) if lead_times_sec else None,
        "std_sec": round(float(np.std(lead_times_sec)), 2) if lead_times_sec else None,
        "lead_times_sec": [round(x, 2) for x in lead_times_sec],
    }

    return {
        "evaluable_episodes": tot_eval,
        "missed_events_count": tot_eval - recalled_evaluable,
        "recalled_evaluable_episodes": recalled_evaluable,
        "event_early_detection_recall": round(recalled_evaluable / tot_eval, 4) if tot_eval > 0 else 0.0,
        "recalled_actionable_episodes": recalled_actionable,
        "actionable_recall": round(recalled_actionable / tot_eval, 4) if tot_eval > 0 else 0.0,
        "lead_time_distribution": lead_stats,
        "healthy_exposure_origins": healthy_exposure_origins,
        "healthy_exposure_days": exposure_days,
        "raw_false_alerts_count": raw_false_alerts,
        "deduplicated_false_alert_episodes": dedup_false_episodes,
        "false_alarms_per_day": false_alarms_per_day,
        "per_event_records": per_event_records,
    }


def compute_priority_weighted_macro(service_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Computes architectural priority-weighted macro metric across 3 core cloud services.

    Formula: (1.5 * Reservations + 1.0 * Crew + 0.5 * Baggage) / 3.0
    Gateway and legacy core are reported separately as interconnect infrastructure.
    """
    weights = {"reservations": 1.5, "crew": 1.0, "baggage": 0.5}
    norm_w = sum(weights.values())

    keys = ["precision", "recall", "f1", "pr_auc", "roc_auc", "brier_score", "ece"]
    weighted_res: Dict[str, Optional[float]] = {}

    for k in keys:
        vals = []
        for s, w in weights.items():
            v = service_metrics[s].get(k)
            if v is not None:
                vals.append((v, w))
        if len(vals) == len(weights):
            score = sum(v * w for v, w in vals) / norm_w
            weighted_res[k] = round(score, 4)
        else:
            weighted_res[k] = None

    return {
        "formula": "(1.5 * reservations + 1.0 * crew + 0.5 * baggage) / 3.0",
        "weights": weights,
        "metrics": weighted_res,
    }


def run_incident_cluster_bootstrap(
    cached_df: List[Dict[str, Any]],
    model_col_a: str,
    model_col_b: Optional[str] = None,
    n_resamples: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Runs incident-cluster bootstrap with 500 replicates for confidence intervals and paired differences."""
    rng = np.random.RandomState(random_state)
    inc_ids = sorted(list(set(row["incident_id"] for row in cached_df)))
    if len(inc_ids) < 2:
        return {"replicates": 0, "status": "insufficient_incidents"}

    inc_to_rows = {inc: [] for inc in inc_ids}
    for row in cached_df:
        inc_to_rows[row["incident_id"]].append(row)

    # Observed point estimates on unresampled data
    y_all = np.array([r["y_event"] for r in cached_df])
    p_a_all = np.array([r[model_col_a] for r in cached_df])
    tp_a_obs = np.sum((p_a_all == 1) & (y_all == 1))
    fp_a_obs = np.sum((p_a_all == 1) & (y_all == 0))
    fn_a_obs = np.sum((p_a_all == 0) & (y_all == 1))
    prec_a_obs = tp_a_obs / (tp_a_obs + fp_a_obs) if (tp_a_obs + fp_a_obs) > 0 else 0.0
    rec_a_obs = tp_a_obs / (tp_a_obs + fn_a_obs) if (tp_a_obs + fn_a_obs) > 0 else 0.0
    f1_a_obs = (2.0 * prec_a_obs * rec_a_obs / (prec_a_obs + rec_a_obs)) if (prec_a_obs + rec_a_obs) > 0 else 0.0

    f1_b_obs = None
    observed_diff_f1 = None
    if model_col_b is not None:
        p_b_all = np.array([r[model_col_b] for r in cached_df])
        tp_b_obs = np.sum((p_b_all == 1) & (y_all == 1))
        fp_b_obs = np.sum((p_b_all == 1) & (y_all == 0))
        fn_b_obs = np.sum((p_b_all == 0) & (y_all == 1))
        prec_b_obs = tp_b_obs / (tp_b_obs + fp_b_obs) if (tp_b_obs + fp_b_obs) > 0 else 0.0
        rec_b_obs = tp_b_obs / (tp_b_obs + fn_b_obs) if (tp_b_obs + fn_b_obs) > 0 else 0.0
        f1_b_obs = (2.0 * prec_b_obs * rec_b_obs / (prec_b_obs + rec_b_obs)) if (prec_b_obs + rec_b_obs) > 0 else 0.0
        observed_diff_f1 = round(float(f1_a_obs - f1_b_obs), 4)

    prec_a_list, rec_a_list, f1_a_list = [], [], []
    diff_f1_list = []

    for _ in range(n_resamples):
        sampled_incs = rng.choice(inc_ids, size=len(inc_ids), replace=True)
        sampled_rows = []
        for inc in sampled_incs:
            sampled_rows.extend(inc_to_rows[inc])

        y_true = np.array([r["y_event"] for r in sampled_rows])
        pred_a = np.array([r[model_col_a] for r in sampled_rows])

        tp_a = np.sum((pred_a == 1) & (y_true == 1))
        fp_a = np.sum((pred_a == 1) & (y_true == 0))
        fn = np.sum((pred_a == 0) & (y_true == 1))

        p_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
        r_a = tp_a / (tp_a + fn) if (tp_a + fn) > 0 else 0.0
        f1_a = (2.0 * p_a * r_a / (p_a + r_a)) if (p_a + r_a) > 0 else 0.0

        prec_a_list.append(p_a)
        rec_a_list.append(r_a)
        f1_a_list.append(f1_a)

        if model_col_b is not None:
            pred_b = np.array([r[model_col_b] for r in sampled_rows])
            tp_b = np.sum((pred_b == 1) & (y_true == 1))
            fp_b = np.sum((pred_b == 1) & (y_true == 0))
            fn_b = np.sum((pred_b == 0) & (y_true == 1))
            p_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
            r_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
            f1_b = (2.0 * p_b * r_b / (p_b + r_b)) if (p_b + r_b) > 0 else 0.0
            diff_f1_list.append(f1_a - f1_b)

    ci_f1 = [round(float(np.percentile(f1_a_list, 2.5)), 4), round(float(np.percentile(f1_a_list, 97.5)), 4)]
    res: Dict[str, Any] = {
        "replicates_computed": len(f1_a_list),
        "observed_f1": round(float(f1_a_obs), 4),
        "bootstrap_mean_f1": round(float(np.mean(f1_a_list)), 4),
        "f1_ci_95": ci_f1,
        "precision_ci_95": [round(float(np.percentile(prec_a_list, 2.5)), 4), round(float(np.percentile(prec_a_list, 97.5)), 4)],
        "recall_ci_95": [round(float(np.percentile(rec_a_list, 2.5)), 4), round(float(np.percentile(rec_a_list, 97.5)), 4)],
    }

    if model_col_b is not None and diff_f1_list:
        ci_diff = [round(float(np.percentile(diff_f1_list, 2.5)), 4), round(float(np.percentile(diff_f1_list, 97.5)), 4)]
        excludes_zero = bool(ci_diff[0] > 0 or ci_diff[1] < 0)
        res["observed_diff_f1"] = observed_diff_f1
        res["bootstrap_mean_diff_f1"] = round(float(np.mean(diff_f1_list)), 4)
        res["diff_f1_ci_95"] = ci_diff
        res["diff_excludes_zero"] = excludes_zero

    return res


def fit_ablation_models(
    train_samples_by_h: Dict[int, Tuple[np.ndarray, np.ndarray]],
    val_samples_by_h: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Dict[int, Any], Dict[int, Any]]:
    """Fits missing reduced models once on training data, calibrated on validation data."""
    # Indices for No-Graph (remove global_max_backlog idx 18, global_max_latency_ratio idx 19)
    idx_no_graph = [i for i in range(35) if i not in (18, 19)]

    # Indices for No-Boundary (remove boundary features 11..17)
    idx_no_boundary = [i for i in range(35) if i not in (11, 12, 13, 14, 15, 16, 17)]

    models_no_graph: Dict[int, Any] = {}
    models_no_boundary: Dict[int, Any] = {}

    for h in [1, 5, 15]:
        X_tr, y_tr = train_samples_by_h[h]
        X_v, y_v = val_samples_by_h[h]

        # 1. No Graph Ablation
        clf_ng = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=42)
        clf_ng.fit(X_tr[:, idx_no_graph], y_tr)
        cal_ng = CalibratedClassifierCV(clf_ng, cv="prefit", method="isotonic")
        cal_ng.fit(X_v[:, idx_no_graph], y_v)
        models_no_graph[h] = {"model": cal_ng, "indices": idx_no_graph}

        # 2. No Boundary Ablation
        clf_nb = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=42)
        clf_nb.fit(X_tr[:, idx_no_boundary], y_tr)
        cal_nb = CalibratedClassifierCV(clf_nb, cv="prefit", method="isotonic")
        cal_nb.fit(X_v[:, idx_no_boundary], y_v)
        models_no_boundary[h] = {"model": cal_nb, "indices": idx_no_boundary}

    return models_no_graph, models_no_boundary


def create_causal_missing_telemetry_cohort(
    incidents: List[Dict[str, Any]],
    dropout_prob: float = 0.15,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """Preprocessing sensitivity experiment: simulates missing observations in raw telemetry streams
    manually carried forward with refreshed timestamps (does not verify inference staleness rejection)."""
    rng = np.random.RandomState(random_state)
    simulated_incidents = []

    for inc in incidents:
        inc_copy = copy.deepcopy(inc)
        svc_tel = inc_copy["service_telemetry"]
        bnd_tel = inc_copy["boundary_telemetry"]
        step_count = inc_copy["step_count"]

        # Simulate packet dropouts per service with causal carry-forward
        for s_id, stream in svc_tel.items():
            last_valid = copy.deepcopy(stream[0])
            for t in range(1, step_count):
                if rng.rand() < dropout_prob:
                    carried = copy.deepcopy(last_valid)
                    carried["timestamp_sec"] = stream[t].get("timestamp_sec", t * SAMPLING_INTERVAL_SECONDS)
                    stream[t] = carried
                else:
                    last_valid = copy.deepcopy(stream[t])

        # Simulate packet dropouts in boundary stream
        last_bnd = copy.deepcopy(bnd_tel[0])
        for t in range(1, step_count):
            if rng.rand() < dropout_prob:
                carried_bnd = copy.deepcopy(last_bnd)
                carried_bnd["timestamp_sec"] = bnd_tel[t].get("timestamp_sec", t * SAMPLING_INTERVAL_SECONDS)
                bnd_tel[t] = carried_bnd
            else:
                last_bnd = copy.deepcopy(bnd_tel[t])

        simulated_incidents.append(inc_copy)
    return simulated_incidents


def evaluate_causal_dropped_telemetry_robustness(
    incidents: List[Dict[str, Any]],
    metas: List[Dict[str, Any]],
    y_true: np.ndarray,
    h_rob: int,
    m_b_rob: Any,
    th_rob: float,
    forecaster: Any,
    zscore_baseline: Optional[Any] = None,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates production incomplete telemetry robustness by removing dropped observations,
    preserving surviving timestamps, and passing incomplete histories through the actual inference
    alignment and staleness-handling path."""
    predictor = DegradationPredictor.get_instance()
    rng = np.random.RandomState(random_state)
    dt = SAMPLING_INTERVAL_SECONDS
    services = [s.value for s in ServiceIdentifier]

    # 1. Short Gap Condition (<= 60s): 15% random packet drops (max 2 consecutive drops -> max 40s gap <= 60s)
    # Dropped observations are REMOVED from the stream completely! Surviving timestamps are preserved.
    dropped_incidents_short = []
    for inc in incidents:
        inc_copy = copy.deepcopy(inc)
        new_svc_tel = {}
        for s_id, stream in inc_copy["service_telemetry"].items():
            new_stream = [stream[0]]
            consec_drops = 0
            for t in range(1, inc_copy["step_count"]):
                if rng.rand() < 0.15 and consec_drops < 2:
                    consec_drops += 1
                else:
                    new_stream.append(stream[t])
                    consec_drops = 0
            new_svc_tel[s_id] = new_stream

        new_bnd_tel = [inc_copy["boundary_telemetry"][0]]
        consec_drops_bnd = 0
        for t in range(1, inc_copy["step_count"]):
            if rng.rand() < 0.15 and consec_drops_bnd < 2:
                consec_drops_bnd += 1
            else:
                new_bnd_tel.append(inc_copy["boundary_telemetry"][t])
                consec_drops_bnd = 0

        inc_copy["service_telemetry"] = new_svc_tel
        inc_copy["boundary_telemetry"] = new_bnd_tel
        dropped_incidents_short.append(inc_copy)

    # Pass incomplete history through production inference alignment path
    X_short_list = []
    carried_forward_flags = []

    for meta in metas:
        inc_id = meta["incident_id"]
        s_id = meta["service_id"]
        origin_t = meta["origin_step"]
        target_ts = origin_t * dt

        d_inc = next(i for i in dropped_incidents_short if i["incident_id"] == inc_id)

        svc_sub = {
            s: [r for r in d_inc["service_telemetry"][s] if r.get("timestamp_sec", 0.0) <= target_ts + 1e-4]
            for s in services
        }
        bnd_sub = [r for r in d_inc["boundary_telemetry"] if r.get("timestamp_sec", 0.0) <= target_ts + 1e-4]

        # Actual inference alignment path
        aligned_svc, aligned_bnd, origin_idx, aligned_ts, warns, _, acct = predictor._validate_and_sanitize_inputs(
            svc_sub, bnd_sub, explicit_timestamp_sec=target_ts
        )
        carried_forward_flags.append(bool(acct["carried_forward_services"]))

        feat_vec, _ = extract_features_at_origin(
            service_id=s_id,
            incident_service_telemetry=aligned_svc,
            incident_boundary_telemetry=aligned_bnd,
            origin_step=origin_idx,
            horizon=h_rob,
            forecaster=forecaster,
            include_workload_forecasts=True,
            physical_timestamp_sec=target_ts,
        )
        X_short_list.append(feat_vec)

    X_short = np.array(X_short_list)
    p_b_short = m_b_rob.predict_event_proba(X_short)
    pred_b_short = (p_b_short >= th_rob).astype(int)
    bin_b_short = compute_binary_metrics(y_true, p_b_short, pred_b_short)

    # Compute genuine Z-score anomaly baseline metrics on perturbed features if baseline provided
    if zscore_baseline is not None:
        z_preds = []
        for idx, meta in enumerate(metas):
            s_id = meta["service_id"]
            curr_lat = float(X_short[idx, 0])
            curr_q = float(X_short[idx, 3])
            z_pred = int(zscore_baseline.predict(s_id, curr_lat, curr_q))
            z_preds.append(z_pred)
        bin_z_short = compute_binary_metrics(y_true, np.array(z_preds, dtype=float), np.array(z_preds, dtype=int))
        zscore_metrics = {
            "f1": bin_z_short["f1"],
            "precision": bin_z_short["precision"],
            "recall": bin_z_short["recall"],
            "pr_auc": None,
        }
    else:
        zscore_metrics = {"f1": None, "precision": None, "recall": None, "pr_auc": None, "note": "not_evaluated"}

    short_results = {
        "description": "15% raw telemetry observations dropped (removed from stream without carry-forward; surviving timestamps preserved) and processed through actual production inference alignment and staleness path (gaps <= 60s)",
        "sample_count": len(y_true),
        "prevalence": bin_b_short["prevalence"],
        "model_b": {"f1": bin_b_short["f1"], "precision": bin_b_short["precision"], "recall": bin_b_short["recall"], "pr_auc": bin_b_short["pr_auc"]},
        "zscore": zscore_metrics,
        "max_gap_seconds": 40.0,
        "staleness_limit_seconds": MAX_TELEMETRY_STALENESS_SECONDS,
        "carried_forward_rate": float(round(sum(carried_forward_flags) / len(carried_forward_flags), 4)),
        "behavior": "causal_carry_forward_with_warnings",
    }

    # 2. Long Gap Condition (> 60s): Introduce an 80-second telemetry blackout preceding target_ts
    rejections_count = 0
    total_long_tests = 0
    for meta in metas:
        if meta["origin_step"] < 4:
            continue
        inc_id = meta["incident_id"]
        origin_t = meta["origin_step"]
        target_ts = origin_t * dt

        orig_inc = next(i for i in incidents if i["incident_id"] == inc_id)
        svc_sub = {
            s: [r for r in orig_inc["service_telemetry"][s] if r.get("timestamp_sec", 0.0) <= target_ts - 60.1]
            for s in services
        }
        bnd_sub = [r for r in orig_inc["boundary_telemetry"] if r.get("timestamp_sec", 0.0) <= target_ts - 60.1]

        total_long_tests += 1
        try:
            predictor._validate_and_sanitize_inputs(svc_sub, bnd_sub, explicit_timestamp_sec=target_ts)
        except ValueError as exc:
            if "Staleness limit exceeded" in str(exc):
                rejections_count += 1
            else:
                raise

    long_results = {
        "description": "Incomplete telemetry with gap exceeding 60 seconds (80s gap) processed through actual production inference alignment and staleness path",
        "sample_count": total_long_tests,
        "observed_gap_seconds": 80.0,
        "staleness_limit_seconds": MAX_TELEMETRY_STALENESS_SECONDS,
        "rejections_count": rejections_count,
        "rejection_rate": float(round(rejections_count / total_long_tests, 4)) if total_long_tests > 0 else 1.0,
        "behavior": "rejected_staleness_limit_exceeded_unavailable",
    }

    return short_results, long_results


def run_task4_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("STARTING TASK 4A COMPREHENSIVE EVALUATION & COMPARISONS")
    print("=" * 80)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    with open(WEIGHTS_PATH, "rb") as f:
        frozen_bundle = pickle.load(f)

    forecaster = WorkloadForecaster.get_instance()

    # 1. Split manifest indexing
    incidents_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for inc in full_data["incidents"]:
        incidents_by_split.setdefault(inc.get("split", "train"), []).append(inc)

    print(f"Loaded dataset: {len(full_data['incidents'])} incidents total.")
    for sp, l in incidents_by_split.items():
        print(f"  {sp}: {len(l)} incidents")

    train_inc = incidents_by_split["train"]
    val_inc = incidents_by_split["val"]
    std_test_inc = incidents_by_split["standard_test"]

    # 2. Fit training-based Z-score baseline
    zscore_baseline = TrainingFittedZScoreBaseline()
    zscore_baseline.fit(train_inc)

    # 3. Extract train and val samples for baseline tuning & ablations
    train_samples_by_h: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    val_samples_by_h: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    val_metas_by_h: Dict[int, List[Dict[str, Any]]] = {}

    for h in [1, 5, 15]:
        X_tr, _, _, _, _, y_tr, _, _, _, _ = extract_split_samples(
            train_inc, horizon=h, include_workload_forecasts=True, forecaster=forecaster, extract_temporal=False
        )
        train_samples_by_h[h] = (X_tr, y_tr)

        X_v, _, _, _, _, y_v, _, _, _, metas_v = extract_split_samples(
            val_inc, horizon=h, include_workload_forecasts=True, forecaster=forecaster, extract_temporal=False
        )
        val_samples_by_h[h] = (X_v, y_v)
        val_metas_by_h[h] = metas_v

    # Tune operating point on validation data using named feature indices
    zscore_thresh = zscore_baseline.tune_operating_point(
        val_samples_by_h[1][0], val_metas_by_h[1], val_samples_by_h[1][1]
    )
    print(f"Fitted Z-Score baseline: tuned operating threshold on validation split = {zscore_thresh:.2f}")

    # 4. Fit lightweight ablation models once on train split
    models_no_graph, models_no_boundary = fit_ablation_models(train_samples_by_h, val_samples_by_h)

    # 5. Extract test samples on standard_test (and temporal tensors for Model C)
    test_data_by_h: Dict[int, Dict[str, Any]] = {}
    for h in [1, 5, 15]:
        print(f"Extracting standard_test samples for H={h}...")
        X, X_t, y_lat, y_q, y_dr, y_ev, y_time, y_typ, y_sev, metas = extract_split_samples(
            std_test_inc, horizon=h, include_workload_forecasts=True, forecaster=forecaster, extract_temporal=True
        )
        test_data_by_h[h] = {
            "X": X,
            "X_temp": X_t,
            "y_lat": y_lat,
            "y_q": y_q,
            "y_dr": y_dr,
            "y_event": y_ev,
            "y_time": y_time,
            "metas": metas,
        }

    # 6. Generate and cache predictions across all models on standard_test
    print("\nGenerating cached predictions across models and baselines...")
    cached_predictions: Dict[int, List[Dict[str, Any]]] = {}
    origin_preds_map: Dict[str, Dict[int, Dict[Tuple[str, str, int], int]]] = {
        m: {1: {}, 5: {}, 15: {}}
        for m in [
            "static_threshold", "zscore_baseline", "model_a_telemetry_only",
            "model_b_telemetry_forecast", "model_c_temporal_comparator",
            "ablation_no_graph", "ablation_no_boundary"
        ]
    }

    static_baseline = StaticThresholdBaseline(ratio=0.75)

    for h in [1, 5, 15]:
        data = test_data_by_h[h]
        X = data["X"]
        X_t = data["X_temp"]
        metas = data["metas"]
        y_ev = data["y_event"]
        th = DEPLOYED_THRESHOLDS[h]

        # Extract baseline inputs using named feature indices with strict validation
        test_baseline_inputs = extract_baseline_inputs(X, metas)

        # Model A (telemetry only, first 22 features)
        m_a = frozen_bundle["models_telemetry_only"][h]
        prob_a = m_a.predict_event_proba(X[:, :22])
        pred_a = (prob_a >= th).astype(int)

        # Model B (telemetry + forecast, 35 features)
        m_b = frozen_bundle["models_telemetry_forecast"][h]
        prob_b = m_b.predict_event_proba(X)
        pred_b = (prob_b >= th).astype(int)

        # Model C (temporal GRU comparator)
        m_c = frozen_bundle["models_temporal_comparator"][h]
        prob_c = m_c.predict_event_proba(X_t)
        pred_c = (prob_c >= th).astype(int)

        # Ablations
        idx_ng = models_no_graph[h]["indices"]
        prob_ng = models_no_graph[h]["model"].predict_proba(X[:, idx_ng])[:, 1]
        pred_ng = (prob_ng >= th).astype(int)

        idx_nb = models_no_boundary[h]["indices"]
        prob_nb = models_no_boundary[h]["model"].predict_proba(X[:, idx_nb])[:, 1]
        pred_nb = (prob_nb >= th).astype(int)

        # Time-to-degradation regression
        m_time = frozen_bundle["models_time_to_onset"][h]
        pred_time = m_time.predict_time_to_onset_sec(X, pred_b)

        cached_rows: List[Dict[str, Any]] = []
        for i in range(len(metas)):
            m = metas[i]
            inp = test_baseline_inputs[i]
            inc_id = m["incident_id"]
            s_id = m["service_id"]
            t = m["origin_step"]

            curr_lat = inp["curr_latency"]
            curr_q = inp["curr_backlog"]
            curr_err = inp["curr_error_rate"]
            curr_dr = inp["curr_sync_drift"]

            # Static threshold decision
            static_alert = int(static_baseline.predict_event_prob(s_id, curr_lat, curr_q, curr_err, curr_dr))

            # Z-score anomaly decision
            z_alert = int(zscore_baseline.predict(s_id, curr_lat, curr_q))

            origin_preds_map["static_threshold"][h][(inc_id, s_id, t)] = static_alert
            origin_preds_map["zscore_baseline"][h][(inc_id, s_id, t)] = z_alert
            origin_preds_map["model_a_telemetry_only"][h][(inc_id, s_id, t)] = int(pred_a[i])
            origin_preds_map["model_b_telemetry_forecast"][h][(inc_id, s_id, t)] = int(pred_b[i])
            origin_preds_map["model_c_temporal_comparator"][h][(inc_id, s_id, t)] = int(pred_c[i])
            origin_preds_map["ablation_no_graph"][h][(inc_id, s_id, t)] = int(pred_ng[i])
            origin_preds_map["ablation_no_boundary"][h][(inc_id, s_id, t)] = int(pred_nb[i])

            y_time_steps = float(data["y_time"][i])
            y_time_sec = float(y_time_steps * SAMPLING_INTERVAL_SECONDS) if y_time_steps >= 0 else None

            cached_rows.append({
                "horizon": h,
                "incident_id": inc_id,
                "service_id": s_id,
                "origin_step": t,
                "timestamp_sec": float(t * SAMPLING_INTERVAL_SECONDS),
                "is_curr_service_degraded": int(m["is_curr_service_degraded"]),
                "label_is_observable": int(m["label_is_observable"]),
                "y_event": int(y_ev[i]),
                "y_time_to_onset_steps": y_time_steps,
                "y_time_to_onset_sec": y_time_sec,
                "static_pred": static_alert,
                "zscore_pred": z_alert,
                "model_a_prob": float(prob_a[i]),
                "model_a_pred": int(pred_a[i]),
                "model_b_prob": float(prob_b[i]),
                "model_b_pred": int(pred_b[i]),
                "model_c_prob": float(prob_c[i]),
                "model_c_pred": int(pred_c[i]),
                "ablation_no_graph_prob": float(prob_ng[i]),
                "ablation_no_graph_pred": int(pred_ng[i]),
                "ablation_no_boundary_prob": float(prob_nb[i]),
                "ablation_no_boundary_pred": int(pred_nb[i]),
                "estimated_time_to_degradation_sec": round(float(pred_time[i]), 1) if pred_time[i] is not None else None,
            })

        cached_predictions[h] = cached_rows

    # 7. Evaluate model performance & comparisons
    print("\nComputing comprehensive performance metrics and early-warning lead times...")
    results_by_horizon: Dict[int, Dict[str, Any]] = {}

    models_to_evaluate = [
        ("Static SLO Threshold Alerting", "static_pred", None, "static_threshold"),
        ("Lightweight Z-Score Anomaly", "zscore_pred", None, "zscore_baseline"),
        ("Model A (Telemetry Only)", "model_a_pred", "model_a_prob", "model_a_telemetry_only"),
        ("Model B (Telemetry + Forecast)", "model_b_pred", "model_b_prob", "model_b_telemetry_forecast"),
        ("Model C (Temporal Comparator GRU)", "model_c_pred", "model_c_prob", "model_c_temporal_comparator"),
    ]

    for h in [1, 5, 15]:
        rows = cached_predictions[h]
        y_true_all = np.array([r["y_event"] for r in rows])
        h_sec = float(h * SAMPLING_INTERVAL_SECONDS)
        h_eval: Dict[str, Any] = {
            "horizon_steps": h,
            "horizon_seconds": h_sec,
            "deployed_threshold": DEPLOYED_THRESHOLDS[h],
        }

        records_map = {(r["incident_id"], r["service_id"], r["origin_step"]): r for r in rows}

        for display_name, pred_col, prob_col, model_key in models_to_evaluate:
            preds_all = np.array([r[pred_col] for r in rows])
            probs_all = np.array([r[prob_col] for r in rows]) if prob_col else None

            # Overall classification metrics
            overall_bin = compute_binary_metrics(y_true_all, probs_all, preds_all)

            # Per-service classification metrics
            svc_metrics: Dict[str, Any] = {}
            for s_id in ALL_SERVICES:
                s_mask = np.array([r["service_id"] == s_id for r in rows])
                y_s = y_true_all[s_mask]
                p_s = preds_all[s_mask]
                pr_s = probs_all[s_mask] if probs_all is not None else None
                svc_metrics[s_id] = compute_binary_metrics(y_s, pr_s, p_s)

            # Priority-weighted macro metric (Reservations: 1.5, Crew: 1.0, Baggage: 0.5)
            macro_3domain = compute_priority_weighted_macro(svc_metrics)

            # Early warning episode evaluation
            ew_metrics = compute_early_warning_metrics(
                incidents=std_test_inc,
                origin_predictions=origin_preds_map[model_key][h],
                cached_records_by_key=records_map,
                horizon=h,
                cooldown_steps=5,
                min_response_time_sec=20.0,
            )

            # Time to degradation MAE (on positive events where alert fired, Model B only)
            if model_key == "model_b_telemetry_forecast":
                valid_time_mask = np.array([r["estimated_time_to_degradation_sec"] is not None for r in rows]) & (y_true_all == 1)
                if np.sum(valid_time_mask) > 0:
                    y_time_true_sec = np.array([r["y_time_to_onset_sec"] for r in rows])[valid_time_mask]
                    y_time_pred_sec = np.array([r["estimated_time_to_degradation_sec"] for r in rows])[valid_time_mask]
                    time_mae = float(np.mean(np.abs(y_time_pred_sec - y_time_true_sec)))
                    time_cov = float(np.sum(valid_time_mask) / max(1, np.sum(y_true_all == 1)))
                else:
                    time_mae = None
                    time_cov = 0.0
            else:
                time_mae = None
                time_cov = None

            h_eval[model_key] = {
                "display_name": display_name,
                "overall_classification": overall_bin,
                "priority_weighted_macro_3domain": macro_3domain,
                "per_service_metrics": svc_metrics,
                "early_warning_metrics": ew_metrics,
                "time_to_degradation_mae_sec": round(time_mae, 2) if time_mae is not None else None,
                "time_to_degradation_coverage": round(time_cov, 4) if time_cov is not None else None,
            }

        # Incident-cluster bootstrap
        boot_b = run_incident_cluster_bootstrap(rows, model_col_a="model_b_pred", n_resamples=500)
        boot_a = run_incident_cluster_bootstrap(rows, model_col_a="model_a_pred", n_resamples=500)
        boot_c = run_incident_cluster_bootstrap(rows, model_col_a="model_c_pred", n_resamples=500)
        boot_b_vs_a = run_incident_cluster_bootstrap(rows, model_col_a="model_b_pred", model_col_b="model_a_pred", n_resamples=500)
        boot_b_vs_c = run_incident_cluster_bootstrap(rows, model_col_a="model_b_pred", model_col_b="model_c_pred", n_resamples=500)
        boot_b_vs_static = run_incident_cluster_bootstrap(rows, model_col_a="model_b_pred", model_col_b="static_pred", n_resamples=500)
        boot_b_vs_zscore = run_incident_cluster_bootstrap(rows, model_col_a="model_b_pred", model_col_b="zscore_pred", n_resamples=500)

        h_eval["bootstrap_uncertainty"] = {
            "model_b": boot_b,
            "model_a": boot_a,
            "model_c": boot_c,
            "diff_model_b_minus_a": boot_b_vs_a,
            "diff_model_b_minus_c": boot_b_vs_c,
            "diff_model_b_minus_static": boot_b_vs_static,
            "diff_model_b_minus_zscore": boot_b_vs_zscore,
        }

        results_by_horizon[h] = h_eval

    # 8. Ablations Evaluation
    print("\nEvaluating feature ablation cohorts (Full Model B vs No-Forecast, No-Graph, No-Boundary)...")
    ablations_results: Dict[str, Any] = {}
    ablation_cohorts = [
        ("Full Model B", "model_b_pred", "model_b_prob", "None (all 35 features retained)"),
        ("No Forecast (Model A)", "model_a_pred", "model_a_prob", "Removed 13 workload forecast features (indices 22..34)"),
        ("No Graph / Neighbour", "ablation_no_graph_pred", "ablation_no_graph_prob", "Removed cross-service neighbour metrics (global_max_backlog, global_max_latency_ratio)"),
        ("No Boundary Gateway", "ablation_no_boundary_pred", "ablation_no_boundary_prob", "Removed 7 boundary gateway & sync drift features (indices 11..17)"),
    ]

    for h in [1, 5, 15]:
        rows = cached_predictions[h]
        y_true_all = np.array([r["y_event"] for r in rows])
        records_map = {(r["incident_id"], r["service_id"], r["origin_step"]): r for r in rows}
        h_ablation = {}

        for name, pred_col, prob_col, desc in ablation_cohorts:
            preds = np.array([r[pred_col] for r in rows])
            probs = np.array([r[prob_col] for r in rows])
            bin_m = compute_binary_metrics(y_true_all, probs, preds)

            model_key = (
                "model_b_telemetry_forecast" if "Full" in name else
                "model_a_telemetry_only" if "No Forecast" in name else
                "ablation_no_graph" if "No Graph" in name else
                "ablation_no_boundary"
            )
            ew_m = compute_early_warning_metrics(
                incidents=std_test_inc,
                origin_predictions=origin_preds_map[model_key][h],
                cached_records_by_key=records_map,
                horizon=h,
            )

            h_ablation[name] = {
                "description": desc,
                "precision": bin_m["precision"],
                "recall": bin_m["recall"],
                "f1": bin_m["f1"],
                "pr_auc": bin_m["pr_auc"],
                "false_alarms_per_day": ew_m["false_alarms_per_day"],
                "actionable_recall": ew_m["actionable_recall"],
                "median_lead_time_sec": ew_m["lead_time_distribution"]["median_sec"],
            }
        ablations_results[f"H={h}"] = h_ablation

    # 9. Robustness Matrix Evaluation
    print("\nEvaluating robustness matrix on held-out stress cohorts and perturbed telemetry...")
    robustness_results: Dict[str, Any] = {}
    stress_splits = [
        ("capacity_loss_test", "Unseen fault-intensity split with physical server/thread pool capacity loss"),
        ("boundary_slowdown_test", "Unseen drift stress split with boundary transit latency & sync drift degradation"),
        ("surge_test", "Unseen workload surge split with abrupt 2.5x traffic bursts"),
    ]

    h_rob = 5
    th_rob = DEPLOYED_THRESHOLDS[h_rob]
    m_b_rob = frozen_bundle["models_telemetry_forecast"][h_rob]

    # A. Clean Reference (standard_test, H=5, 25 incidents)
    X_std = test_data_by_h[h_rob]["X"]
    y_std = test_data_by_h[h_rob]["y_event"]
    metas_std = test_data_by_h[h_rob]["metas"]

    clean_inputs = extract_baseline_inputs(X_std, metas_std)
    p_b_std = m_b_rob.predict_event_proba(X_std)
    pred_b_std = (p_b_std >= th_rob).astype(int)
    bin_b_std = compute_binary_metrics(y_std, p_b_std, pred_b_std)

    pred_z_std = np.array([
        zscore_baseline.predict(clean_inputs[i]["service_id"], clean_inputs[i]["curr_latency"], clean_inputs[i]["curr_backlog"])
        for i in range(len(y_std))
    ])
    bin_z_std = compute_binary_metrics(y_std, None, pred_z_std)

    robustness_results["clean_reference"] = {
        "description": "Standard clean test cohort (standard_test, 25 incidents at H=5)",
        "sample_count": len(y_std),
        "prevalence": bin_b_std["prevalence"],
        "model_b": {"f1": bin_b_std["f1"], "precision": bin_b_std["precision"], "recall": bin_b_std["recall"], "pr_auc": bin_b_std["pr_auc"]},
        "zscore": {"f1": bin_z_std["f1"], "precision": bin_z_std["precision"], "recall": bin_z_std["recall"], "pr_auc": None},
    }

    # B1. Preprocessing Sensitivity Experiment (15% manual carry-forward with refreshed timestamps)
    causal_missing_inc = create_causal_missing_telemetry_cohort(std_test_inc, dropout_prob=0.15, random_state=42)
    X_miss, _, _, _, _, y_miss, _, _, _, metas_miss = extract_split_samples(
        causal_missing_inc, horizon=h_rob, include_workload_forecasts=True, forecaster=forecaster, extract_temporal=False
    )
    p_b_miss = m_b_rob.predict_event_proba(X_miss)
    pred_b_miss = (p_b_miss >= th_rob).astype(int)
    bin_b_miss = compute_binary_metrics(y_miss, p_b_miss, pred_b_miss)

    miss_inputs = extract_baseline_inputs(X_miss, metas_miss)
    pred_z_miss = np.array([
        zscore_baseline.predict(miss_inputs[i]["service_id"], miss_inputs[i]["curr_latency"], miss_inputs[i]["curr_backlog"])
        for i in range(len(y_miss))
    ])
    bin_z_miss = compute_binary_metrics(y_miss, None, pred_z_miss)

    robustness_results["causal_missing_telemetry_pipeline"] = {
        "description": "15% raw telemetry dropouts simulated with manual carry-forward and refreshed timestamps (preprocessing sensitivity experiment; does not test inference staleness rejection)",
        "sample_count": len(y_miss),
        "prevalence": bin_b_miss["prevalence"],
        "model_b": {"f1": bin_b_miss["f1"], "precision": bin_b_miss["precision"], "recall": bin_b_miss["recall"], "pr_auc": bin_b_miss["pr_auc"]},
        "zscore": {"f1": bin_z_miss["f1"], "precision": bin_z_miss["precision"], "recall": bin_z_miss["recall"], "pr_auc": None},
    }

    # B2. Genuine Incomplete Telemetry Alignment & Staleness Robustness (through production inference alignment path)
    short_rob_res, long_rob_res = evaluate_causal_dropped_telemetry_robustness(
        incidents=std_test_inc,
        metas=metas_std,
        y_true=y_std,
        h_rob=h_rob,
        m_b_rob=m_b_rob,
        th_rob=th_rob,
        forecaster=forecaster,
        zscore_baseline=zscore_baseline,
    )
    robustness_results["causal_incomplete_telemetry_alignment_within_60s"] = short_rob_res
    robustness_results["causal_incomplete_telemetry_staleness_exceeded_gt_60s"] = long_rob_res

    # C. Feature-Zeroing Sensitivity (20% random feature cells zeroed)
    rng_noise = np.random.RandomState(42)
    X_zeroed = copy.deepcopy(X_std)
    mask_drop = rng_noise.rand(*X_std.shape) < 0.20
    X_zeroed[mask_drop] = 0.0

    p_b_zero = m_b_rob.predict_event_proba(X_zeroed)
    pred_b_zero = (p_b_zero >= th_rob).astype(int)
    bin_b_zero = compute_binary_metrics(y_std, p_b_zero, pred_b_zero)

    robustness_results["feature_zeroing_sensitivity"] = {
        "description": "20% artificial feature zeroing sensitivity experiment",
        "sample_count": len(y_std),
        "prevalence": bin_b_std["prevalence"],
        "model_b": {"f1": bin_b_zero["f1"], "precision": bin_b_zero["precision"], "recall": bin_b_zero["recall"], "pr_auc": bin_b_zero["pr_auc"]},
    }

    # D. Feature-Space Noise Sensitivity (+/- 5% continuous Gaussian noise)
    noise_matrix = rng_noise.normal(0.0, 0.05, size=X_std.shape) * np.std(X_std, axis=0, keepdims=True)
    X_noisy = np.maximum(0.0, X_std + noise_matrix)

    # Preserve discrete categorical / configured capacity fields
    idx_has_cap = get_feature_index("has_known_capacity", schema="telemetry")
    X_noisy[:, idx_has_cap] = X_std[:, idx_has_cap]

    p_b_noisy = m_b_rob.predict_event_proba(X_noisy)
    pred_b_noisy = (p_b_noisy >= th_rob).astype(int)
    bin_b_noisy = compute_binary_metrics(y_std, p_b_noisy, pred_b_noisy)

    noisy_inputs = extract_baseline_inputs(X_noisy, metas_std)
    z_noisy_preds = np.array([
        zscore_baseline.predict(noisy_inputs[i]["service_id"], noisy_inputs[i]["curr_latency"], noisy_inputs[i]["curr_backlog"])
        for i in range(len(y_std))
    ])
    bin_z_noisy = compute_binary_metrics(y_std, None, z_noisy_preds)

    robustness_results["feature_space_noise_sensitivity"] = {
        "description": "Artificial Gaussian noise (+/- 5% sigma) applied to continuous telemetry feature space",
        "sample_count": len(y_std),
        "prevalence": bin_b_std["prevalence"],
        "model_b": {"f1": bin_b_noisy["f1"], "precision": bin_b_noisy["precision"], "recall": bin_b_noisy["recall"], "pr_auc": bin_b_noisy["pr_auc"]},
        "zscore": {"f1": bin_z_noisy["f1"], "precision": bin_z_noisy["precision"], "recall": bin_z_noisy["recall"], "pr_auc": None},
    }

    # E. Natural Held-Out Stress Splits
    for split_key, split_desc in stress_splits:
        inc_list = incidents_by_split.get(split_key, [])
        if not inc_list:
            continue
        X_s, _, _, _, _, y_s, _, _, _, metas_s = extract_split_samples(
            inc_list, horizon=h_rob, include_workload_forecasts=True, forecaster=forecaster, extract_temporal=False
        )
        p_b_s = m_b_rob.predict_event_proba(X_s)
        pred_b_s = (p_b_s >= th_rob).astype(int)
        bin_b_s = compute_binary_metrics(y_s, p_b_s, pred_b_s)

        stress_inputs = extract_baseline_inputs(X_s, metas_s)
        pred_z_s = np.array([
            zscore_baseline.predict(stress_inputs[i]["service_id"], stress_inputs[i]["curr_latency"], stress_inputs[i]["curr_backlog"])
            for i in range(len(y_s))
        ])
        bin_z_s = compute_binary_metrics(y_s, None, pred_z_s)

        robustness_results[split_key] = {
            "description": split_desc,
            "incident_count": len(inc_list),
            "sample_count": len(y_s),
            "prevalence": bin_b_s["prevalence"],
            "model_b": {"f1": bin_b_s["f1"], "precision": bin_b_s["precision"], "recall": bin_b_s["recall"], "pr_auc": bin_b_s["pr_auc"]},
            "zscore": {"f1": bin_z_s["f1"], "precision": bin_z_s["precision"], "recall": bin_z_s["recall"], "pr_auc": None},
        }

    # 10. Generate Task 4 Visualizations
    print("\nGenerating Task 4 plots (PR curves, reliability diagram, lead-time histograms, ablation chart)...")
    generate_task4_plots(results_by_horizon, cached_predictions, ablations_results)

    # 11. Write Evaluation Protocol, Model Card, and Limitations
    write_evaluation_protocol()
    write_model_card(results_by_horizon, ablations_results, robustness_results)
    write_limitations_document()

    # 12. Save machine-readable outputs
    task4_payload = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract_version": CONTRACT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_provenance": {
            "total_incidents": 275,
            "splits": {k: len(v) for k, v in incidents_by_split.items()},
            "status": "Previously inspected holdouts evaluated under predeclared protocol without parameter retuning.",
        },
        "results_by_horizon": results_by_horizon,
        "ablations": ablations_results,
        "robustness": robustness_results,
    }

    with open(RESULTS_DIR / "task4_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(task4_payload, f, indent=2)

    # Save cached predictions (compact per-origin records)
    with open(RESULTS_DIR / "cached_predictions.json", "w", encoding="utf-8") as f:
        json.dump(cached_predictions, f)

    # 13. Generate programmatic reconciled tables
    generate_report_markdown_tables(task4_payload)

    print("\n" + "=" * 80)
    print("TASK 4A COMPREHENSIVE EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return task4_payload


def generate_task4_plots(
    results_by_h: Dict[int, Any],
    cached_preds: Dict[int, List[Dict[str, Any]]],
    ablations: Dict[str, Any],
) -> None:
    """Generates the 4 required visualization deliverables."""
    # Plot 1: Precision-Recall Curve Comparison (H=5)
    plt.figure(figsize=(8, 6))
    h = 5
    rows = cached_preds[h]
    y_true = np.array([r["y_event"] for r in rows])

    models_to_plot = [
        ("Model B (Telemetry + Forecast)", "model_b_prob", "#1f77b4"),
        ("Model A (Telemetry Only)", "model_a_prob", "#ff7f0e"),
        ("Model C (Temporal GRU)", "model_c_prob", "#2ca02c"),
    ]

    for label, prob_col, color in models_to_plot:
        probs = np.array([r[prob_col] for r in rows])
        thresholds = np.linspace(0.01, 0.99, 100)
        precs, recs = [], []
        for th in thresholds:
            preds = (probs >= th).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))
            p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precs.append(p)
            recs.append(r)
        recs = [1.0] + recs + [0.0]
        precs = [float(np.mean(y_true))] + precs + [1.0]
        pr_auc = average_precision_score(y_true, probs)
        plt.plot(recs, precs, label=f"{label} (AP = {pr_auc:.3f})", color=color, linewidth=2)

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curves at Forward Horizon H=5 (100s)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pr_curve_comparison.png", dpi=200)
    plt.close()

    # Plot 2: Reliability Diagram (10 Bins, probability 1.0 included in final bin)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    colors = {1: "#1f77b4", 5: "#ff7f0e", 15: "#2ca02c"}
    for h in [1, 5, 15]:
        rows_h = cached_preds[h]
        y_tr = np.array([r["y_event"] for r in rows_h])
        y_pr = np.array([r["model_b_prob"] for r in rows_h])
        color = colors[h]

        bin_edges = np.linspace(0.0, 1.0, 11)
        conf_list, acc_list = [], []
        for b in range(10):
            low, high = bin_edges[b], bin_edges[b + 1]
            mask = (y_pr >= low) & (y_pr <= high) if b == 9 else (y_pr >= low) & (y_pr < high)
            if np.sum(mask) > 0:
                conf_list.append(float(np.mean(y_pr[mask])))
                acc_list.append(float(np.mean(y_tr[mask])))

        plt.plot(conf_list, acc_list, marker="o", linewidth=2, label=f"Model B H={h} ({h*20}s)", color=color)

    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Empirical Degradation Frequency", fontsize=12)
    plt.title("Reliability Diagram (10 Fixed Bins, Model B, [0, 1] Inclusive)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "reliability_diagram.png", dpi=200)
    plt.close()

    # Plot 3: Lead-Time Distribution Histogram across Horizons (Model B)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for idx_h, h in enumerate([1, 5, 15]):
        ax = axes[idx_h]
        ew_m = results_by_h[h]["model_b_telemetry_forecast"]["early_warning_metrics"]
        lead_times = ew_m["lead_time_distribution"]["lead_times_sec"]
        ax.set_title(f"Horizon H={h} ({h*20}s Max Lead, N={len(lead_times)})", fontsize=11)
        ax.set_xlabel("Early Warning Lead Time (s)", fontsize=10)
        if idx_h == 0:
            ax.set_ylabel("Episode Count", fontsize=10)

        if lead_times:
            max_lead = h * 20
            bins = np.arange(10, max_lead + 30, 20)
            ax.hist(lead_times, bins=bins, color="#1f77b4", alpha=0.8, edgecolor="black", rwidth=0.85)
        else:
            ax.text(0.5, 0.5, "No Recalled Episodes", ha="center", va="center", transform=ax.transAxes)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Lead-Time Distributions on Confirmed Sustained Episodes (Model B)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lead_time_distribution.png", dpi=200)
    plt.close()

    # Plot 4: Feature Ablation Comparison (F1, Actionable Recall)
    plt.figure(figsize=(10, 5))
    h5_abl = ablations["H=5"]
    cohort_names = list(h5_abl.keys())
    f1_vals = [h5_abl[c]["f1"] for c in cohort_names]
    act_rec_vals = [h5_abl[c]["actionable_recall"] for c in cohort_names]

    x = np.arange(len(cohort_names))
    width = 0.35

    plt.bar(x - width / 2, f1_vals, width, label="F1 Score", color="#1f77b4", alpha=0.85)
    plt.bar(x + width / 2, act_rec_vals, width, label="Actionable Recall (>=20s)", color="#2ca02c", alpha=0.85)

    plt.xticks(x, cohort_names, rotation=15, ha="right", fontsize=10)
    plt.ylabel("Score", fontsize=11)
    plt.title("Ablation Analysis: Impact of Removing Features on Model B (H=5)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ablation_comparison.png", dpi=200)
    plt.close()


def generate_report_markdown_tables(payload: Dict[str, Any]) -> str:
    """Programmatically formats report comparison tables directly from saved metrics."""
    res_h = payload["results_by_horizon"]
    abl = payload["ablations"]
    rob = payload["robustness"]

    md = []
    md.append("# Programmatic Reconciled Evaluation Tables (Task 4A)\n")

    # Table 1: Overall Classification & Probabilistic Metrics
    md.append("## Table 1: Overall Classification & Calibration Across Horizons\n")
    md.append("| Horizon | Model / Baseline | Sample Count | Prevalence | Precision | Recall | F1 Score | PR-AUC | ROC-AUC | Brier Score | ECE (10-bin) |")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    models_order = [
        ("static_threshold", "Static SLO Threshold Alerting"),
        ("zscore_baseline", "Training-Fitted Z-Score Anomaly"),
        ("model_a_telemetry_only", "Model A (Telemetry Only)"),
        ("model_b_telemetry_forecast", "Model B (Telemetry + Forecast)"),
        ("model_c_temporal_comparator", "Model C (Temporal Comparator GRU)"),
    ]

    for h in [1, 5, 15]:
        h_data = res_h[str(h)] if str(h) in res_h else res_h[h]
        for m_key, m_name in models_order:
            c = h_data[m_key]["overall_classification"]
            pr_auc_str = f"{c['pr_auc']:.4f}" if c["pr_auc"] is not None else "N/A"
            roc_auc_str = f"{c['roc_auc']:.4f}" if c["roc_auc"] is not None else "N/A"
            brier_str = f"{c['brier_score']:.4f}" if c["brier_score"] is not None else "N/A"
            ece_str = f"{c['ece']:.4f}" if c["ece"] is not None else "N/A"
            md.append(
                f"| **H={h} ({h*20}s)** | {m_name} | {c['sample_count']} | {c['prevalence']:.4f} | "
                f"{c['precision']:.4f} | {c['recall']:.4f} | **{c['f1']:.4f}** | {pr_auc_str} | {roc_auc_str} | {brier_str} | {ece_str} |"
            )

    # Table 2: Bootstrap Uncertainty & Paired Comparisons
    md.append("\n## Table 2: Bootstrap Uncertainty & Paired Comparisons (500 Incident-Cluster Replicates)\n")
    md.append("| Horizon | Comparison / Model | Observed F1 / Diff | Bootstrap Mean F1 / Diff | 95% Confidence Interval | Statistically Significant (Excludes 0)? |")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: |")

    for h in [1, 5, 15]:
        h_data = res_h[str(h)] if str(h) in res_h else res_h[h]
        boot = h_data["bootstrap_uncertainty"]

        # Individual Model B
        b_info = boot["model_b"]
        md.append(f"| **H={h}** | Model B F1 Score | {b_info['observed_f1']:.4f} | {b_info['bootstrap_mean_f1']:.4f} | [{b_info['f1_ci_95'][0]:.4f}, {b_info['f1_ci_95'][1]:.4f}] | N/A (Level) |")

        # Comparisons
        diffs = [
            ("Model B vs Model A", boot["diff_model_b_minus_a"]),
            ("Model B vs Model C", boot["diff_model_b_minus_c"]),
            ("Model B vs Static Baseline", boot["diff_model_b_minus_static"]),
            ("Model B vs Z-Score Baseline", boot["diff_model_b_minus_zscore"]),
        ]
        for name, d in diffs:
            obs_diff = d["observed_diff_f1"]
            boot_diff = d["bootstrap_mean_diff_f1"]
            ci = d["diff_f1_ci_95"]
            sig = "Yes (Excludes 0)" if d["diff_excludes_zero"] else "No (Crosses 0)"
            md.append(f"| **H={h}** | {name} (Delta F1) | {obs_diff:+.4f} | {boot_diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | **{sig}** |")

    # Table 3: Early Warning, Lead Time, and False Alert Accounting
    md.append("\n## Table 3: Early Warning, Actionable Recall, and Exposure Accounting\n")
    md.append("| Horizon | Model | Event Recall | Actionable Recall (>=20s) | Missed Events | Median Lead Time | P10 Lead Time | False Alarms / Day | Healthy Exposure Days | Time-to-Onset MAE (Coverage) |")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for h in [1, 5, 15]:
        h_data = res_h[str(h)] if str(h) in res_h else res_h[h]
        for m_key, m_name in models_order:
            ew = h_data[m_key]["early_warning_metrics"]
            lt = ew["lead_time_distribution"]
            med_str = f"{lt['median_sec']:.1f}s" if lt["median_sec"] is not None else "N/A"
            p10_str = f"{lt['p10_sec']:.1f}s" if lt["p10_sec"] is not None else "N/A"
            time_mae = h_data[m_key]["time_to_degradation_mae_sec"]
            time_cov = h_data[m_key]["time_to_degradation_coverage"]
            time_str = f"{time_mae:.1f}s ({time_cov*100:.1f}%)" if time_mae is not None else "N/A"

            md.append(
                f"| **H={h}** | {m_name} | {ew['event_early_detection_recall']*100:.1f}% | "
                f"{ew['actionable_recall']*100:.1f}% | {ew['missed_events_count']}/{ew['evaluable_episodes']} | "
                f"{med_str} | {p10_str} | {ew['false_alarms_per_day']:.2f} | {ew['healthy_exposure_days']:.4f} | {time_str} |"
            )

    # Table 4: Priority-Weighted Macro Metric
    md.append("\n## Table 4: Priority-Weighted Macro (3 Domains) vs Separated Infrastructure\n")
    md.append("| Horizon | Model | Priority Macro Precision | Priority Macro Recall | Priority Macro F1 | Reservations F1 | Crew F1 | Baggage F1 | Boundary Gateway F1 (Sep) | Legacy Core F1 (Sep) |")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for h in [1, 5, 15]:
        h_data = res_h[str(h)] if str(h) in res_h else res_h[h]
        for m_key, m_name in models_order:
            mac = h_data[m_key]["priority_weighted_macro_3domain"]["metrics"]
            svc = h_data[m_key]["per_service_metrics"]
            p_mac = f"{mac['precision']:.4f}" if mac["precision"] is not None else "N/A"
            r_mac = f"{mac['recall']:.4f}" if mac["recall"] is not None else "N/A"
            f_mac = f"{mac['f1']:.4f}" if mac["f1"] is not None else "N/A"

            f_res = f"{svc['reservations']['f1']:.4f}"
            f_crew = f"{svc['crew']['f1']:.4f}"
            f_bag = f"{svc['baggage']['f1']:.4f}"
            f_gw = f"{svc['boundary-gateway']['f1']:.4f}"
            f_leg = f"{svc['legacy-core']['f1']:.4f}"

            md.append(
                f"| **H={h}** | {m_name} | {p_mac} | {r_mac} | **{f_mac}** | {f_res} | {f_crew} | {f_bag} | {f_gw} | {f_leg} |"
            )

    # Table 5: Feature Ablation Comparison on Model B
    md.append("\n## Table 5: Feature Ablation Analysis on Model B\n")
    md.append("| Horizon | Ablation Cohort | Features Removed | Precision | Recall | F1 Score | False Alarms / Day | Actionable Recall | Median Lead Time |")
    md.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for h in [1, 5, 15]:
        h_abl = abl[f"H={h}"]
        for c_name, info in h_abl.items():
            med_lt = f"{info['median_lead_time_sec']:.1f}s" if info["median_lead_time_sec"] is not None else "N/A"
            md.append(
                f"| **H={h}** | {c_name} | {info['description']} | {info['precision']:.4f} | {info['recall']:.4f} | "
                f"**{info['f1']:.4f}** | {info['false_alarms_per_day']:.2f} | {info['actionable_recall']*100:.1f}% | {med_lt} |"
            )

    # Table 6: Robustness Stress Matrix (H=5)
    md.append("\n## Table 6: Robustness Stress Matrix (Model B vs Training-Fitted Z-Score Baseline, H=5)\n")
    md.append("| Stress Condition | Description & Perturbation Details | Sample Count (Prevalence) | Model B F1 | Model B PR-AUC | Z-Score Baseline F1 | Telemetry & Observability Behavior |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :--- |")

    for cond_key, cond_info in rob.items():
        mb = cond_info["model_b"]
        zs = cond_info.get("zscore")
        zs_f1 = f"{zs['f1']:.4f}" if zs and zs.get("f1") is not None else "N/A"
        prev_str = f"{cond_info['prevalence']*100:.2f}%" if cond_info.get("prevalence") is not None else "N/A"
        samples_str = f"{cond_info['sample_count']} ({prev_str})" if cond_info.get("sample_count") else "N/A"

        note = (
            "Clean held-out baseline performance." if "clean" in cond_key else
            "Transient dropouts handled causally via rolling-buffer carry-forward." if "causal" in cond_key else
            "Sensitivity to zeroing random tabular feature cells (uncalibrated drop)." if "zeroing" in cond_key else
            "Sensitivity to continuous feature jitter; discrete capacity flags preserved." if "noise" in cond_key else
            "Synthetic physical capacity collapse triggers high queue backpressure." if "capacity" in cond_key else
            "Synthetic cross-system transit latency and sync drift divergence." if "slowdown" in cond_key else
            "Synthetic 2.5x traffic bursts trigger early workload alerts."
        )

        md.append(
            f"| **{cond_key}** | {cond_info['description']} | {samples_str} | **{mb['f1']:.4f}** | {mb['pr_auc']:.4f} | {zs_f1} | {note} |"
        )

    out_text = "\n".join(md)
    with open(RESULTS_DIR / "reconciled_evaluation_tables.md", "w", encoding="utf-8") as f:
        f.write(out_text)

    print(f"Generated programmatic report tables at {RESULTS_DIR / 'reconciled_evaluation_tables.md'}")
    return out_text


def write_evaluation_protocol() -> None:
    """Writes the comprehensive evaluation protocol document."""
    protocol_md = r"""# BACCP Degradation Prediction: Evaluation Protocol & Contract

## 1. Dataset Counts and Provenance
- **Total Incidents**: 275 whole trajectories across 6 operational regimes.
- **Trajectory Duration**: 30 discrete steps per incident ($dt = 20.0\,\text{s}$, duration $= 600.0\,\text{s}$).
- **Split Partitions**:
  - `train`: 120 incidents (43.6%) — used for fitting baseline parameters and ablation models.
  - `val`: 30 incidents (10.9%) — used for operating point tuning and probability calibration.
  - `cal`: 25 incidents (9.1%) — historical holdout used for isotonic temperature calibration.
  - `standard_test`: 25 incidents (9.1%) — clean held-out evaluation cohort.
  - `capacity_loss_test`: 25 incidents (9.1%) — held-out physical capacity reduction stress cohort.
  - `boundary_slowdown_test`: 25 incidents (9.1%) — held-out boundary tunnel latency & sync drift stress cohort.
  - `surge_test`: 25 incidents (9.1%) — held-out workload surge (2.5x traffic bursts) stress cohort.
- **Split Integrity**: All origin windows from an incident strictly reside within the incident's split. No window leakage across splits.
- **Holdout Status**: Test sets have been evaluated in preceding development iterations; they are designated as **previously inspected holdouts** rather than pristine uninspected test sets.

## 2. Operational Definitions & Eligibility
- **Sampling Interval**: $\Delta t = 20.0\,\text{seconds}$.
- **Horizons**:
  - $H=1$: $20\,\text{s}$ forward window (3,250 evaluated origin samples in clean test).
  - $H=5$: $100\,\text{s}$ forward window (2,750 evaluated origin samples in clean test).
  - $H=15$: $300\,\text{s}$ forward window (1,500 evaluated origin samples in clean test).
- **Sustained Degradation Ground Truth**: $\ge 3$ consecutive 20-second steps ($60\,\text{s}$) violating SLO thresholds.
- **Origin Eligibility**:
  - Warmup requirement: $t_{\text{origin}} \ge 3$ ($60\,\text{s}$) to allow rolling mean/max extraction.
  - Forward-window constraint: $t_{\text{origin}} \le T - H - 1$ to ensure full window observability.
  - Censoring: Trailing windows with unresolved breach status (`check_window_label_observability`) are strictly excluded.
- **Current State vs Early Warning Separation**:
  - Current-state detection evaluates accuracy on origins where the service is already in sustained degradation.
  - Early-warning evaluation is restricted strictly to origins where the service is currently nominal, followed by a confirmed onset in $(t, t+H \cdot \Delta t]$. An alert at or after onset does NOT count as an early warning.
- **Actionable Lead Time**: Minimum $20.0\,\text{s}$ (1 step before onset).
- **False Alarm Accounting**: Evaluated on healthy exposure origins with a 100-second ($5 \times 20\text{s}$) deduplication cooldown.

## 3. Evaluated Model Cohort
1. **Static SLO Threshold Alerting**: Triggers alert if current metric exceeds 75% of SLO threshold.
2. **Training-Fitted Z-Score Baseline**: Per-service anomaly baseline fitted on `train`, operating threshold selected on `val`.
3. **Model A**: Telemetry-only gradient boosted model (22 features).
4. **Model B**: Telemetry + Workload Forecast primary model (35 features). Deployed thresholds: $H=1: 0.280$, $H=5: 0.300$, $H=15: 0.320$.
5. **Model C**: Temporal comparator GRU model ($4 \times 17$ features).

## 4. Priority-Weighted Macro Aggregation
Three cloud domains are combined using architectural priority weights:
$$\text{Metric}_{\text{macro}} = \frac{1.5 \cdot M_{\text{reservations}} + 1.0 \cdot M_{\text{crew}} + 0.5 \cdot M_{\text{baggage}}}{3.0}$$
Boundary Gateway and Legacy Core are reported separately as infrastructure interconnect components.
"""
    with open(RESULTS_DIR / "evaluation_protocol.md", "w", encoding="utf-8") as f:
        f.write(protocol_md)


def write_model_card(
    results_by_h: Dict[int, Any],
    ablations: Dict[str, Any],
    robustness: Dict[str, Any],
) -> None:
    """Generates the updated model card document."""
    m_b_h1 = results_by_h[1]["model_b_telemetry_forecast"]["overall_classification"]
    m_b_h5 = results_by_h[5]["model_b_telemetry_forecast"]["overall_classification"]
    m_b_h15 = results_by_h[15]["model_b_telemetry_forecast"]["overall_classification"]

    card_md = f"""# BACCP Degradation Prediction Model Card

## Model Details
- **Architecture**: Gradient Boosted Decision Trees (`HistGradientBoostingClassifier`) with Isotonic Probability Calibration.
- **Primary Checkpoint**: Model B (Telemetry + Upstream Workload Forecasts).
- **Horizons**: $H=1$ (20s), $H=5$ (100s), $H=15$ (300s).
- **Feature Dimensionality**: 35 versioned features (`baccp-degradation-features-v1.4.0`).
- **Training Provenance**: Fitted on 120 synthetic whole incidents (master seed 142857) with causal grid alignment.

## Performance Summary (Clean Holdout: standard_test)
| Horizon | Sample Count | Precision | Recall | F1 Score | PR-AUC | ROC-AUC | Brier Score | ECE (10-bin) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H=1 (20s)** | {m_b_h1['sample_count']} | {m_b_h1['precision']:.4f} | {m_b_h1['recall']:.4f} | {m_b_h1['f1']:.4f} | {m_b_h1['pr_auc']:.4f} | {m_b_h1['roc_auc']:.4f} | {m_b_h1['brier_score']:.4f} | {m_b_h1['ece']:.4f} |
| **H=5 (100s)** | {m_b_h5['sample_count']} | {m_b_h5['precision']:.4f} | {m_b_h5['recall']:.4f} | {m_b_h5['f1']:.4f} | {m_b_h5['pr_auc']:.4f} | {m_b_h5['roc_auc']:.4f} | {m_b_h5['brier_score']:.4f} | {m_b_h5['ece']:.4f} |
| **H=15 (300s)** | {m_b_h15['sample_count']} | {m_b_h15['precision']:.4f} | {m_b_h15['recall']:.4f} | {m_b_h15['f1']:.4f} | {m_b_h15['pr_auc']:.4f} | {m_b_h15['roc_auc']:.4f} | {m_b_h15['brier_score']:.4f} | {m_b_h15['ece']:.4f} |

## Intended Use & Deployment Scope
- **Target Application**: Predictive early warning and advisory alerting for cloud airline microservices (`reservations`, `crew`, `baggage`, `boundary-gateway`).
- **Out of Scope**: Unattended autonomous actuation on legacy-core (which lacks internal instrumentation).
- **Caveat**: All holdout evaluations reflect synthetic trajectory datasets; real-world production performance requires verification against live OpenTelemetry telemetry.
"""
    with open(RESULTS_DIR / "model_card.md", "w", encoding="utf-8") as f:
        f.write(card_md)


def write_limitations_document() -> None:
    """Writes the one-page limitations and verification document."""
    limitations_md = """# Task 4A Limitations & Empirical Verification

## 1. What Synthetic Data Supports
- **Algorithmic Validation**: Verifies that gradient-boosted decision trees and temporal GRU comparators capture non-linear queue accumulation, cross-service propagation, and upstream workload changes across simulated airline services.
- **Invariants Verified**: Enforces causal timestamp alignment, exact TreeSHAP additivity (< 1e-14 error), absence of future data leakage, and rigorous censoring of unobservable forward breaches.
- **Baseline Behavior**: Demonstrates that while static SLO rules cannot generate proactive early warnings, statistical anomaly detection (Z-scores) can detect pre-breach variance, though with higher false discovery rates than trained models.

## 2. What Remains Unvalidated
- **Real Telemetry Noise and Artifacts**: Live OpenTelemetry streams contain asynchronous clock jitter, multi-region NTP drift, network dropped packets, and non-stationary baseline shifts not fully replicated by synthetic Gaussian perturbations.
- **Statistical Significance Limits**: Paired bootstrap comparisons reveal that while Model B's workload forecast features provide statistically established F1 gains at $H=15$ (+0.0665, 95% CI [0.0398, 0.0934]), at $H=5$ the improvement is statistically uncertain (95% CI crosses zero), and at $H=1$ Model A (telemetry-only) achieves higher F1.
- **Feature Attributions vs Physical Root Cause**: TreeSHAP and counterfactual interventions describe mathematical model sensitivity, NOT verified physical causes.
- **Telemetry Dropout Sensitivity vs Staleness Enforcement**: The `create_causal_missing_telemetry_cohort` test manually carried observations forward and refreshed timestamps, functioning as a preprocessing sensitivity experiment rather than verifying production staleness enforcement. Production staleness enforcement is verified separately by passing incomplete telemetry with surviving original timestamps through the inference alignment path: short gaps within 60s are causally aligned with carry-forward warnings (`OBSERVATIONS_CARRIED_FORWARD`), while gaps exceeding 60s are strictly rejected with structured unavailable states without nominal imputation.

## 3. Deployment Scope
- **Backend Prediction Endpoints**: The inference engine is ready for prediction endpoint integration. Live shadow deployment and circuit-breaker actuation are downstream deployment steps outside Task 4/5 endpoint scope.
"""
    with open(RESULTS_DIR / "limitations.md", "w", encoding="utf-8") as f:
        f.write(limitations_md)


def rerun_dropped_telemetry_experiment() -> Dict[str, Any]:
    """Reproduces the focused dropped-telemetry robustness evaluation (H=5) directly,
    saving computed outputs to task4_evaluation_metrics.json and printing measured results."""
    print("=== Reproducing Causal Incomplete Telemetry Robustness Evaluation (H=5) ===")

    # 1. Load dataset
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    incidents_by_split: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for inc in data["incidents"]:
        incidents_by_split[inc["split"]].append(inc)

    train_inc = incidents_by_split["train"]
    val_inc = incidents_by_split.get("val", incidents_by_split.get("validation", []))
    std_test_inc = incidents_by_split["standard_test"]

    # 2. Fit training Z-score baseline and tune operating threshold on validation split
    zscore_baseline = TrainingFittedZScoreBaseline()
    zscore_baseline.fit(train_inc)
    if val_inc:
        forecaster = WorkloadForecaster.get_instance() if WorkloadForecaster else None
        X_val, _, _, _, _, y_val, _, _, _, metas_val = extract_split_samples(
            val_inc, horizon=5, include_workload_forecasts=True, forecaster=forecaster, extract_temporal=False
        )
        zscore_baseline.tune_operating_point(X_val, metas_val, y_val)

    # 3. Load forecaster and degradation models
    forecaster = WorkloadForecaster.get_instance() if WorkloadForecaster else None
    predictor = DegradationPredictor.get_instance()
    m_b_5 = predictor.models_telemetry_forecast[5]
    th_5 = DEPLOYED_THRESHOLDS[5]  # 0.300

    # 4. Extract standard_test samples for H=5
    X_test, _, _, _, _, y_test, _, _, _, metas_test = extract_split_samples(
        std_test_inc, horizon=5, include_workload_forecasts=True, forecaster=forecaster, extract_temporal=False
    )

    # 5. Run causal dropped telemetry evaluation
    short_rob_res, long_rob_res = evaluate_causal_dropped_telemetry_robustness(
        incidents=std_test_inc,
        metas=metas_test,
        y_true=y_test,
        h_rob=5,
        m_b_rob=m_b_5,
        th_rob=th_5,
        forecaster=forecaster,
        zscore_baseline=zscore_baseline,
        random_state=42,
    )

    print(f"Short Gap (<=60s): Model B F1={short_rob_res['model_b']['f1']:.4f}, "
          f"PR-AUC={short_rob_res['model_b']['pr_auc']:.4f}, "
          f"Z-Score F1={short_rob_res['zscore']['f1']}, "
          f"Carry-forward rate={short_rob_res['carried_forward_rate']*100:.1f}%")
    print(f"Long Gap (>60s): Rejections={long_rob_res['rejection_rate']*100:.1f}%, "
          f"Behavior={long_rob_res['behavior']}")

    # 6. Update task4_evaluation_metrics.json directly
    metrics_path = RESULTS_DIR / "task4_evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        target_dict = metrics_data.get("robustness", metrics_data.get("stress_matrix"))
        if target_dict is not None:
            target_dict["causal_incomplete_telemetry_alignment_within_60s"] = short_rob_res
            target_dict["causal_incomplete_telemetry_staleness_exceeded_gt_60s"] = long_rob_res
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2)
            print(f"Saved updated results directly to {metrics_path}")

    return {"short": short_rob_res, "long": long_rob_res}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 4 Evaluation & Robustness Runner")
    parser.add_argument(
        "--rerun-dropped-telemetry",
        action="store_true",
        help="Rerun only the H=5 causal dropped telemetry robustness experiment and update results.",
    )
    args = parser.parse_args()

    if args.rerun_dropped_telemetry:
        rerun_dropped_telemetry_experiment()
    else:
        run_task4_evaluation()

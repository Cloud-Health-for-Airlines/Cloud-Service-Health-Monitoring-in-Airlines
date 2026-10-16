"""Model Training and Artifact Serialization for Degradation Prediction.

Version: baccp-degradation-train-v1.2.0

Trains:
- Model A: HistGradientBoosting on Telemetry Only (22 features)
- Model B: HistGradientBoosting on Telemetry + Workload Forecasts (35 features)
- Model C: PyTorch Temporal GRU Comparator with Dependency Neighbor Context
- Target 2: Conditional Time-to-Onset Regressors (censoring-aware)
- Target 3: Degradation Type & Severity Classifiers (SLO-based)
- System-Level Calibrator: Isotonic probability calibration on aggregated origins
- Calibrates event probabilities on disjoint calibration partition
- Tunes alert thresholds strictly on validation partition after applying calibration
- Serializes locked models bundle to weights/degradation_prediction/degradation_models_bundle.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .data_contract import (
        BOUNDARY_SYNC_DRIFT_THRESHOLD,
        CONTRACT_VERSION,
        SAMPLING_INTERVAL_SECONDS,
        DegradationSeverity,
        DegradationType,
        ServiceIdentifier,
        SLO_THRESHOLDS,
        classify_service_step_type_and_severity,
        detect_confirmed_sustained_episodes,
        get_operational_state_at_origin,
        check_window_label_observability,
        is_service_step_degraded,
        is_boundary_step_degraded,
        is_system_step_degraded,
    )
    from .features import (
        FEATURE_SCHEMA_VERSION,
        TELEMETRY_ONLY_FEATURE_NAMES,
        TELEMETRY_PLUS_FORECAST_FEATURE_NAMES,
        assert_feature_schema,
        extract_features_at_origin,
        extract_temporal_sequence_tensor,
    )
    from .models import (
        CausalQueueRolloutBaseline,
        ConditionalTimeToOnsetRegressor,
        DegradationTypeAndSeverityClassifier,
        LearnedDegradationHorizonModel,
        LearnedTemporalHorizonModel,
        PersistenceDegradationBaseline,
        SystemLevelCalibrator,
    )
except ImportError:
    from degradation_prediction.data_contract import (
        BOUNDARY_SYNC_DRIFT_THRESHOLD,
        CONTRACT_VERSION,
        SAMPLING_INTERVAL_SECONDS,
        DegradationSeverity,
        DegradationType,
        ServiceIdentifier,
        SLO_THRESHOLDS,
        classify_service_step_type_and_severity,
        detect_confirmed_sustained_episodes,
        get_operational_state_at_origin,
        check_window_label_observability,
        is_service_step_degraded,
        is_boundary_step_degraded,
        is_system_step_degraded,
    )
    from degradation_prediction.features import (
        FEATURE_SCHEMA_VERSION,
        TELEMETRY_ONLY_FEATURE_NAMES,
        TELEMETRY_PLUS_FORECAST_FEATURE_NAMES,
        assert_feature_schema,
        extract_features_at_origin,
        extract_temporal_sequence_tensor,
    )
    from degradation_prediction.models import (
        CausalQueueRolloutBaseline,
        ConditionalTimeToOnsetRegressor,
        DegradationTypeAndSeverityClassifier,
        LearnedDegradationHorizonModel,
        LearnedTemporalHorizonModel,
        PersistenceDegradationBaseline,
        SystemLevelCalibrator,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_MODELS_DIR = REPO_ROOT / "ai-models"
DATA_PATH = AI_MODELS_DIR / "data" / "degradation_prediction" / "degradation_dataset.json"
WEIGHTS_DIR = AI_MODELS_DIR / "weights" / "degradation_prediction"
DEFAULT_WEIGHTS_BUNDLE_PATH = WEIGHTS_DIR / "degradation_models_bundle.pkl"

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


def extract_split_samples(
    incidents: List[Dict[str, Any]],
    horizon: int,
    include_workload_forecasts: bool,
    forecaster: Optional[Any],
    min_origin: int = 3,
    extract_temporal: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
    List[Dict[str, Any]],
]:
    """Extracts features, temporal tensors, and aligned multi-target labels for all origins and services in a partition.

    Target definitions:
    1. Continuous endpoint at t+H: y_lat, y_backlog, y_drift
    2. Event target: Confirmed sustained degradation onset (>= 3 consecutive breaches) in (t, t+H]
    3. Time to onset: Steps to next confirmed degradation onset (T_onset >= t), or -1.0 if censored/healthy
    4. Type & Severity: Observable type and severity classes at onset
    """
    X_list = []
    X_temp_list = []
    y_lat_list = []
    y_backlog_list = []
    y_drift_list = []
    y_event_list = []
    y_time_list = []
    y_type_list = []
    y_sev_list = []
    meta_list = []

    services = [s.value for s in ServiceIdentifier]

    for inc in incidents:
        inc_id = inc["incident_id"]
        regime = inc["regime"]
        svc_tel = inc["service_telemetry"]
        bnd_tel = inc["boundary_telemetry"]
        step_count = inc["step_count"]

        # Precompute sustained episodes across incident
        sys_episodes = detect_confirmed_sustained_episodes(svc_tel, bnd_tel, target_service_id=None, min_consecutive=3)
        svc_episodes_map = {
            s: detect_confirmed_sustained_episodes(svc_tel, bnd_tel, target_service_id=s, min_consecutive=3)
            for s in services
        }

        # Precompute step-level breach indicators for label observability
        step_breached_sys = np.array([
            is_system_step_degraded({s: svc_tel[s][step_i] for s in services}, bnd_tel[step_i])
            for step_i in range(step_count)
        ], dtype=int)

        step_breached_svc = {
            s: np.array([
                (is_boundary_step_degraded(bnd_tel[step_i]) or is_service_step_degraded(s, svc_tel[s][step_i]))
                if s == ServiceIdentifier.BOUNDARY_GATEWAY.value else
                is_service_step_degraded(s, svc_tel[s][step_i])
                for step_i in range(step_count)
            ], dtype=int)
            for s in services
        }

        # Origin steps
        max_origin = step_count - horizon - 1
        if max_origin < min_origin:
            continue

        for t in range(min_origin, max_origin + 1):
            target_step = t + horizon

            # Causal operational state derived strictly from observations through origin t ONLY
            sys_state_at_t = get_operational_state_at_origin(breach_indicators_up_to_t=step_breached_sys[: t + 1])
            is_curr_sys_degraded = 1 if sys_state_at_t["is_degraded"] else 0

            # System-wide forward-window sustained onset in (t, t+H]
            is_sys_window_degraded = 1 if any(
                t < ep["first_breach_step"] <= target_step for ep in sys_episodes
            ) else 0

            # Observability check for system window
            sys_window_observable = check_window_label_observability(
                target_step=target_step, step_count=step_count, breach_indicators=step_breached_sys
            )

            for s_id in services:
                phys_t = float(svc_tel[s_id][t].get("timestamp_sec", t * SAMPLING_INTERVAL_SECONDS))
                # Tabular feature vector at origin t
                feat_vec, aux = extract_features_at_origin(
                    service_id=s_id,
                    incident_service_telemetry=svc_tel,
                    incident_boundary_telemetry=bnd_tel,
                    origin_step=t,
                    horizon=horizon,
                    forecaster=forecaster,
                    include_workload_forecasts=include_workload_forecasts,
                    physical_timestamp_sec=phys_t,
                )

                # Temporal sequence tensor (4, 16) with neighbor context
                if extract_temporal:
                    feat_tensor = extract_temporal_sequence_tensor(
                        service_id=s_id,
                        incident_service_telemetry=svc_tel,
                        incident_boundary_telemetry=bnd_tel,
                        origin_step=t,
                        horizon=horizon,
                        window_size=4,
                        forecaster=forecaster,
                        physical_timestamp_sec=phys_t,
                    )
                else:
                    feat_tensor = np.zeros((4, 16), dtype=np.float32)

                # Continuous ground truth targets at t+H
                target_s = svc_tel[s_id][target_step]
                target_b = bnd_tel[target_step]

                y_lat = float(target_s.get("latency_ms", 15.0))
                y_backlog = float(target_s.get("backlog_requests", 0.0))
                y_drift = float(target_b.get("sync_drift_score", 12.0))

                # Service sustained operational state at origin t
                s_state_at_t = get_operational_state_at_origin(breach_indicators_up_to_t=step_breached_svc[s_id][: t + 1])
                is_curr_s_degraded = 1 if s_state_at_t["is_degraded"] else 0

                # Target 1: Service sustained degradation onset in (t, t+H]
                s_eps = svc_episodes_map[s_id]
                matching_window_eps = [
                    ep for ep in s_eps if t < ep["first_breach_step"] <= target_step
                ]
                is_s_window_degraded = 1 if len(matching_window_eps) > 0 else 0

                s_window_observable = check_window_label_observability(
                    target_step=target_step, step_count=step_count, breach_indicators=step_breached_svc[s_id]
                )

                # Target 2: Time to next confirmed degradation onset
                future_eps = [ep for ep in s_eps if ep["first_breach_step"] >= t]
                if future_eps:
                    steps_to_onset = float(future_eps[0]["first_breach_step"] - t)
                else:
                    steps_to_onset = -1.0  # Censored / no future onset in trajectory

                # Target 3: Observable type & severity
                if is_s_window_degraded == 1:
                    onset_step = matching_window_eps[0]["first_breach_step"]
                    step_data = svc_tel[s_id][onset_step]
                    bnd_data = bnd_tel[onset_step] if s_id == ServiceIdentifier.BOUNDARY_GATEWAY.value else None
                    deg_t, deg_s = classify_service_step_type_and_severity(s_id, step_data, boundary_data=bnd_data)
                    type_str = deg_t.value
                    sev_str = deg_s.value
                else:
                    type_str = DegradationType.NOMINAL.value
                    sev_str = DegradationSeverity.NOMINAL.value

                X_list.append(feat_vec)
                X_temp_list.append(feat_tensor)
                y_lat_list.append(y_lat)
                y_backlog_list.append(y_backlog)
                y_drift_list.append(y_drift)
                y_event_list.append(is_s_window_degraded)
                y_time_list.append(steps_to_onset)
                y_type_list.append(type_str)
                y_sev_list.append(sev_str)

                meta_list.append({
                    "incident_id": inc_id,
                    "regime": regime,
                    "service_id": s_id,
                    "origin_step": t,
                    "target_step": target_step,
                    "horizon": horizon,
                    "is_curr_service_degraded": is_curr_s_degraded,
                    "is_curr_system_degraded": int(is_curr_sys_degraded),
                    "is_curr_system_healthy": int(not is_curr_sys_degraded),
                    "is_system_window_degraded": is_sys_window_degraded,
                    "sys_window_observable": sys_window_observable,
                    "label_is_observable": s_window_observable,
                    "is_full_history": int(t >= 15),
                    "is_censored": bool(steps_to_onset < 0),
                    "steps_to_onset": steps_to_onset,
                })

    X_mat = np.array(X_list, dtype=np.float64)
    expected_schema = "combined" if include_workload_forecasts else "telemetry"
    assert_feature_schema(X_mat, schema=expected_schema)

    return (
        X_mat,
        np.array(X_temp_list, dtype=np.float32),
        np.array(y_lat_list, dtype=np.float64),
        np.array(y_backlog_list, dtype=np.float64),
        np.array(y_drift_list, dtype=np.float64),
        np.array(y_event_list, dtype=np.int32),
        np.array(y_time_list, dtype=np.float32),
        y_type_list,
        y_sev_list,
        meta_list,
    )


def train_and_bundle_models(
    dataset_path: Path = DATA_PATH,
    output_bundle_path: Path = DEFAULT_WEIGHTS_BUNDLE_PATH,
) -> Dict[str, Any]:
    """Full training pipeline for degradation prediction with multi-target and temporal comparator models."""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r") as f:
        data_payload = json.load(f)

    meta = data_payload["metadata"]
    incidents_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for inc in data_payload["incidents"]:
        split = inc.get("split", "train")
        incidents_by_split.setdefault(split, []).append(inc)

    train_incs = incidents_by_split.get("train", [])
    val_incs = incidents_by_split.get("val", [])
    cal_incs = incidents_by_split.get("cal", [])

    print(f"Loaded incidents: train={len(train_incs)}, val={len(val_incs)}, cal={len(cal_incs)}")

    # Instantiate upstream workload forecaster
    forecaster = None
    if WorkloadForecaster is not None:
        try:
            forecaster = WorkloadForecaster.get_instance()
            print("Successfully loaded upstream WorkloadForecaster for out-of-fold feature extraction.")
        except Exception as exc:
            print(f"Warning: Could not load WorkloadForecaster: {exc}. Using internal simulation fallback.")

    horizons = [1, 5, 15]

    models_telemetry_only: Dict[int, LearnedDegradationHorizonModel] = {}
    models_telemetry_forecast: Dict[int, LearnedDegradationHorizonModel] = {}
    models_temporal_comparator: Dict[int, LearnedTemporalHorizonModel] = {}
    models_time_to_onset: Dict[int, ConditionalTimeToOnsetRegressor] = {}
    models_type_severity: Dict[int, DegradationTypeAndSeverityClassifier] = {}
    models_system_calibrator: Dict[int, SystemLevelCalibrator] = {}

    training_stats: Dict[str, Any] = {}

    for h in horizons:
        print(f"\n================ Training Models for Horizon H={h} ({h*20}s) ================")

        # 1. Extract samples
        print(f"[Horizon H={h}] Extracting features for train, val, cal...")
        (
            X_tr_a, X_tr_temp, y_lat_tr, y_q_tr, y_dr_tr, y_ev_tr, y_tm_tr, y_tp_tr, y_sv_tr, _
        ) = extract_split_samples(train_incs, horizon=h, include_workload_forecasts=False, forecaster=forecaster)

        (
            X_val_a, X_val_temp, _, _, _, y_ev_val, y_tm_val, _, _, meta_val
        ) = extract_split_samples(val_incs, horizon=h, include_workload_forecasts=False, forecaster=forecaster)

        (
            X_cal_a, X_cal_temp, _, _, _, y_ev_cal, y_tm_cal, _, _, meta_cal
        ) = extract_split_samples(cal_incs, horizon=h, include_workload_forecasts=False, forecaster=forecaster)

        # Combined features (Model B) - reuse temporal tensors from Model A
        X_tr_b, _, _, _, _, _, _, _, _, _ = extract_split_samples(
            train_incs, horizon=h, include_workload_forecasts=True, forecaster=forecaster, extract_temporal=False
        )
        X_val_b, _, _, _, _, _, _, _, _, _ = extract_split_samples(
            val_incs, horizon=h, include_workload_forecasts=True, forecaster=forecaster, extract_temporal=False
        )
        X_cal_b, _, _, _, _, _, _, _, _, _ = extract_split_samples(
            cal_incs, horizon=h, include_workload_forecasts=True, forecaster=forecaster, extract_temporal=False
        )

        # 2. Train Model A (Telemetry Only)
        model_a = LearnedDegradationHorizonModel(
            horizon=h, include_workload_forecasts=False, random_state=42 + h
        )
        model_a.fit_train(X_tr_a, y_lat_tr, y_q_tr, y_dr_tr, y_ev_tr)
        model_a.calibrate(X_cal_a, y_ev_cal)
        thresh_a = model_a.tune_threshold(X_val_a, y_ev_val)
        models_telemetry_only[h] = model_a
        print(f"[Model A - Telemetry Only] Trained ({len(X_tr_a)} samples). Calibrated. Val threshold = {thresh_a}")

        # 3. Train Model B (Telemetry + Forecasts)
        model_b = LearnedDegradationHorizonModel(
            horizon=h, include_workload_forecasts=True, random_state=142 + h
        )
        model_b.fit_train(X_tr_b, y_lat_tr, y_q_tr, y_dr_tr, y_ev_tr)
        model_b.calibrate(X_cal_b, y_ev_cal)
        thresh_b = model_b.tune_threshold(X_val_b, y_ev_val)
        models_telemetry_forecast[h] = model_b
        print(f"[Model B - Telemetry + Forecasts] Trained ({len(X_tr_b)} samples). Calibrated. Val threshold = {thresh_b}")

        # 4. Train Model C (Temporal GRU Comparator)
        model_c = LearnedTemporalHorizonModel(
            horizon=h, input_dim=X_tr_temp.shape[-1], hidden_dim=32, random_state=242 + h
        )
        model_c.fit_train(X_tr_temp, y_ev_tr, y_tm_tr, epochs=12)
        model_c.calibrate(X_cal_temp, y_ev_cal)
        thresh_c = model_c.tune_threshold(X_val_temp, y_ev_val)
        models_temporal_comparator[h] = model_c
        print(f"[Model C - Temporal GRU] Trained ({len(X_tr_temp)} sequences). Calibrated. Val threshold = {thresh_c}")

        # 5. Train Conditional Time-to-Onset Regressor
        observed_mask = (y_ev_tr == 1) & (y_tm_tr >= 0)
        time_reg = ConditionalTimeToOnsetRegressor(random_state=342 + h)
        if np.sum(observed_mask) >= 10:
            time_reg.fit(X_tr_b[observed_mask], y_tm_tr[observed_mask])
            print(f"[Time-to-Onset Regressor] Trained conditionally on {np.sum(observed_mask)} observed degradation events.")
        else:
            print("[Time-to-Onset Regressor] Insufficient positive events in training split; initialized fallback.")
        models_time_to_onset[h] = time_reg

        # 6. Train Degradation Type & Severity Classifier
        type_sev_clf = DegradationTypeAndSeverityClassifier(random_state=442 + h)
        type_sev_clf.fit(X_tr_b, y_tp_tr, y_sv_tr)
        models_type_severity[h] = type_sev_clf
        print("[Type & Severity Classifier] Trained multi-class classifiers for observable degradation types and severities.")

        # 7. Fit System-Level Calibrator on aggregated calibration origins
        prob_b_cal = model_b.predict_event_proba(X_cal_b)
        cal_origin_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for idx, m in enumerate(meta_cal):
            key = (m["incident_id"], m["origin_step"])
            if key not in cal_origin_map:
                cal_origin_map[key] = {
                    "y_sys": m["is_system_window_degraded"],
                    "probs": [],
                }
            cal_origin_map[key]["probs"].append(prob_b_cal[idx])

        comp_risks_cal = np.array([max(o["probs"]) for o in cal_origin_map.values()])
        y_sys_cal = np.array([o["y_sys"] for o in cal_origin_map.values()], dtype=int)

        sys_calibrator = SystemLevelCalibrator()
        sys_calibrator.fit(comp_risks_cal, y_sys_cal)
        models_system_calibrator[h] = sys_calibrator
        print(f"[System Calibrator] Fitted system-level calibrator on {len(comp_risks_cal)} calibration origins.")

        training_stats[str(h)] = {
            "train_samples": len(X_tr_a),
            "val_samples": len(X_val_a),
            "cal_samples": len(X_cal_a),
            "model_a_threshold": thresh_a,
            "model_b_threshold": thresh_b,
            "model_c_threshold": thresh_c,
            "observed_onset_events_trained": int(np.sum(observed_mask)),
        }

    # Save bundle
    output_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "metadata": {
            "contract_version": CONTRACT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "dataset_sha256": meta.get("dataset_sha256"),
            "horizons": horizons,
            "services": [s.value for s in ServiceIdentifier],
            "feature_names_telemetry_only": TELEMETRY_ONLY_FEATURE_NAMES,
            "feature_names_telemetry_forecast": TELEMETRY_PLUS_FORECAST_FEATURE_NAMES,
            "training_stats": training_stats,
            "created_timestamp": time.time(),
        },
        "models_telemetry_only": models_telemetry_only,
        "models_telemetry_forecast": models_telemetry_forecast,
        "models_temporal_comparator": models_temporal_comparator,
        "models_time_to_onset": models_time_to_onset,
        "models_type_severity": models_type_severity,
        "models_system_calibrator": models_system_calibrator,
    }

    with open(output_bundle_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSuccessfully serialized complete degradation models bundle to {output_bundle_path}")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and bundle degradation models.")
    parser.add_argument("--dataset", type=str, default=str(DATA_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_WEIGHTS_BUNDLE_PATH))
    args = parser.parse_args()

    train_and_bundle_models(Path(args.dataset), Path(args.output))

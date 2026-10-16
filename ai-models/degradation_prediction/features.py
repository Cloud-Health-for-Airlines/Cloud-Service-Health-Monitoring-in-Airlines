"""Feature Extraction and Workload Forecast Integration for Degradation Prediction.

Version: baccp-degradation-features-v1.0.0

Extracts causal features at forecast origin t for horizon H in {1, 5, 15} steps:
1. Telemetry-Only Features (Ablation group A):
   - Current latency, rolling stats, latency velocity ΔL
   - Current backlog Q(t), backlog velocity ΔQ
   - Current throughput μ(t), throughput ratio μ / λ
   - Error rate, dropped requests
   - Boundary signals: Gateway queue, sync drift ε(t), velocity Δε, divergence |Φ - Ψ|
   - Cross-service max pressure indicators
2. Workload-Forecast Features (Ablation group B):
   - Out-of-fold point forecast λ_hat(t+H) from WorkloadForecaster
   - Conformal interval bounds, width
   - Signed capacity margin, available headroom, forecasted overload
   - Unknown legacy capacity flag (preserves legacy as None without assumption)
   - Upstream reliability warning & fallback flags
   - Trapezoidal arrival rollout assumption:
     λ_bar = (λ(t) + λ_hat(t+H)) / 2
     A_hat = λ_bar * (H * Δt)
     Q_hat_rollout = max(0, Q(t) + A_hat - C_eff * H * Δt)
     L_hat_rollout = L_0 + (Q_hat_rollout / C_eff) * 1000 ms
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from .data_contract import (
        AdaptedWorkloadForecast,
        CONFIGURED_SERVICE_CAPACITIES,
        NOMINAL_BASELINES,
        SAMPLING_INTERVAL_SECONDS,
        ServiceIdentifier,
        SLO_THRESHOLDS,
        WorkloadForecastState,
    )
except ImportError:
    from degradation_prediction.data_contract import (
        AdaptedWorkloadForecast,
        CONFIGURED_SERVICE_CAPACITIES,
        NOMINAL_BASELINES,
        SAMPLING_INTERVAL_SECONDS,
        ServiceIdentifier,
        SLO_THRESHOLDS,
        WorkloadForecastState,
    )

# Upstream WorkloadForecaster
REPO_ROOT = Path(__file__).resolve().parents[2]
AI_MODELS_DIR = REPO_ROOT / "ai-models"
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

FEATURE_SCHEMA_VERSION = "baccp-degradation-features-v1.4.0"

# Feature Provenance Classification
FEATURE_PROVENANCE_CATALOG: Dict[str, str] = {
    # Measured live telemetry (20 features)
    "curr_latency_ms": "measured",
    "latency_roll_mean_3": "measured",
    "latency_delta_1": "measured",
    "curr_backlog": "measured",
    "backlog_roll_max_3": "measured",
    "backlog_delta_1": "measured",
    "curr_offered_rate": "measured",
    "curr_completed_rate": "measured",
    "throughput_ratio": "measured",
    "curr_error_rate": "measured",
    "curr_dropped_requests": "measured",
    "gw_queue_depth": "measured",
    "gw_sync_drift": "measured",
    "gw_drift_delta_1": "measured",
    "rate_divergence_abs": "measured",
    "rate_divergence_ratio": "measured",
    "boundary_latency_ms": "measured",
    "boundary_error_rate": "measured",
    "global_max_backlog": "measured",
    "global_max_latency_ratio": "measured",
    # Configured contract constants (2 features)
    "has_known_capacity": "configured",
    "known_capacity_value": "configured",
    # Inferred from upstream out-of-sample WorkloadForecaster (13 features)
    "wf_forecast_demand": "inferred",
    "wf_forecast_ratio": "inferred",
    "wf_interval_lower": "inferred",
    "wf_interval_upper": "inferred",
    "wf_interval_width": "inferred",
    "wf_capacity_margin": "inferred",
    "wf_available_headroom": "inferred",
    "wf_forecasted_overload": "inferred",
    "wf_is_overloaded": "inferred",
    "wf_fallback_active": "inferred",
    "wf_rollout_arrivals": "inferred",
    "wf_rollout_backlog": "inferred",
    "wf_rollout_latency_proj": "inferred",
}

# Ordered list of feature names for Telemetry Only (22 features)
TELEMETRY_ONLY_FEATURE_NAMES = [
    "curr_latency_ms",
    "latency_roll_mean_3",
    "latency_delta_1",
    "curr_backlog",
    "backlog_roll_max_3",
    "backlog_delta_1",
    "curr_offered_rate",
    "curr_completed_rate",
    "throughput_ratio",
    "curr_error_rate",
    "curr_dropped_requests",
    "gw_queue_depth",
    "gw_sync_drift",
    "gw_drift_delta_1",
    "rate_divergence_abs",
    "rate_divergence_ratio",
    "boundary_latency_ms",
    "boundary_error_rate",
    "global_max_backlog",
    "global_max_latency_ratio",
    "has_known_capacity",
    "known_capacity_value",
]

# Ordered list of feature names for Telemetry + Forecasts (35 features)
TELEMETRY_PLUS_FORECAST_FEATURE_NAMES = TELEMETRY_ONLY_FEATURE_NAMES + [
    "wf_forecast_demand",
    "wf_forecast_ratio",
    "wf_interval_lower",
    "wf_interval_upper",
    "wf_interval_width",
    "wf_capacity_margin",
    "wf_available_headroom",
    "wf_forecasted_overload",
    "wf_is_overloaded",
    "wf_fallback_active",
    "wf_rollout_arrivals",
    "wf_rollout_backlog",
    "wf_rollout_latency_proj",
]

# Versioned feature index lookups (eliminates magic indices)
FEATURE_INDEX_LOOKUP_TELEMETRY: Dict[str, int] = {
    name: idx for idx, name in enumerate(TELEMETRY_ONLY_FEATURE_NAMES)
}
FEATURE_INDEX_LOOKUP_COMBINED: Dict[str, int] = {
    name: idx for idx, name in enumerate(TELEMETRY_PLUS_FORECAST_FEATURE_NAMES)
}


def get_feature_index(feature_name: str, schema: str = "combined") -> int:
    """Returns the versioned feature column index for a given feature name.

    Raises KeyError if feature_name is not in the specified schema.
    """
    lookup = FEATURE_INDEX_LOOKUP_COMBINED if schema == "combined" else FEATURE_INDEX_LOOKUP_TELEMETRY
    if feature_name not in lookup:
        raise KeyError(
            f"Feature '{feature_name}' not found in schema '{schema}'. "
            f"Available features: {list(lookup.keys())}"
        )
    return lookup[feature_name]


def assert_feature_schema(X: np.ndarray, schema: str = "combined") -> None:
    """Asserts that the input feature matrix matches the expected schema dimensionality."""
    expected_len = (
        len(TELEMETRY_PLUS_FORECAST_FEATURE_NAMES)
        if schema == "combined"
        else len(TELEMETRY_ONLY_FEATURE_NAMES)
    )
    actual_len = X.shape[-1]
    if actual_len != expected_len:
        raise ValueError(
            f"Feature dimension mismatch for schema '{schema}': "
            f"expected {expected_len} features ({'Model B' if schema == 'combined' else 'Model A'}), "
            f"but got {actual_len} features."
        )


# Documented dependency graph and edge directions
# Cloud services -> boundary-gateway -> legacy-core
SERVICE_TOPOLOGY = {
    "reservations": {"upstream": [], "downstream": ["boundary-gateway"]},
    "crew": {"upstream": [], "downstream": ["boundary-gateway"]},
    "baggage": {"upstream": [], "downstream": ["boundary-gateway"]},
    "boundary-gateway": {"upstream": ["reservations", "crew", "baggage"], "downstream": ["legacy-core"]},
    "legacy-core": {"upstream": ["boundary-gateway"], "downstream": []},
}

TEMPORAL_SEQUENCE_FEATURE_NAMES = [
    "target_latency_ms",
    "target_backlog_requests",
    "target_offered_rate",
    "target_completed_rate",
    "target_error_rate",
    "target_dropped_requests",
    "neighbor_latency_ms",
    "neighbor_backlog_requests",
    "boundary_sync_drift_pct",
    "boundary_queue_depth",
    "upstream_pressure_rate",
    "has_known_capacity",
    "capacity_headroom",
    "forecast_demand_req_s",
    "forecast_interval_width",
    "forecast_overload_req_s",
]


def adapt_workload_forecast(
    service_id: str,
    horizon: int,
    history: List[float],
    forecaster: Optional[Any],
    current_offered_rps: float,
    physical_timestamp_sec: Optional[float] = None,
    step_index: Optional[int] = None,
) -> AdaptedWorkloadForecast:
    """Shared, strictly validated adapter translating WorkloadForecaster responses for degradation models.

    Preserves:
    - Units (RPS)
    - Conformal interval bounds, width, and nominal coverage (0.90)
    - Capacity semantics (signed margin, headroom, overload, and unknown legacy capacity)
    - Fallback status and explicit reliability warnings

    Distinguishes:
    1. VALID_TRAINED_PREDICTION: Successful prediction matching required schema.
    2. INTENTIONAL_PERSISTENCE_FALLBACK: Forecaster explicitly None or persistence baseline requested.
    3. MISSING_ARTIFACT: Forecaster provided but lacks weights/calibrator for service/horizon.
    4. INVALID_RESPONSE: Response returned but missing required contract keys (raises ValueError).
    5. PREDICTION_EXCEPTION: Runtime exception during forecaster inference (raises RuntimeError).
    """
    known_cap = CONFIGURED_SERVICE_CAPACITIES.get(service_id)
    has_known_cap = known_cap is not None
    known_cap_val = float(known_cap) if has_known_cap else 0.0

    # 1. State 2: Intentional Persistence Fallback
    if forecaster is None:
        fc_demand = float(current_offered_rps)
        ci_low = fc_demand * 0.85
        ci_high = fc_demand * 1.15
        ci_width = max(0.0, ci_high - ci_low)
        cap_margin = (known_cap_val - fc_demand) if has_known_cap else None
        headroom = max(0.0, cap_margin) if cap_margin is not None else None
        overload = max(0.0, -cap_margin) if cap_margin is not None else None

        return AdaptedWorkloadForecast(
            forecast_demand_rps=fc_demand,
            interval_lower_rps=ci_low,
            interval_upper_rps=ci_high,
            interval_width_rps=ci_width,
            nominal_coverage=None,
            is_calibrated=False,
            calibration_method="heuristic_uncalibrated_fixed_percentage",
            configured_capacity_rps=known_cap,
            capacity_margin_rps=cap_margin,
            available_headroom_rps=headroom,
            overload_rps=overload,
            has_known_capacity=has_known_cap,
            is_overload=1.0 if (overload is not None and overload > 0.0) else 0.0,
            fallback_triggered=True,
            fallback_reason="Intentional persistence fallback: forecaster is None",
            fallback_active=1.0,
            reliability_status="intentional_persistence_fallback",
            reliability_warnings=[
                "Forecaster instance not provided; using intentional persistence fallback.",
                "Conformal prediction intervals are heuristic (±15%) and uncalibrated; nominal coverage is unavailable.",
            ],
            prediction_state=WorkloadForecastState.INTENTIONAL_PERSISTENCE_FALLBACK,
        )

    # 2. State 3: Missing Artifact
    if hasattr(forecaster, "models"):
        if service_id not in forecaster.models or horizon not in forecaster.models[service_id]:
            raise FileNotFoundError(
                f"Missing artifact: Forecaster has no model weights for service '{service_id}' at horizon {horizon}."
            )

    # 3. Invoke Upstream Forecaster
    try:
        raw_res = forecaster.predict(
            service_id=service_id,
            history=history,
            horizon=horizon,
            model_name="auto_fallback",
            step_index=step_index,
            timestamp_sec=physical_timestamp_sec,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing artifact during forecaster prediction: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(
            f"Prediction exception in WorkloadForecaster for service '{service_id}' horizon {horizon}: {exc}"
        ) from exc

    # 4. State 4: Validate Response Schema (Must NOT silently fall back)
    if not isinstance(raw_res, dict):
        raise ValueError(f"Invalid WorkloadForecaster response: expected dict, got {type(raw_res)}")

    missing_keys = []
    if "point_forecast_rps" not in raw_res:
        missing_keys.append("point_forecast_rps")
    if "prediction_interval_90" not in raw_res or not isinstance(raw_res["prediction_interval_90"], dict):
        missing_keys.append("prediction_interval_90")
    else:
        for k in ["lower_rps", "upper_rps", "interval_width_rps"]:
            if k not in raw_res["prediction_interval_90"]:
                missing_keys.append(f"prediction_interval_90.{k}")
    if "capacity_metrics" not in raw_res or not isinstance(raw_res["capacity_metrics"], dict):
        missing_keys.append("capacity_metrics")
    else:
        for k in ["capacity_margin_rps", "available_headroom_rps", "overload_rps"]:
            if k not in raw_res["capacity_metrics"]:
                missing_keys.append(f"capacity_metrics.{k}")
    if "fallback_triggered" not in raw_res:
        missing_keys.append("fallback_triggered")
    if "reliability_metadata" not in raw_res or not isinstance(raw_res["reliability_metadata"], dict):
        missing_keys.append("reliability_metadata")
    else:
        if "reliability_status" not in raw_res["reliability_metadata"]:
            missing_keys.append("reliability_metadata.reliability_status")

    if missing_keys:
        raise ValueError(
            f"Invalid WorkloadForecaster response schema: missing required keys {missing_keys}. "
            "Supposedly successful responses must adhere to the contract and cannot be silently substituted."
        )

    # 5. State 1: Valid Trained Prediction
    fc_demand = float(raw_res["point_forecast_rps"])
    pi = raw_res["prediction_interval_90"]
    ci_low = float(pi["lower_rps"])
    ci_high = float(pi["upper_rps"])
    ci_width = float(pi["interval_width_rps"])
    nominal_cov = float(pi.get("nominal_coverage", 0.90))

    cm = raw_res["capacity_metrics"]
    cap_val = cm.get("configured_capacity_rps")
    cap_margin = float(cm["capacity_margin_rps"]) if cm["capacity_margin_rps"] is not None else None
    headroom = float(cm["available_headroom_rps"]) if cm["available_headroom_rps"] is not None else None
    overload = float(cm["overload_rps"]) if cm["overload_rps"] is not None else None

    rel_meta = raw_res["reliability_metadata"]
    rel_status = str(rel_meta.get("reliability_status", "nominal_in_distribution"))
    rel_warnings = list(rel_meta.get("confidence_warnings", []))
    fallback_trig = bool(raw_res.get("fallback_triggered", False))
    fallback_reason = raw_res.get("fallback_reason")

    selected_model = raw_res.get("selected_model", "")
    fallback_active = 1.0 if (fallback_trig or selected_model != "gradient_boosting" or len(rel_warnings) > 0) else 0.0

    return AdaptedWorkloadForecast(
        forecast_demand_rps=fc_demand,
        interval_lower_rps=ci_low,
        interval_upper_rps=ci_high,
        interval_width_rps=ci_width,
        nominal_coverage=nominal_cov,
        is_calibrated=True,
        calibration_method=str(pi.get("calibration_method", "exact_finite_sample_conformal_quantile")),
        configured_capacity_rps=cap_val,
        capacity_margin_rps=cap_margin,
        available_headroom_rps=headroom,
        overload_rps=overload,
        has_known_capacity=has_known_cap,
        is_overload=1.0 if (overload is not None and overload > 0.0) else 0.0,
        fallback_triggered=fallback_trig,
        fallback_reason=fallback_reason,
        fallback_active=fallback_active,
        reliability_status=rel_status,
        reliability_warnings=rel_warnings,
        prediction_state=WorkloadForecastState.VALID_TRAINED_PREDICTION,
    )


def extract_temporal_sequence_tensor(
    service_id: str,
    incident_service_telemetry: Dict[str, List[Dict[str, Any]]],
    incident_boundary_telemetry: List[Dict[str, Any]],
    origin_step: int,
    horizon: int = 1,
    window_size: int = 4,
    forecaster: Optional[Any] = None,
    physical_timestamp_sec: Optional[float] = None,
) -> np.ndarray:
    """Extracts a (window_size, num_features) temporal sequence tensor for GRU/TCN models.

    Uses strictly historical observations in [origin_step - window_size + 1, origin_step]
    with dependency neighbor context, capacity headroom, and forecast workload.
    """
    dt = SAMPLING_INTERVAL_SECONDS
    t = origin_step
    seq_steps = [max(0, t - window_size + 1 + step) for step in range(window_size)]

    # Compute forecast at origin using shared validated adapter
    s_hist = incident_service_telemetry[service_id][: t + 1]
    curr_s = s_hist[-1]
    offered_hist = [float(x.get("offered_rate", 20.0)) for x in s_hist]
    curr_offered = offered_hist[-1]

    phys_origin_ts = physical_timestamp_sec if physical_timestamp_sec is not None else float(curr_s.get("timestamp_sec", t * dt))
    phys_step_idx = int(curr_s.get("step_index", round(phys_origin_ts / dt)))

    adapted = adapt_workload_forecast(
        service_id=service_id,
        horizon=horizon,
        history=offered_hist,
        forecaster=forecaster,
        current_offered_rps=curr_offered,
        physical_timestamp_sec=phys_origin_ts,
        step_index=phys_step_idx,
    )

    fc_demand = adapted.forecast_demand_rps
    ci_width = adapted.interval_width_rps
    overload = float(adapted.overload_rps) if adapted.overload_rps is not None else 0.0

    cap = CONFIGURED_SERVICE_CAPACITIES.get(service_id)
    has_cap = 1.0 if cap is not None else 0.0

    timestep_vectors = []
    for step in seq_steps:
        s_data = incident_service_telemetry[service_id][step]
        b_data = incident_boundary_telemetry[step]

        target_lat = float(s_data.get("latency_ms", 15.0))
        target_q = float(s_data.get("backlog_requests", 0.0))
        target_off = float(s_data.get("offered_rate", 20.0))
        target_comp = float(s_data.get("completed_rate", target_off))
        target_err = float(s_data.get("error_rate", 0.0))
        target_drop = float(s_data.get("dropped_requests", 0.0))

        # Neighbor context
        gw_q = float(b_data.get("gateway_queue_depth", 0.0))
        gw_drift = float(b_data.get("sync_drift_score", 12.0))

        if service_id in ["reservations", "crew", "baggage"]:
            # Downstream neighbor is boundary-gateway & legacy-core
            leg_s = incident_service_telemetry["legacy-core"][step]
            neighbor_lat = float(leg_s.get("latency_ms", 68.0))
            neighbor_q = float(leg_s.get("backlog_requests", 0.0))
            upstream_pressure = target_off
        elif service_id == "boundary-gateway":
            # Upstream neighbors are cloud services; downstream is legacy
            cloud_rates = [
                float(incident_service_telemetry[cs][step].get("offered_rate", 20.0))
                for cs in ["reservations", "crew", "baggage"]
            ]
            upstream_pressure = sum(cloud_rates)
            leg_s = incident_service_telemetry["legacy-core"][step]
            neighbor_lat = float(leg_s.get("latency_ms", 68.0))
            neighbor_q = float(leg_s.get("backlog_requests", 0.0))
        else:  # legacy-core
            # Upstream is boundary-gateway
            neighbor_lat = float(b_data.get("boundary_latency_ms", 20.0))
            neighbor_q = gw_q
            upstream_pressure = float(b_data.get("gateway_submission_rate", 50.0))

        headroom = max(0.0, cap - target_off) if cap is not None else 0.0

        vec = [
            target_lat,
            target_q,
            target_off,
            target_comp,
            target_err,
            target_drop,
            neighbor_lat,
            neighbor_q,
            gw_drift,
            gw_q,
            upstream_pressure,
            has_cap,
            headroom,
            fc_demand,
            ci_width,
            overload,
        ]
        timestep_vectors.append(vec)

    return np.array(timestep_vectors, dtype=np.float32)


def extract_features_at_origin(
    service_id: str,
    incident_service_telemetry: Dict[str, List[Dict[str, Any]]],
    incident_boundary_telemetry: List[Dict[str, Any]],
    origin_step: int,
    horizon: int,
    forecaster: Optional[Any] = None,
    include_workload_forecasts: bool = True,
    physical_timestamp_sec: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extracts the feature vector at forecast origin t for horizon H without future data leakage."""
    dt = SAMPLING_INTERVAL_SECONDS
    t = origin_step

    s_history = incident_service_telemetry[service_id][: t + 1]
    b_history = incident_boundary_telemetry[: t + 1]

    curr_s = s_history[-1]
    prev_s = s_history[-2] if len(s_history) >= 2 else curr_s
    curr_b = b_history[-1]
    prev_b = b_history[-2] if len(b_history) >= 2 else curr_b

    # Preserve physical timestamps and physical step index
    phys_origin_ts = physical_timestamp_sec if physical_timestamp_sec is not None else float(curr_s.get("timestamp_sec", origin_step * dt))
    phys_step_idx = int(curr_s.get("step_index", round(phys_origin_ts / dt)))

    # 1. Telemetry-Only features
    curr_lat = float(curr_s.get("latency_ms", 15.0))
    lat_window = [float(x.get("latency_ms", 15.0)) for x in s_history[-3:]]
    lat_roll_mean = float(np.mean(lat_window))
    lat_delta = curr_lat - float(prev_s.get("latency_ms", 15.0))

    curr_backlog = float(curr_s.get("backlog_requests", 0.0))
    q_window = [float(x.get("backlog_requests", 0.0)) for x in s_history[-3:]]
    q_roll_max = float(np.max(q_window))
    q_delta = curr_backlog - float(prev_s.get("backlog_requests", 0.0))

    curr_offered = float(curr_s.get("offered_rate", 20.0))
    curr_completed = float(curr_s.get("completed_rate", curr_offered))
    throughput_ratio = float(curr_completed / max(1.0, curr_offered))

    curr_err = float(curr_s.get("error_rate", 0.0))
    curr_drops = float(curr_s.get("dropped_requests", 0.0))

    # Boundary metrics
    gw_q = float(curr_b.get("gateway_queue_depth", 0.0))
    gw_drift = float(curr_b.get("sync_drift_score", 12.0))
    gw_drift_delta = gw_drift - float(prev_b.get("sync_drift_score", 12.0))

    sub_rate = float(curr_b.get("gateway_submission_rate", curr_offered))
    comp_rate = float(curr_b.get("legacy_completion_rate", curr_completed))
    rate_diff = abs(sub_rate - comp_rate)
    rate_div_ratio = rate_diff / (sub_rate + 1e-4)

    b_lat = float(curr_b.get("boundary_latency_ms", 20.0))
    b_err = float(curr_b.get("boundary_error_rate", 0.0))

    # Cross-service pressure
    max_b = max([float(incident_service_telemetry[k][t].get("backlog_requests", 0.0)) for k in incident_service_telemetry])
    max_lat_ratio = max([
        float(incident_service_telemetry[k][t].get("latency_ms", 15.0)) / max(1.0, NOMINAL_BASELINES[k]["latency_ms"])
        for k in incident_service_telemetry
    ])

    nominal_cap = CONFIGURED_SERVICE_CAPACITIES.get(service_id)
    has_known_cap = 1.0 if nominal_cap is not None else 0.0
    known_cap_val = float(nominal_cap) if nominal_cap is not None else 0.0

    telemetry_features = [
        curr_lat,
        lat_roll_mean,
        lat_delta,
        curr_backlog,
        q_roll_max,
        q_delta,
        curr_offered,
        curr_completed,
        throughput_ratio,
        curr_err,
        curr_drops,
        gw_q,
        gw_drift,
        gw_drift_delta,
        rate_diff,
        rate_div_ratio,
        b_lat,
        b_err,
        max_b,
        max_lat_ratio,
        has_known_cap,
        known_cap_val,
    ]

    aux_info: Dict[str, Any] = {
        "service_id": service_id,
        "origin_step": origin_step,
        "horizon": horizon,
        "timestamp_sec": phys_origin_ts,
    }

    if not include_workload_forecasts:
        return np.array(telemetry_features, dtype=np.float64), aux_info

    # 2. Workload forecast features via shared validated adapter
    offered_history = [float(x.get("offered_rate", 20.0)) for x in s_history]

    adapted = adapt_workload_forecast(
        service_id=service_id,
        horizon=horizon,
        history=offered_history,
        forecaster=forecaster,
        current_offered_rps=curr_offered,
        physical_timestamp_sec=phys_origin_ts,
        step_index=phys_step_idx,
    )

    fc_demand = adapted.forecast_demand_rps
    fc_ratio = fc_demand / max(1.0, curr_offered)
    ci_low = adapted.interval_lower_rps
    ci_high = adapted.interval_upper_rps
    ci_width = adapted.interval_width_rps
    cap_margin = float(adapted.capacity_margin_rps) if adapted.capacity_margin_rps is not None else 0.0
    headroom = float(adapted.available_headroom_rps) if adapted.available_headroom_rps is not None else 0.0
    overload = float(adapted.overload_rps) if adapted.overload_rps is not None else 0.0
    is_ovld = adapted.is_overload
    fallback_active = adapted.fallback_active

    # Causal step-by-step rollout (bounded by buffer limits B_max and Little's Law)
    from .models import simulate_step_by_step_rollout

    mean_arrival_rate = (curr_offered + fc_demand) / 2.0
    rollout_arrivals = mean_arrival_rate * (horizon * dt)

    roll_res = simulate_step_by_step_rollout(
        service_id=service_id,
        curr_backlog=curr_backlog,
        curr_offered=curr_offered,
        forecast_demand=fc_demand,
        curr_completed=curr_completed,
        curr_sync_drift=gw_drift,
        horizon=horizon,
        dt=dt,
    )
    rollout_backlog = roll_res["backlog_requests"]
    rollout_latency_proj = roll_res["latency_ms"]

    forecast_features = [
        fc_demand,
        fc_ratio,
        ci_low,
        ci_high,
        ci_width,
        cap_margin,
        headroom,
        overload,
        is_ovld,
        fallback_active,
        rollout_arrivals,
        rollout_backlog,
        rollout_latency_proj,
    ]

    all_features = telemetry_features + forecast_features
    aux_info["forecast_demand"] = fc_demand
    aux_info["rollout_backlog"] = rollout_backlog
    aux_info["rollout_latency_proj"] = rollout_latency_proj
    aux_info["workload_forecast"] = adapted.to_dict()

    return np.array(all_features, dtype=np.float64), aux_info

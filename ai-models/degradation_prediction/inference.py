"""Inference Engine for BACCP Service and Boundary Degradation Prediction.

Version: baccp-degradation-inference-v1.3.0

Serves:
1. Continuous predictions at t+H (latency_ms, backlog_requests, boundary_sync_drift_pct)
2. Calibrated forward-window sustained degradation risk (3-consecutive breach rule)
3. Target 2: Alert-gated conditional time to degradation (seconds), or None if healthy/unsupported
4. Target 3: Observable degradation type and severity classes (including boundary drift)
5. System-wide composite risk score and calibrated system degradation probability (scoped strictly to full Model B)
6. Model variants: Model A (Telemetry Only), Model B (Telemetry + Forecasts), Model C (Temporal GRU)
7. Full input sanitization, timestamp-based stream alignment, data provenance tracking, and schema assertions
"""

from __future__ import annotations

import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from .data_contract import (
        BOUNDARY_SYNC_DRIFT_CRITICAL,
        BOUNDARY_SYNC_DRIFT_THRESHOLD,
        CONFIGURED_SERVICE_CAPACITIES,
        CONTRACT_VERSION,
        MAX_TELEMETRY_STALENESS_SECONDS,
        NOMINAL_BASELINES,
        SAMPLING_INTERVAL_SECONDS,
        DegradationSeverity,
        DegradationType,
        ServiceIdentifier,
        SLO_THRESHOLDS,
        classify_boundary_step_type_and_severity,
        classify_service_step_type_and_severity,
        compute_sync_drift_score,
        get_operational_state_at_origin,
    )
    from .features import (
        FEATURE_SCHEMA_VERSION,
        TELEMETRY_ONLY_FEATURE_NAMES,
        TELEMETRY_PLUS_FORECAST_FEATURE_NAMES,
        assert_feature_schema,
        extract_features_at_origin,
        extract_temporal_sequence_tensor,
    )
except ImportError:
    from degradation_prediction.data_contract import (
        BOUNDARY_SYNC_DRIFT_CRITICAL,
        BOUNDARY_SYNC_DRIFT_THRESHOLD,
        CONFIGURED_SERVICE_CAPACITIES,
        CONTRACT_VERSION,
        MAX_TELEMETRY_STALENESS_SECONDS,
        NOMINAL_BASELINES,
        SAMPLING_INTERVAL_SECONDS,
        DegradationSeverity,
        DegradationType,
        ServiceIdentifier,
        SLO_THRESHOLDS,
        classify_boundary_step_type_and_severity,
        classify_service_step_type_and_severity,
        compute_sync_drift_score,
        get_operational_state_at_origin,
    )
    from degradation_prediction.features import (
        FEATURE_SCHEMA_VERSION,
        TELEMETRY_ONLY_FEATURE_NAMES,
        TELEMETRY_PLUS_FORECAST_FEATURE_NAMES,
        assert_feature_schema,
        extract_features_at_origin,
        extract_temporal_sequence_tensor,
    )

try:
    from .alert_policy import ALERT_THRESHOLDS, evaluate_alert_decision, get_alert_threshold
except ImportError:
    from degradation_prediction.alert_policy import (
        ALERT_THRESHOLDS,
        evaluate_alert_decision,
        get_alert_threshold,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_MODELS_DIR = REPO_ROOT / "ai-models"
DEFAULT_WEIGHTS_PATH = AI_MODELS_DIR / "weights" / "degradation_prediction" / "degradation_models_bundle.pkl"

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


class InputReliabilityInfo(dict):
    """Container for input reliability tracking that supports both string equality and dictionary lookup."""

    def __init__(self, status: str, **kwargs):
        super().__init__(status=status, **kwargs)
        self.status = status

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.get("status") == other
        return super().__eq__(other)

    def __str__(self) -> str:
        return str(self.get("status", ""))


class DegradationPredictor:
    """Production inference engine for BACCP degradation and boundary drift prediction."""

    _instance: Optional["DegradationPredictor"] = None

    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS_PATH):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Degradation weights bundle not found at {self.weights_path}. Run train.py first."
            )

        with open(self.weights_path, "rb") as f:
            bundle = pickle.load(f)

        self.metadata = bundle["metadata"]
        self.models_telemetry_only = bundle["models_telemetry_only"]
        self.models_telemetry_forecast = bundle["models_telemetry_forecast"]
        self.models_temporal_comparator = bundle.get("models_temporal_comparator", {})
        self.models_time_to_onset = bundle.get("models_time_to_onset", {})
        self.models_type_severity = bundle.get("models_type_severity", {})
        self.models_system_calibrator = bundle.get("models_system_calibrator", {})

        self.supported_horizons = list(bundle["metadata"]["horizons"])
        self.supported_services = list(bundle["metadata"]["services"])

        self.forecaster = None
        self.forecaster_load_error: Optional[str] = None
        if WorkloadForecaster is not None:
            try:
                self.forecaster = WorkloadForecaster.get_instance()
            except Exception as exc:
                self.forecaster_load_error = str(exc)
        else:
            self.forecaster_load_error = "WorkloadForecaster could not be imported"

    @classmethod
    def get_instance(cls, weights_path: Path = DEFAULT_WEIGHTS_PATH) -> "DegradationPredictor":
        """Singleton accessor for efficient serving."""
        if cls._instance is None:
            cls._instance = cls(weights_path=weights_path)
        return cls._instance

    def _validate_and_sanitize_inputs(
        self,
        service_telemetry: Dict[str, List[Dict[str, Any]]],
        boundary_telemetry: Optional[List[Dict[str, Any]]],
        explicit_origin_step: Optional[int] = None,
        explicit_timestamp_sec: Optional[float] = None,
    ) -> Tuple[
        Dict[str, List[Dict[str, Any]]],
        List[Dict[str, Any]],
        int,
        float,
        List[str],
        List[str],
        Dict[str, Any],
    ]:
        """Validates telemetry streams, aligns by explicit timestamp, sanitizes non-finite values,
        computes boundary drift via repository formula when missing, and tracks all imputations.
        """
        dt = SAMPLING_INTERVAL_SECONDS
        warnings: List[str] = []
        imputed_services: List[str] = []
        imputed_fields_count = 0
        padded_services: List[str] = []
        carried_forward_services: List[str] = []

        # 1. Inspect and sanitize service streams
        cleaned_raw_svc: Dict[str, List[Dict[str, Any]]] = {}
        stream_lengths: Dict[str, int] = {}

        for s in self.supported_services:
            if s not in service_telemetry or len(service_telemetry[s]) == 0:
                warnings.append(f"Missing telemetry stream for service '{s}'. Synthesizing nominal baseline.")
                imputed_services.append(s)
                nominal_lat = NOMINAL_BASELINES[s]["latency_ms"]
                cleaned_raw_svc[s] = [
                    {
                        "latency_ms": nominal_lat,
                        "backlog_requests": 0.0,
                        "offered_rate": 20.0,
                        "completed_rate": 20.0,
                        "error_rate": 0.001,
                        "dropped_requests": 0.0,
                        "timestamp_sec": float(step_i * dt),
                        "step_index": step_i,
                    }
                    for step_i in range(4)
                ]
            else:
                s_list = []
                for idx, step_dict in enumerate(service_telemetry[s]):
                    clean_dict = dict(step_dict)
                    # Assign explicit timestamp_sec if missing
                    if "timestamp_sec" not in clean_dict:
                        clean_dict["timestamp_sec"] = float(clean_dict.get("step_index", idx) * dt)
                    else:
                        clean_dict["timestamp_sec"] = float(clean_dict["timestamp_sec"])
                    if "step_index" not in clean_dict:
                        clean_dict["step_index"] = int(round(clean_dict["timestamp_sec"] / dt))
                    else:
                        clean_dict["step_index"] = int(clean_dict["step_index"])

                    for k in ["latency_ms", "backlog_requests", "offered_rate", "completed_rate", "error_rate", "dropped_requests"]:
                        val = clean_dict.get(k, 0.0)
                        if val is None or math.isnan(val) or math.isinf(val):
                            imputed_fields_count += 1
                            clean_dict[k] = NOMINAL_BASELINES[s].get(k, 0.0)

                    s_list.append(clean_dict)

                # Validate monotonic ordering
                ts_vals = [x["timestamp_sec"] for x in s_list]
                if any(ts_vals[i] > ts_vals[i + 1] for i in range(len(ts_vals) - 1)):
                    warnings.append(f"Non-monotonic timestamps detected in service '{s}'. Sorting chronologically.")
                    s_list = sorted(s_list, key=lambda x: x["timestamp_sec"])

                # Check and deduplicate timestamps
                dedup_list = []
                seen_ts = {}
                has_duplicates = False
                for record in s_list:
                    r_ts = record["timestamp_sec"]
                    if r_ts in seen_ts:
                        has_duplicates = True
                        dedup_list[seen_ts[r_ts]] = record
                    else:
                        seen_ts[r_ts] = len(dedup_list)
                        dedup_list.append(record)
                if has_duplicates:
                    warnings.append(f"Duplicate timestamps detected in service '{s}'. Kept latest observation per timestamp.")
                    s_list = dedup_list

                # Check sampling gaps
                for i in range(len(s_list) - 1):
                    gap = s_list[i + 1]["timestamp_sec"] - s_list[i]["timestamp_sec"]
                    if gap > 1.5 * dt:
                        warnings.append(f"Sampling gap detected in service '{s}': {gap:.1f}s > {1.5*dt:.1f}s at step {i}.")

                cleaned_raw_svc[s] = s_list

            stream_lengths[s] = len(cleaned_raw_svc[s])

        # 2. Determine common alignment timestamp
        available_services = [s for s in self.supported_services if s not in imputed_services]
        if available_services:
            latest_available_ts = min(cleaned_raw_svc[s][-1]["timestamp_sec"] for s in available_services)
        else:
            latest_available_ts = cleaned_raw_svc[self.supported_services[0]][-1]["timestamp_sec"]

        if explicit_timestamp_sec is not None:
            target_ts = float(explicit_timestamp_sec)
        elif explicit_origin_step is not None:
            target_ts = float(explicit_origin_step * dt)
            if target_ts > latest_available_ts:
                warnings.append(f"Requested origin step {explicit_origin_step} ({target_ts}s) exceeds available stream coverage ({latest_available_ts}s). Clamping to {latest_available_ts}s.")
                target_ts = latest_available_ts
        else:
            target_ts = float(latest_available_ts)

        # 3. Clean and sanitize existing boundary telemetry early if provided
        cleaned_raw_bnd: List[Dict[str, Any]] = []
        boundary_missing = (boundary_telemetry is None or len(boundary_telemetry) == 0)
        if not boundary_missing:
            b_list = []
            for idx, b_step in enumerate(boundary_telemetry):
                clean_b = dict(b_step)
                if "timestamp_sec" not in clean_b:
                    clean_b["timestamp_sec"] = float(clean_b.get("step_index", idx) * dt)
                else:
                    clean_b["timestamp_sec"] = float(clean_b["timestamp_sec"])
                if "step_index" not in clean_b:
                    clean_b["step_index"] = int(round(clean_b["timestamp_sec"] / dt))
                else:
                    clean_b["step_index"] = int(clean_b["step_index"])

                for k in ["gateway_queue_depth", "sync_drift_score", "boundary_latency_ms", "boundary_error_rate"]:
                    val = clean_b.get(k, 0.0)
                    if val is None or math.isnan(val) or math.isinf(val):
                        imputed_fields_count += 1
                        clean_b[k] = 0.0

                b_list.append(clean_b)

            b_list = sorted(b_list, key=lambda x: x["timestamp_sec"])
            dedup_b = []
            seen_b_ts = {}
            has_b_dups = False
            for record in b_list:
                r_ts = record["timestamp_sec"]
                if r_ts in seen_b_ts:
                    has_b_dups = True
                    dedup_b[seen_b_ts[r_ts]] = record
                else:
                    seen_b_ts[r_ts] = len(dedup_b)
                    dedup_b.append(record)
            if has_b_dups:
                warnings.append("Duplicate timestamps detected in boundary stream. Kept latest observation per timestamp.")
            cleaned_raw_bnd = dedup_b

        # 4. Causal As-Of Validation:
        # Every consumed observation must have timestamp_sec <= target_ts.
        # NEVER backfill from future. Reject if all records are in the future.
        # Enforce documented staleness limit (MAX_TELEMETRY_STALENESS_SECONDS = 60.0s).
        earliest_obs_ts = target_ts
        for s in available_services:
            sub = [x for x in cleaned_raw_svc[s] if x["timestamp_sec"] <= target_ts + 1e-4]
            if len(sub) == 0:
                earliest_ts = cleaned_raw_svc[s][0]["timestamp_sec"]
                raise ValueError(
                    f"Causal alignment violation: service '{s}' has no historical observations at or before "
                    f"prediction timestamp {target_ts:.1f}s (earliest observation at {earliest_ts:.1f}s). "
                    "Future observations cannot be used during alignment."
                )

            latest_obs_ts = sub[-1]["timestamp_sec"]
            staleness = target_ts - latest_obs_ts
            if staleness > MAX_TELEMETRY_STALENESS_SECONDS:
                raise ValueError(
                    f"Staleness limit exceeded for service '{s}': latest available observation is at {latest_obs_ts:.1f}s, "
                    f"which is {staleness:.1f}s behind target prediction timestamp {target_ts:.1f}s "
                    f"(maximum permitted staleness is {MAX_TELEMETRY_STALENESS_SECONDS:.1f}s). Cannot predict on stale telemetry."
                )
            earliest_obs_ts = min(earliest_obs_ts, sub[0]["timestamp_sec"])

        if not boundary_missing:
            sub_b = [x for x in cleaned_raw_bnd if x["timestamp_sec"] <= target_ts + 1e-4]
            if len(sub_b) == 0:
                earliest_b_ts = cleaned_raw_bnd[0]["timestamp_sec"]
                raise ValueError(
                    f"Causal alignment violation: boundary telemetry has no historical observations at or before "
                    f"prediction timestamp {target_ts:.1f}s (earliest observation at {earliest_b_ts:.1f}s). "
                    "Future observations cannot be used during alignment."
                )
            b_latest_ts = sub_b[-1]["timestamp_sec"]
            b_staleness = target_ts - b_latest_ts
            if b_staleness > MAX_TELEMETRY_STALENESS_SECONDS:
                raise ValueError(
                    f"Staleness limit exceeded for boundary telemetry: latest observation is at {b_latest_ts:.1f}s, "
                    f"which is {b_staleness:.1f}s behind target prediction timestamp {target_ts:.1f}s "
                    f"(maximum permitted staleness is {MAX_TELEMETRY_STALENESS_SECONDS:.1f}s)."
                )
            earliest_obs_ts = min(earliest_obs_ts, sub_b[0]["timestamp_sec"])

        # 5. Regular Sampling Grid Construction:
        # Require matching physical timestamps on a regular sampling grid with dt=20s.
        # Eliminates cross-service timestamp misalignment (e.g. [0,20,40,60,80] vs [0,0,20,60,80]).
        span_steps = int(round((target_ts - earliest_obs_ts) / dt)) + 1
        grid_len = max(4, span_steps)
        grid_timestamps = [float(round(target_ts - (grid_len - 1 - k) * dt, 4)) for k in range(grid_len)]
        aligned_origin_index = grid_len - 1

        aligned_svc: Dict[str, List[Dict[str, Any]]] = {}
        for s in self.supported_services:
            if s in imputed_services:
                base_dict = cleaned_raw_svc[s][0]
                syn_list = [
                    {
                        **base_dict,
                        "timestamp_sec": grid_timestamps[step_i],
                        "source_timestamp_sec": grid_timestamps[step_i],
                        "step_index": int(round(grid_timestamps[step_i] / dt)),
                        "is_grid_aligned": True,
                        "is_carried_forward": False,
                        "is_padded": True,
                    }
                    for step_i in range(grid_len)
                ]
                aligned_svc[s] = syn_list
                continue

            sub_all = [x for x in cleaned_raw_svc[s] if x["timestamp_sec"] <= target_ts + 1e-4]
            svc_aligned_list = []
            for k, t_k in enumerate(grid_timestamps):
                eligible = [x for x in sub_all if x["timestamp_sec"] <= t_k + 1e-4]
                if len(eligible) > 0:
                    best_rec = eligible[-1]
                    stale_k = t_k - best_rec["timestamp_sec"]
                    if stale_k > MAX_TELEMETRY_STALENESS_SECONDS:
                        raise ValueError(
                            f"Staleness limit exceeded for service '{s}' at grid timestamp {t_k:.1f}s: "
                            f"latest observation at {best_rec['timestamp_sec']:.1f}s is {stale_k:.1f}s behind "
                            f"(maximum permitted staleness is {MAX_TELEMETRY_STALENESS_SECONDS:.1f}s)."
                        )
                    aligned_rec = dict(best_rec)
                    aligned_rec["timestamp_sec"] = t_k
                    aligned_rec["step_index"] = int(round(t_k / dt))
                    aligned_rec["source_timestamp_sec"] = float(best_rec["timestamp_sec"])
                    aligned_rec["is_grid_aligned"] = True
                    aligned_rec["is_padded"] = False
                    if stale_k > 1e-4:
                        aligned_rec["is_carried_forward"] = True
                        aligned_rec["staleness_seconds"] = round(stale_k, 2)
                        if s not in carried_forward_services:
                            carried_forward_services.append(s)
                            warnings.append(
                                f"Observation for service '{s}' carried forward from {best_rec['timestamp_sec']:.1f}s to {t_k:.1f}s "
                                f"(staleness {stale_k:.1f}s <= limit {MAX_TELEMETRY_STALENESS_SECONDS:.1f}s)."
                            )
                    else:
                        aligned_rec["is_carried_forward"] = False
                        aligned_rec["staleness_seconds"] = 0.0
                else:
                    # History padding for steps before earliest observation
                    earliest_rec = sub_all[0]
                    aligned_rec = dict(earliest_rec)
                    aligned_rec["timestamp_sec"] = t_k
                    aligned_rec["step_index"] = int(round(t_k / dt))
                    aligned_rec["source_timestamp_sec"] = float(earliest_rec["timestamp_sec"])
                    aligned_rec["is_grid_aligned"] = True
                    aligned_rec["is_padded"] = True
                    aligned_rec["is_carried_forward"] = False
                    aligned_rec["staleness_seconds"] = 0.0
                    if s not in padded_services:
                        padded_services.append(s)
                        warnings.append(
                            f"Short history for service '{s}' at grid timestamp {t_k:.1f}s "
                            f"(stream started at {earliest_rec['timestamp_sec']:.1f}s). Left-padded earliest step."
                        )
                svc_aligned_list.append(aligned_rec)
            aligned_svc[s] = svc_aligned_list

        # 6. Align or synthesize boundary telemetry on the identical regular sampling grid
        aligned_bnd: List[Dict[str, Any]] = []
        if boundary_missing:
            warnings.append("Boundary telemetry stream missing. Deriving drift score from gateway and legacy telemetry via repository formula.")
            for idx in range(grid_len):
                gw = aligned_svc["boundary-gateway"][idx]
                leg = aligned_svc["legacy-core"][idx]
                sub_rate = float(gw.get("offered_rate", 20.0))
                comp_rate = float(leg.get("completed_rate", 20.0))
                q_depth = float(gw.get("backlog_requests", 0.0))
                b_lat = float(gw.get("latency_ms", 20.0))
                b_err = float(gw.get("error_rate", 0.0))

                drift_score = compute_sync_drift_score(
                    gateway_submission_rate=sub_rate,
                    legacy_completion_rate=comp_rate,
                    gateway_queue_depth=q_depth,
                    boundary_latency_ms=b_lat,
                    nominal_latency_ms=15.0,
                    boundary_error_rate=b_err,
                )
                aligned_bnd.append({
                    "timestamp_sec": float(gw["timestamp_sec"]),
                    "step_index": int(gw.get("step_index", idx)),
                    "source_timestamp_sec": float(gw.get("source_timestamp_sec", gw["timestamp_sec"])),
                    "gateway_submission_rate": sub_rate,
                    "legacy_completion_rate": comp_rate,
                    "gateway_queue_depth": q_depth,
                    "boundary_latency_ms": b_lat,
                    "boundary_error_rate": b_err,
                    "sync_drift_score": drift_score,
                    "is_grid_aligned": True,
                    "is_carried_forward": bool(gw.get("is_carried_forward", False)),
                    "is_padded": bool(gw.get("is_padded", False)),
                })
        else:
            sub_b_all = [x for x in cleaned_raw_bnd if x["timestamp_sec"] <= target_ts + 1e-4]
            bnd_aligned_list = []
            for k, t_k in enumerate(grid_timestamps):
                eligible_b = [x for x in sub_b_all if x["timestamp_sec"] <= t_k + 1e-4]
                if len(eligible_b) > 0:
                    best_b = eligible_b[-1]
                    stale_b_k = t_k - best_b["timestamp_sec"]
                    if stale_b_k > MAX_TELEMETRY_STALENESS_SECONDS:
                        raise ValueError(
                            f"Staleness limit exceeded for boundary telemetry at grid timestamp {t_k:.1f}s: "
                            f"latest observation at {best_b['timestamp_sec']:.1f}s is {stale_b_k:.1f}s behind "
                            f"(maximum permitted staleness is {MAX_TELEMETRY_STALENESS_SECONDS:.1f}s)."
                        )
                    aligned_b = dict(best_b)
                    aligned_b["timestamp_sec"] = t_k
                    aligned_b["step_index"] = int(round(t_k / dt))
                    aligned_b["source_timestamp_sec"] = float(best_b["timestamp_sec"])
                    aligned_b["is_grid_aligned"] = True
                    aligned_b["is_padded"] = False
                    if stale_b_k > 1e-4:
                        aligned_b["is_carried_forward"] = True
                        aligned_b["staleness_seconds"] = round(stale_b_k, 2)
                        warnings.append(
                            f"Boundary observation carried forward to grid timestamp {t_k:.1f}s "
                            f"(source at {best_b['timestamp_sec']:.1f}s, staleness {stale_b_k:.1f}s <= limit {MAX_TELEMETRY_STALENESS_SECONDS:.1f}s)."
                        )
                    else:
                        aligned_b["is_carried_forward"] = False
                        aligned_b["staleness_seconds"] = 0.0
                else:
                    earliest_b = sub_b_all[0]
                    aligned_b = dict(earliest_b)
                    aligned_b["timestamp_sec"] = t_k
                    aligned_b["step_index"] = int(round(t_k / dt))
                    aligned_b["source_timestamp_sec"] = float(earliest_b["timestamp_sec"])
                    aligned_b["is_grid_aligned"] = True
                    aligned_b["is_padded"] = True
                    aligned_b["is_carried_forward"] = False
                    aligned_b["staleness_seconds"] = 0.0
                bnd_aligned_list.append(aligned_b)
            aligned_bnd = bnd_aligned_list

        if imputed_fields_count > 0:
            warnings.append(f"Non-finite or missing telemetry values detected and sanitized ({imputed_fields_count} fields imputed with nominal defaults).")

        accounting_info = {
            "imputed_fields_count": imputed_fields_count,
            "padded_services": padded_services,
            "carried_forward_services": carried_forward_services,
            "boundary_telemetry_missing": boundary_missing,
            "stream_lengths_before_alignment": stream_lengths,
            "aligned_origin_index": aligned_origin_index,
            "aligned_timestamp_sec": target_ts,
            "aligned_length": grid_len,
            "grid_timestamps": grid_timestamps,
            "max_telemetry_staleness_limit_sec": MAX_TELEMETRY_STALENESS_SECONDS,
        }

        return aligned_svc, aligned_bnd, aligned_origin_index, target_ts, warnings, imputed_services, accounting_info

    def predict(
        self,
        service_telemetry: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        boundary_telemetry: Optional[List[Dict[str, Any]]] = None,
        horizons: Optional[List[int]] = None,
        service_id: Optional[str] = None,
        include_workload_forecasts: bool = True,
        model_variant: Optional[str] = None,
        origin_step: Optional[int] = None,
        data_provenance: Optional[str] = None,
        service_telemetry_buffers: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        boundary_telemetry_buffer: Optional[List[Dict[str, Any]]] = None,
        target_services: Optional[List[str]] = None,
        prediction_timestamp_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generates continuous degradation predictions, event risks, estimated onset times, and type/severity.

        Parameters:
        - service_telemetry / service_telemetry_buffers: Mapping from service ID to list of chronological telemetry dictionaries
        - boundary_telemetry / boundary_telemetry_buffer: Optional list of chronological boundary telemetry dictionaries
        - horizons: Optional subset of horizons in [1, 5, 15]
        - service_id / target_services: Optional target services to predict
        - include_workload_forecasts: Ablation toggle (True=Model B, False=Model A Telemetry-Only)
        - model_variant: Optional explicit variant ("model_a_telemetry_only", "model_b_telemetry_forecast", "model_c_temporal_comparator")
        - origin_step: Optional explicit step origin (defaults to latest available step)
        - prediction_timestamp_sec: Optional explicit alignment timestamp
        - data_provenance: Optional explicit provenance ("LIVE_MEASURED", "SYNTHETIC_SIMULATED", "DEGRADED_IMPUTED")
        """
        # Resolve aliases
        svc_input = service_telemetry if service_telemetry is not None else service_telemetry_buffers
        if svc_input is None:
            raise ValueError("service_telemetry or service_telemetry_buffers must be provided.")
        bnd_input = boundary_telemetry if boundary_telemetry is not None else boundary_telemetry_buffer

        eval_services: List[str]
        if target_services is not None:
            eval_services = list(target_services)
        elif service_id is not None:
            eval_services = [service_id]
        else:
            eval_services = list(self.supported_services)

        # Strict input validation
        for s_target in eval_services:
            if s_target not in self.supported_services:
                raise ValueError(
                    f"Unsupported service_id '{s_target}'. Must be one of {self.supported_services}"
                )

        eval_horizons = horizons if horizons is not None else self.supported_horizons
        for h in eval_horizons:
            if h not in self.supported_horizons:
                raise ValueError(
                    f"Unsupported horizon '{h}'. Must be one of {self.supported_horizons}"
                )

        supported_variants = [
            "model_a_telemetry_only",
            "model_b_telemetry_forecast",
            "model_c_temporal_comparator",
        ]
        if model_variant is not None and model_variant not in supported_variants:
            raise ValueError(
                f"Unsupported model_variant '{model_variant}'. Must be one of {supported_variants}"
            )

        supported_provenances = ["LIVE_MEASURED", "SYNTHETIC_SIMULATED", "DEGRADED_IMPUTED"]
        if data_provenance is not None and data_provenance not in supported_provenances:
            raise ValueError(
                f"Unsupported data_provenance '{data_provenance}'. Must be one of {supported_provenances}"
            )

        # Validate and align telemetry streams
        (
            clean_svc,
            clean_bnd,
            current_origin,
            aligned_ts,
            warnings,
            imputed_services,
            accounting_info,
        ) = self._validate_and_sanitize_inputs(
            svc_input,
            bnd_input,
            explicit_origin_step=origin_step,
            explicit_timestamp_sec=prediction_timestamp_sec,
        )

        # Source provenance vs input reliability separation
        source_provenance = data_provenance if data_provenance is not None else "SYNTHETIC_SIMULATED"
        rel_warnings: List[str] = []
        if accounting_info["boundary_telemetry_missing"]:
            rel_warnings.append("MISSING_BOUNDARY_TELEMETRY")
        if accounting_info["imputed_fields_count"] > 0:
            rel_warnings.append("NON_FINITE_FIELDS_IMPUTED")
        if len(accounting_info["padded_services"]) > 0:
            rel_warnings.append("HISTORY_PADDED")
        if len(accounting_info["carried_forward_services"]) > 0:
            rel_warnings.append("OBSERVATIONS_CARRIED_FORWARD")

        if len(imputed_services) > 0 or accounting_info["imputed_fields_count"] > 0 or len(accounting_info["padded_services"]) > 0:
            rel_status = "DEGRADED_IMPUTED"
            warnings.append(f"Input streams required imputation or short-padding. Reliability: DEGRADED_IMPUTED.")
        elif source_provenance == "LIVE_MEASURED":
            rel_status = "NOMINAL_MEASURED"
        else:
            rel_status = "SYNTHETIC_SIMULATED"
            warnings.append("Note: Telemetry inputs classified as SYNTHETIC_SIMULATED. eBPF discovery captures tcp_connection_attempt only; application latency, backlog, and error rates are simulated/imputed.")

        input_reliability = InputReliabilityInfo(
            status=rel_status,
            has_history_padding=bool(len(accounting_info["padded_services"]) > 0),
            has_carried_forward=bool(len(accounting_info["carried_forward_services"]) > 0),
            boundary_stream_missing=bool(accounting_info["boundary_telemetry_missing"]),
            imputed_fields_count=accounting_info["imputed_fields_count"],
            imputed_services=imputed_services,
            padded_services=accounting_info["padded_services"],
            carried_forward_services=accounting_info["carried_forward_services"],
            reliability_warnings=rel_warnings,
        )

        warnings.append("Note: legacy-core configured capacity is uninstrumented (None). Preserved without synthetic assumption.")

        # Resolve active model variant
        if model_variant is None:
            active_variant = "model_b_telemetry_forecast" if include_workload_forecasts else "model_a_telemetry_only"
        else:
            active_variant = model_variant

        use_temporal = (active_variant == "model_c_temporal_comparator")
        use_forecasts = (active_variant in ["model_b_telemetry_forecast", "model_c_temporal_comparator"])

        # Prerequisite 3: Verify intended artifact loaded for forecast-enhanced models
        if use_forecasts and self.forecaster is None:
            err_msg = self.forecaster_load_error or "WorkloadForecaster instance is None"
            raise RuntimeError(
                f"Forecast-enhanced model variant '{active_variant}' requires a loaded WorkloadForecaster artifact, "
                f"but the artifact failed to load: {err_msg}. "
                "Will not silently interpret a missing/failed artifact as an intentional persistence baseline."
            )

        services_output: Dict[str, Any] = {}

        target_services = eval_services

        for s in target_services:
            curr_s = clean_svc[s][current_origin]
            curr_b = clean_bnd[current_origin]

            # Current operational state at origin
            thresh = SLO_THRESHOLDS.get(s, {})
            is_currently_degraded = (
                curr_s.get("latency_ms", 0.0) >= thresh.get("latency_ms", 100.0)
                or curr_s.get("backlog_requests", 0.0) >= thresh.get("backlog_requests", 50.0)
                or curr_s.get("error_rate", 0.0) >= thresh.get("error_rate", 0.05)
                or (s == ServiceIdentifier.BOUNDARY_GATEWAY.value and curr_b.get("sync_drift_score", 0.0) >= BOUNDARY_SYNC_DRIFT_THRESHOLD)
            )

            # Classify current step type and severity using boundary context
            if s == ServiceIdentifier.BOUNDARY_GATEWAY.value:
                cur_t, cur_sev = classify_service_step_type_and_severity(s, curr_s, curr_b)
            else:
                cur_t, cur_sev = classify_service_step_type_and_severity(s, curr_s, None)

            current_summary = {
                "latency_ms": round(float(curr_s.get("latency_ms", 15.0)), 2),
                "backlog_requests": round(float(curr_s.get("backlog_requests", 0.0)), 2),
                "error_rate": round(float(curr_s.get("error_rate", 0.0)), 4),
                "boundary_sync_drift_pct": round(float(curr_b.get("sync_drift_score", 12.0)), 2),
                "is_currently_degraded": bool(is_currently_degraded),
                "current_degradation_type": cur_t.value,
                "current_severity": cur_sev.value,
            }

            predictions_by_h: Dict[str, Any] = {}

            for h in eval_horizons:
                if h not in self.models_telemetry_forecast:
                    continue
                h_str = str(h)

                # 1. Feature extraction & schema assertion
                feat_vec, aux = extract_features_at_origin(
                    service_id=s,
                    incident_service_telemetry=clean_svc,
                    incident_boundary_telemetry=clean_bnd,
                    origin_step=current_origin,
                    horizon=h,
                    forecaster=self.forecaster,
                    include_workload_forecasts=use_forecasts,
                    physical_timestamp_sec=aligned_ts,
                )
                assert_feature_schema(feat_vec.reshape(1, -1), schema="combined" if use_forecasts else "telemetry")

                # Continuous prediction from boosting regressor
                model_boost = self.models_telemetry_forecast[h] if use_forecasts else self.models_telemetry_only[h]
                cont_preds = model_boost.predict_continuous(feat_vec.reshape(1, -1))
                pred_lat = float(cont_preds["latency_ms"][0])
                pred_backlog = float(cont_preds["backlog_requests"][0])
                pred_drift = float(cont_preds["boundary_sync_drift_pct"][0])

                # Event risk prediction
                if use_temporal and h in self.models_temporal_comparator:
                    seq_tensor = extract_temporal_sequence_tensor(
                        service_id=s,
                        incident_service_telemetry=clean_svc,
                        incident_boundary_telemetry=clean_bnd,
                        origin_step=current_origin,
                        horizon=h,
                        window_size=4,
                        forecaster=self.forecaster,
                        physical_timestamp_sec=aligned_ts,
                    )
                    m_temp = self.models_temporal_comparator[h]
                    event_risk = float(m_temp.predict_event_proba(seq_tensor.reshape(1, 4, -1))[0])
                else:
                    event_risk = float(model_boost.predict_event_proba(feat_vec.reshape(1, -1))[0])

                alert_issued, alert_thresh = evaluate_alert_decision(event_risk, h)

                # Target 2: Estimated time to degradation onset (seconds)
                # Semantics: Alert-gated conditional onset prediction. Returns None when alert is not active.
                time_reg = self.models_time_to_onset.get(h)
                pred_time_sec = None
                if time_reg is not None and alert_issued:
                    if not use_forecasts:
                        feat_vec_35, _ = extract_features_at_origin(
                            service_id=s,
                            incident_service_telemetry=clean_svc,
                            incident_boundary_telemetry=clean_bnd,
                            origin_step=current_origin,
                            horizon=h,
                            forecaster=self.forecaster,
                            include_workload_forecasts=True,
                            physical_timestamp_sec=aligned_ts,
                        )
                    else:
                        feat_vec_35 = feat_vec
                    time_res = time_reg.predict_time_to_onset_sec(feat_vec_35.reshape(1, -1), np.array([1]))
                    pred_time_sec = time_res[0] if time_res else None

                # Target 3: Observable degradation type & severity
                type_sev_clf = self.models_type_severity.get(h)
                pred_type = DegradationType.NOMINAL.value
                pred_sev = DegradationSeverity.NOMINAL.value
                if type_sev_clf is not None and alert_issued:
                    if not use_forecasts:
                        feat_vec_35, _ = extract_features_at_origin(
                            service_id=s,
                            incident_service_telemetry=clean_svc,
                            incident_boundary_telemetry=clean_bnd,
                            origin_step=current_origin,
                            horizon=h,
                            forecaster=self.forecaster,
                            include_workload_forecasts=True,
                            physical_timestamp_sec=aligned_ts,
                        )
                    else:
                        feat_vec_35 = feat_vec
                    types_out, sevs_out = type_sev_clf.predict(feat_vec_35.reshape(1, -1), np.array([1]))
                    pred_type = types_out[0]
                    pred_sev = sevs_out[0]

                # Special boundary-gateway drift handling: if drift breaches SLO, reflect boundary drift type
                if s == ServiceIdentifier.BOUNDARY_GATEWAY.value and pred_drift >= BOUNDARY_SYNC_DRIFT_THRESHOLD and alert_issued:
                    if pred_type == DegradationType.NOMINAL.value:
                        pred_type = DegradationType.BOUNDARY_DRIFT_DETERIORATION.value
                        pred_sev = DegradationSeverity.CRITICAL.value if pred_drift >= 70.0 else (
                            DegradationSeverity.MAJOR.value if pred_drift >= 55.0 else DegradationSeverity.MINOR.value
                        )
                    elif pred_type != DegradationType.BOUNDARY_DRIFT_DETERIORATION.value:
                        pred_type = DegradationType.COMPOUND_MULTIPLE.value

                # Upstream forecast metadata if available
                upstream_info = None
                if use_forecasts and "forecast_demand" in aux:
                    cap = CONFIGURED_SERVICE_CAPACITIES.get(s)
                    upstream_info = {
                        "forecast_demand_req_s": round(float(aux["forecast_demand"]), 2),
                        "nominal_capacity_req_s": cap,
                        "rollout_projected_backlog": round(float(aux["rollout_backlog"]), 2),
                        "rollout_projected_latency_ms": round(float(aux["rollout_latency_proj"]), 2),
                        "rollout_assumption": "step_by_step_bounded_causal_rollout",
                    }

                predictions_by_h[str(h)] = {
                    "horizon_steps": h,
                    "horizon_seconds": float(h * SAMPLING_INTERVAL_SECONDS),
                    "predicted_continuous": {
                        "latency_ms": round(pred_lat, 2),
                        "backlog_requests": round(pred_backlog, 2),
                        "boundary_sync_drift_pct": round(pred_drift, 2),
                    },
                    "degradation_event_risk": round(event_risk, 4),
                    "calibrated_event_probability_raw": float(event_risk),
                    "alert_issued": alert_issued,
                    "alert_threshold": round(alert_thresh, 3),
                    "estimated_time_to_degradation_sec": pred_time_sec,
                    "time_to_degradation_semantics": "alert_gated_conditional_prediction",
                    "predicted_degradation_type": pred_type,
                    "predicted_severity": pred_sev,
                    "slo_thresholds": thresh,
                    "upstream_forecast_integration": upstream_info,
                }
                predictions_by_h[h_str]["predicted_event_probability"] = round(event_risk, 4)
                predictions_by_h[h_str]["predicted_event_probability_raw"] = float(event_risk)

            services_output[s] = {
                "service_id": s,
                "current_state": current_summary,
                "predictions_by_horizon": predictions_by_h,
                "horizons": predictions_by_h,
            }

        curr_b_state = clean_bnd[current_origin]
        drift_score = float(curr_b_state.get("sync_drift_score", 12.0))
        drift_status = "CRITICAL" if drift_score >= 70.0 else ("DEGRADED" if drift_score >= 45.0 else "HEALTHY")

        # Top-level system prediction with strict calibration scoping
        system_predictions_by_h: Dict[str, Any] = {}
        for h in eval_horizons:
            h_str = str(h)
            svc_risks = [
                services_output[s]["predictions_by_horizon"][h_str]["degradation_event_risk"]
                for s in target_services
                if h_str in services_output[s]["predictions_by_horizon"]
            ]
            composite_risk = max(svc_risks) if svc_risks else 0.0

            # Calibration Scope Enforcement:
            # System calibrator was fitted on Model B's maximum across ALL 5 supported services.
            # It must ONLY be applied when:
            # 1. active_variant == "model_b_telemetry_forecast"
            # 2. All 5 supported services are evaluated (len(target_services) == len(self.supported_services))
            can_calibrate_system = (
                active_variant == "model_b_telemetry_forecast"
                and len(target_services) == len(self.supported_services)
                and h in self.models_system_calibrator
            )

            if can_calibrate_system:
                sys_cal = self.models_system_calibrator[h]
                cal_sys_risk = round(float(sys_cal.predict_proba(np.array([composite_risk]))[0]), 4)
                cal_status = "CALIBRATED"
                cal_note = "Calibrated system-event probability via isotonic regression on 5-service composite risk."
            else:
                cal_sys_risk = None
                if len(target_services) < len(self.supported_services):
                    cal_status = "UNAVAILABLE_SUBSET_SCOPE"
                    cal_note = "System calibrator requires all 5 services; cannot calibrate a single-service or subset risk."
                else:
                    cal_status = f"UNAVAILABLE_FOR_MODEL_VARIANT_{active_variant.upper()}"
                    cal_note = f"System calibrator is fitted for Model B and is not applicable to {active_variant}."

            sys_alert = any(
                services_output[s]["predictions_by_horizon"][h_str]["alert_issued"]
                for s in target_services
                if h_str in services_output[s]["predictions_by_horizon"]
            )

            valid_times = [
                services_output[s]["predictions_by_horizon"][h_str]["estimated_time_to_degradation_sec"]
                for s in target_services
                if h_str in services_output[s]["predictions_by_horizon"]
                and services_output[s]["predictions_by_horizon"][h_str]["estimated_time_to_degradation_sec"] is not None
            ]
            earliest_time = min(valid_times) if valid_times else None

            system_predictions_by_h[h_str] = {
                "horizon_steps": h,
                "horizon_seconds": float(h * SAMPLING_INTERVAL_SECONDS),
                "system_composite_risk_score": round(float(composite_risk), 4),
                "system_degradation_risk": cal_sys_risk if cal_sys_risk is not None else round(float(composite_risk), 4),
                "system_calibrated_degradation_risk": cal_sys_risk,
                "system_calibration_status": cal_status,
                "system_calibration_note": cal_note,
                "system_alert_issued": bool(sys_alert),
                "system_alert_policy": "SERVICE_DISJUNCTION",
                "earliest_time_to_degradation_sec": earliest_time,
            }

        history_len = len(clean_svc[self.supported_services[0]])
        is_full_history = bool(history_len >= 15)

        return {
            "contract_version": CONTRACT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_variant": active_variant,
            "origin_step": current_origin,
            "timestamp_sec": aligned_ts,
            "alignment_timestamp_sec": aligned_ts,
            "input_reliability": input_reliability,
            "data_provenance": source_provenance,
            "imputed_services": imputed_services,
            "input_accounting": accounting_info,
            "is_full_history": is_full_history,
            "history_length_steps": history_len,
            "boundary_status": {
                "sync_drift_score": round(drift_score, 2),
                "status": drift_status,
                "gateway_queue_depth": round(float(curr_b_state.get("gateway_queue_depth", 0.0)), 2),
            },
            "system_prediction": {
                "predictions_by_horizon": system_predictions_by_h,
            },
            "system_aggregate_predictions": system_predictions_by_h,
            "services": services_output,
            "per_service_predictions": services_output,
            "warnings": warnings,
        }

    # Backward compatibility alias
    predict_degradation = predict

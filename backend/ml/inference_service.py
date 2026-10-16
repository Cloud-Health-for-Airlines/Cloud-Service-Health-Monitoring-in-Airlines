"""Inference Service for BACCP Backend.

Coordinates:
- Workload forecasting (/forecast/workload)
- Degradation prediction (/predict/degradation)
- Degradation explanations (/explain/{service})

Key principles:
1. Load model bundles once per process, not once per request.
2. Graceful 503 fallback if model artifacts or dependencies are missing.
3. Bounded LRU caching keyed by (source/snapshot, service, horizon, model_version).
4. Asynchronous ThreadPoolExecutor (max 2 workers) for CPU-heavy explanation work.
5. Strict snapshot consistency across predictions and explanations.
6. JSON serialization without NaN or Inf.
"""

from __future__ import annotations

import collections
import concurrent.futures
import copy
import logging
import math
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("baccp.ml.service")

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_MODELS_DIR = REPO_ROOT / "ai-models"

from backend.ml.alert_policy import (
    ALERT_THRESHOLDS,
    evaluate_alert_decision,
    get_alert_threshold,
)
from backend.ml.metrics import BACCPMLMetrics
from backend.ml.telemetry_adapter import (
    HORIZON_SECONDS_MAP,
    SAMPLING_INTERVAL_SECONDS,
    SUPPORTED_HORIZONS,
    SUPPORTED_SERVICES,
    TelemetryAdapter,
    TelemetryError,
    TelemetrySnapshot,
)

SUPPORTED_MODEL_VARIANTS = (
    "model_a_telemetry_only",
    "model_b_telemetry_forecast",
    "model_c_temporal_comparator",
)


def sanitize_floats(obj: Any) -> Any:
    """Recursively converts non-finite floats (NaN, Inf) to None for safe JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_floats(v) for v in obj)
    return obj


class MLInferenceService:
    """Singleton inference coordinator and endpoint handler."""

    _instance: Optional["MLInferenceService"] = None

    def __init__(
        self,
        telemetry_adapter: Optional[TelemetryAdapter] = None,
        max_cache_size: int = 256,
        max_workers: int = 2,
        explanation_timeout_seconds: float = 10.0,
    ):
        self.telemetry_adapter = telemetry_adapter or TelemetryAdapter()
        self.metrics = BACCPMLMetrics.get_instance()
        self.max_cache_size = max_cache_size
        self.explanation_timeout_seconds = explanation_timeout_seconds

        # Bounded LRU Cache: (cache_type, snapshot_id, service_id, horizon, model_variant) -> response_dict
        self._cache_lock = threading.Lock()
        self._cache: collections.OrderedDict[Tuple[str, str, str, int, str], Dict[str, Any]] = collections.OrderedDict()

        # Thread pool executor bounded to at most 2 CPU workers
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="baccp-ml-worker",
        )
        # Bounded concurrency guard for explanation requests (max 4 concurrent or queued)
        self._explanation_semaphore = threading.BoundedSemaphore(value=4)

        # Lazy model references
        self.forecaster = None
        self.predictor = None
        self.explainer = None
        self.models_available: bool = False
        self.model_load_error: Optional[str] = None

        self._load_models_once()

    @classmethod
    def get_instance(cls) -> "MLInferenceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_models_once(self) -> None:
        """Attempts to load frozen model bundles once per process."""
        try:
            import sys
            if str(AI_MODELS_DIR) not in sys.path:
                sys.path.insert(0, str(AI_MODELS_DIR))
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))

            from workload_forecasting.inference import WorkloadForecaster
            from degradation_prediction.inference import DegradationPredictor
            from degradation_prediction.explanation import DegradationExplainer

            self.forecaster = WorkloadForecaster.get_instance()
            self.predictor = DegradationPredictor.get_instance()
            self.explainer = DegradationExplainer.get_instance(predictor=self.predictor)

            self.models_available = True
            logger.info("Successfully loaded frozen ML model bundles (WorkloadForecaster, DegradationPredictor, DegradationExplainer).")
        except Exception as exc:
            self.models_available = False
            self.model_load_error = str(exc)
            logger.warning(
                f"Optional ML model bundles could not be initialized: {exc}. "
                "Backend will serve existing routes normally; ML endpoints will return structured 503."
            )

    def _get_from_cache(self, key: Tuple[str, str, str, int, str]) -> Optional[Dict[str, Any]]:
        """Retrieves cached result and marks as recently used (thread-safe)."""
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return copy.deepcopy(self._cache[key])
            return None

    def _put_in_cache(self, key: Tuple[str, str, str, int, str], value: Dict[str, Any]) -> None:
        """Stores result in bounded LRU cache (thread-safe)."""
        with self._cache_lock:
            if len(self._cache) >= self.max_cache_size:
                self._cache.popitem(last=False)
            self._cache[key] = copy.deepcopy(value)

    def clear_cache_for_testing(self) -> None:
        """Clears LRU cache for test isolation."""
        with self._cache_lock:
            self._cache.clear()

    # -------------------------------------------------------------------------
    # 1. GET /forecast/workload?service=...&horizon=...
    # -------------------------------------------------------------------------
    def handle_workload_forecast(
        self,
        service: str,
        horizon: int,
        replay_step: Optional[int] = None,
        replay_time_sec: Optional[float] = None,
        incident_id: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Handles GET /forecast/workload."""
        if not self.models_available or self.forecaster is None:
            return 503, {
                "error": "ML_MODELS_UNAVAILABLE",
                "message": "Workload forecasting model bundle is unavailable.",
                "detail": self.model_load_error,
            }

        # 1. Retrieve telemetry snapshot
        snap_or_err = self.telemetry_adapter.get_telemetry_snapshot(
            service_id=service,
            horizon=horizon,
            replay_step=replay_step,
            replay_time_sec=replay_time_sec,
            incident_id=incident_id,
        )
        if isinstance(snap_or_err, TelemetryError):
            self.metrics.set_telemetry_unavailable(service, snap_or_err.staleness_seconds or 0.0)
            return snap_or_err.status_code, snap_or_err.to_dict()

        snapshot: TelemetrySnapshot = snap_or_err
        cache_key = ("forecast", snapshot.snapshot_id, service, horizon, "auto_fallback")

        # 2. Check bounded cache
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            cached["cached"] = True
            return 200, cached

        # 3. Perform inference
        try:
            history = snapshot.get_service_history(service, field="offered_rate")
            if not history:
                return 422, {
                    "error": "UNAVAILABLE_INSUFFICIENT_TELEMETRY",
                    "message": f"Offered rate history for service '{service}' is empty.",
                    "service": service,
                }

            wf_res = self.forecaster.predict(
                service_id=service,
                history=history,
                horizon=horizon,
                timestamp_sec=snapshot.timestamp_sec,
            )

            # 1. Resolve pending forecasts whose target timestamp has arrived at current snapshot clock
            self.metrics.resolve_pending_forecasts(
                incident_id=snapshot.incident_id or snapshot.data_source,
                current_clock_sec=snapshot.timestamp_sec,
                service_telemetry=snapshot.service_telemetry,
            )

            # 2. Record new pending forecast for future observation matching when clock advances
            target_ts = snapshot.timestamp_sec + float(horizon * SAMPLING_INTERVAL_SECONDS)
            pred_demand = float(wf_res.get("point_forecast_rps", wf_res.get("predicted_demand", 0.0)))
            self.metrics.record_workload_forecast(
                incident_id=snapshot.incident_id or snapshot.data_source,
                service=service,
                horizon=horizon,
                predicted_demand=pred_demand,
                origin_timestamp_sec=snapshot.timestamp_sec,
                target_timestamp_sec=target_ts,
            )

            cap_metrics = wf_res.get("capacity_metrics", {})
            pred_interval = wf_res.get("prediction_interval_90", wf_res.get("conformal_interval"))
            reliability_meta = wf_res.get("reliability_metadata", {})

            response = {
                "service": service,
                "horizon_steps": horizon,
                "horizon_seconds": HORIZON_SECONDS_MAP.get(horizon, float(horizon * SAMPLING_INTERVAL_SECONDS)),
                "observation_timestamp_sec": snapshot.timestamp_sec,
                "target_timestamp_sec": target_ts,
                "data_source": snapshot.data_source,
                "telemetry_mode": snapshot.telemetry_mode,
                "data_provenance": snapshot.data_provenance,
                "incident_id": snapshot.incident_id,
                "snapshot_id": snapshot.snapshot_id,
                "freshness_seconds": snapshot.freshness_seconds,
                "model_name": wf_res.get("selected_model", wf_res.get("model_name", "gradient_boosting")),
                "model_version": "baccp-workload-v1.1.0",
                "predicted_demand_req_s": round(pred_demand, 2),
                "prediction_interval_90": pred_interval,
                "conformal_interval": pred_interval,
                "conformal_coverage": 0.90,
                "capacity_margin_req_s": cap_metrics.get("capacity_margin_rps"),
                "available_headroom_req_s": cap_metrics.get("available_headroom_rps"),
                "is_overloaded": cap_metrics.get("is_overloaded", False),
                "candidate_models": wf_res.get("candidate_forecasts", wf_res.get("candidate_models")),
                "reliability": {
                    "status": reliability_meta.get("reliability_status", "nominal_in_distribution"),
                    "is_stale": snapshot.freshness_seconds > 60.0,
                    "is_carried_forward": snapshot.is_carried_forward,
                    "warnings": snapshot.warnings + reliability_meta.get("confidence_warnings", []),
                },
                "cached": False,
            }

            clean_resp = sanitize_floats(response)
            self._put_in_cache(cache_key, clean_resp)
            return 200, clean_resp
        except Exception as exc:
            logger.exception("Error during workload forecast inference")
            return 500, {"error": "INFERENCE_ERROR", "message": str(exc), "service": service}

    # -------------------------------------------------------------------------
    # 2. GET /predict/degradation?service=...&horizon=...
    # -------------------------------------------------------------------------
    def handle_degradation_prediction(
        self,
        service: str,
        horizon: int,
        replay_step: Optional[int] = None,
        replay_time_sec: Optional[float] = None,
        incident_id: Optional[str] = None,
        model_variant: str = "model_b_telemetry_forecast",
    ) -> Tuple[int, Dict[str, Any]]:
        """Handles GET /predict/degradation."""
        if model_variant not in SUPPORTED_MODEL_VARIANTS:
            return 400, {
                "error": "UNSUPPORTED_MODEL_VARIANT",
                "message": f"Model variant '{model_variant}' is not supported. Supported variants: {list(SUPPORTED_MODEL_VARIANTS)}",
                "service": service,
            }

        if not self.models_available or self.predictor is None:
            return 503, {
                "error": "ML_MODELS_UNAVAILABLE",
                "message": "Degradation prediction model bundle is unavailable.",
                "detail": self.model_load_error,
            }

        # 1. Retrieve telemetry snapshot
        snap_or_err = self.telemetry_adapter.get_telemetry_snapshot(
            service_id=service,
            horizon=horizon,
            replay_step=replay_step,
            replay_time_sec=replay_time_sec,
            incident_id=incident_id,
        )
        if isinstance(snap_or_err, TelemetryError):
            self.metrics.set_telemetry_unavailable(service, snap_or_err.staleness_seconds or 0.0)
            return snap_or_err.status_code, snap_or_err.to_dict()

        snapshot: TelemetrySnapshot = snap_or_err
        cache_key = ("predict", snapshot.snapshot_id, service, horizon, model_variant)

        # 2. Check bounded cache
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            cached["cached"] = True
            return 200, cached

        # 3. Perform inference
        try:
            pred_raw = self.predictor.predict(
                service_telemetry=snapshot.service_telemetry,
                boundary_telemetry=snapshot.boundary_telemetry,
                service_id=service,
                horizons=[horizon],
                model_variant=model_variant,
                prediction_timestamp_sec=snapshot.timestamp_sec,
                data_provenance=snapshot.data_provenance,
            )

            svc_data = pred_raw["services"][service]
            h_data = svc_data["predictions_by_horizon"][str(horizon)]
            curr_state = svc_data["current_state"]

            raw_event_prob = float(h_data.get("calibrated_event_probability_raw", h_data.get("event_risk_raw", h_data["degradation_event_risk"])))
            # Preserve core inference alert decision and threshold consistently
            if "alert_issued" in h_data:
                alert_issued = bool(h_data["alert_issued"])
                alert_thresh = float(h_data.get("alert_threshold", evaluate_alert_decision(raw_event_prob, horizon)[1]))
            else:
                alert_issued, alert_thresh = evaluate_alert_decision(raw_event_prob, horizon)

            time_to_deg = h_data.get("estimated_time_to_degradation_sec")

            time_reason = None
            if time_to_deg is None:
                time_reason = "service_nominal_no_active_alert" if not alert_issued else "unresolved_censored_trajectory"

            # Record Prometheus metrics (with strict deduplication on snapshot ID)
            self.metrics.record_degradation_prediction(
                service=service,
                horizon=horizon,
                probability=raw_event_prob,
                alert_issued=alert_issued,
                snapshot_id=snapshot.snapshot_id,
                freshness_seconds=snapshot.freshness_seconds,
            )

            # Expose sufficient probability precision if needed to avoid confusing displays
            if not alert_issued and round(raw_event_prob, 4) >= alert_thresh:
                calibrated_prob_disp = round(raw_event_prob, 5)
            else:
                calibrated_prob_disp = round(raw_event_prob, 4)

            response = {
                "service": service,
                "horizon_steps": horizon,
                "horizon_seconds": HORIZON_SECONDS_MAP.get(horizon, float(horizon * SAMPLING_INTERVAL_SECONDS)),
                "observation_timestamp_sec": snapshot.timestamp_sec,
                "target_timestamp_sec": snapshot.timestamp_sec + float(horizon * SAMPLING_INTERVAL_SECONDS),
                "data_source": snapshot.data_source,
                "telemetry_mode": snapshot.telemetry_mode,
                "data_provenance": snapshot.data_provenance,
                "incident_id": snapshot.incident_id,
                "snapshot_id": snapshot.snapshot_id,
                "model_variant": model_variant,
                "model_version": "baccp-degradation-inference-v1.3.0",
                "input_reliability": pred_raw.get("input_reliability", {}),
                "calibrated_event_probability": calibrated_prob_disp,
                "alert_threshold": round(alert_thresh, 3),
                "advisory_alert_decision": alert_issued,
                "alert_issued": alert_issued,
                "current_observed_state": {
                    "latency_ms": curr_state.get("latency_ms"),
                    "backlog_requests": curr_state.get("backlog_requests"),
                    "error_rate": curr_state.get("error_rate"),
                    "boundary_sync_drift_pct": curr_state.get("boundary_sync_drift_pct"),
                    "is_currently_degraded": curr_state.get("is_currently_degraded", False),
                    "current_degradation_type": curr_state.get("current_degradation_type", "NOMINAL"),
                    "current_severity": curr_state.get("current_severity", "NOMINAL"),
                },
                "predicted_future_risk": {
                    "continuous_predictions": h_data.get("continuous_predictions", {}),
                    "estimated_time_to_degradation_seconds": time_to_deg,
                    "time_to_degradation_reason": time_reason,
                    "predicted_degradation_type": h_data.get("predicted_degradation_type", "NOMINAL"),
                    "predicted_severity": h_data.get("predicted_severity", "NOMINAL"),
                    "slo_thresholds": h_data.get("slo_thresholds", {}),
                    "upstream_workload_forecast": h_data.get("upstream_workload_forecast"),
                },
                "cached": False,
            }

            clean_resp = sanitize_floats(response)
            self._put_in_cache(cache_key, clean_resp)
            return 200, clean_resp
        except Exception as exc:
            logger.exception("Error during degradation prediction inference")
            return 500, {"error": "INFERENCE_ERROR", "message": str(exc), "service": service}

    # -------------------------------------------------------------------------
    # 3. GET /explain/{service}?horizon=...
    # -------------------------------------------------------------------------
    def handle_explanation(
        self,
        service: str,
        horizon: int,
        replay_step: Optional[int] = None,
        replay_time_sec: Optional[float] = None,
        incident_id: Optional[str] = None,
        model_variant: str = "model_b_telemetry_forecast",
        timeout: Optional[float] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Handles GET /explain/{service} asynchronously using bounded thread pool."""
        if model_variant not in SUPPORTED_MODEL_VARIANTS:
            return 400, {
                "error": "UNSUPPORTED_MODEL_VARIANT",
                "message": f"Model variant '{model_variant}' is not supported. Supported variants: {list(SUPPORTED_MODEL_VARIANTS)}",
                "service": service,
            }

        if not self.models_available or self.explainer is None:
            return 503, {
                "error": "ML_MODELS_UNAVAILABLE",
                "message": "Degradation explanation engine is unavailable.",
                "detail": self.model_load_error,
            }

        # 1. Retrieve telemetry snapshot
        snap_or_err = self.telemetry_adapter.get_telemetry_snapshot(
            service_id=service,
            horizon=horizon,
            replay_step=replay_step,
            replay_time_sec=replay_time_sec,
            incident_id=incident_id,
        )
        if isinstance(snap_or_err, TelemetryError):
            self.metrics.set_telemetry_unavailable(service, snap_or_err.staleness_seconds or 0.0)
            return snap_or_err.status_code, snap_or_err.to_dict()

        snapshot: TelemetrySnapshot = snap_or_err
        cache_key = ("explain", snapshot.snapshot_id, service, horizon, model_variant)

        # 2. Check bounded cache
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            cached["cached"] = True
            return 200, cached

        # 3. Run CPU-heavy explanation in ThreadPoolExecutor with bounded queue
        if not self._explanation_semaphore.acquire(blocking=False):
            return 503, {
                "error": "EXPLANATION_CAPACITY_EXCEEDED",
                "message": "Explanation queue is at maximum capacity (max 4 active/queued requests). Please retry shortly.",
            }

        permit_released = False
        permit_lock = threading.Lock()

        def _release_permit(*args):
            nonlocal permit_released
            with permit_lock:
                if not permit_released:
                    permit_released = True
                    try:
                        self._explanation_semaphore.release()
                    except ValueError:
                        pass

        def _compute():
            return self.explainer.explain_prediction(
                service_telemetry=snapshot.service_telemetry,
                boundary_telemetry=snapshot.boundary_telemetry,
                service_id=service,
                horizon=horizon,
                model_variant=model_variant,
                prediction_timestamp_sec=snapshot.timestamp_sec,
            )

        eff_timeout = timeout if timeout is not None else getattr(self, "explanation_timeout_seconds", 10.0)

        try:
            try:
                future = self._executor.submit(_compute)
            except Exception:
                _release_permit()
                raise

            future.add_done_callback(_release_permit)

            raw_expl_res = future.result(timeout=eff_timeout)

            expl_payload = raw_expl_res["explanation"]
            pred_res = raw_expl_res["prediction"]

            svc_pred = pred_res["services"][service]
            h_pred = svc_pred["predictions_by_horizon"][str(horizon)]
            graph_loc = expl_payload.get("graph_localization", {})

            # Map graph localization states honestly:
            # An ambiguous tie must NOT be reported as localized!
            is_tie = bool(graph_loc.get("is_tie", False))
            is_abstained = bool(graph_loc.get("is_abstained", False))
            suspected = graph_loc.get("suspected_contributing_node")

            if is_tie:
                loc_status = "abstain_ambiguous_tie"
            elif is_abstained or suspected == "unknown_unlocalized":
                loc_status = "abstain_unknown_unlocalized"
            else:
                loc_status = "localized"

            raw_event_risk = float(h_pred.get("calibrated_event_probability_raw", h_pred.get("event_risk_raw", h_pred["degradation_event_risk"])))
            # Preserve core inference alert decision and threshold consistently
            if "alert_issued" in h_pred:
                alert_issued = bool(h_pred["alert_issued"])
                alert_thresh = float(h_pred.get("alert_threshold", evaluate_alert_decision(raw_event_risk, horizon)[1]))
            else:
                alert_issued, alert_thresh = evaluate_alert_decision(raw_event_risk, horizon)

            # Expose sufficient probability precision if needed to avoid confusing displays
            if not alert_issued and round(raw_event_risk, 4) >= alert_thresh:
                calibrated_prob_disp = round(raw_event_risk, 5)
            else:
                calibrated_prob_disp = round(raw_event_risk, 4)

            response = {
                "service": service,
                "horizon_steps": horizon,
                "horizon_seconds": HORIZON_SECONDS_MAP.get(horizon, float(horizon * SAMPLING_INTERVAL_SECONDS)),
                "observation_timestamp_sec": snapshot.timestamp_sec,
                "data_source": snapshot.data_source,
                "telemetry_mode": snapshot.telemetry_mode,
                "data_provenance": snapshot.data_provenance,
                "incident_id": snapshot.incident_id,
                "snapshot_id": snapshot.snapshot_id,
                "model_variant": model_variant,
                "model_version": "baccp-degradation-explainer-v1.0.0",
                "calibrated_event_probability": calibrated_prob_disp,
                "alert_threshold": round(alert_thresh, 3),
                "advisory_alert_decision": alert_issued,
                "alert_issued": alert_issued,
                "attribution_method": expl_payload.get("attribution_method"),
                "attribution_target": expl_payload.get("attribution_target"),
                "expected_value": expl_payload.get("expected_value"),
                "output_value": expl_payload.get("output_value"),
                "reconstruction_error": expl_payload.get("reconstruction_error"),
                "top_risk_increasing_drivers": expl_payload.get("top_risk_increasing_drivers", []),
                "top_risk_decreasing_drivers": expl_payload.get("top_risk_decreasing_drivers", []),
                "category_attributions": expl_payload.get("category_attributions", {}),
                "localization": {
                    "suspected_contributing_node": suspected,
                    "propagation_bottleneck": graph_loc.get("propagation_bottleneck"),
                    "localization_status": loc_status,
                    "is_tie": is_tie,
                    "is_abstained": is_abstained,
                    "abstention_reason": graph_loc.get("abstention_reason"),
                    "candidate_rankings": graph_loc.get("candidate_rankings", []),
                },
                "incident_brief": expl_payload.get("incident_brief"),
                "methodology_disclaimer": expl_payload.get("methodology_disclaimer"),
                "cached": False,
            }

            clean_resp = sanitize_floats(response)
            self._put_in_cache(cache_key, clean_resp)
            return 200, clean_resp
        except concurrent.futures.TimeoutError:
            return 504, {"error": "EXPLANATION_TIMEOUT", "message": f"Explanation computation exceeded {eff_timeout:.1f}s timeout."}
        except ValueError as val_err:
            return 400, {"error": "INVALID_EXPLANATION_REQUEST", "message": str(val_err)}
        except Exception as exc:
            logger.exception("Error during explanation generation")
            return 500, {"error": "EXPLANATION_ERROR", "message": str(exc), "service": service}

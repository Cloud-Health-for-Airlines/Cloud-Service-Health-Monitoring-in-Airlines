"""Reusable forecasting inference interface for BACCP per-service workload forecasting.

Version: baccp-inference-v1.1.0

Provides high-performance, validated single-service and multi-horizon inference:
- Validates input history, non-finite values, and safely handles short history padding.
- Extracts features strictly consistent with training pipeline and continuous phase anchoring.
- Preserves predictions from ALL candidate models.
- Implements training-derived fallback policy when workload exceeds training distribution.
- Computes explicit signed capacity margin, available headroom, and overload.
- Exposes transparent reliability metadata, empirical calibration status, and distribution shift warnings.
"""

from __future__ import annotations

import math
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .data_contract import CONFIGURED_SERVICE_CAPACITIES, ServiceIdentifier
from .features import CONTEXT_WINDOW_STEPS, SUPPORTED_HORIZONS, extract_features_at_origin
from .models import compute_capacity_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "ai-models" / "weights" / "workload_forecasting" / "workload_models_bundle.pkl"


class WorkloadForecaster:
    """Production inference engine for BACCP workload forecasting."""

    _instance: Optional["WorkloadForecaster"] = None

    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS_PATH):
        self.weights_path = weights_path
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights bundle not found at {self.weights_path}. Run train.py first."
            )

        with open(self.weights_path, "rb") as f:
            bundle = pickle.load(f)

        self.metadata = bundle["metadata"]
        self.models = bundle["models"]
        self.calibrators = bundle["calibrators"]
        self.supported_services = set(self.metadata["services"])
        self.supported_horizons = set(self.metadata["horizons"])
        self.feature_schema = self.metadata.get("feature_schema", {})
        self.training_domain_bounds = self.metadata.get("training_domain_bounds", {})
        self.fallback_config = self.metadata.get("fallback_policy", {})

    @classmethod
    def get_instance(cls, weights_path: Path = DEFAULT_WEIGHTS_PATH) -> "WorkloadForecaster":
        """Singleton accessor for efficient serving."""
        if cls._instance is None:
            cls._instance = cls(weights_path=weights_path)
        return cls._instance

    def _prepare_history_array(self, raw_history: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, bool]:
        """Validates input finiteness and safely pads history if shorter than context window."""
        arr = np.asarray(raw_history, dtype=np.float64)
        if len(arr) == 0:
            raise ValueError("History array cannot be empty.")
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("History array contains NaN or infinite values.")

        padded = False
        if len(arr) < CONTEXT_WINDOW_STEPS:
            # Pad earliest observation to satisfy the minimum 15-step context window
            pad_len = CONTEXT_WINDOW_STEPS - len(arr)
            pad_values = np.full(pad_len, arr[0], dtype=np.float64)
            arr = np.concatenate([pad_values, arr])
            padded = True

        return arr, padded

    def predict(
        self,
        service_id: str,
        history: Union[List[float], np.ndarray],
        horizon: int = 1,
        model_name: str = "auto_fallback",
        step_index: Optional[int] = None,
        timestamp_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generates a point forecast, conformal interval, candidate predictions, and capacity metrics.

        Parameters:
        - service_id: Service identifier in {reservations, crew, baggage, boundary-gateway, legacy-core}
        - history: Historical sequence of offered requests/sec up to current time t
        - horizon: Forecast step horizon in {1, 5, 15} (20s, 100s, 300s)
        - model_name: 'auto_fallback' (recommended), 'gradient_boosting', 'autoregressive_ridge',
                      'persistence', or 'seasonal_naive'
        - step_index: Optional discrete telemetry step index for consistent micro-cycle phase anchoring
        - timestamp_sec: Optional elapsed time in seconds for continuous micro-cycle phase anchoring

        Returns:
        - Standardized JSON-serializable dictionary with predictions, candidate forecasts,
          capacity headroom, and reliability metadata.
        """
        if service_id not in self.supported_services:
            raise ValueError(
                f"Unsupported service '{service_id}'. Must be one of {sorted(self.supported_services)}"
            )
        if horizon not in self.supported_horizons:
            raise ValueError(
                f"Unsupported horizon {horizon}. Must be one of {sorted(self.supported_horizons)}"
            )

        valid_model_choices = {
            "auto_fallback",
            "gradient_boosting",
            "autoregressive_ridge",
            "persistence",
            "seasonal_naive",
        }
        if model_name not in valid_model_choices:
            raise ValueError(
                f"Unknown model_name '{model_name}'. Must be one of {sorted(valid_model_choices)}"
            )

        arr, padded = self._prepare_history_array(history)
        y_t = float(arr[-1])

        # Track whether micro-cycle phase is anchored to physical time
        is_phase_anchored = bool(step_index is not None or timestamp_sec is not None)

        # Extract features at origin t using consistent 20-feature schema
        features = extract_features_at_origin(
            arr,
            step_index=step_index,
            timestamp_sec=timestamp_sec,
        )
        X = features.reshape(1, -1)

        # 1. Compute predictions for ALL candidate models
        available_models = self.models[service_id][horizon]
        candidate_forecasts: Dict[str, float] = {}
        for m_key, m_obj in available_models.items():
            pred_val = float(m_obj.predict(X)[0])
            candidate_forecasts[m_key] = round(max(0.0, pred_val), 3)

        # 2. Derive out-of-distribution status and fallback decision
        bounds = self.training_domain_bounds.get(
            service_id, {"min_rps": 0.0, "max_rps": 1e9, "surge_threshold_rps": 1e9}
        )
        min_train = bounds["min_rps"]
        max_train = bounds["max_rps"]
        surge_threshold = bounds["surge_threshold_rps"]

        is_out_of_range = bool(y_t < min_train or y_t > max_train)
        is_surge = bool(y_t > surge_threshold)

        fallback_triggered = False
        fallback_reason: Optional[str] = None

        if model_name == "auto_fallback":
            if is_surge:
                selected_model = "autoregressive_ridge"
                fallback_triggered = True
                fallback_reason = (
                    f"Workload at origin ({y_t:.2f} RPS) exceeds training distribution maximum "
                    f"({surge_threshold:.2f} RPS); switched to autoregressive_ridge for linear extrapolation."
                )
            else:
                selected_model = "gradient_boosting"
                fallback_triggered = False
                fallback_reason = None
        else:
            selected_model = model_name
            fallback_triggered = False
            fallback_reason = None

        point_pred = candidate_forecasts[selected_model]

        # 3. Retrieve conformal interval for selected model
        calibrator = self.calibrators[service_id][horizon][selected_model]
        lower_bound, upper_bound = calibrator.get_intervals(np.array([point_pred]))
        lower = max(0.0, float(lower_bound[0]))
        upper = float(upper_bound[0])

        # Enforce interval ordering: lower <= point <= upper
        lower = min(lower, point_pred)
        upper = max(upper, point_pred)

        # 4. Explicit capacity metrics
        capacity_metrics = compute_capacity_metrics(service_id, point_pred)

        # 5. Transparent reliability metadata & warnings list
        warnings_list: List[str] = []
        if is_out_of_range:
            warnings_list.append(
                f"WARNING: Workload at origin ({y_t:.2f} RPS) is outside training distribution [{min_train:.2f}, {max_train:.2f}] RPS. "
                "Conformal intervals calibrated on stationary operations do not guarantee coverage under distribution shift."
            )
        if padded:
            warnings_list.append(
                f"WARNING: Input history ({len(history)} steps) was shorter than required context window ({CONTEXT_WINDOW_STEPS}); "
                "earliest observation was left-padded."
            )
        if not is_phase_anchored:
            warnings_list.append(
                "NOTICE: Neither timestamp_sec nor step_index was provided; micro-cycle phase is unanchored (defaulted to neutral phase 0). "
                "Provide timestamp_sec or step_index to ensure continuous phase alignment."
            )

        reliability_status = (
            "insufficient_history" if padded else (
                "out_of_distribution_surge" if is_surge else (
                    "out_of_training_range" if is_out_of_range else "nominal_in_distribution"
                )
            )
        )

        reliability_metadata = {
            "reliability_status": reliability_status,
            "is_out_of_training_range": is_out_of_range,
            "training_workload_range_rps": [min_train, max_train],
            "surge_threshold_rps": surge_threshold,
            "input_workload_rps": round(y_t, 3),
            "input_history_length": len(history),
            "missing_history_padded": padded,
            "is_phase_anchored": is_phase_anchored,
            "exchangeability_assumed": False,
            "confidence_warnings": warnings_list,
        }

        return {
            "service_id": service_id,
            "horizon_steps": horizon,
            "horizon_seconds": float(horizon * 20.0),
            "selected_model": selected_model,
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "last_observed_rps": round(y_t, 3),
            "point_forecast_rps": point_pred,
            "prediction_interval_90": {
                "lower_rps": round(lower, 3),
                "upper_rps": round(upper, 3),
                "interval_width_rps": round(upper - lower, 3),
                "nominal_coverage": 0.90,
                "calibration_method": "exact_finite_sample_conformal_quantile",
                "calibrator_source_model": selected_model,
                "is_composite_policy_calibrated": False,
            },
            "candidate_forecasts": candidate_forecasts,
            "capacity_metrics": capacity_metrics,
            "reliability_metadata": reliability_metadata,
        }

    def predict_all_horizons(
        self,
        service_id: str,
        history: Union[List[float], np.ndarray],
        model_name: str = "auto_fallback",
        step_index: Optional[int] = None,
        timestamp_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generates predictions across all supported horizons H in {1, 5, 15}."""
        forecasts = {}
        for h in sorted(self.supported_horizons):
            forecasts[str(h)] = self.predict(
                service_id=service_id,
                history=history,
                horizon=h,
                model_name=model_name,
                step_index=step_index,
                timestamp_sec=timestamp_sec,
            )

        return {
            "service_id": service_id,
            "model_name": model_name,
            "forecasts_by_horizon": forecasts,
        }


# Convenient module-level accessor
def get_forecaster() -> WorkloadForecaster:
    return WorkloadForecaster.get_instance()

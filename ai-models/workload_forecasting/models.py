"""Workload forecasting model architectures and conformal prediction interval calibration.

Version: baccp-models-v1.1.0

Implements:
1. PersistenceForecaster: Last observed workload baseline.
2. SeasonalNaiveForecaster: S=15 seasonal lag with persistence fallback.
3. AutoregressiveForecaster: Regularized Ridge linear autoregression on scaled features.
4. GradientBoostingForecaster: Scikit-learn HistGradientBoostingRegressor on lagged features.
5. ConformalIntervalCalibrator: Exact finite-sample conformal quantile interval calibrator
   with explicit sample-size sufficiency checks.
6. compute_capacity_metrics: Explicit signed capacity margin, available headroom, and overload.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .data_contract import CONFIGURED_SERVICE_CAPACITIES


class PersistenceForecaster:
    """Baseline 1: Persistence (last observed workload at forecast origin t)."""

    def __init__(self, horizon: int):
        self.horizon = horizon

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceForecaster":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Feature 0 is lag_0 (y_t)
        return np.maximum(0.0, X[:, 0].copy())


class SeasonalNaiveForecaster:
    """Baseline 2: Seasonal Naive with documented period S=15 (300s) and persistence fallback."""

    def __init__(self, horizon: int, seasonal_period: int = 15):
        self.horizon = horizon
        self.seasonal_period = seasonal_period

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SeasonalNaiveForecaster":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Effective seasonal lag: target is t + H.
        # Reference point in past cycle is (t + H) - S.
        # Relative to origin t, this is S - H steps into the past.
        # If H == 15, S - H == 0 -> lag_0 (index 0).
        # If H == 5, S - H == 10 -> lag_9 (index 6, closest available lag <= 10).
        # If H == 1, S - H == 14 -> lag_14 (index 7).
        lag_steps_back = self.seasonal_period - self.horizon

        if lag_steps_back == 14:
            pred = X[:, 7].copy()  # lag_14
        elif lag_steps_back == 10:
            pred = X[:, 6].copy()  # lag_9
        elif lag_steps_back == 0:
            pred = X[:, 0].copy()  # lag_0
        else:
            pred = X[:, 0].copy()

        return np.maximum(0.0, pred)


class AutoregressiveForecaster:
    """Baseline 3: Statistical Regularized Autoregression (Ridge Linear Regression)."""

    def __init__(self, horizon: int, alpha: float = 1.0):
        self.horizon = horizon
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AutoregressiveForecaster":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return np.maximum(0.0, preds)


class GradientBoostingForecaster:
    """Learned Forecaster: Gradient Boosting on lagged workload and phase features."""

    def __init__(
        self,
        horizon: int,
        learning_rate: float = 0.05,
        max_iter: int = 120,
        max_depth: int = 5,
        min_samples_leaf: int = 15,
        l2_regularization: float = 0.1,
        random_state: int = 42,
    ):
        self.horizon = horizon
        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingForecaster":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.maximum(0.0, preds)


class ConformalIntervalCalibrator:
    """Calibrates nominal 90% prediction intervals using exact finite-sample empirical quantiles.

    Assumption & Distribution Shift Warning:
    Standard split conformal prediction assumes exchangeability of calibration and test points.
    In time series telemetry, rolling windows within the same incident exhibit strong serial
    autocorrelation, and severe operational surges constitute distribution shifts.
    Consequently, nominal 90% intervals achieve ~90% empirical coverage under stationary
    in-distribution operations, but CANNOT guarantee coverage during extreme out-of-distribution surges.
    """

    def __init__(self, target_coverage: float = 0.90):
        self.target_coverage = target_coverage
        self.quantile_value: Optional[float] = None
        self.calibration_sample_count: int = 0

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes exact finite-sample (1 - alpha) quantile with sample size sufficiency check."""
        residuals = np.abs(y_true - y_pred)
        n = len(residuals)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Check minimum sample size required for nominal coverage:
        # For target_coverage 1 - alpha (e.g. 0.90), minimum sample size is ceil(1 / alpha) - 1 (e.g. 9).
        min_required = math.ceil(1.0 / (1.0 - self.target_coverage)) - 1
        if n < min_required:
            raise ValueError(
                f"Insufficient calibration samples ({n}) to calibrate nominal {self.target_coverage:.0%} interval. "
                f"Minimum mathematically required is {min_required} samples."
            )

        target_idx = math.ceil((n + 1) * self.target_coverage) - 1
        if target_idx >= n:
            raise ValueError(
                f"Calibration sample size ({n}) is insufficient for exact finite-sample quantile index {target_idx}."
            )

        self.quantile_value = float(np.sort(residuals)[target_idx])
        self.calibration_sample_count = n
        return self.quantile_value

    def get_intervals(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns symmetric conformal bounds around predictions clipped to non-negative."""
        if self.quantile_value is None:
            raise RuntimeError("Calibrator has not been fitted on calibration data yet")
        lower = np.maximum(0.0, y_pred - self.quantile_value)
        upper = y_pred + self.quantile_value
        return lower, upper


def compute_capacity_metrics(
    service_id: str,
    forecast_rps: float,
) -> Dict[str, Any]:
    """Computes explicit signed capacity margin, available headroom, and overload.

    Formulas:
      capacity_margin = capacity - forecast
      available_headroom = max(0, capacity_margin)
      overload = max(0, -capacity_margin)

    Returns:
      Dictionary with capacity metrics. For uninstrumented legacy services where
      capacity is unknown, all numerical metrics are explicitly None.
    """
    cap = CONFIGURED_SERVICE_CAPACITIES.get(service_id)
    if cap is None:
        return {
            "configured_capacity_rps": None,
            "capacity_type": "unknown_uninstrumented",
            "capacity_margin_rps": None,
            "available_headroom_rps": None,
            "overload_rps": None,
            "forecast_to_capacity_ratio": None,
            "is_capacity_exceeded": None,
        }

    margin = round(cap - forecast_rps, 3)
    headroom = round(max(0.0, margin), 3)
    overload = round(max(0.0, -margin), 3)
    ratio = round(forecast_rps / cap, 4)
    exceeded = bool(forecast_rps > cap)

    return {
        "configured_capacity_rps": cap,
        "capacity_type": "configured_simulation_assumption",
        "capacity_margin_rps": margin,
        "available_headroom_rps": headroom,
        "overload_rps": overload,
        "forecast_to_capacity_ratio": ratio,
        "is_capacity_exceeded": exceeded,
    }

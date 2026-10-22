"""Baseline and trained machine learning forecasting models for airline passenger demand.

Implements:
1. Standard time-series metrics: MAE, RMSE, sMAPE (symmetric percentage error), MAPE.
2. Naive baselines: LastValueBaseline and SeasonalNaiveBaseline (s=7 for weekly cycles).
3. Trained lightweight model: LightweightAutoRegressiveForecaster using lagged autoregression,
   rolling statistical momentum, and calendar trigonometric features.
4. Conformal / residual prediction intervals (90% uncertainty bounds).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: (1/N) * sum(|y - y_hat|)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: sqrt((1/N) * sum((y - y_hat)^2))."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (sMAPE) in [0.0, 200.0]%.
    
    Formula: (100% / N) * sum( 2 * |y - y_hat| / (|y| + |y_hat| + eps) )
    Avoids asymmetric penalties and division-by-zero on low values.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    return float(np.mean(np.abs(y_pred - y_true) / denominator) * 100.0)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE) in [0.0, inf)%.
    
    Assumption: y_true > 0 strictly for all observations. Validated for passenger counts.
    """
    if np.any(y_true <= 0):
        raise ValueError("MAPE assumption violated: y_true contains zero or negative values.")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


@dataclass
class ForecastResult:
    """Structured forecast outputs with prediction horizons and uncertainty intervals."""
    model_name: str
    target_column: str
    horizon_steps: int
    timestamps: List[str]
    forecast_values: List[float]
    lower_bounds_90: List[float]
    upper_bounds_90: List[float]
    metrics: Dict[str, float] = field(default_factory=dict)
    provenance_note: str = "Forecast produced strictly using historical past data; uncertainty bounds derived from validation residuals."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_column": self.target_column,
            "horizon_steps": self.horizon_steps,
            "timestamps": self.timestamps,
            "forecast_values": [round(v, 2) for v in self.forecast_values],
            "lower_bounds_90": [round(v, 2) for v in self.lower_bounds_90],
            "upper_bounds_90": [round(v, 2) for v in self.upper_bounds_90],
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "provenance_note": self.provenance_note,
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": self.timestamps,
            "forecast": self.forecast_values,
            "lower_90": self.lower_bounds_90,
            "upper_90": self.upper_bounds_90,
        })


class LastValueBaseline:
    """Naive persistence baseline: y_hat(t+h) = y(t).
    
    Forecasts the most recently observed value forward across all horizons.
    """
    def __init__(self):
        self.last_value: Optional[float] = None
        self.target_column: str = "enplaned_passengers"

    def fit(self, train_df: pd.DataFrame, target_column: str = "enplaned_passengers") -> LastValueBaseline:
        self.target_column = target_column
        self.last_value = float(train_df[target_column].iloc[-1])
        return self

    def predict(
        self,
        horizon_steps: int,
        history: pd.DataFrame,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        date_column: str = "date",
    ) -> ForecastResult:
        if self.last_value is None and history.empty:
            raise RuntimeError("Model must be fit or non-empty history provided before prediction.")

        last_val = float(history[self.target_column].iloc[-1]) if not history.empty else self.last_value
        
        last_dt = pd.to_datetime(history[date_column].iloc[-1]) if not history.empty else pd.to_datetime(start_date or "2025-01-01")
        future_dates = [
            (last_dt + datetime.timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(horizon_steps)
        ]
        preds = [last_val] * horizon_steps
        
        return ForecastResult(
            model_name="LastValueBaseline",
            target_column=self.target_column,
            horizon_steps=horizon_steps,
            timestamps=future_dates,
            forecast_values=preds,
            lower_bounds_90=preds,
            upper_bounds_90=preds,
        )


class SeasonalNaiveBaseline:
    """Seasonal naive baseline: y_hat(t+h) = y(t + h - s).
    
    For daily airline demand with weekly seasonality, seasonal cycle s = 7 days.
    """
    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.target_column: str = "enplaned_passengers"
        self.recent_season: List[float] = []

    def fit(self, train_df: pd.DataFrame, target_column: str = "enplaned_passengers") -> SeasonalNaiveBaseline:
        self.target_column = target_column
        values = train_df[target_column].values
        self.recent_season = [float(v) for v in values[-self.season_length:]]
        return self

    def predict(
        self,
        horizon_steps: int,
        history: pd.DataFrame,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        date_column: str = "date",
    ) -> ForecastResult:
        if not history.empty:
            hist_vals = list(history[self.target_column].values)
            season = hist_vals[-self.season_length:] if len(hist_vals) >= self.season_length else self.recent_season
            last_dt = pd.to_datetime(history[date_column].iloc[-1])
        else:
            season = self.recent_season
            last_dt = pd.to_datetime(start_date or "2025-01-01")

        future_dates = [
            (last_dt + datetime.timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(horizon_steps)
        ]
        
        preds = []
        for h in range(horizon_steps):
            idx = (len(season) - self.season_length + (h % self.season_length)) % len(season)
            preds.append(float(season[idx]))

        return ForecastResult(
            model_name=f"SeasonalNaiveBaseline(s={self.season_length})",
            target_column=self.target_column,
            horizon_steps=horizon_steps,
            timestamps=future_dates,
            forecast_values=preds,
            lower_bounds_90=preds,
            upper_bounds_90=preds,
        )


class LightweightAutoRegressiveForecaster:
    """Lightweight autoregressive Ridge forecasting model for airline demand.
    
    Features constructed strictly without future leakage:
    - AutoRegressive Lags: y(t-1), y(t-2), y(t-7), y(t-14)
    - Past Rolling Statistics: 7-day rolling mean, 14-day rolling mean
    - Calendar Fourier Features: Day-of-week sine/cosine, Month sine/cosine
    - L2 Ridge regularization with validation-calibrated conformal uncertainty bounds.
    """

    def __init__(self, alpha: float = 1.0, max_lags: int = 14):
        self.alpha = alpha
        self.max_lags = max_lags
        self.target_column: str = "enplaned_passengers"
        self.date_column: str = "date"
        self.model: Optional[Ridge] = None
        self.scaler: StandardScaler = StandardScaler()
        self.val_residual_std: float = 0.0
        self.feature_names: List[str] = []

    def _extract_features_for_row(
        self,
        past_values: List[float],
        dt: pd.Timestamp,
    ) -> List[float]:
        """Construct feature vector for a single time step using only past observations."""
        feats = []
        # 1. Lags: t-1, t-2, t-3, t-7, t-14
        for lag in (1, 2, 3, 7, 14):
            val = past_values[-lag] if len(past_values) >= lag else past_values[-1]
            feats.append(val)

        # 2. Rolling statistics (strictly past)
        roll7 = float(np.mean(past_values[-7:])) if len(past_values) >= 7 else past_values[-1]
        roll14 = float(np.mean(past_values[-14:])) if len(past_values) >= 14 else past_values[-1]
        feats.append(roll7)
        feats.append(roll14)

        # 3. Calendar trigonometric representations
        dow = dt.weekday() # 0 to 6
        feats.append(math.sin(2.0 * math.pi * dow / 7.0))
        feats.append(math.cos(2.0 * math.pi * dow / 7.0))

        month = dt.month # 1 to 12
        feats.append(math.sin(2.0 * math.pi * (month - 1) / 12.0))
        feats.append(math.cos(2.0 * math.pi * (month - 1) / 12.0))

        return feats

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        target_column: str = "enplaned_passengers",
        date_column: str = "date",
    ) -> LightweightAutoRegressiveForecaster:
        self.target_column = target_column
        self.date_column = date_column

        # Sort strictly chronologically
        df = train_df.sort_values(by=date_column).reset_index(drop=True)
        dates = pd.to_datetime(df[date_column])
        values = [float(v) for v in df[target_column].values]

        X_train: List[List[float]] = []
        y_train: List[float] = []

        for i in range(self.max_lags, len(values)):
            past = values[:i]
            dt = dates.iloc[i]
            feats = self._extract_features_for_row(past, dt)
            X_train.append(feats)
            y_train.append(values[i])

        X_train_arr = np.array(X_train)
        y_train_arr = np.array(y_train)

        # Scale features strictly based on training distribution
        X_train_scaled = self.scaler.fit_transform(X_train_arr)

        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model.fit(X_train_scaled, y_train_arr)

        self.feature_names = [
            "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
            "roll_mean_7", "roll_mean_14",
            "dow_sin", "dow_cos", "month_sin", "month_cos",
        ]

        # Calibrate residual uncertainty variance on validation split if provided
        if val_df is not None and len(val_df) > 0:
            val_eval = self.evaluate(val_df, history_df=train_df)
            self.val_residual_std = float(val_eval["rmse"])
        else:
            # Fallback to training in-sample residual std
            train_preds = self.model.predict(X_train_scaled)
            self.val_residual_std = float(np.std(y_train_arr - train_preds))

        return self

    def predict(
        self,
        horizon_steps: int,
        history: pd.DataFrame,
        date_column: str = "date",
    ) -> ForecastResult:
        if self.model is None:
            raise RuntimeError("Model must be fit prior to predict.")

        hist_df = history.sort_values(by=date_column).reset_index(drop=True)
        dates = pd.to_datetime(hist_df[date_column])
        past_values = [float(v) for v in hist_df[self.target_column].values]
        last_dt = dates.iloc[-1]

        future_dates: List[str] = []
        forecast_values: List[float] = []
        lower_bounds: List[float] = []
        upper_bounds: List[float] = []

        # 90% confidence z-multiplier (1.645)
        z_90 = 1.645

        # Recursive multi-step generation
        for h in range(1, horizon_steps + 1):
            next_dt = last_dt + datetime.timedelta(days=h)
            feats = self._extract_features_for_row(past_values, next_dt)
            feats_scaled = self.scaler.transform(np.array([feats]))
            pred_val = float(self.model.predict(feats_scaled)[0])
            pred_val = max(0.0, pred_val)

            # Horizon-scaled uncertainty expansion
            uncertainty = z_90 * self.val_residual_std * math.sqrt(1.0 + 0.04 * (h - 1))
            lower_b = max(0.0, pred_val - uncertainty)
            upper_b = pred_val + uncertainty

            future_dates.append(next_dt.strftime("%Y-%m-%d"))
            forecast_values.append(pred_val)
            lower_bounds.append(lower_b)
            upper_bounds.append(upper_b)

            # Append prediction to history for next recursive step
            past_values.append(pred_val)

        return ForecastResult(
            model_name="LightweightAutoRegressiveForecaster(Ridge)",
            target_column=self.target_column,
            horizon_steps=horizon_steps,
            timestamps=future_dates,
            forecast_values=forecast_values,
            lower_bounds_90=lower_bounds,
            upper_bounds_90=upper_bounds,
        )

    def evaluate(
        self,
        test_df: pd.DataFrame,
        history_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Evaluate multi-step forecasts over a held-out test split."""
        horizon = len(test_df)
        forecast = self.predict(horizon_steps=horizon, history=history_df)
        
        y_true = np.array([float(v) for v in test_df[self.target_column].values])
        y_pred = np.array(forecast.forecast_values)

        mae = compute_mae(y_true, y_pred)
        rmse = compute_rmse(y_true, y_pred)
        smape = compute_smape(y_true, y_pred)
        mape = compute_mape(y_true, y_pred) if np.all(y_true > 0) else float("nan")

        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "smape": round(smape, 2),
            "mape": round(mape, 2),
        }

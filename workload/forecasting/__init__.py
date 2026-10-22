"""Airline Demand Forecasting Package for BACCP.

Provides:
- Ingestion and schema validation of BTS historical airline demand data.
- Chronological, non-shuffled time-series splits preventing future data leakage.
- Naive baselines (LastValueBaseline, SeasonalNaiveBaseline).
- Trained LightweightAutoRegressiveForecaster (Ridge) with uncertainty bounds.
- Standard evaluation metrics: MAE, RMSE, sMAPE, MAPE.
- Calibrated DemandToWorkloadMapper integrating forecasts into simulated service workloads.
- End-to-end AirlineDemandForecastingPipeline.
"""

from __future__ import annotations

from .dataset import (
    DatasetValidationError,
    create_chronological_splits,
    load_airline_demand_dataset,
)
from .models import (
    ForecastResult,
    LastValueBaseline,
    LightweightAutoRegressiveForecaster,
    SeasonalNaiveBaseline,
    compute_mae,
    compute_mape,
    compute_rmse,
    compute_smape,
)
from .pipeline import AirlineDemandForecastingPipeline
from .workload_mapper import DemandConversionConfig, DemandToWorkloadMapper

__all__ = [
    "load_airline_demand_dataset",
    "create_chronological_splits",
    "DatasetValidationError",
    "LastValueBaseline",
    "SeasonalNaiveBaseline",
    "LightweightAutoRegressiveForecaster",
    "ForecastResult",
    "compute_mae",
    "compute_rmse",
    "compute_smape",
    "compute_mape",
    "DemandConversionConfig",
    "DemandToWorkloadMapper",
    "AirlineDemandForecastingPipeline",
]

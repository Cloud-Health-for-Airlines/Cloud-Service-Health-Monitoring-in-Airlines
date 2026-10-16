"""BACCP Workload Forecasting Module.

Provides reproducible per-service workload forecasting across multiple horizons
(H=1: 20s, H=5: 100s, H=15: 300s) with conformal prediction intervals and capacity headroom.
"""

from __future__ import annotations

from .data_contract import (
    CONFIGURED_SERVICE_CAPACITIES,
    CONTRACT_VERSION,
    SERVICE_GENERATION_MAPPING,
    GenerationType,
    IncidentTrajectory,
    ServiceIdentifier,
    WorkloadRecord,
    validate_workload_record,
)
from .features import (
    CONTEXT_WINDOW_STEPS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    SEASONAL_PERIOD_STEPS,
    SUPPORTED_HORIZONS,
    build_forecasting_dataset_for_service,
    extract_features_at_origin,
)
from .inference import WorkloadForecaster, get_forecaster
from .models import (
    AutoregressiveForecaster,
    ConformalIntervalCalibrator,
    GradientBoostingForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    compute_capacity_metrics,
)

__all__ = [
    "WorkloadForecaster",
    "get_forecaster",
    "WorkloadRecord",
    "IncidentTrajectory",
    "ServiceIdentifier",
    "GenerationType",
    "CONFIGURED_SERVICE_CAPACITIES",
    "CONTRACT_VERSION",
    "SERVICE_GENERATION_MAPPING",
    "validate_workload_record",
    "CONTEXT_WINDOW_STEPS",
    "SEASONAL_PERIOD_STEPS",
    "SUPPORTED_HORIZONS",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "extract_features_at_origin",
    "build_forecasting_dataset_for_service",
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
    "AutoregressiveForecaster",
    "GradientBoostingForecaster",
    "ConformalIntervalCalibrator",
    "compute_capacity_metrics",
]

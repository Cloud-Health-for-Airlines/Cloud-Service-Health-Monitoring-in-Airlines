"""BACCP Workload Management, Public Datasets, and Demand Forecasting Package."""

from __future__ import annotations

from .datasets.google_cluster import (
    GoogleClusterTraceAdapter,
    GoogleClusterTaskEvent,
    GoogleClusterTaskUsage,
    EVENT_TYPES,
    map_to_airline_tier,
)
from .generator import (
    RealisticAirlineWorkloadGenerator,
    WorkloadScenarioConfig,
    WorkloadObservation,
    WorkloadSnapshot,
    DataProvenance,
    CANONICAL_5_SERVICES,
    EXPANDED_9_SERVICES,
)
from .forecasting import (
    AirlineDemandForecastingPipeline,
    DemandConversionConfig,
    DemandToWorkloadMapper,
    ForecastResult,
    LastValueBaseline,
    LightweightAutoRegressiveForecaster,
    SeasonalNaiveBaseline,
    compute_mae,
    compute_mape,
    compute_rmse,
    compute_smape,
    create_chronological_splits,
    load_airline_demand_dataset,
)

__version__ = "1.0.0"

__all__ = [
    "GoogleClusterTraceAdapter",
    "GoogleClusterTaskEvent",
    "GoogleClusterTaskUsage",
    "EVENT_TYPES",
    "map_to_airline_tier",
    "RealisticAirlineWorkloadGenerator",
    "WorkloadScenarioConfig",
    "WorkloadObservation",
    "WorkloadSnapshot",
    "DataProvenance",
    "CANONICAL_5_SERVICES",
    "EXPANDED_9_SERVICES",
    "AirlineDemandForecastingPipeline",
    "DemandConversionConfig",
    "DemandToWorkloadMapper",
    "ForecastResult",
    "LastValueBaseline",
    "LightweightAutoRegressiveForecaster",
    "SeasonalNaiveBaseline",
    "compute_mae",
    "compute_mape",
    "compute_rmse",
    "compute_smape",
    "create_chronological_splits",
    "load_airline_demand_dataset",
]


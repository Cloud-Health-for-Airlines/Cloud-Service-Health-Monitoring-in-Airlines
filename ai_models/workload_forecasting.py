"""Standard Python package bridge for ai-models/workload_forecasting."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
AI_MODELS_DIR = REPO_ROOT / "ai-models"
if str(AI_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(AI_MODELS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_mod = import_module("workload_forecasting")

WorkloadForecaster = getattr(_mod, "WorkloadForecaster")
get_forecaster = getattr(_mod, "get_forecaster")
WorkloadRecord = getattr(_mod, "WorkloadRecord")
IncidentTrajectory = getattr(_mod, "IncidentTrajectory")
ServiceIdentifier = getattr(_mod, "ServiceIdentifier")
GenerationType = getattr(_mod, "GenerationType")
CONFIGURED_SERVICE_CAPACITIES = getattr(_mod, "CONFIGURED_SERVICE_CAPACITIES")
SERVICE_GENERATION_MAPPING = getattr(_mod, "SERVICE_GENERATION_MAPPING")
PersistenceForecaster = getattr(_mod, "PersistenceForecaster")
SeasonalNaiveForecaster = getattr(_mod, "SeasonalNaiveForecaster")
AutoregressiveForecaster = getattr(_mod, "AutoregressiveForecaster")
GradientBoostingForecaster = getattr(_mod, "GradientBoostingForecaster")
ConformalIntervalCalibrator = getattr(_mod, "ConformalIntervalCalibrator")
compute_capacity_metrics = getattr(_mod, "compute_capacity_metrics")
extract_features_at_origin = getattr(_mod, "extract_features_at_origin")
build_forecasting_dataset_for_service = getattr(_mod, "build_forecasting_dataset_for_service")
FEATURE_SCHEMA_VERSION = getattr(_mod, "FEATURE_SCHEMA_VERSION")
CONTRACT_VERSION = getattr(_mod, "CONTRACT_VERSION")

__all__ = [
    "WorkloadForecaster",
    "get_forecaster",
    "WorkloadRecord",
    "IncidentTrajectory",
    "ServiceIdentifier",
    "GenerationType",
    "CONFIGURED_SERVICE_CAPACITIES",
    "CONTRACT_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "SERVICE_GENERATION_MAPPING",
    "extract_features_at_origin",
    "build_forecasting_dataset_for_service",
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
    "AutoregressiveForecaster",
    "GradientBoostingForecaster",
    "ConformalIntervalCalibrator",
    "compute_capacity_metrics",
]

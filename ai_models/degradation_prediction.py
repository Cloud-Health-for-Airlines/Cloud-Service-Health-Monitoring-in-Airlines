"""Standard Python package bridge for ai-models/degradation_prediction."""

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

_mod = import_module("degradation_prediction")

DegradationPredictor = getattr(_mod, "DegradationPredictor")
DegradationExplainer = getattr(_mod, "DegradationExplainer")
CONTRACT_VERSION = getattr(_mod, "CONTRACT_VERSION")
FEATURE_SCHEMA_VERSION = getattr(_mod, "FEATURE_SCHEMA_VERSION")
SAMPLING_INTERVAL_SECONDS = getattr(_mod, "SAMPLING_INTERVAL_SECONDS")
SUPPORTED_HORIZONS = getattr(_mod, "SUPPORTED_HORIZONS")
SLO_THRESHOLDS = getattr(_mod, "SLO_THRESHOLDS")
NOMINAL_BASELINES = getattr(_mod, "NOMINAL_BASELINES")
CONFIGURED_SERVICE_CAPACITIES = getattr(_mod, "CONFIGURED_SERVICE_CAPACITIES")
BOUNDARY_SYNC_DRIFT_THRESHOLD = getattr(_mod, "BOUNDARY_SYNC_DRIFT_THRESHOLD")
BOUNDARY_SYNC_DRIFT_CRITICAL = getattr(_mod, "BOUNDARY_SYNC_DRIFT_CRITICAL")
DegradationRegime = getattr(_mod, "DegradationRegime")
DegradationIncident = getattr(_mod, "DegradationIncident")
ServiceIdentifier = getattr(_mod, "ServiceIdentifier")
is_step_degraded = getattr(_mod, "is_step_degraded")
extract_features_at_origin = getattr(_mod, "extract_features_at_origin")
TELEMETRY_ONLY_FEATURE_NAMES = getattr(_mod, "TELEMETRY_ONLY_FEATURE_NAMES")
TELEMETRY_PLUS_FORECAST_FEATURE_NAMES = getattr(_mod, "TELEMETRY_PLUS_FORECAST_FEATURE_NAMES")
PersistenceDegradationBaseline = getattr(_mod, "PersistenceDegradationBaseline")
CausalQueueRolloutBaseline = getattr(_mod, "CausalQueueRolloutBaseline")
LearnedDegradationHorizonModel = getattr(_mod, "LearnedDegradationHorizonModel")

__all__ = [
    "DegradationPredictor",
    "DegradationExplainer",
    "CONTRACT_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "SAMPLING_INTERVAL_SECONDS",
    "SUPPORTED_HORIZONS",
    "SLO_THRESHOLDS",
    "NOMINAL_BASELINES",
    "CONFIGURED_SERVICE_CAPACITIES",
    "BOUNDARY_SYNC_DRIFT_THRESHOLD",
    "BOUNDARY_SYNC_DRIFT_CRITICAL",
    "DegradationRegime",
    "DegradationIncident",
    "ServiceIdentifier",
    "is_step_degraded",
    "extract_features_at_origin",
    "TELEMETRY_ONLY_FEATURE_NAMES",
    "TELEMETRY_PLUS_FORECAST_FEATURE_NAMES",
    "PersistenceDegradationBaseline",
    "CausalQueueRolloutBaseline",
    "LearnedDegradationHorizonModel",
]

"""BACCP Degradation Prediction Module.

Workload-Aware Service and Boundary Degradation Prediction (Task 2).
"""

from .data_contract import (
    BOUNDARY_SYNC_DRIFT_CRITICAL,
    BOUNDARY_SYNC_DRIFT_THRESHOLD,
    CONFIGURED_SERVICE_CAPACITIES,
    CONTRACT_VERSION,
    DegradationIncident,
    DegradationRegime,
    NOMINAL_BASELINES,
    SAMPLING_INTERVAL_SECONDS,
    SLO_THRESHOLDS,
    SUPPORTED_HORIZONS,
    ServiceIdentifier,
    is_step_degraded,
)
from .features import (
    FEATURE_SCHEMA_VERSION,
    TELEMETRY_ONLY_FEATURE_NAMES,
    TELEMETRY_PLUS_FORECAST_FEATURE_NAMES,
    extract_features_at_origin,
)
from .explanation import DegradationExplainer
from .inference import DegradationPredictor
from .models import (
    CausalQueueRolloutBaseline,
    LearnedDegradationHorizonModel,
    PersistenceDegradationBaseline,
)

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

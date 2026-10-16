"""BACCP Backend ML Integration Package."""

from .alert_policy import (
    ALERT_THRESHOLDS,
    DEFAULT_ALERT_THRESHOLD,
    evaluate_alert_decision,
    get_alert_threshold,
)
from .inference_service import MLInferenceService, sanitize_floats
from .metrics import BACCPMLMetrics
from .telemetry_adapter import (
    HORIZON_SECONDS_MAP,
    MAX_TELEMETRY_STALENESS_SECONDS,
    SAMPLING_INTERVAL_SECONDS,
    SUPPORTED_HORIZONS,
    SUPPORTED_SERVICES,
    TelemetryAdapter,
    TelemetryError,
    TelemetrySnapshot,
)

__all__ = [
    "MLInferenceService",
    "sanitize_floats",
    "BACCPMLMetrics",
    "ALERT_THRESHOLDS",
    "DEFAULT_ALERT_THRESHOLD",
    "get_alert_threshold",
    "evaluate_alert_decision",
    "TelemetryAdapter",
    "TelemetrySnapshot",
    "TelemetryError",
    "SUPPORTED_SERVICES",
    "SUPPORTED_HORIZONS",
    "SAMPLING_INTERVAL_SECONDS",
    "MAX_TELEMETRY_STALENESS_SECONDS",
    "HORIZON_SECONDS_MAP",
]

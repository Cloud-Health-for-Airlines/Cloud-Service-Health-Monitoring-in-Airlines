"""Prometheus Metrics Collector for BACCP Workload Forecasting & Degradation Prediction.

Exposes bounded, standard Prometheus metrics:
- baccp_predicted_degradation_probability: Calibrated risk gauge by service and horizon.
- baccp_advisory_alert_episodes_total: Deduplicated advisory alert episodes counter.
- baccp_workload_forecast_absolute_error: Workload absolute error gauge (matched on future observation arrival).
- baccp_telemetry_freshness_seconds: Input telemetry freshness gauge by service.
- baccp_telemetry_available: Binary telemetry availability indicator by service.
"""

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .alert_policy import ALERT_THRESHOLDS, get_alert_threshold

logger = logging.getLogger("baccp.ml.metrics")

# Bounded label domains
SUPPORTED_SERVICES = ["reservations", "crew", "baggage", "boundary-gateway", "legacy-core"]
SUPPORTED_HORIZONS = [1, 5, 15]
SAMPLING_INTERVAL_SECONDS = 20.0

# Max pending forecasts retained in memory before expiration
MAX_PENDING_FORECASTS = 500
PENDING_FORECAST_TTL_SECONDS = 600.0  # 10 minutes


class BACCPMLMetrics:
    """Thread-safe in-memory Prometheus metrics registry for BACCP ML models."""

    _instance: Optional["BACCPMLMetrics"] = None

    def __init__(self):
        self._lock = threading.Lock()

        # 1. Degradation event probability gauge: (service, horizon) -> float
        self.degradation_probabilities: Dict[Tuple[str, int], float] = {}

        # 2. Advisory alert episodes counter: (service, horizon) -> int
        self.alert_episode_counts: Dict[Tuple[str, int], int] = {
            (s, h): 0 for s in SUPPORTED_SERVICES for h in SUPPORTED_HORIZONS
        }

        # Episode state tracking for strict deduplication
        # (service, horizon) -> {"in_alert": bool, "last_snapshot_id": str, "last_updated": float}
        self.alert_episode_state: Dict[Tuple[str, int], Dict[str, Any]] = {
            (s, h): {"in_alert": False, "last_snapshot_id": "", "last_updated": 0.0}
            for s in SUPPORTED_SERVICES
            for h in SUPPORTED_HORIZONS
        }

        # 3. Workload forecast absolute error gauge: (service, horizon) -> float (None if unavailable)
        self.forecast_absolute_errors: Dict[Tuple[str, int], Optional[float]] = {}

        # Pending forecast records waiting for target observation:
        # Key: (incident_id, service, horizon, origin_timestamp_sec, target_timestamp_sec)
        # Value: {"predicted_demand": float, "origin_time_sec": float, "target_time_sec": float, ...}
        self.pending_forecasts: Dict[Tuple[str, str, int, float, float], Dict[str, Any]] = {}

        # 4. Telemetry freshness and availability gauges: service -> float
        # Availability begins at 0 (unavailable) until verified via snapshot checking.
        self.telemetry_freshness: Dict[str, float] = {s: 0.0 for s in SUPPORTED_SERVICES}
        self.telemetry_available: Dict[str, int] = {s: 0 for s in SUPPORTED_SERVICES}

    @classmethod
    def get_instance(cls) -> "BACCPMLMetrics":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset_for_testing(self) -> None:
        """Resets all metrics state for test isolation."""
        with self._lock:
            self.degradation_probabilities.clear()
            self.alert_episode_counts = {
                (s, h): 0 for s in SUPPORTED_SERVICES for h in SUPPORTED_HORIZONS
            }
            self.alert_episode_state = {
                (s, h): {"in_alert": False, "last_snapshot_id": "", "last_updated": 0.0}
                for s in SUPPORTED_SERVICES
                for h in SUPPORTED_HORIZONS
            }
            self.forecast_absolute_errors.clear()
            self.pending_forecasts.clear()
            self.telemetry_freshness = {s: 0.0 for s in SUPPORTED_SERVICES}
            self.telemetry_available = {s: 0 for s in SUPPORTED_SERVICES}

    def record_degradation_prediction(
        self,
        service: str,
        horizon: int,
        probability: float,
        alert_issued: bool,
        snapshot_id: str,
        freshness_seconds: float = 0.0,
    ) -> None:
        """Records degradation probability and updates deduplicated alert episode count."""
        if service not in SUPPORTED_SERVICES or horizon not in SUPPORTED_HORIZONS:
            return

        with self._lock:
            key = (service, horizon)
            thresh = get_alert_threshold(horizon)
            if not alert_issued and round(float(probability), 4) >= thresh:
                self.degradation_probabilities[key] = float(probability)
            else:
                self.degradation_probabilities[key] = round(float(probability), 4)
            self.telemetry_freshness[service] = round(float(freshness_seconds), 2)
            self.telemetry_available[service] = 1

            # Episode deduplication logic:
            # Repeated GETs or cache hits for the SAME snapshot must not increment counter.
            # Only increment when state transitions from nominal to alert, or when a new episode begins.
            state = self.alert_episode_state[key]
            last_snap = state.get("last_snapshot_id", "")
            was_alert = state.get("in_alert", False)

            if snapshot_id != last_snap:
                # New telemetry snapshot
                if alert_issued and not was_alert:
                    # Transition into new alert episode
                    self.alert_episode_counts[key] += 1
                    state["in_alert"] = True
                elif not alert_issued:
                    # Returned to nominal state
                    state["in_alert"] = False

                state["last_snapshot_id"] = snapshot_id
                state["last_updated"] = time.time()
            else:
                # Identical snapshot (repeated request / cache hit) -> DO NOT increment
                pass

    def set_telemetry_unavailable(self, service: str, freshness_seconds: float = 0.0) -> None:
        """Marks telemetry as unavailable for a service on stale or failed telemetry."""
        with self._lock:
            if service in self.telemetry_available:
                self.telemetry_available[service] = 0
            if service in self.telemetry_freshness:
                self.telemetry_freshness[service] = round(float(freshness_seconds), 2)

    def record_workload_forecast(
        self,
        incident_id: str,
        service: str,
        horizon: int,
        predicted_demand: float,
        origin_timestamp_sec: float,
        target_timestamp_sec: float,
    ) -> None:
        """Registers a pending workload forecast waiting for a future observation."""
        if service not in SUPPORTED_SERVICES or horizon not in SUPPORTED_HORIZONS:
            return

        with self._lock:
            self._expire_stale_pending_forecasts_locked()

            f_key = (
                str(incident_id or "default"),
                service,
                horizon,
                round(float(origin_timestamp_sec), 2),
                round(float(target_timestamp_sec), 2),
            )
            if f_key not in self.pending_forecasts:
                if len(self.pending_forecasts) >= MAX_PENDING_FORECASTS:
                    # Evict oldest entry
                    oldest_k = next(iter(self.pending_forecasts))
                    self.pending_forecasts.pop(oldest_k, None)

                self.pending_forecasts[f_key] = {
                    "incident_id": str(incident_id or "default"),
                    "service": service,
                    "horizon": horizon,
                    "predicted_demand": float(predicted_demand),
                    "origin_time_sec": float(origin_timestamp_sec),
                    "target_time_sec": float(target_timestamp_sec),
                    "created_at": time.time(),
                }

    def resolve_pending_forecasts(
        self,
        incident_id: str,
        current_clock_sec: float,
        service_telemetry: Dict[str, List[Dict[str, Any]]],
    ) -> List[Tuple[str, int, float]]:
        """Resolves pending forecasts whose target timestamp has arrived (target <= current_clock_sec)
        using observations available in the current snapshot for matching incident_id.
        """
        resolved: List[Tuple[str, int, float]] = []
        with self._lock:
            self._expire_stale_pending_forecasts_locked()

            inc_str = str(incident_id or "default")
            keys_to_resolve = [
                k for k, rec in self.pending_forecasts.items()
                if rec["incident_id"] == inc_str and rec["target_time_sec"] <= current_clock_sec + 1e-4
            ]

            for k in keys_to_resolve:
                rec = self.pending_forecasts.pop(k)
                s = rec["service"]
                h = rec["horizon"]
                target_ts = rec["target_time_sec"]
                pred = rec["predicted_demand"]

                # Find matching observation in current stream
                stream = service_telemetry.get(s, [])
                actual_rate: Optional[float] = None
                for r in stream:
                    rec_ts = r.get("timestamp_sec", r.get("step_index", 0) * SAMPLING_INTERVAL_SECONDS)
                    if abs(rec_ts - target_ts) < 1.0:
                        actual_rate = float(r.get("offered_rate", 0.0))
                        break

                if actual_rate is not None:
                    abs_err = round(abs(actual_rate - pred), 3)
                    self.forecast_absolute_errors[(s, h)] = abs_err
                    resolved.append((s, h, abs_err))

        return resolved

    def _expire_stale_pending_forecasts_locked(self) -> None:
        """Removes expired pending forecast records (must be called with _lock held)."""
        now = time.time()
        to_del = [
            k for k, v in self.pending_forecasts.items()
            if now - v.get("created_at", now) > PENDING_FORECAST_TTL_SECONDS
        ]
        for k in to_del:
            self.pending_forecasts.pop(k, None)

    def format_prometheus_lines(self) -> List[str]:
        """Formats ML metrics as standard Prometheus exposition text."""
        lines: List[str] = []

        with self._lock:
            # 1. Predicted degradation probability gauge
            lines.append("# HELP baccp_predicted_degradation_probability Calibrated failure degradation probability by service and horizon")
            lines.append("# TYPE baccp_predicted_degradation_probability gauge")
            for (s, h), prob in sorted(self.degradation_probabilities.items()):
                thresh = get_alert_threshold(h)
                state = self.alert_episode_state.get((s, h), {})
                if not state.get("in_alert", False) and round(prob, 4) >= thresh:
                    lines.append(f'baccp_predicted_degradation_probability{{service="{s}",horizon="{h}"}} {prob:.5f}')
                else:
                    lines.append(f'baccp_predicted_degradation_probability{{service="{s}",horizon="{h}"}} {prob:.4f}')

            # 2. Advisory alert episodes counter
            lines.append("# HELP baccp_advisory_alert_episodes_total Total deduplicated advisory alert episodes by service and horizon")
            lines.append("# TYPE baccp_advisory_alert_episodes_total counter")
            for (s, h), count in sorted(self.alert_episode_counts.items()):
                lines.append(f'baccp_advisory_alert_episodes_total{{service="{s}",horizon="{h}"}} {count}')

            # 3. Workload forecast absolute error gauge (only report services/horizons with matched future observations)
            lines.append("# HELP baccp_workload_forecast_absolute_error Mean absolute demand error of workload forecast after future observation arrives")
            lines.append("# TYPE baccp_workload_forecast_absolute_error gauge")
            for (s, h), err in sorted(self.forecast_absolute_errors.items()):
                if err is not None:
                    lines.append(f'baccp_workload_forecast_absolute_error{{service="{s}",horizon="{h}"}} {err:.3f}')

            # 4. Telemetry freshness gauge
            lines.append("# HELP baccp_telemetry_freshness_seconds Elapsed seconds since latest telemetry observation by service")
            lines.append("# TYPE baccp_telemetry_freshness_seconds gauge")
            for s, freshness in sorted(self.telemetry_freshness.items()):
                lines.append(f'baccp_telemetry_freshness_seconds{{service="{s}"}} {freshness:.2f}')

            # 5. Telemetry availability gauge
            lines.append("# HELP baccp_telemetry_available Telemetry stream availability status (1=available, 0=unavailable)")
            lines.append("# TYPE baccp_telemetry_available gauge")
            for s, avail in sorted(self.telemetry_available.items()):
                lines.append(f'baccp_telemetry_available{{service="{s}"}} {avail}')

        return lines

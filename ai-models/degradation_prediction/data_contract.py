"""BACCP Workload-Aware Service and Boundary Degradation Prediction Data Contract.

Version: baccp-degradation-contract-v1.1.0

Formalizes the data contract for predicting future service degradation and boundary
synchronization deterioration:
1. Target definitions and scope hierarchy:
   - Endpoint continuous targets at t+H:
     * latency_ms: Service round-trip/processing latency in milliseconds
     * backlog_requests: Unserviced pending queue depth (requests)
     * boundary_sync_drift_pct: Boundary synchronization drift score ε(t) in [8.5, 99.5]%
   - Forward window event targets over (t, t+H]:
     * service_event: Binary indicator {0, 1} whether service i breaches latency, backlog, or error SLO in (t, t+H].
     * boundary_event: Binary indicator {0, 1} whether boundary-gateway breaches SLO or drift ε(tau) >= 45.0% in (t, t+H].
     * system_event: Binary indicator {0, 1} whether ANY service or boundary SLO is breached in (t, t+H].
       Counted once per forecast origin.
2. Valid Early-Warning Constraint:
   - An alert issued at origin t for horizon H is a correct onset early warning IF AND ONLY IF:
     0 < onset_time - alert_time <= H * 20 seconds.
   - Alerts issued when the system is already degraded (onset_time <= alert_time) are classified
     as ongoing detections, not early warnings.
   - Alerts issued too far in advance (onset_time - alert_time > H * 20s) are out-of-horizon
     premature alerts, not true positives for (t, t+H].
3. Feature Provenance Classification:
   - MEASURED: Extracted from live telemetry streams (latency, queue, rates, drift, errors).
   - CONFIGURED: Architectural contracts and provisioned baselines (capacities, thresholds).
   - INFERRED: Derived from out-of-sample upstream WorkloadForecaster (forecast, intervals, rollout).
   - SIMULATOR_ONLY: Internal simulator state (hidden capacity, fault schedules) STRICTLY EXCLUDED.
4. Horizons:
   - H in {1, 5, 15} steps at 20-second sampling intervals (20s, 100s, 300s).
5. SLO Thresholds (Simulation assumptions documented when not measured):
   - Cloud latency SLO: 100.0 ms (nominal 15-25 ms)
   - Boundary gateway latency SLO: 200.0 ms (nominal 15-35 ms)
   - Legacy core latency SLO: 250.0 ms (nominal 15-70 ms)
   - Backlog SLO: 50.0 requests (nominal 0-10 requests)
   - Boundary sync drift SLO: 45.0% (nominal < 45.0%, critical >= 70.0%)
   - Error rate SLO: 5.0% (nominal < 0.5%)
6. Regimes:
   - nominal, gradual_ramp, abrupt_burst, capacity_reduction, boundary_slowdown, recovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


CONTRACT_VERSION = "baccp-degradation-contract-v1.5.0"
SAMPLING_INTERVAL_SECONDS = 20.0
SUPPORTED_HORIZONS = [1, 5, 15]  # 20s, 100s, 300s
MAX_TELEMETRY_STALENESS_SECONDS = 60.0  # Documented maximum permissible telemetry staleness (3 sampling intervals)


class WorkloadForecastState(str, Enum):
    VALID_TRAINED_PREDICTION = "VALID_TRAINED_PREDICTION"
    INTENTIONAL_PERSISTENCE_FALLBACK = "INTENTIONAL_PERSISTENCE_FALLBACK"
    MISSING_ARTIFACT = "MISSING_ARTIFACT"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    PREDICTION_EXCEPTION = "PREDICTION_EXCEPTION"


@dataclass
class AdaptedWorkloadForecast:
    """Validated, normalized workload forecast matching upstream WorkloadForecaster contract."""
    forecast_demand_rps: float
    interval_lower_rps: float
    interval_upper_rps: float
    interval_width_rps: float
    nominal_coverage: Optional[float]
    is_calibrated: bool
    calibration_method: str
    configured_capacity_rps: Optional[float]
    capacity_margin_rps: Optional[float]
    available_headroom_rps: Optional[float]
    overload_rps: Optional[float]
    has_known_capacity: bool
    is_overload: float
    fallback_triggered: bool
    fallback_reason: Optional[str]
    fallback_active: float
    reliability_status: str
    reliability_warnings: List[str]
    prediction_state: WorkloadForecastState
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Service Priority Weights for multi-domain aggregation
SERVICE_PRIORITY_WEIGHTS: Dict[str, float] = {
    "reservations": 1.5,
    "crew": 1.0,
    "baggage": 0.5,
}


class DataProvenance(str, Enum):
    LIVE_MEASURED = "LIVE_MEASURED"              # Verified runtime telemetry from APM / instrumentation
    SYNTHETIC_SIMULATED = "SYNTHETIC_SIMULATED"  # Discrete-event simulation telemetry
    CONFIGURED = "CONFIGURED"                    # Architectural contract constant
    INFERRED = "INFERRED"                        # Upstream ML / statistical model output
    DEGRADED_IMPUTED = "DEGRADED_IMPUTED"        # Missing/padded telemetry stream


class FeatureProvenance(str, Enum):
    MEASURED = "measured"              # Live telemetry observable in production
    CONFIGURED = "configured"          # Contract / architecture provisioned constant
    INFERRED = "inferred"              # Out-of-sample upstream ML forecast
    SIMULATOR_ONLY = "simulator_only"  # Ground-truth simulation state (strictly barred)


class DegradationConfirmationState(str, Enum):
    HEALTHY = "healthy"                          # Operating within SLO thresholds
    SINGLE_BREACH = "single_breach"              # Isolated 1-step breach snapshot
    CONFIRMED_SUSTAINED = "confirmed_sustained"  # Confirmed by >= 3 consecutive breached snapshots
    ONGOING = "ongoing"                          # Sustained degradation already active
    RECOVERY = "recovery"                        # Post-degradation recovery (>= 3 consecutive healthy snapshots)


class DegradationType(str, Enum):
    NOMINAL = "NOMINAL"
    LATENCY_BREACH = "LATENCY_BREACH"
    BACKLOG_SATURATION = "BACKLOG_SATURATION"
    ERROR_SPIKE = "ERROR_SPIKE"
    BOUNDARY_DRIFT_DETERIORATION = "BOUNDARY_DRIFT_DETERIORATION"
    COMPOUND_MULTIPLE = "COMPOUND_MULTIPLE"


class DegradationSeverity(str, Enum):
    NOMINAL = "NOMINAL"    # Level 0: No SLO breach
    MINOR = "MINOR"        # Level 1: 1.0x <= metric < 1.5x SLO
    MAJOR = "MAJOR"        # Level 2: 1.5x <= metric < 2.5x SLO
    CRITICAL = "CRITICAL"  # Level 3: >= 2.5x SLO, drift >= 70.0%, or dropped requests > 0


class TargetScope(str, Enum):
    SERVICE_SPECIFIC = "service_specific"
    BOUNDARY_SPECIFIC = "boundary_specific"
    SYSTEM_WIDE = "system_wide"


class DegradationRegime(str, Enum):
    NOMINAL = "nominal"
    GRADUAL_RAMP = "gradual_ramp"
    ABRUPT_BURST = "abrupt_burst"
    CAPACITY_REDUCTION = "capacity_reduction"
    BOUNDARY_SLOWDOWN = "boundary_slowdown"
    RECOVERY = "recovery"


class ServiceIdentifier(str, Enum):
    RESERVATIONS = "reservations"
    CREW = "crew"
    BAGGAGE = "baggage"
    BOUNDARY_GATEWAY = "boundary-gateway"
    LEGACY_CORE = "legacy-core"


# Documented SLO thresholds (simulation assumptions for unmeasured tiers)
SLO_THRESHOLDS: Dict[str, Dict[str, float]] = {
    ServiceIdentifier.RESERVATIONS.value: {
        "latency_ms": 100.0,
        "backlog_requests": 50.0,
        "error_rate": 0.05,
    },
    ServiceIdentifier.CREW.value: {
        "latency_ms": 100.0,
        "backlog_requests": 50.0,
        "error_rate": 0.05,
    },
    ServiceIdentifier.BAGGAGE.value: {
        "latency_ms": 100.0,
        "backlog_requests": 50.0,
        "error_rate": 0.05,
    },
    ServiceIdentifier.BOUNDARY_GATEWAY.value: {
        "latency_ms": 200.0,
        "backlog_requests": 50.0,
        "error_rate": 0.05,
        "sync_drift_pct": 45.0,  # Repository threshold config.sync_drift_threshold
    },
    ServiceIdentifier.LEGACY_CORE.value: {
        "latency_ms": 250.0,
        "backlog_requests": 50.0,
        "error_rate": 0.05,
    },
}

# Ground-truth configured nominal capacities (requests/second)
# NOTE: Simulation assumptions for cloud & gateway tiers.
# Legacy-core capacity is explicitly unknown (None).
CONFIGURED_SERVICE_CAPACITIES: Dict[str, Optional[float]] = {
    ServiceIdentifier.RESERVATIONS.value: 100.0,
    ServiceIdentifier.CREW.value: 60.0,
    ServiceIdentifier.BAGGAGE.value: 75.0,
    ServiceIdentifier.BOUNDARY_GATEWAY.value: 220.0,
    ServiceIdentifier.LEGACY_CORE.value: None,  # Explicitly unknown legacy mainframe
}

# Nominal operating baselines
NOMINAL_BASELINES: Dict[str, Dict[str, Any]] = {
    ServiceIdentifier.RESERVATIONS.value: {"latency_ms": 18.5, "nominal_capacity": 100.0},
    ServiceIdentifier.CREW.value: {"latency_ms": 22.1, "nominal_capacity": 60.0},
    ServiceIdentifier.BAGGAGE.value: {"latency_ms": 15.4, "nominal_capacity": 75.0},
    ServiceIdentifier.BOUNDARY_GATEWAY.value: {"latency_ms": 20.0, "nominal_capacity": 220.0},
    ServiceIdentifier.LEGACY_CORE.value: {"latency_ms": 68.2, "nominal_capacity": None},  # Unknown legacy
}

BOUNDARY_SYNC_DRIFT_THRESHOLD = 45.0
BOUNDARY_SYNC_DRIFT_CRITICAL = 70.0


def compute_sync_drift_score(
    gateway_submission_rate: float,
    legacy_completion_rate: float,
    gateway_queue_depth: float,
    boundary_latency_ms: float,
    nominal_latency_ms: float = 15.0,
    boundary_error_rate: float = 0.0,
    delta: float = 1e-4,
) -> float:
    """Computes boundary synchronization drift ε(t) in [8.5, 99.5]% using the repository formula."""
    phi = max(0.0, float(gateway_submission_rate))
    psi = max(0.0, float(legacy_completion_rate))
    rate_diff = abs(phi - psi)
    base_drift = (rate_diff / (phi + delta)) * 100.0
    queue_drift = min(40.0, (float(gateway_queue_depth) / 50.0) * 20.0)
    lat_ratio = max(1.0, float(boundary_latency_ms) / max(1.0, float(nominal_latency_ms)))
    lat_drift = min(35.0, math.log2(lat_ratio) * 10.0) if lat_ratio > 1.0 else 0.0
    error_drift = min(50.0, float(boundary_error_rate) * 250.0)
    total_drift = base_drift + queue_drift + lat_drift + error_drift
    return round(max(8.5, min(99.5, total_drift)), 2)



@dataclass
class ServiceTelemetryStep:
    """Observable telemetry snapshot for a single service at a discrete step."""
    timestamp_sec: float
    step_index: int
    service_id: str
    offered_rate: float        # Incoming offered requests/sec
    completed_rate: float      # Processed throughput requests/sec
    backlog_requests: float    # Queue depth (pending requests)
    latency_ms: float          # Measured response latency (ms)
    error_rate: float          # Fraction of failed/timed-out requests [0.0, 1.0]
    dropped_requests: float    # Requests dropped due to buffer overflow in step
    effective_capacity: Optional[float] = None  # Telemetry measurement if instrumented (None for legacy)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BoundaryTelemetryStep:
    """Observable telemetry snapshot for the boundary synchronization gateway."""
    timestamp_sec: float
    step_index: int
    gateway_submission_rate: float
    legacy_completion_rate: float
    gateway_queue_depth: float
    boundary_latency_ms: float
    boundary_error_rate: float
    sync_drift_score: float     # ε(t) computed via existing repo definition

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DegradationIncident:
    """Full time-series trajectory of an operational or degraded incident."""
    incident_id: str
    regime: str
    duration_sec: float
    step_count: int
    service_telemetry: Dict[str, List[Dict[str, Any]]]
    boundary_telemetry: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def is_service_step_degraded(service_id: str, step_data: Dict[str, Any]) -> bool:
    """Checks whether a single service breaches its observable SLO at the given step."""
    thresh = SLO_THRESHOLDS.get(service_id, {})
    return (
        step_data.get("latency_ms", 0.0) >= thresh.get("latency_ms", 100.0)
        or step_data.get("backlog_requests", 0.0) >= thresh.get("backlog_requests", 50.0)
        or step_data.get("error_rate", 0.0) >= thresh.get("error_rate", 0.05)
    )


def is_boundary_step_degraded(boundary_data: Dict[str, Any]) -> bool:
    """Checks whether the boundary link breaches synchronization drift SLO."""
    return boundary_data.get("sync_drift_score", 0.0) >= BOUNDARY_SYNC_DRIFT_THRESHOLD


def is_system_step_degraded(
    service_telemetry_at_step: Dict[str, Dict[str, Any]],
    boundary_telemetry_at_step: Dict[str, Any],
) -> bool:
    """Checks whether ANY service or boundary entity breaches SLO at the given step."""
    if is_boundary_step_degraded(boundary_telemetry_at_step):
        return True
    for s_id, s_data in service_telemetry_at_step.items():
        if is_service_step_degraded(s_id, s_data):
            return True
    return False


# Compatibility alias
is_step_degraded = is_system_step_degraded


def classify_service_step_type_and_severity(
    service_id: str,
    step_data: Dict[str, Any],
    boundary_data: Optional[Dict[str, Any]] = None,
) -> Tuple[DegradationType, DegradationSeverity]:
    """Classifies observable degradation type and severity using documented SLO exceedance rules.

    Supports:
    - Service metric breaches (latency, backlog, error_rate)
    - Boundary synchronization drift breaches (for boundary-gateway)
    - Compound breaches when multiple indicators fail simultaneously

    Severity Levels:
    - NOMINAL: All metrics strictly within SLO thresholds.
    - MINOR: 1.0x <= max(metric / SLO) < 1.5x (or drift in [45.0, 55.0)%)
    - MAJOR: 1.5x <= max(metric / SLO) < 2.5x (or drift in [55.0, 70.0)%)
    - CRITICAL: max(metric / SLO) >= 2.5x, dropped_requests > 0, or drift >= 70.0%.
    """
    thresh = SLO_THRESHOLDS.get(service_id, {})
    slo_lat = thresh.get("latency_ms", 100.0)
    slo_q = thresh.get("backlog_requests", 50.0)
    slo_err = thresh.get("error_rate", 0.05)

    lat = float(step_data.get("latency_ms", 0.0))
    q = float(step_data.get("backlog_requests", 0.0))
    err = float(step_data.get("error_rate", 0.0))
    drops = float(step_data.get("dropped_requests", 0.0))

    r_lat = lat / max(1.0, slo_lat)
    r_q = q / max(1.0, slo_q)
    r_err = err / max(1e-4, slo_err)

    active_types: List[DegradationType] = []
    if r_lat >= 1.0:
        active_types.append(DegradationType.LATENCY_BREACH)
    if r_q >= 1.0:
        active_types.append(DegradationType.BACKLOG_SATURATION)
    if r_err >= 1.0:
        active_types.append(DegradationType.ERROR_SPIKE)

    # Check boundary synchronization drift for boundary-gateway
    drift_val = 0.0
    drift_sev = DegradationSeverity.NOMINAL
    if service_id == ServiceIdentifier.BOUNDARY_GATEWAY.value and boundary_data is not None:
        drift_val = float(boundary_data.get("sync_drift_score", 0.0))
        if drift_val >= BOUNDARY_SYNC_DRIFT_THRESHOLD:
            active_types.append(DegradationType.BOUNDARY_DRIFT_DETERIORATION)
            if drift_val >= BOUNDARY_SYNC_DRIFT_CRITICAL:
                drift_sev = DegradationSeverity.CRITICAL
            elif drift_val >= 55.0:
                drift_sev = DegradationSeverity.MAJOR
            else:
                drift_sev = DegradationSeverity.MINOR

    if not active_types:
        return DegradationType.NOMINAL, DegradationSeverity.NOMINAL

    deg_type = DegradationType.COMPOUND_MULTIPLE if len(active_types) > 1 else active_types[0]

    max_ratio = max(r_lat, r_q, r_err)
    if max_ratio >= 2.5 or drops > 0.0:
        svc_sev = DegradationSeverity.CRITICAL
    elif max_ratio >= 1.5:
        svc_sev = DegradationSeverity.MAJOR
    elif max_ratio >= 1.0:
        svc_sev = DegradationSeverity.MINOR
    else:
        svc_sev = DegradationSeverity.NOMINAL

    sev_rank = {
        DegradationSeverity.NOMINAL: 0,
        DegradationSeverity.MINOR: 1,
        DegradationSeverity.MAJOR: 2,
        DegradationSeverity.CRITICAL: 3,
    }
    severity = drift_sev if sev_rank[drift_sev] > sev_rank[svc_sev] else svc_sev

    return deg_type, severity


def classify_boundary_step_type_and_severity(
    boundary_data: Dict[str, Any]
) -> Tuple[DegradationType, DegradationSeverity]:
    """Classifies boundary synchronization drift degradation type and severity."""
    drift = float(boundary_data.get("sync_drift_score", 0.0))
    if drift < BOUNDARY_SYNC_DRIFT_THRESHOLD:
        return DegradationType.NOMINAL, DegradationSeverity.NOMINAL

    if drift >= BOUNDARY_SYNC_DRIFT_CRITICAL:
        severity = DegradationSeverity.CRITICAL
    elif drift >= 55.0:
        severity = DegradationSeverity.MAJOR
    else:
        severity = DegradationSeverity.MINOR

    return DegradationType.BOUNDARY_DRIFT_DETERIORATION, severity


def detect_confirmed_sustained_episodes(
    service_telemetry: Dict[str, List[Dict[str, Any]]],
    boundary_telemetry: List[Dict[str, Any]],
    target_service_id: Optional[str] = None,
    min_consecutive: int = 3,
    min_recovery_consecutive: int = 3,
) -> List[Dict[str, Any]]:
    """Detects confirmed sustained degradation episodes requiring >= min_consecutive consecutive breached snapshots.

    Operational Definitions:
    - Degradation Confirmation: >= min_consecutive (3) consecutive breached snapshots.
    - First Breached Snapshot: step idx (t_onset = idx * 20s), earliest breach of sustained sequence.
    - Recovery: confirmed return to healthy state for >= min_recovery_consecutive (3) consecutive snapshots.
      The recovery_step is the first healthy snapshot of this confirmed recovery sequence.
    - Transient healthy dips (< min_recovery_consecutive) do NOT terminate an ongoing confirmed episode.
    """
    dt = SAMPLING_INTERVAL_SECONDS
    step_count = len(boundary_telemetry)

    # Step-level breach indicators
    breached = np.zeros(step_count, dtype=int)
    for t in range(step_count):
        b_data = boundary_telemetry[t]
        if target_service_id is None:
            curr_s = {s: service_telemetry[s][t] for s in [s.value for s in ServiceIdentifier]}
            if is_system_step_degraded(curr_s, b_data):
                breached[t] = 1
        elif target_service_id == ServiceIdentifier.BOUNDARY_GATEWAY.value:
            if is_boundary_step_degraded(b_data) or is_service_step_degraded(target_service_id, service_telemetry[target_service_id][t]):
                breached[t] = 1
        else:
            if is_service_step_degraded(target_service_id, service_telemetry[target_service_id][t]):
                breached[t] = 1

    episodes: List[Dict[str, Any]] = []
    idx = 0
    ep_num = 1

    while idx < step_count:
        if breached[idx] == 1:
            streak = 0
            j = idx
            while j < step_count and breached[j] == 1:
                streak += 1
                j += 1

            if streak >= min_consecutive:
                first_breach_step = idx
                confirmation_step = idx + min_consecutive - 1

                # Scan forward for confirmed recovery (min_recovery_consecutive healthy steps)
                curr = j
                recovered = False
                recovery_step = step_count

                while curr < step_count:
                    if breached[curr] == 0:
                        h_streak = 0
                        h_start = curr
                        while curr < step_count and breached[curr] == 0:
                            h_streak += 1
                            curr += 1
                        if h_streak >= min_recovery_consecutive:
                            recovered = True
                            recovery_step = h_start
                            break
                    else:
                        curr += 1

                episodes.append({
                    "episode_id": f"ep_{ep_num:02d}",
                    "first_breach_step": first_breach_step,
                    "first_breach_time_sec": float(first_breach_step * dt),
                    "confirmation_step": confirmation_step,
                    "confirmation_time_sec": float(confirmation_step * dt),
                    "recovery_step": recovery_step,
                    "recovery_time_sec": float(recovery_step * dt),
                    "duration_steps": recovery_step - first_breach_step,
                    "duration_sec": float((recovery_step - first_breach_step) * dt),
                    "is_born_degraded": bool(first_breach_step == 0),
                    "is_recovered": recovered,
                })
                ep_num += 1
                idx = curr
            else:
                idx = j
        else:
            idx += 1

    return episodes


def get_operational_state_at_origin(
    service_telemetry_up_to_t: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    boundary_telemetry_up_to_t: Optional[List[Dict[str, Any]]] = None,
    target_service_id: Optional[str] = None,
    min_consecutive: int = 3,
    min_recovery_consecutive: int = 3,
    breach_indicators_up_to_t: Optional[Union[List[int], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Derives current operational degradation state using observations available through origin t ONLY.

    Does NOT use future information or whole-incident hindsight.
    - Transitions from HEALTHY to DEGRADED only when >= min_consecutive consecutive breaches are observed up to t.
    - Transitions from DEGRADED to HEALTHY only when >= min_recovery_consecutive consecutive healthy snapshots are observed up to t.
    """
    if breach_indicators_up_to_t is not None:
        step_count = len(breach_indicators_up_to_t)
        if step_count == 0:
            return {
                "is_degraded": False,
                "active_breach_streak": 0,
                "active_healthy_streak": 0,
                "confirmed_degraded_since_step": None,
            }
        is_degraded = False
        breach_streak = 0
        healthy_streak = 0
        confirmed_step = None
        for s, step_breached in enumerate(breach_indicators_up_to_t):
            if step_breached:
                breach_streak += 1
                healthy_streak = 0
                if not is_degraded and breach_streak >= min_consecutive:
                    is_degraded = True
                    confirmed_step = s
            else:
                healthy_streak += 1
                breach_streak = 0
                if is_degraded and healthy_streak >= min_recovery_consecutive:
                    is_degraded = False
                    confirmed_step = None
        return {
            "is_degraded": is_degraded,
            "active_breach_streak": breach_streak,
            "active_healthy_streak": healthy_streak,
            "confirmed_degraded_since_step": confirmed_step,
        }

    if boundary_telemetry_up_to_t is None:
        return {
            "is_degraded": False,
            "active_breach_streak": 0,
            "active_healthy_streak": 0,
            "confirmed_degraded_since_step": None,
        }

    step_count = len(boundary_telemetry_up_to_t)
    if step_count == 0:
        return {
            "is_degraded": False,
            "active_breach_streak": 0,
            "active_healthy_streak": 0,
            "confirmed_degraded_since_step": None,
        }

    is_degraded = False
    breach_streak = 0
    healthy_streak = 0
    confirmed_step = None

    for s in range(step_count):
        b_data = boundary_telemetry_up_to_t[s]
        if target_service_id is None:
            curr_s = {k: service_telemetry_up_to_t[k][s] for k in service_telemetry_up_to_t}
            step_breached = is_system_step_degraded(curr_s, b_data)
        elif target_service_id == ServiceIdentifier.BOUNDARY_GATEWAY.value:
            step_breached = is_boundary_step_degraded(b_data) or is_service_step_degraded(
                target_service_id, service_telemetry_up_to_t[target_service_id][s]
            )
        else:
            step_breached = is_service_step_degraded(
                target_service_id, service_telemetry_up_to_t[target_service_id][s]
            )

        if step_breached:
            breach_streak += 1
            healthy_streak = 0
            if not is_degraded and breach_streak >= min_consecutive:
                is_degraded = True
                confirmed_step = s
        else:
            healthy_streak += 1
            breach_streak = 0
            if is_degraded and healthy_streak >= min_recovery_consecutive:
                is_degraded = False
                confirmed_step = None

    return {
        "is_degraded": is_degraded,
        "active_breach_streak": breach_streak,
        "active_healthy_streak": healthy_streak,
        "confirmed_degraded_since_step": confirmed_step,
    }


def check_window_label_observability(
    target_step: int,
    step_count: int,
    breach_indicators: np.ndarray,
    min_consecutive: int = 3,
) -> bool:
    """Verifies whether the classification window through target_step has complete follow-up.

    When degradation requires 3 consecutive breaches, any breach starting at step tau in (t, target_step]
    requires observing tau + 2 to confirm whether it is sustained or transient.
    If the recording ends with an unresolved breach streak (< min_consecutive), that window
    is right-truncated and cannot be definitively labeled negative.
    Returns False if an unresolved trailing breach exists; True otherwise.
    """
    if target_step + (min_consecutive - 1) < step_count:
        return True

    # Check trailing streak at end of recording
    trailing_streak = 0
    k = step_count - 1
    while k >= 0 and breach_indicators[k] == 1:
        trailing_streak += 1
        k -= 1

    if 0 < trailing_streak < min_consecutive:
        trailing_start = step_count - trailing_streak
        if trailing_start <= target_step:
            return False

    return True



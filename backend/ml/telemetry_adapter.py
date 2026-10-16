"""Telemetry Adapter for BACCP Backend.

Provides structured access to service and boundary telemetry streams:
1. Validates service IDs and supported step horizons (H=1, 5, 15 at 20s sampling interval).
2. Distinguishes live telemetry mode from configured incident replay mode in EVERY response.
3. Live mode: checks real available columns (eBPF TCP connections, static graph latency).
   Never silently imputes nominal values; returns structured UNAVAILABLE states on missing telemetry.
4. Replay mode: loads verified saved incident data, preserves causal timestamps, enforces
   replay-clock freshness (MAX_TELEMETRY_STALENESS_SECONDS = 60.0s).
5. Exposes future observation matching for workload forecast error verification.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("baccp.telemetry_adapter")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "ai-models" / "data" / "degradation_prediction" / "degradation_dataset.json"

SUPPORTED_SERVICES = {
    "reservations",
    "crew",
    "baggage",
    "boundary-gateway",
    "legacy-core",
}

SUPPORTED_HORIZONS = {1, 5, 15}
SAMPLING_INTERVAL_SECONDS = 20.0
MAX_TELEMETRY_STALENESS_SECONDS = 60.0

HORIZON_SECONDS_MAP = {
    1: 20.0,
    5: 100.0,
    15: 300.0,
}


class TelemetryError:
    """Structured representation of telemetry access or validation errors."""

    def __init__(
        self,
        error_code: str,
        message: str,
        status_code: int = 400,
        missing_fields: Optional[List[str]] = None,
        available_metrics: Optional[List[str]] = None,
        staleness_seconds: Optional[float] = None,
        telemetry_mode: str = "replay",
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.missing_fields = missing_fields or []
        self.available_metrics = available_metrics or []
        self.staleness_seconds = staleness_seconds
        self.telemetry_mode = telemetry_mode

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
            "telemetry_mode": self.telemetry_mode,
        }
        if self.missing_fields:
            d["missing_required_metrics"] = self.missing_fields
        if self.available_metrics:
            d["available_live_metrics"] = self.available_metrics
        if self.staleness_seconds is not None:
            d["staleness_seconds"] = round(self.staleness_seconds, 2)
            d["max_staleness_limit_seconds"] = MAX_TELEMETRY_STALENESS_SECONDS
        return d


class TelemetrySnapshot:
    """Clean, causally-aligned telemetry snapshot with explicit provenance and metadata."""

    def __init__(
        self,
        snapshot_id: str,
        timestamp_sec: float,
        telemetry_mode: str,
        data_source: str,
        data_provenance: str,
        service_telemetry: Dict[str, List[Dict[str, Any]]],
        boundary_telemetry: Optional[List[Dict[str, Any]]],
        origin_step: int,
        freshness_seconds: float = 0.0,
        incident_id: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        is_carried_forward: bool = False,
    ):
        self.snapshot_id = snapshot_id
        self.timestamp_sec = float(timestamp_sec)
        self.telemetry_mode = telemetry_mode
        self.data_source = data_source
        self.data_provenance = data_provenance
        self.service_telemetry = service_telemetry
        self.boundary_telemetry = boundary_telemetry
        self.origin_step = int(origin_step)
        self.freshness_seconds = float(freshness_seconds)
        self.incident_id = incident_id
        self.warnings = warnings or []
        self.is_carried_forward = is_carried_forward

    def get_service_history(self, service_id: str, field: str = "offered_rate") -> List[float]:
        """Extracts numerical history sequence for a service attribute."""
        stream = self.service_telemetry.get(service_id, [])
        return [float(rec.get(field, 0.0)) for rec in stream]


class TelemetryAdapter:
    """Unified telemetry adapter supporting explicit Live and Replay modes."""

    def __init__(
        self,
        mode: Optional[str] = None,
        dataset_path: Path = DEFAULT_DATASET_PATH,
        default_incident_id: Optional[str] = None,
    ):
        # Mode resolution: env BACCP_TELEMETRY_MODE > parameter > default 'replay'
        env_mode = os.environ.get("BACCP_TELEMETRY_MODE", "").lower()
        if env_mode in ("live", "replay"):
            self.mode = env_mode
        elif mode in ("live", "replay"):
            self.mode = mode
        else:
            self.mode = "replay"

        self.dataset_path = dataset_path
        self._dataset_cache: Optional[Dict[str, Any]] = None
        self._incidents_by_id: Dict[str, Dict[str, Any]] = {}
        self.default_incident_id = default_incident_id

        # Replay clock state
        self.current_replay_step: int = 15
        self.current_replay_time_sec: float = 300.0

        if self.mode == "replay":
            self._load_saved_dataset()

    def _load_saved_dataset(self) -> None:
        """Loads incident dataset for replay mode."""
        if not self.dataset_path.exists():
            logger.warning(f"Saved incident dataset not found at {self.dataset_path}. Replay mode will be unavailable.")
            return

        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._dataset_cache = data
            for inc in data.get("incidents", []):
                self._incidents_by_id[inc["incident_id"]] = inc

            if self.default_incident_id is None:
                # Prefer standard_test incident, or first available incident
                std_inc = next((i["incident_id"] for i in data.get("incidents", []) if i.get("split") == "standard_test"), None)
                self.default_incident_id = std_inc or (data["incidents"][0]["incident_id"] if data.get("incidents") else None)

            logger.info(f"Loaded replay dataset: {len(self._incidents_by_id)} incidents (default: {self.default_incident_id})")
        except Exception as exc:
            logger.error(f"Failed to load replay dataset: {exc}")

    def validate_request_params(self, service_id: str, horizon: int) -> Optional[TelemetryError]:
        """Validates service ID and horizon parameters."""
        if not service_id:
            return TelemetryError(
                error_code="INVALID_SERVICE",
                message="Query parameter 'service' is required.",
                status_code=400,
                telemetry_mode=self.mode,
            )

        if service_id not in SUPPORTED_SERVICES:
            return TelemetryError(
                error_code="UNSUPPORTED_SERVICE",
                message=f"Service '{service_id}' is not supported. Supported services: {sorted(SUPPORTED_SERVICES)}",
                status_code=400,
                telemetry_mode=self.mode,
            )

        if horizon not in SUPPORTED_HORIZONS:
            return TelemetryError(
                error_code="UNSUPPORTED_HORIZON",
                message=f"Horizon '{horizon}' is not supported. Supported step horizons: {sorted(SUPPORTED_HORIZONS)} (20s, 100s, 300s)",
                status_code=400,
                telemetry_mode=self.mode,
            )

        return None

    def get_telemetry_snapshot(
        self,
        service_id: str,
        horizon: int,
        replay_step: Optional[int] = None,
        replay_time_sec: Optional[float] = None,
        incident_id: Optional[str] = None,
    ) -> Union[TelemetrySnapshot, TelemetryError]:
        """Provides telemetry snapshot for the given service and horizon."""
        val_err = self.validate_request_params(service_id, horizon)
        if val_err:
            return val_err

        if replay_time_sec is not None:
            try:
                r_ts = float(replay_time_sec)
                if math.isnan(r_ts) or math.isinf(r_ts) or r_ts < 0.0:
                    return TelemetryError(
                        error_code="INVALID_REPLAY_PARAMETER",
                        message=f"Parameter 'replay_time_sec' must be a non-negative finite number (got {replay_time_sec}).",
                        status_code=400,
                        telemetry_mode=self.mode,
                    )
            except (ValueError, TypeError):
                return TelemetryError(
                    error_code="INVALID_REPLAY_PARAMETER",
                    message=f"Parameter 'replay_time_sec' must be a non-negative finite number (got {replay_time_sec}).",
                    status_code=400,
                    telemetry_mode=self.mode,
                )

        if replay_step is not None:
            try:
                r_st = int(replay_step)
                if r_st < 0:
                    return TelemetryError(
                        error_code="INVALID_REPLAY_PARAMETER",
                        message=f"Parameter 'replay_step' must be a non-negative integer (got {replay_step}).",
                        status_code=400,
                        telemetry_mode=self.mode,
                    )
            except (ValueError, TypeError):
                return TelemetryError(
                    error_code="INVALID_REPLAY_PARAMETER",
                    message=f"Parameter 'replay_step' must be a non-negative integer (got {replay_step}).",
                    status_code=400,
                    telemetry_mode=self.mode,
                )

        if self.mode == "live":
            return self._get_live_telemetry_snapshot(service_id, horizon)
        else:
            return self._get_replay_telemetry_snapshot(
                service_id=service_id,
                horizon=horizon,
                replay_step=replay_step,
                replay_time_sec=replay_time_sec,
                incident_id=incident_id,
            )

    def _get_live_telemetry_snapshot(self, service_id: str, horizon: int) -> TelemetryError:
        """Inspects live telemetry source and reports live data gaps."""
        # Live eBPF discovery captures TCP connection attempts only.
        # Application-level metrics (measured latency percentiles, queue depth, error rates)
        # are not instrumented in the live backend environment.
        # DO NOT silently impute nominal values.
        return TelemetryError(
            error_code="UNAVAILABLE_INSUFFICIENT_TELEMETRY",
            message=(
                f"Live telemetry source lacks required multi-metric application history for service '{service_id}'. "
                "eBPF probe captures tcp_connection_attempt only. Application latency percentiles, queue backlog, "
                "and error rates are not collected in live mode. To test predictions and explanations, "
                "configure replay mode (BACCP_TELEMETRY_MODE=replay)."
            ),
            status_code=503,
            missing_fields=[
                "latency_ms",
                "backlog_requests",
                "offered_rate",
                "completed_rate",
                "error_rate",
                "boundary_sync_drift_pct",
            ],
            available_metrics=["tcp_connection_attempt", "avg_latency_ms (static graph edge)"],
            telemetry_mode="live",
        )

    def _get_replay_telemetry_snapshot(
        self,
        service_id: str,
        horizon: int,
        replay_step: Optional[int] = None,
        replay_time_sec: Optional[float] = None,
        incident_id: Optional[str] = None,
    ) -> Union[TelemetrySnapshot, TelemetryError]:
        """Builds causally aligned snapshot from saved incident data with replay clock freshness."""
        if not self._incidents_by_id:
            return TelemetryError(
                error_code="REPLAY_DATASET_UNAVAILABLE",
                message="Replay incident dataset is not loaded.",
                status_code=503,
                telemetry_mode="replay",
            )

        inc_key = incident_id or self.default_incident_id
        if inc_key not in self._incidents_by_id:
            return TelemetryError(
                error_code="INCIDENT_NOT_FOUND",
                message=f"Incident '{inc_key}' not found in replay dataset.",
                status_code=400,
                telemetry_mode="replay",
            )

        inc = self._incidents_by_id[inc_key]
        step_count = inc["step_count"]

        # Replay clock resolution
        if replay_time_sec is not None:
            clock_ts = float(replay_time_sec)
            origin_t = int(round(clock_ts / SAMPLING_INTERVAL_SECONDS))
        elif replay_step is not None:
            origin_t = int(replay_step)
            clock_ts = float(origin_t * SAMPLING_INTERVAL_SECONDS)
        else:
            origin_t = self.current_replay_step
            clock_ts = self.current_replay_time_sec

        # Bounds validation without rewinding clock based on forecast horizon
        if origin_t < 3:
            return TelemetryError(
                error_code="UNAVAILABLE_INSUFFICIENT_TELEMETRY",
                message=(
                    f"Insufficient history: requested replay origin step {origin_t} (time {clock_ts:.1f}s) "
                    "has fewer than 4 observation steps (minimum 60.0s context window required)."
                ),
                status_code=422,
                telemetry_mode="replay",
            )


        # Causal slice strictly up to clock_ts
        svc_tel = inc["service_telemetry"]
        bnd_tel = inc["boundary_telemetry"]

        svc_sliced: Dict[str, List[Dict[str, Any]]] = {}
        per_service_freshness: Dict[str, float] = {}

        for s, stream in svc_tel.items():
            records = [
                copy.deepcopy(rec)
                for rec in stream
                if rec.get("timestamp_sec", rec.get("step_index", 0) * SAMPLING_INTERVAL_SECONDS) <= clock_ts + 1e-4
            ]
            if not records:
                return TelemetryError(
                    error_code="UNAVAILABLE_INSUFFICIENT_TELEMETRY",
                    message=f"No telemetry observations available for service '{s}' before timestamp {clock_ts:.1f}s.",
                    status_code=422,
                    telemetry_mode="replay",
                )
            svc_sliced[s] = records
            s_latest_ts = records[-1].get("timestamp_sec", records[-1].get("step_index", 0) * SAMPLING_INTERVAL_SECONDS)
            per_service_freshness[s] = max(0.0, clock_ts - s_latest_ts)

        bnd_sliced = [
            copy.deepcopy(rec)
            for rec in bnd_tel
            if rec.get("timestamp_sec", rec.get("step_index", 0) * SAMPLING_INTERVAL_SECONDS) <= clock_ts + 1e-4
        ]
        if not bnd_sliced:
            return TelemetryError(
                error_code="UNAVAILABLE_INSUFFICIENT_TELEMETRY",
                message=f"No boundary gateway telemetry observations available before timestamp {clock_ts:.1f}s.",
                status_code=422,
                telemetry_mode="replay",
            )
        bnd_latest_ts = bnd_sliced[-1].get("timestamp_sec", bnd_sliced[-1].get("step_index", 0) * SAMPLING_INTERVAL_SECONDS)
        bnd_freshness = max(0.0, clock_ts - bnd_latest_ts)

        # 1. Freshness check for requested service specifically:
        # A fresh neighbour MUST NOT hide a stale requested service!
        req_svc_freshness = per_service_freshness.get(service_id, 0.0)
        if req_svc_freshness > MAX_TELEMETRY_STALENESS_SECONDS:
            return TelemetryError(
                error_code="STALENESS_LIMIT_EXCEEDED",
                message=(
                    f"Telemetry for requested service '{service_id}' is stale: latest observation is {req_svc_freshness:.1f}s behind "
                    f"replay clock {clock_ts:.1f}s (maximum permitted staleness is {MAX_TELEMETRY_STALENESS_SECONDS:.1f}s). "
                    "Cannot predict on stale telemetry."
                ),
                status_code=422,
                staleness_seconds=req_svc_freshness,
                telemetry_mode="replay",
            )

        # 2. Freshness check for boundary stream:
        if bnd_freshness > MAX_TELEMETRY_STALENESS_SECONDS:
            return TelemetryError(
                error_code="STALENESS_LIMIT_EXCEEDED",
                message=(
                    f"Boundary gateway telemetry is stale: latest observation is {bnd_freshness:.1f}s behind "
                    f"replay clock {clock_ts:.1f}s (maximum permitted staleness is {MAX_TELEMETRY_STALENESS_SECONDS:.1f}s). "
                    "Cannot predict on stale boundary telemetry."
                ),
                status_code=422,
                staleness_seconds=bnd_freshness,
                telemetry_mode="replay",
            )

        snapshot_id = f"replay:{inc_key}:step_{origin_t}:t_{clock_ts:.1f}"

        warnings: List[str] = []
        is_carried = False
        freshness_sec = req_svc_freshness
        if freshness_sec > 1e-4:
            is_carried = True
            warnings.append(
                f"Telemetry observation for '{service_id}' carried forward by {freshness_sec:.1f}s to replay clock {clock_ts:.1f}s "
                f"(staleness <= {MAX_TELEMETRY_STALENESS_SECONDS:.1f}s limit)."
            )
        if bnd_freshness > 1e-4:
            warnings.append(
                f"Boundary gateway telemetry observation carried forward by {bnd_freshness:.1f}s to replay clock {clock_ts:.1f}s."
            )

        return TelemetrySnapshot(
            snapshot_id=snapshot_id,
            timestamp_sec=clock_ts,
            telemetry_mode="replay",
            data_source="REPLAY_INCIDENT_STREAM",
            data_provenance="SYNTHETIC_SIMULATED",
            service_telemetry=svc_sliced,
            boundary_telemetry=bnd_sliced,
            origin_step=origin_t,
            freshness_seconds=freshness_sec,
            incident_id=inc_key,
            warnings=warnings,
            is_carried_forward=is_carried,
        )

    def get_future_observation(
        self,
        service_id: str,
        target_timestamp_sec: float,
        incident_id: Optional[str] = None,
        max_available_timestamp_sec: Optional[float] = None,
    ) -> Optional[float]:
        """Retrieves actual offered rate observation at target timestamp ONLY if available
        at or before max_available_timestamp_sec (prevents premature observation exposure)."""
        if max_available_timestamp_sec is not None and target_timestamp_sec > max_available_timestamp_sec + 1e-4:
            return None

        inc_key = incident_id or self.default_incident_id
        if inc_key not in self._incidents_by_id:
            return None

        inc = self._incidents_by_id[inc_key]
        stream = inc["service_telemetry"].get(service_id, [])

        for rec in stream:
            rec_ts = rec.get("timestamp_sec", rec.get("step_index", 0) * SAMPLING_INTERVAL_SECONDS)
            if abs(rec_ts - target_timestamp_sec) < 1.0:
                return float(rec.get("offered_rate", 0.0))

        return None

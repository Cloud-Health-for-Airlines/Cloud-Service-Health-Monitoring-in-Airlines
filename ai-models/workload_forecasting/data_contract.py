"""BACCP Workload Forecasting Data Contract.

Version: baccp-workload-contract-v1.1.0

Explicitly formalizes the per-service telemetry schema, units, sampling rates,
provenance, and boundary assumptions for workload forecasting.
Maintains clear distinctions between:
1. Offered workload (incoming customer demand in requests/sec)
2. Completed throughput (processed transactions in requests/sec)
3. Configured capacity (simulation assumption in requests/sec, or None if unknown)
4. Signed capacity margin, available headroom, and overload:
   capacity_margin = capacity - forecast
   available_headroom = max(0, capacity_margin)
   overload = max(0, -capacity_margin)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional


CONTRACT_VERSION = "baccp-workload-contract-v1.1.0"


class GenerationType(str, Enum):
    CLOUD_NATIVE = "cloud-native"
    BOUNDARY_GATEWAY = "boundary-gateway"
    LEGACY = "legacy"


class ServiceIdentifier(str, Enum):
    RESERVATIONS = "reservations"
    CREW = "crew"
    BAGGAGE = "baggage"
    BOUNDARY_GATEWAY = "boundary-gateway"
    LEGACY_CORE = "legacy-core"


# Ground-truth configured nominal capacities (requests/second)
# NOTE: These are configured simulation assumptions for cloud & gateway tiers.
# Legacy-core capacity is explicitly unknown (uninstrumented legacy mainframe).
CONFIGURED_SERVICE_CAPACITIES: Dict[str, Optional[float]] = {
    ServiceIdentifier.RESERVATIONS.value: 100.0,
    ServiceIdentifier.CREW.value: 60.0,
    ServiceIdentifier.BAGGAGE.value: 75.0,
    ServiceIdentifier.BOUNDARY_GATEWAY.value: 220.0,
    ServiceIdentifier.LEGACY_CORE.value: None,  # Explicitly unknown/uninstrumented
}

SERVICE_GENERATION_MAPPING: Dict[str, GenerationType] = {
    ServiceIdentifier.RESERVATIONS.value: GenerationType.CLOUD_NATIVE,
    ServiceIdentifier.CREW.value: GenerationType.CLOUD_NATIVE,
    ServiceIdentifier.BAGGAGE.value: GenerationType.CLOUD_NATIVE,
    ServiceIdentifier.BOUNDARY_GATEWAY.value: GenerationType.BOUNDARY_GATEWAY,
    ServiceIdentifier.LEGACY_CORE.value: GenerationType.LEGACY,
}


@dataclass(frozen=True)
class WorkloadRecord:
    """A single discrete-time snapshot of telemetry for a single service."""
    incident_id: str
    service_id: str
    generation_type: str
    step_index: int
    elapsed_time_sec: float
    sampling_interval_sec: float
    offered_workload_rps: float
    completed_throughput_rps: float
    configured_capacity_rps: Optional[float]
    capacity_margin_rps: Optional[float]
    available_headroom_rps: Optional[float]
    overload_rps: Optional[float]
    capacity_utilization: Optional[float]
    capacity_type: str
    data_provenance: str
    scenario_name: str
    scenario_intensity: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IncidentTrajectory:
    """Complete time-series trajectory for an independent incident across all services."""
    incident_id: str
    scenario_name: str
    scenario_intensity: float
    seed: int
    total_steps: int
    sampling_interval_sec: float
    duration_seconds: float
    records_by_service: Dict[str, List[WorkloadRecord]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "scenario_name": self.scenario_name,
            "scenario_intensity": self.scenario_intensity,
            "seed": self.seed,
            "total_steps": self.total_steps,
            "sampling_interval_sec": self.sampling_interval_sec,
            "duration_seconds": self.duration_seconds,
            "records_by_service": {
                svc: [rec.to_dict() for rec in recs]
                for svc, recs in self.records_by_service.items()
            },
        }


def validate_workload_record(record: WorkloadRecord) -> None:
    """Validates record adherence to contract constraints."""
    if record.step_index < 0:
        raise ValueError(f"Invalid step_index {record.step_index}: must be >= 0")
    if record.sampling_interval_sec <= 0:
        raise ValueError("sampling_interval_sec must be positive")
    if record.offered_workload_rps < 0:
        raise ValueError(f"offered_workload_rps cannot be negative: {record.offered_workload_rps}")
    if record.completed_throughput_rps < 0:
        raise ValueError(f"completed_throughput_rps cannot be negative: {record.completed_throughput_rps}")

    if record.configured_capacity_rps is not None:
        if record.configured_capacity_rps <= 0:
            raise ValueError(f"configured_capacity_rps must be positive when specified: {record.configured_capacity_rps}")
        expected_margin = record.configured_capacity_rps - record.offered_workload_rps
        expected_headroom = max(0.0, expected_margin)
        expected_overload = max(0.0, -expected_margin)

        if record.capacity_margin_rps is not None and not math.isclose(
            record.capacity_margin_rps, expected_margin, rel_tol=1e-3, abs_tol=1e-3
        ):
            raise ValueError(f"Inconsistent capacity_margin: expected {expected_margin}, got {record.capacity_margin_rps}")

        if record.available_headroom_rps is not None and not math.isclose(
            record.available_headroom_rps, expected_headroom, rel_tol=1e-3, abs_tol=1e-3
        ):
            raise ValueError(f"Inconsistent available_headroom: expected {expected_headroom}, got {record.available_headroom_rps}")

        if record.overload_rps is not None and not math.isclose(
            record.overload_rps, expected_overload, rel_tol=1e-3, abs_tol=1e-3
        ):
            raise ValueError(f"Inconsistent overload: expected {expected_overload}, got {record.overload_rps}")

        if record.capacity_type != "configured_simulation_assumption":
            raise ValueError(f"Inconsistent capacity_type: expected 'configured_simulation_assumption', got {record.capacity_type}")
    else:
        if (
            record.capacity_margin_rps is not None
            or record.available_headroom_rps is not None
            or record.overload_rps is not None
            or record.capacity_utilization is not None
        ):
            raise ValueError("All capacity metrics must be None when configured_capacity_rps is None")
        if record.capacity_type != "unknown_uninstrumented":
            raise ValueError(f"Inconsistent capacity_type: expected 'unknown_uninstrumented', got {record.capacity_type}")

    if record.data_provenance not in ("synthetic", "measured"):
        raise ValueError(f"Invalid provenance: {record.data_provenance}")

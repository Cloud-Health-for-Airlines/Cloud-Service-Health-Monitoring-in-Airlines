"""Airline IT Workload Generator Package for BACCP."""

from __future__ import annotations

from .models import (
    DataProvenance,
    WorkloadObservation,
    WorkloadScenarioConfig,
    WorkloadSnapshot,
)
from .patterns import (
    CANONICAL_5_SERVICES,
    EXPANDED_9_SERVICES,
    SERVICE_TIERS,
    build_canonical_edges,
    build_expanded_edges,
    get_diurnal_multiplier,
    get_ramp_multiplier,
    get_spike_multiplier,
    load_bts_airline_profile,
)
from .airline_workload import RealisticAirlineWorkloadGenerator

__all__ = [
    "RealisticAirlineWorkloadGenerator",
    "WorkloadScenarioConfig",
    "WorkloadObservation",
    "WorkloadSnapshot",
    "DataProvenance",
    "CANONICAL_5_SERVICES",
    "EXPANDED_9_SERVICES",
    "SERVICE_TIERS",
    "build_canonical_edges",
    "build_expanded_edges",
    "get_diurnal_multiplier",
    "get_ramp_multiplier",
    "get_spike_multiplier",
    "load_bts_airline_profile",
]

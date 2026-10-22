"""Calibrated mapping from forecasted airline passenger demand to simulated IT service workloads.

Implements an explicit, configurable conversion pipeline:
Forecasted Passenger Demand (BTS)
    ↓
Configurable Demand-to-Request Conversion
    ↓
Service-Specific Arrival Rates (Search, Reservations, Payment, Check-in, Baggage, Crew)
    ↓
Computing Resource Demand Estimates (Google Cluster Traces)
    ↓
BACCP Workload / Cloud Telemetry Pipeline (Orchestrator & GNN)

Maintains strict conceptual separation between public passenger demand and generic cluster computing characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..generator.airline_workload import RealisticAirlineWorkloadGenerator
from ..generator.models import (
    DataProvenance,
    WorkloadObservation,
    WorkloadScenarioConfig,
    WorkloadSnapshot,
)
from ..generator.patterns import CANONICAL_5_SERVICES, EXPANDED_9_SERVICES, get_diurnal_multiplier


@dataclass
class DemandConversionConfig:
    """Configurable calibration parameters translating passenger volume into IT transactions."""
    reference_daily_passengers: float = 2_150_000.0  # Baseline empirical US domestic volume
    nominal_reservations_base_tps: float = 28.0       # Calibrated TPS at reference passenger volume
    nominal_search_base_tps: float = 180.0
    nominal_payment_base_tps: float = 12.0
    nominal_checkin_base_tps: float = 22.0
    nominal_crew_base_tps: float = 14.0
    nominal_baggage_base_tps: float = 16.0
    nominal_notifications_base_tps: float = 35.0
    
    # Operational ratios
    search_to_book_ratio: float = 15.0                # 15 searches per completed reservation
    payment_to_book_ratio: float = 0.43
    checkin_passenger_ratio: float = 0.95
    checked_bags_per_passenger: float = 0.75
    
    # Conversion assumptions documentation
    conversion_rationale: str = (
        "Passenger enplanement forecasts scale system transaction rates proportionally. "
        "Peak passenger travel banks drive reservation and check-in volumes, which in turn "
        "drive browse queries via calibrated search-to-book ratios. "
        "Resource demands (CPU/Memory) are subsequently derived using Google Cluster trace profiles."
    )


class DemandToWorkloadMapper:
    """Bridges forecasted airline demand to the BACCP simulation and cloud monitoring pipeline."""

    def __init__(
        self,
        conversion_config: Optional[DemandConversionConfig] = None,
        generator: Optional[RealisticAirlineWorkloadGenerator] = None,
    ):
        self.config = conversion_config or DemandConversionConfig()
        self.generator = generator or RealisticAirlineWorkloadGenerator()

    def compute_demand_multiplier(self, forecasted_passengers: float) -> float:
        """Compute normalized demand intensity multiplier relative to reference passenger volume."""
        if forecasted_passengers <= 0:
            raise ValueError(f"Forecasted passenger count must be positive. Got: {forecasted_passengers}")
        return float(forecasted_passengers / self.config.reference_daily_passengers)

    def convert_demand_to_service_tps(
        self,
        forecasted_passengers: float,
        hour_of_day: float = 9.0,
        day_of_week_mult: float = 1.0,
    ) -> Dict[str, float]:
        """Convert forecasted daily passenger enplanements into instantaneous service request rates (TPS)."""
        demand_mult = self.compute_demand_multiplier(forecasted_passengers)
        diurnal_factor = get_diurnal_multiplier(hour_of_day)
        total_scaling = demand_mult * diurnal_factor * day_of_week_mult

        res_tps = max(1.0, self.config.nominal_reservations_base_tps * total_scaling)
        search_tps = max(5.0, res_tps * self.config.search_to_book_ratio)
        payment_tps = max(1.0, res_tps * self.config.payment_to_book_ratio)
        checkin_tps = max(1.0, self.config.nominal_checkin_base_tps * total_scaling)
        crew_tps = max(1.0, self.config.nominal_crew_base_tps * total_scaling)
        baggage_tps = max(1.0, self.config.nominal_baggage_base_tps * total_scaling)
        notif_tps = max(2.0, self.config.nominal_notifications_base_tps * total_scaling)

        # Gateway submission rate: sum of legacy-bound calls
        # 85% reservations, 50% payment, 60% check-in, 70% crew, 40% baggage
        gw_tps = (
            res_tps * 0.85
            + payment_tps * 0.50
            + checkin_tps * 0.60
            + crew_tps * 0.70
            + baggage_tps * 0.40
        )
        legacy_tps = min(75.0, gw_tps)

        return {
            "flight-search": round(search_tps, 2),
            "reservations": round(res_tps, 2),
            "payment": round(payment_tps, 2),
            "check-in": round(checkin_tps, 2),
            "crew": round(crew_tps, 2),
            "baggage": round(baggage_tps, 2),
            "notifications": round(notif_tps, 2),
            "boundary-gateway": round(gw_tps, 2),
            "legacy-core": round(legacy_tps, 2),
        }

    def create_workload_config_from_forecast(
        self,
        forecasted_passengers: float,
        scenario_name: str = "forecast_driven",
        duration_seconds: float = 600.0,
        sampling_interval_sec: float = 20.0,
        start_hour: float = 9.0,
        day_of_week: str = "Monday",
        services: Optional[List[str]] = None,
        random_seed: int = 42,
        active_fault: Optional[str] = None,
        fault_level: Optional[str] = None,
        fault_start_sec: Optional[float] = None,
        spike_start_sec: Optional[float] = None,
        spike_magnitude: float = 3.0,
    ) -> WorkloadScenarioConfig:
        """Create a WorkloadScenarioConfig parameterized directly from forecasted passenger demand."""
        demand_intensity = self.compute_demand_multiplier(forecasted_passengers)
        return WorkloadScenarioConfig(
            scenario_name=scenario_name,
            duration_seconds=duration_seconds,
            sampling_interval_sec=sampling_interval_sec,
            demand_intensity=round(demand_intensity, 4),
            start_hour=start_hour,
            day_of_week=day_of_week,
            services=services,
            random_seed=random_seed,
            active_fault=active_fault,
            fault_level=fault_level,
            fault_start_sec=fault_start_sec,
            spike_start_sec=spike_start_sec,
            spike_magnitude=spike_magnitude,
        )

    def generate_workload_from_forecast(
        self,
        forecasted_passengers: float,
        scenario_name: str = "forecast_driven",
        duration_seconds: float = 600.0,
        sampling_interval_sec: float = 20.0,
        start_hour: float = 9.0,
        day_of_week: str = "Monday",
        services: Optional[List[str]] = None,
        random_seed: int = 42,
        active_fault: Optional[str] = None,
        fault_level: Optional[str] = None,
        fault_start_sec: Optional[float] = None,
        spike_start_sec: Optional[float] = None,
        spike_magnitude: float = 3.0,
    ) -> List[WorkloadSnapshot]:
        """Generate full workload snapshots from forecasted demand with enriched provenance."""
        cfg = self.create_workload_config_from_forecast(
            forecasted_passengers=forecasted_passengers,
            scenario_name=scenario_name,
            duration_seconds=duration_seconds,
            sampling_interval_sec=sampling_interval_sec,
            start_hour=start_hour,
            day_of_week=day_of_week,
            services=services,
            random_seed=random_seed,
            active_fault=active_fault,
            fault_level=fault_level,
            fault_start_sec=fault_start_sec,
            spike_start_sec=spike_start_sec,
            spike_magnitude=spike_magnitude,
        )
        snapshots = self.generator.generate_workload(cfg)

        # Enrich provenance records with forecasting calibration details
        demand_mult = self.compute_demand_multiplier(forecasted_passengers)
        for snap in snapshots:
            for obs in snap.observations.values():
                obs.provenance.forecast_values["forecasted_daily_passengers"] = round(forecasted_passengers, 1)
                obs.provenance.forecast_values["calibrated_demand_multiplier"] = round(demand_mult, 4)
                obs.provenance.assumptions["demand_to_request_mapping"] = self.config.conversion_rationale

        return snapshots

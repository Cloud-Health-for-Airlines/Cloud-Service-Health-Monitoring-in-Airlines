"""Data models, observation schemas, and configuration contracts for the airline IT workload generator.

Represents workload observations with clear provenance distinguishing:
- Observed dataset values (e.g., Google Cluster resource traces, BTS hourly departure distributions)
- Forecast values (e.g., projected request rate, expected peak demand)
- Simulated values (e.g., discrete queue length, latency, error rate)
- Explicit engineering assumptions
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DataProvenance:
    """Clear attribution and traceability for workload metrics.
    
    Prevents confounding observed empirical trace data with simulated queue dynamics
    or forecasting assumptions.
    """
    observed_dataset_values: Dict[str, Any] = field(default_factory=dict)
    forecast_values: Dict[str, Any] = field(default_factory=dict)
    simulated_values: Dict[str, Any] = field(default_factory=dict)
    assumptions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkloadObservation:
    """A single service-level workload observation at a discrete point in time."""
    timestamp: str                         # ISO-8601 UTC timestamp string
    timestamp_sec: float                   # Simulation epoch seconds
    service_id: str                        # Service identifier (e.g., "reservations", "flight-search")
    service_tier: str                      # "cloud-native", "boundary-gateway", or "legacy"
    request_count: float                   # Transaction or request count in window
    request_rate: float                    # Requests per second (TPS)
    cpu_demand: float                      # Estimated normalized CPU demand [0.0, 1.0]
    memory_demand: float                   # Estimated normalized memory demand [0.0, 1.0]
    queue_length: float                    # Pending queue depth / worker backlog
    latency_ms: float                      # Observed or simulated round-trip latency in ms
    error_rate: float                      # Failed / timed-out request fraction [0.0, 1.0]
    workload_scenario: str                 # Scenario name (e.g., "normal_diurnal", "sudden_spike")
    provenance: DataProvenance             # Provenance distinguishing observed vs simulated

    def to_dict(self) -> Dict[str, Any]:
        """Serialize observation to structured dictionary."""
        return {
            "timestamp": self.timestamp,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "service_id": self.service_id,
            "service_tier": self.service_tier,
            "request_count": round(self.request_count, 2),
            "request_rate": round(self.request_rate, 2),
            "cpu_demand": round(self.cpu_demand, 4),
            "memory_demand": round(self.memory_demand, 4),
            "queue_length": round(self.queue_length, 2),
            "latency_ms": round(self.latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "workload_scenario": self.workload_scenario,
            "provenance": self.provenance.to_dict(),
        }

    def to_orchestrator_node_metric(self) -> Dict[str, float]:
        """Extract RED metric subset compatible with BACCPCloudOrchestrator and GNN."""
        return {
            "request_rate": round(self.request_rate, 2),
            "latency_ms": round(self.latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "cpu_load": round(self.cpu_demand, 3),
            "queue_depth": round(self.queue_length, 2),
        }


@dataclass
class WorkloadSnapshot:
    """Multi-service system snapshot at a discrete simulation step."""
    timestamp_sec: float
    iso_timestamp: str
    scenario: str
    observations: Dict[str, WorkloadObservation] = field(default_factory=dict)
    sync_drift_score: float = 12.4
    active_fault: Optional[str] = None
    fault_level: Optional[str] = None

    def to_orchestrator_telemetry(self) -> Dict[str, Dict[str, float]]:
        """Format as telemetry payload accepted by BACCPCloudOrchestrator.run_pipeline."""
        return {
            svc_id: obs.to_orchestrator_node_metric()
            for svc_id, obs in self.observations.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize snapshot to structured dictionary."""
        return {
            "timestamp_sec": round(self.timestamp_sec, 3),
            "iso_timestamp": self.iso_timestamp,
            "scenario": self.scenario,
            "sync_drift_score": round(self.sync_drift_score, 2),
            "active_fault": self.active_fault,
            "fault_level": self.fault_level,
            "observations": {
                k: v.to_dict() for k, v in self.observations.items()
            },
        }


@dataclass
class WorkloadScenarioConfig:
    """Configurable parameters for workload generation scenarios."""
    scenario_name: str = "normal_diurnal"
    duration_seconds: float = 600.0         # Total duration (default 10 minutes)
    sampling_interval_sec: float = 20.0     # Time between snapshots (default 20 seconds)
    demand_intensity: float = 1.0           # Overall traffic multiplier (1.0 = nominal)
    random_seed: int = 42                   # Deterministic random seed
    services: Optional[List[str]] = None    # List of services to simulate (None = canonical 5)
    
    # Time-of-day / Day-of-week context
    start_hour: float = 9.0                 # 09:00 AM (peak morning bank)
    day_of_week: str = "Monday"             # Business travel day
    
    # Sudden Spike configuration
    spike_start_sec: Optional[float] = None
    spike_duration_sec: float = 60.0
    spike_magnitude: float = 3.0            # e.g., 3x arrival rate jump
    
    # Gradual Ramp configuration
    ramp_start_sec: Optional[float] = None
    ramp_duration_sec: float = 300.0
    ramp_end_factor: float = 2.5            # e.g., 2.5x growth over ramp duration
    
    # Chaos injection configuration
    active_fault: Optional[str] = None      # "network-delay", "connection-drop", "batch-job-stall"
    fault_level: Optional[str] = None       # "low", "medium", "high"
    fault_start_sec: Optional[float] = None
    fault_duration_sec: Optional[float] = None
    
    # Public trace data integration
    use_cluster_traces: bool = True
    cluster_trace_path: Optional[str] = None
    bts_profile_path: Optional[str] = None

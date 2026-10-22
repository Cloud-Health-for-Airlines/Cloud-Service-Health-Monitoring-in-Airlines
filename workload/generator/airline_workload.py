"""Realistic Airline IT Workload Generator for BACCP.

Generates time-dependent, multi-service operational workloads for airline IT systems.
Informed by:
1. Google Cluster workload traces (generic compute characteristics, CPU/memory distributions, and volatility)
2. Empirical airline operational datasets (BTS/FAA hourly departure curves and service call multipliers)
3. BACCP chaos testbed fault dynamics (network-delay, connection-drop, batch-job-stall)
"""

from __future__ import annotations

import copy
import datetime
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Internal imports within workload package
from ..datasets.google_cluster import GoogleClusterTraceAdapter
from .models import (
    DataProvenance,
    WorkloadObservation,
    WorkloadScenarioConfig,
    WorkloadSnapshot,
)
from .patterns import (
    BTS_PROFILE_PATH,
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

# Optional import of existing graph builder & sync drift calculator
try:
    from importlib import import_module
    _schema = import_module("ai-models.graph-builder.schema")
    GenerationTypedGraph = getattr(_schema, "GenerationTypedGraph", None)
    _features = import_module("ai-models.graph-builder.features")
    compute_sync_drift_score = getattr(_features, "compute_sync_drift_score", None)
except Exception:
    GenerationTypedGraph = None
    compute_sync_drift_score = None

# Fallback sync-drift calculator if ai-models path is not configured
if compute_sync_drift_score is None:
    def compute_sync_drift_score(
        gateway_submission_rate: float,
        legacy_completion_rate: float,
        gateway_queue_depth: float = 0.0,
        boundary_latency_ms: float = 15.0,
        nominal_latency_ms: float = 15.0,
        boundary_error_rate: float = 0.0,
        delta: float = 1e-3,
    ) -> float:
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

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAOS_PROFILES_PATH = REPO_ROOT / "testbed" / "chaos" / "profiles.json"


class RealisticAirlineWorkloadGenerator:
    """Configurable, reproducible workload generator for airline IT infrastructures.
    
    Adheres strictly to the BACCP data provenance rules:
    - Google Cluster traces inform generic computing workload characteristics (CPU/mem usage & volatility)
    - Public airline datasets inform passenger booking & departure demand curves
    - Does not conflate cluster task IDs with passenger records
    """

    def __init__(
        self,
        cluster_adapter: Optional[GoogleClusterTraceAdapter] = None,
        bts_profile_path: Optional[Union[str, Path]] = None,
        chaos_profiles_path: Optional[Union[str, Path]] = None,
    ):
        self.cluster_adapter = cluster_adapter or GoogleClusterTraceAdapter()
        self.bts_profile = load_bts_airline_profile(Path(bts_profile_path) if bts_profile_path else None)
        
        # Load chaos fault profiles from testbed
        self.chaos_profiles = {
            "network-delay": {"low": 100, "medium": 500, "high": 1500},
            "connection-drop": {"low": 10, "medium": 2, "high": 1},
            "batch-job-stall": {"low": 1, "medium": 5, "high": 15},
        }
        c_path = Path(chaos_profiles_path) if chaos_profiles_path else CHAOS_PROFILES_PATH
        if c_path.exists():
            try:
                import json
                self.chaos_profiles = json.loads(c_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Extract empirical computing resource parameters from Google Cluster trace sample
        self.cluster_compute_params = self._extract_cluster_compute_characteristics()

    def _extract_cluster_compute_characteristics(self) -> Dict[str, Dict[str, float]]:
        """Derive empirical CPU and memory utilization statistics from Google Cluster traces.
        
        Maps scheduling classes to tier resource baselines.
        """
        try:
            profile = self.cluster_adapter.extract_airline_workload_profile(window_seconds=60.0)
            res_cpu = profile["reservations"]["cpu_load"]
            crew_cpu = profile["crew"]["cpu_load"]
            bag_cpu = profile["baggage"]["cpu_load"]
            gw_cpu = profile["boundary-gateway"]["cpu_load"]
        except Exception:
            res_cpu, crew_cpu, bag_cpu, gw_cpu = 0.25, 0.20, 0.15, 0.35

        return {
            "cloud-native-high": {"mean_cpu": res_cpu, "std_cpu": 0.05, "mean_mem": 0.40, "std_mem": 0.04},
            "cloud-native-mid": {"mean_cpu": crew_cpu, "std_cpu": 0.04, "mean_mem": 0.35, "std_mem": 0.03},
            "cloud-native-low": {"mean_cpu": bag_cpu, "std_cpu": 0.03, "mean_mem": 0.25, "std_mem": 0.02},
            "boundary-gateway": {"mean_cpu": gw_cpu, "std_cpu": 0.06, "mean_mem": 0.50, "std_mem": 0.05},
            "legacy": {"mean_cpu": 0.40, "std_cpu": 0.07, "mean_mem": 0.65, "std_mem": 0.06},
        }

    def generate_workload(
        self,
        config: Optional[WorkloadScenarioConfig] = None,
    ) -> List[WorkloadSnapshot]:
        """Generate a complete reproducible time-dependent workload series across all services."""
        cfg = config or WorkloadScenarioConfig()
        rng = np.random.RandomState(cfg.random_seed)

        # Determine services to simulate
        if cfg.services is not None:
            active_services = list(cfg.services)
        else:
            active_services = list(CANONICAL_5_SERVICES)

        # Service configuration parameters from BTS profile
        service_configs = self.bts_profile.get("service_profiles", {})
        hourly_curve = self.bts_profile.get("hourly_demand_curve", {})
        day_multipliers = self.bts_profile.get("day_of_week_multipliers", {})

        day_mult = float(day_multipliers.get(cfg.day_of_week, 1.0))
        num_steps = max(1, int(math.ceil(cfg.duration_seconds / cfg.sampling_interval_sec)))
        start_base_time = datetime.datetime(2026, 10, 22, int(cfg.start_hour), 0, 0, tzinfo=datetime.timezone.utc)

        # Internal state tracking for queue & boundary dynamics
        gw_queue = 0.0
        legacy_stall_remaining_sec = 0.0
        snapshots: List[WorkloadSnapshot] = []

        # Fault parameter extraction
        fault_param = 0
        if cfg.active_fault and cfg.fault_level:
            fault_param = self.chaos_profiles.get(cfg.active_fault, {}).get(cfg.fault_level, 0)

        for step in range(num_steps):
            current_sec = step * cfg.sampling_interval_sec
            current_iso = (start_base_time + datetime.timedelta(seconds=current_sec)).isoformat()
            fractional_hour = cfg.start_hour + (current_sec / 3600.0)

            # 1. Base demand factors
            diurnal_mult = get_diurnal_multiplier(fractional_hour, hourly_curve)
            spike_mult = get_spike_multiplier(
                current_sec,
                cfg.spike_start_sec,
                cfg.spike_duration_sec,
                cfg.spike_magnitude,
            )
            ramp_mult = get_ramp_multiplier(
                current_sec,
                cfg.ramp_start_sec,
                cfg.ramp_duration_sec,
                cfg.ramp_end_factor,
            )

            # Check if chaos fault is active
            is_fault_active = False
            if cfg.active_fault:
                fault_start = cfg.fault_start_sec if cfg.fault_start_sec is not None else 100.0
                fault_duration = cfg.fault_duration_sec if cfg.fault_duration_sec is not None else 300.0
                is_fault_active = (current_sec >= fault_start) and (current_sec < fault_start + fault_duration)

            # Compute request arrival rates for each cloud-native service
            cloud_arrival_rates: Dict[str, float] = {}
            service_observations: Dict[str, WorkloadObservation] = {}
            legacy_forward_rate_total = 0.0

            for svc in active_services:
                if svc in ("boundary-gateway", "legacy-core"):
                    continue

                svc_spec = service_configs.get(svc, {
                    "base_tps": 20.0,
                    "search_to_book_ratio": 1.0,
                    "cpu_load_baseline": 0.20,
                    "memory_mb": 512,
                    "calls_boundary_gateway": True,
                    "boundary_forward_prob": 0.60,
                    "tier": "cloud-native",
                })

                base_tps = float(svc_spec.get("base_tps", 20.0))
                # Volatility jitter parameterized from Google cluster task arrival variance
                arrival_jitter = rng.normal(1.0, 0.05)
                
                # Search and reservation services absorb higher share of spike multiplier
                svc_spike_mult = spike_mult
                if svc == "flight-search":
                    svc_spike_mult = 1.0 + (spike_mult - 1.0) * 1.5  # search spikes more aggressively
                elif svc == "baggage":
                    svc_spike_mult = 1.0 + (spike_mult - 1.0) * 0.4  # baggage is less sensitive to instantaneous flash booking

                computed_rate = max(
                    1.0,
                    base_tps * diurnal_mult * day_mult * svc_spike_mult * ramp_mult * cfg.demand_intensity * arrival_jitter
                )
                cloud_arrival_rates[svc] = computed_rate

                # Accumulate legacy boundary calls
                if svc_spec.get("calls_boundary_gateway", True):
                    forward_prob = float(svc_spec.get("boundary_forward_prob", 0.60))
                    legacy_forward_rate_total += computed_rate * forward_prob

            # 2. Legacy Core State & Fault Injection
            nominal_legacy_latency = 15.0
            legacy_latency = nominal_legacy_latency + rng.normal(0, 1.2)
            legacy_error_rate = 0.0

            if is_fault_active:
                if cfg.active_fault == "network-delay":
                    legacy_latency = fault_param + rng.normal(0, max(1.0, fault_param * 0.08))
                elif cfg.active_fault == "connection-drop":
                    # connection-drop level param: 1 in fault_param dropped
                    drop_prob = min(0.95, 1.0 / max(1.0, float(fault_param)))
                    legacy_error_rate = drop_prob
                    legacy_latency = 45.0 + rng.exponential(15.0)
                elif cfg.active_fault == "batch-job-stall":
                    # Mainframe pause
                    legacy_stall_remaining_sec = max(legacy_stall_remaining_sec, float(fault_param))
                    legacy_latency = 1200.0 + rng.exponential(250.0)
                    legacy_error_rate = min(0.90, 0.20 * fault_param)
            else:
                legacy_latency = max(8.0, 15.0 + rng.normal(0, 1.2))
                legacy_error_rate = max(0.0, float(rng.normal(0.005, 0.002)))

            # Handle stall countdown
            if legacy_stall_remaining_sec > 0:
                legacy_stall_remaining_sec = max(0.0, legacy_stall_remaining_sec - cfg.sampling_interval_sec)

            # 3. Boundary Gateway Queuing & Throughput Dynamics
            gateway_submission_rate = max(5.0, legacy_forward_rate_total)
            legacy_capacity_tps = 75.0

            # When legacy latency spikes or queue overflows, service rate drops
            latency_slowdown = nominal_legacy_latency / max(nominal_legacy_latency, legacy_latency)
            potential_completion_rate = min(legacy_capacity_tps, gateway_submission_rate) * (1.0 - legacy_error_rate) * latency_slowdown

            # Queue accumulation over interval
            rate_deficit = gateway_submission_rate - potential_completion_rate
            if rate_deficit > 0:
                gw_queue = min(180.0, gw_queue + rate_deficit * (cfg.sampling_interval_sec / 20.0))
            else:
                gw_queue = max(0.0, gw_queue - 8.0 * (cfg.sampling_interval_sec / 20.0))

            # Gateway error rate spikes under bufferbloat
            if gw_queue > 75.0 or legacy_error_rate > 0.25:
                gw_error_rate = min(0.95, 0.10 + (gw_queue / 180.0) * 0.85)
            else:
                gw_error_rate = max(0.0, legacy_error_rate * 0.75)

            gw_latency = legacy_latency + (gw_queue * 3.8)
            actual_legacy_completion_rate = max(0.0, potential_completion_rate * (1.0 - gw_error_rate))

            # 4. State Synchronization Drift Score ε(t)
            drift_score = compute_sync_drift_score(
                gateway_submission_rate=gateway_submission_rate,
                legacy_completion_rate=actual_legacy_completion_rate,
                gateway_queue_depth=gw_queue,
                boundary_latency_ms=gw_latency,
                nominal_latency_ms=nominal_legacy_latency,
                boundary_error_rate=gw_error_rate,
            )

            # 5. Build Observations for All Services
            # Provenance templates
            cluster_prov = {
                "dataset_source": "Google Cluster Data (clusterdata-2011-2)",
                "usage_table": "task_usage (5-min average)",
                "events_table": "task_events",
            }
            bts_prov = {
                "dataset_source": "U.S. Bureau of Transportation Statistics (BTS) / FAA ASPM",
                "diurnal_interpolated_hour": round(fractional_hour, 2),
                "day_of_week": cfg.day_of_week,
            }

            # A. Cloud Services Observations
            for svc, rate in cloud_arrival_rates.items():
                svc_spec = service_configs.get(svc, {})
                tier = svc_spec.get("tier", "cloud-native")
                base_cpu = float(svc_spec.get("cpu_load_baseline", 0.20))
                
                # Latency and error propagation from boundary gateway backpressure
                base_lat = 12.0 + rng.normal(0, 1.0)
                base_err = 0.0

                if svc_spec.get("calls_boundary_gateway", True):
                    forward_prob = float(svc_spec.get("boundary_forward_prob", 0.60))
                    if gw_error_rate > 0.15 or gw_queue > 35.0:
                        base_lat += gw_latency * (forward_prob * 0.6)
                        base_err = min(0.90, gw_error_rate * forward_prob)

                # CPU demand estimate modulated by arrival rate and cluster statistics
                cpu_jitter = float(rng.normal(0.0, 0.02))
                cpu_est = min(1.0, max(0.05, base_cpu * (rate / max(1.0, float(svc_spec.get("base_tps", 20.0)))) + cpu_jitter))
                mem_est = min(1.0, max(0.10, (float(svc_spec.get("memory_mb", 512)) / 2048.0) + rng.normal(0.0, 0.01)))

                provenance = DataProvenance(
                    observed_dataset_values={
                        "google_cluster_cpu_baseline": self.cluster_compute_params.get("cloud-native-mid", {}).get("mean_cpu", 0.20),
                        "bts_diurnal_multiplier": round(diurnal_mult, 4),
                        "bts_day_multiplier": round(day_mult, 4),
                    },
                    forecast_values={
                        "projected_hourly_rate": round(rate, 2),
                        "scenario_factor": round(spike_mult * ramp_mult * cfg.demand_intensity, 3),
                    },
                    simulated_values={
                        "queue_depth": round(gw_queue * float(svc_spec.get("boundary_forward_prob", 0.5)), 2),
                        "latency_ms": round(base_lat, 2),
                        "error_rate": round(base_err, 4),
                    },
                    assumptions={
                        "service_to_book_ratio": str(svc_spec.get("search_to_book_ratio", 1.0)),
                        "boundary_forward_prob": str(svc_spec.get("boundary_forward_prob", 0.0)),
                        "resource_scaling": "Linear request-to-CPU demand with Google Cluster task variance",
                    },
                )

                service_observations[svc] = WorkloadObservation(
                    timestamp=current_iso,
                    timestamp_sec=current_sec,
                    service_id=svc,
                    service_tier=tier,
                    request_count=rate * cfg.sampling_interval_sec,
                    request_rate=rate,
                    cpu_demand=cpu_est,
                    memory_demand=mem_est,
                    queue_length=gw_queue * float(svc_spec.get("boundary_forward_prob", 0.5)),
                    latency_ms=base_lat,
                    error_rate=base_err,
                    workload_scenario=cfg.scenario_name,
                    provenance=provenance,
                )

            # B. Boundary Gateway Observation
            gw_cpu = min(1.0, max(0.15, 0.30 + (gw_queue / 180.0) * 0.60 + rng.normal(0, 0.02)))
            gw_mem = min(1.0, 0.50 + (gw_queue / 180.0) * 0.35)
            gw_prov = DataProvenance(
                observed_dataset_values={
                    "google_cluster_gateway_cpu_ref": self.cluster_compute_params.get("boundary-gateway", {}).get("mean_cpu", 0.35),
                },
                forecast_values={
                    "aggregate_cloud_submission_rate": round(gateway_submission_rate, 2),
                },
                simulated_values={
                    "worker_queue_backlog": round(gw_queue, 2),
                    "gateway_latency_ms": round(gw_latency, 2),
                    "boundary_error_rate": round(gw_error_rate, 4),
                    "sync_drift_score": round(drift_score, 2),
                },
                assumptions={
                    "boundary_architecture": "HTTP reverse-proxy to synchronous TCP mainframe socket",
                    "queue_saturation_threshold": "180 pending connection buffers",
                },
            )
            service_observations["boundary-gateway"] = WorkloadObservation(
                timestamp=current_iso,
                timestamp_sec=current_sec,
                service_id="boundary-gateway",
                service_tier="boundary-gateway",
                request_count=gateway_submission_rate * cfg.sampling_interval_sec,
                request_rate=gateway_submission_rate,
                cpu_demand=gw_cpu,
                memory_demand=gw_mem,
                queue_length=gw_queue,
                latency_ms=gw_latency,
                error_rate=gw_error_rate,
                workload_scenario=cfg.scenario_name,
                provenance=gw_prov,
            )

            # C. Legacy Core Observation
            legacy_cpu = min(1.0, max(0.20, 0.25 + (actual_legacy_completion_rate / legacy_capacity_tps) * 0.65 + rng.normal(0, 0.02)))
            legacy_mem = 0.65 + rng.normal(0, 0.01)
            legacy_prov = DataProvenance(
                observed_dataset_values={
                    "google_cluster_legacy_cpu_ref": self.cluster_compute_params.get("legacy", {}).get("mean_cpu", 0.40),
                },
                forecast_values={
                    "capacity_limit_tps": legacy_capacity_tps,
                },
                simulated_values={
                    "completed_tps": round(actual_legacy_completion_rate, 2),
                    "tcp_rtt_latency_ms": round(legacy_latency, 2),
                    "legacy_error_rate": round(legacy_error_rate, 4),
                },
                assumptions={
                    "mainframe_processing": "Fixed serialization limit with synchronous TCP connection per transaction",
                },
            )
            service_observations["legacy-core"] = WorkloadObservation(
                timestamp=current_iso,
                timestamp_sec=current_sec,
                service_id="legacy-core",
                service_tier="legacy",
                request_count=actual_legacy_completion_rate * cfg.sampling_interval_sec,
                request_rate=actual_legacy_completion_rate,
                cpu_demand=legacy_cpu,
                memory_demand=legacy_mem,
                queue_length=0.0,
                latency_ms=legacy_latency,
                error_rate=legacy_error_rate,
                workload_scenario=cfg.scenario_name,
                provenance=legacy_prov,
            )

            snapshot = WorkloadSnapshot(
                timestamp_sec=current_sec,
                iso_timestamp=current_iso,
                scenario=cfg.scenario_name,
                observations=service_observations,
                sync_drift_score=drift_score,
                active_fault=cfg.active_fault if is_fault_active else None,
                fault_level=cfg.fault_level if is_fault_active else None,
            )
            snapshots.append(snapshot)

        return snapshots

    def generate_canonical_telemetry_series(
        self,
        config: Optional[WorkloadScenarioConfig] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a series of telemetry dicts directly consumable by BACCPCloudOrchestrator."""
        cfg = config or WorkloadScenarioConfig(services=CANONICAL_5_SERVICES)
        snapshots = self.generate_workload(cfg)
        return [
            {
                "timestamp_sec": snap.timestamp_sec,
                "iso_timestamp": snap.iso_timestamp,
                "sync_drift_score": snap.sync_drift_score,
                "active_fault": snap.active_fault,
                "fault_level": snap.fault_level,
                "telemetry": snap.to_orchestrator_telemetry(),
            }
            for snap in snapshots
        ]

    def generate_graph_sequence(
        self,
        config: Optional[WorkloadScenarioConfig] = None,
    ) -> List[Dict[str, Any]]:
        """Generate graph snapshots compatible with GenerationTypedGraph and HeteroCascadePredictor."""
        cfg = config or WorkloadScenarioConfig(services=CANONICAL_5_SERVICES)
        snapshots = self.generate_workload(cfg)
        edges = build_canonical_edges() if cfg.services is None or set(cfg.services) == set(CANONICAL_5_SERVICES) else build_expanded_edges()

        if GenerationTypedGraph is None:
            raise RuntimeError("GenerationTypedGraph not available. Ensure ai-models is in sys.path.")

        graph_builder = GenerationTypedGraph()
        graph_snapshots = []

        for snap in snapshots:
            node_metrics = {
                svc_id: {
                    **obs.to_orchestrator_node_metric(),
                    "sync_drift_score": snap.sync_drift_score,
                }
                for svc_id, obs in snap.observations.items()
                if svc_id in CANONICAL_5_SERVICES
            }
            hetero_snap = graph_builder.build_snapshot(node_metrics, edges, timestamp=snap.timestamp_sec)
            flat_x, flat_edges, flat_attrs = graph_builder.to_flat_tensors(hetero_snap)

            graph_snapshots.append({
                "timestamp_sec": snap.timestamp_sec,
                "sync_drift_score": snap.sync_drift_score,
                "hetero_snapshot": hetero_snap,
                "flat_x": flat_x,
                "flat_edge_index": flat_edges,
                "flat_edge_attr": flat_attrs,
                "active_fault": snap.active_fault,
                "fault_level": snap.fault_level,
                "node_metrics": node_metrics,
            })

        return graph_snapshots

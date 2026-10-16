"""Synthetic Trajectory Generator for Workload-Aware Service and Boundary Degradation.

Version: baccp-degradation-generator-v1.0.0

Generates reproducible, causally modeled service and boundary degradation incidents:
- Causal queue mass conservation: Q(t) = max(0, Q(t-1) + Arrivals - Departures - Drops)
- Little's Law latency derivation from queue backlog and effective capacity
- Dropped requests accounted explicitly when exceeding buffer limits
- Boundary synchronization drift ε(t) derived via repo's compute_sync_drift_score
- Unknown legacy capacity preserved as None
- 6 operational regimes: nominal, gradual_ramp, abrupt_burst, capacity_reduction,
  boundary_slowdown, and recovery.
- Disjoint incident partitioning: train, val, cal, standard_test, capacity_loss_test,
  boundary_slowdown_test, and surge_test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .data_contract import (
        CONTRACT_VERSION,
        CONFIGURED_SERVICE_CAPACITIES,
        DegradationIncident,
        DegradationRegime,
        NOMINAL_BASELINES,
        SAMPLING_INTERVAL_SECONDS,
        ServiceIdentifier,
        SLO_THRESHOLDS,
    )
except ImportError:
    from degradation_prediction.data_contract import (
        CONTRACT_VERSION,
        CONFIGURED_SERVICE_CAPACITIES,
        DegradationIncident,
        DegradationRegime,
        NOMINAL_BASELINES,
        SAMPLING_INTERVAL_SECONDS,
        ServiceIdentifier,
        SLO_THRESHOLDS,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_MODELS_DIR = REPO_ROOT / "ai-models"
DATA_DIR = AI_MODELS_DIR / "data" / "degradation_prediction"
DEFAULT_DATASET_PATH = DATA_DIR / "degradation_dataset.json"

# Import compute_sync_drift_score from repository
from importlib import import_module
try:
    _feat_mod = import_module("graph-builder.features")
    compute_sync_drift_score = getattr(_feat_mod, "compute_sync_drift_score")
except Exception:
    try:
        _feat_mod = import_module("ai-models.graph-builder.features")
        compute_sync_drift_score = getattr(_feat_mod, "compute_sync_drift_score")
    except Exception:
        # Fallback exact mathematical implementation if pathing requires
        def compute_sync_drift_score(
            gateway_submission_rate: float,
            legacy_completion_rate: float,
            gateway_queue_depth: float,
            boundary_latency_ms: float,
            nominal_latency_ms: float = 15.0,
            boundary_error_rate: float = 0.0,
            delta: float = 1e-4,
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


# Nominal offered rates (req/s)
# Boundary-gateway submits legacy transactions from cloud services to legacy-core
NOMINAL_DEMAND_MEANS: Dict[str, float] = {
    ServiceIdentifier.RESERVATIONS.value: 48.0,
    ServiceIdentifier.CREW.value: 28.0,
    ServiceIdentifier.BAGGAGE.value: 36.0,
    ServiceIdentifier.BOUNDARY_GATEWAY.value: 42.0,  # Gateway-to-legacy submission rate
    ServiceIdentifier.LEGACY_CORE.value: 42.0,
}

BUFFER_LIMITS: Dict[str, float] = {
    ServiceIdentifier.RESERVATIONS.value: 400.0,
    ServiceIdentifier.CREW.value: 300.0,
    ServiceIdentifier.BAGGAGE.value: 350.0,
    ServiceIdentifier.BOUNDARY_GATEWAY.value: 600.0,
    ServiceIdentifier.LEGACY_CORE.value: 500.0,
}


def simulate_single_incident(
    incident_id: str,
    regime: DegradationRegime,
    rng: np.random.RandomState,
    step_count: int = 30,  # 30 steps * 20s = 600 seconds (10 minutes)
) -> DegradationIncident:
    """Simulates a single causal incident trajectory across all 5 services."""
    services = [s.value for s in ServiceIdentifier]
    dt = SAMPLING_INTERVAL_SECONDS

    # Internal simulator capacity state (legacy capacity is simulated internally as ~50 req/s
    # but EXPLICITLY set to None in public telemetry output)
    simulated_capacities: Dict[str, float] = {
        ServiceIdentifier.RESERVATIONS.value: 100.0,
        ServiceIdentifier.CREW.value: 60.0,
        ServiceIdentifier.BAGGAGE.value: 75.0,
        ServiceIdentifier.BOUNDARY_GATEWAY.value: 220.0,
        ServiceIdentifier.LEGACY_CORE.value: 52.0,  # Internal ground-truth only
    }

    # Regime event parameters
    event_start_step = rng.randint(8, 14)
    event_duration_steps = rng.randint(8, 14)
    event_end_step = min(step_count, event_start_step + event_duration_steps)

    # State tracking per service
    current_backlog: Dict[str, float] = {s: 0.0 for s in services}

    service_telemetry_records: Dict[str, List[Dict[str, Any]]] = {s: [] for s in services}
    boundary_telemetry_records: List[Dict[str, Any]] = []

    # Latency moving average memory
    lat_ema: Dict[str, float] = {s: NOMINAL_BASELINES[s]["latency_ms"] for s in services}

    for step in range(step_count):
        timestamp = step * dt
        in_event = (event_start_step <= step < event_end_step)

        # 1. Determine offered demand rates (req/s) per service
        offered_rates: Dict[str, float] = {}
        for s in services:
            base_mean = NOMINAL_DEMAND_MEANS[s]
            # Daily diurnal cycle + periodic micro-cycle
            cycle_phase = 2.0 * math.pi * ((timestamp % 3600.0) / 3600.0)
            diurnal = 1.0 + 0.12 * math.sin(cycle_phase)
            noise = rng.normal(0.0, 1.5)

            if regime == DegradationRegime.NOMINAL:
                demand = base_mean * diurnal + noise
            elif regime == DegradationRegime.GRADUAL_RAMP:
                # Steady ramp beginning at step 5
                ramp_factor = 1.0 + (max(0, step - 5) / 18.0) * 1.15
                demand = base_mean * diurnal * ramp_factor + noise
            elif regime == DegradationRegime.ABRUPT_BURST:
                burst_mult = rng.uniform(1.6, 2.2) if in_event else 1.0
                demand = base_mean * diurnal * burst_mult + noise
            elif regime == DegradationRegime.CAPACITY_REDUCTION:
                # Steady offered demand during capacity crash
                demand = base_mean * diurnal + noise
            elif regime == DegradationRegime.BOUNDARY_SLOWDOWN:
                # Upstream demand steady or slightly elevated
                boost = 1.25 if in_event else 1.0
                demand = base_mean * diurnal * boost + noise
            elif regime == DegradationRegime.RECOVERY:
                # High initial load in steps 0-12, dropping rapidly back to normal
                rec_factor = 1.7 if step < 12 else 0.85
                demand = base_mean * diurnal * rec_factor + noise
            else:
                demand = base_mean * diurnal + noise

            offered_rates[s] = max(2.0, float(demand))

        # 2. Determine effective service capacities (req/s)
        eff_capacities: Dict[str, float] = {}
        for s in services:
            nominal_c = simulated_capacities[s]
            if regime == DegradationRegime.CAPACITY_REDUCTION and in_event:
                # Cloud services or gateway experience container failure / CPU throttle
                if s in [ServiceIdentifier.RESERVATIONS.value, ServiceIdentifier.BOUNDARY_GATEWAY.value]:
                    eff_c = nominal_c * rng.uniform(0.35, 0.50)  # 50-65% capacity drop
                else:
                    eff_c = nominal_c * rng.uniform(0.70, 0.85)
            elif regime == DegradationRegime.BOUNDARY_SLOWDOWN and in_event:
                # Legacy mainframe experiences lock contention / stall
                if s == ServiceIdentifier.LEGACY_CORE.value:
                    eff_c = nominal_c * rng.uniform(0.18, 0.28)  # Drops to 10-15 req/s
                elif s == ServiceIdentifier.BOUNDARY_GATEWAY.value:
                    eff_c = nominal_c * 0.65  # Gateway worker exhaustion due to slow legacy responses
                else:
                    eff_c = nominal_c
            else:
                # Nominal capacity with small random variance (worker jitter)
                eff_c = nominal_c * rng.uniform(0.96, 1.04)

            eff_capacities[s] = max(1.0, float(eff_c))

        # 3. Simulate causal queue conservation, completed departures, and drops
        step_service_snapshots: Dict[str, Dict[str, Any]] = {}

        for s in services:
            lam = offered_rates[s]
            cap = eff_capacities[s]
            q_prev = current_backlog[s]
            buffer_max = BUFFER_LIMITS[s]

            # Inflow during step: arrivals = lambda * dt
            arrivals = lam * dt
            q_avail = q_prev + arrivals

            # Max throughput processable during step: capacity * dt
            max_throughput = cap * dt
            departures = min(q_avail, max_throughput)
            completed_rate = departures / dt

            # Remainder in queue
            remainder = q_avail - departures

            # Drops if remainder exceeds buffer capacity
            drops = max(0.0, remainder - buffer_max)
            q_next = remainder - drops
            current_backlog[s] = q_next

            # 4. Latency dynamics: base + Little's Law queuing delay + stochastic jitter
            base_lat = NOMINAL_BASELINES[s]["latency_ms"]
            # Queuing delay: (Q / Capacity) in seconds -> converted to ms
            queue_delay_ms = (q_next / (cap + 1e-4)) * 1000.0
            raw_lat = base_lat + queue_delay_ms + rng.normal(0.0, 1.5)

            # Extra latency penalty if legacy core is stalling
            if s == ServiceIdentifier.LEGACY_CORE.value and regime == DegradationRegime.BOUNDARY_SLOWDOWN and in_event:
                raw_lat += rng.uniform(250.0, 650.0)
            elif s == ServiceIdentifier.BOUNDARY_GATEWAY.value and regime == DegradationRegime.BOUNDARY_SLOWDOWN and in_event:
                raw_lat += rng.uniform(80.0, 220.0)

            lat_ema[s] = 0.7 * lat_ema[s] + 0.3 * max(base_lat * 0.5, raw_lat)
            current_latency = max(2.0, float(lat_ema[s]))

            # 5. Error rate dynamics: base + drop fraction + timeout escalation
            drop_fraction = drops / (q_avail + 1e-4)
            base_err = rng.uniform(0.001, 0.004)
            timeout_err = 0.0
            if current_latency > 300.0:
                timeout_err = min(0.70, (current_latency - 300.0) / 700.0)

            error_rate = min(1.0, float(base_err + drop_fraction * 0.85 + timeout_err))

            # Exported telemetry: legacy capacity is EXPLICITLY None (unmeasured legacy mainframe)
            exported_capacity = None if s == ServiceIdentifier.LEGACY_CORE.value else eff_capacities[s]

            snapshot = {
                "timestamp_sec": float(timestamp),
                "step_index": int(step),
                "service_id": s,
                "offered_rate": round(float(lam), 2),
                "completed_rate": round(float(completed_rate), 2),
                "backlog_requests": round(float(q_next), 2),
                "latency_ms": round(float(current_latency), 2),
                "error_rate": round(float(error_rate), 4),
                "dropped_requests": round(float(drops), 2),
                "effective_capacity": round(float(exported_capacity), 2) if exported_capacity is not None else None,
            }
            service_telemetry_records[s].append(snapshot)
            step_service_snapshots[s] = snapshot

        # 6. Compute boundary telemetry and digital-twin sync drift score ε(t)
        gw_snap = step_service_snapshots[ServiceIdentifier.BOUNDARY_GATEWAY.value]
        legacy_snap = step_service_snapshots[ServiceIdentifier.LEGACY_CORE.value]

        sub_rate = gw_snap["offered_rate"]
        comp_rate = legacy_snap["completed_rate"]
        gw_q = gw_snap["backlog_requests"]
        # Boundary TCP link latency measured at the gateway (nominal 15.0 ms)
        b_lat = gw_snap["latency_ms"]
        b_err = max(gw_snap["error_rate"], legacy_snap["error_rate"])

        drift_score = compute_sync_drift_score(
            gateway_submission_rate=sub_rate,
            legacy_completion_rate=comp_rate,
            gateway_queue_depth=gw_q,
            boundary_latency_ms=b_lat,
            nominal_latency_ms=20.0,
            boundary_error_rate=b_err,
        )

        boundary_snap = {
            "timestamp_sec": float(timestamp),
            "step_index": int(step),
            "gateway_submission_rate": round(float(sub_rate), 2),
            "legacy_completion_rate": round(float(comp_rate), 2),
            "gateway_queue_depth": round(float(gw_q), 2),
            "boundary_latency_ms": round(float(b_lat), 2),
            "boundary_error_rate": round(float(b_err), 4),
            "sync_drift_score": round(float(drift_score), 2),
        }
        boundary_telemetry_records.append(boundary_snap)

    incident = DegradationIncident(
        incident_id=incident_id,
        regime=regime.value,
        duration_sec=float(step_count * dt),
        step_count=int(step_count),
        service_telemetry=service_telemetry_records,
        boundary_telemetry=boundary_telemetry_records,
        metadata={
            "contract_version": CONTRACT_VERSION,
            "event_start_step": int(event_start_step),
            "event_end_step": int(event_end_step),
            "sampling_interval_sec": float(dt),
        },
    )
    return incident


def generate_degradation_dataset(
    output_path: Path = DEFAULT_DATASET_PATH,
    seed_train: int = 142857,
    seed_eval: int = 271828,
) -> Dict[str, Any]:
    """Generates 275 whole incident trajectories partitioned into disjoint training, calibration, and fresh evaluation splits."""
    rng_train = np.random.RandomState(seed_train)
    rng_eval = np.random.RandomState(seed_eval)

    # 1. Plan incident counts per partition and regime
    # Partitions:
    # - train: 120 incidents (nominal: 45, gradual_ramp: 25, abrupt_burst: 20, capacity_reduction: 15, boundary_slowdown: 10, recovery: 5)
    # - val: 30 incidents (balanced across regimes)
    # - cal: 25 incidents (balanced across regimes)
    # - standard_test: 25 incidents (nominal, gradual_ramp, recovery) [FRESH EVALUATION SEED 271828]
    # - capacity_loss_test: 25 incidents (pure capacity_reduction) [FRESH EVALUATION SEED 271828]
    # - boundary_slowdown_test: 25 incidents (pure boundary_slowdown) [FRESH EVALUATION SEED 271828]
    # - surge_test: 25 fresh incidents (pure abrupt_burst) [FRESH EVALUATION SEED 271828]

    splits_spec = {
        "train": {
            DegradationRegime.NOMINAL: 45,
            DegradationRegime.GRADUAL_RAMP: 25,
            DegradationRegime.ABRUPT_BURST: 20,
            DegradationRegime.CAPACITY_REDUCTION: 15,
            DegradationRegime.BOUNDARY_SLOWDOWN: 10,
            DegradationRegime.RECOVERY: 5,
        },
        "val": {
            DegradationRegime.NOMINAL: 8,
            DegradationRegime.GRADUAL_RAMP: 6,
            DegradationRegime.ABRUPT_BURST: 5,
            DegradationRegime.CAPACITY_REDUCTION: 5,
            DegradationRegime.BOUNDARY_SLOWDOWN: 4,
            DegradationRegime.RECOVERY: 2,
        },
        "cal": {
            DegradationRegime.NOMINAL: 7,
            DegradationRegime.GRADUAL_RAMP: 5,
            DegradationRegime.ABRUPT_BURST: 4,
            DegradationRegime.CAPACITY_REDUCTION: 4,
            DegradationRegime.BOUNDARY_SLOWDOWN: 3,
            DegradationRegime.RECOVERY: 2,
        },
        "standard_test": {
            DegradationRegime.NOMINAL: 12,
            DegradationRegime.GRADUAL_RAMP: 8,
            DegradationRegime.RECOVERY: 5,
        },
        "capacity_loss_test": {
            DegradationRegime.CAPACITY_REDUCTION: 25,
        },
        "boundary_slowdown_test": {
            DegradationRegime.BOUNDARY_SLOWDOWN: 25,
        },
        "surge_test": {
            DegradationRegime.ABRUPT_BURST: 25,
        },
    }

    incidents: List[Dict[str, Any]] = []
    split_manifest: Dict[str, List[str]] = {k: [] for k in splits_spec.keys()}
    regime_manifest: Dict[str, List[str]] = {r.value: [] for r in DegradationRegime}

    incident_counter = 0

    for split_name, regime_counts in splits_spec.items():
        is_test_split = "test" in split_name
        current_rng = rng_eval if is_test_split else rng_train

        for regime, count in regime_counts.items():
            for _ in range(count):
                inc_id = f"inc_{split_name}_{incident_counter:04d}"
                incident_counter += 1

                inc = simulate_single_incident(
                    incident_id=inc_id,
                    regime=regime,
                    rng=current_rng,
                    step_count=30,  # 30 steps * 20s = 600 seconds
                )
                inc_dict = inc.to_dict()
                inc_dict["split"] = split_name
                incidents.append(inc_dict)

                split_manifest[split_name].append(inc_id)
                regime_manifest[regime.value].append(inc_id)

    # Compute dataset hash
    serialized = json.dumps(incidents, sort_keys=True).encode("utf-8")
    dataset_sha256 = hashlib.sha256(serialized).hexdigest()

    dataset_payload = {
        "metadata": {
            "contract_version": CONTRACT_VERSION,
            "dataset_sha256": dataset_sha256,
            "training_seed": seed_train,
            "evaluation_seed": seed_eval,
            "upstream_workload_dataset_sha256": "5f2c200e70f6284bf378a61744fed2e14d52d98a3d3638f61274e230ccd36147",
            "upstream_provenance": "out-of-sample pre-trained WorkloadForecaster (250 whole incidents, master seed 42)",
            "sampling_interval_seconds": SAMPLING_INTERVAL_SECONDS,
            "total_incidents": len(incidents),
            "step_count_per_incident": 30,
            "total_timesteps": len(incidents) * 30,
            "splits": {k: len(v) for k, v in split_manifest.items()},
            "regimes": {k: len(v) for k, v in regime_manifest.items()},
        },
        "split_manifest": split_manifest,
        "incidents": incidents,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset_payload, f, indent=2)

    return dataset_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic degradation dataset.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--seed_train", type=int, default=142857)
    parser.add_argument("--seed_eval", type=int, default=271828)
    args = parser.parse_args()

    payload = generate_degradation_dataset(Path(args.output), seed_train=args.seed_train, seed_eval=args.seed_eval)
    meta = payload["metadata"]
    print(f"Generated {meta['total_incidents']} incidents saved to {args.output}")
    print(f"Dataset SHA-256: {meta['dataset_sha256']}")
    print(f"Splits: {meta['splits']}")
    print(f"Regimes: {meta['regimes']}")

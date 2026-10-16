"""Deterministic, CPU-friendly Workload Data Generator for BACCP.

Produces multi-service offered workload and completed throughput time-series trajectories
under realistic airline IT operational regimes with zero Docker/kernel dependencies.

Key Properties:
- 20-second sampling interval (Δt = 20s).
- Trajectory duration: 90 steps (1,800s = 30 minutes).
- Defined micro-cycle seasonal period: S = 15 steps (300s = 5 minutes, simulator assumption).
- Explicit incident-level partitioning:
  * Train: 120 incidents (standard regime)
  * Validation: 30 incidents (standard regime)
  * Calibration: 25 incidents (standard regime)
  * Test Standard: 25 incidents (standard regime)
  * Test Unseen Surge Diagnostic: 25 incidents (examined diagnostic surge)
  * Test Unseen Surge Locked: 25 incidents (fresh, independently seeded locked surge evaluation)
- Explicit distinction between incoming offered demand, completed throughput,
  signed capacity margin, available headroom, and overload.
- Provenance strictly labeled as synthetic throughout.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data_contract import (
    CONFIGURED_SERVICE_CAPACITIES,
    CONTRACT_VERSION,
    SERVICE_GENERATION_MAPPING,
    GenerationType,
    IncidentTrajectory,
    ServiceIdentifier,
    WorkloadRecord,
    validate_workload_record,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DATA_DIR = REPO_ROOT / "ai-models" / "data" / "workload_forecasting"
OUTPUT_RESULTS_DIR = REPO_ROOT / "results" / "workload_forecasting"

# Scenario definitions and nominal distributions
SCENARIOS = {
    "steady_nominal": {"weight": 0.35, "intensity_range": (0.85, 1.15)},
    "periodic_wave": {"weight": 0.25, "intensity_range": (0.90, 1.25)},
    "morning_ramp": {"weight": 0.20, "intensity_range": (1.10, 1.45)},
    "burst_surge": {"weight": 0.20, "intensity_range": (1.30, 1.75)},
}

UNSEEN_SCENARIO = "heldout_extreme_surge"
UNSEEN_INTENSITY_RANGE = (2.20, 2.85)

# Service base demand parameters (requests/second)
SERVICE_BASE_DEMAND = {
    ServiceIdentifier.RESERVATIONS.value: {"mean": 45.0, "std": 4.5, "noise_std": 2.2},
    ServiceIdentifier.CREW.value: {"mean": 25.0, "std": 2.5, "noise_std": 1.4},
    ServiceIdentifier.BAGGAGE.value: {"mean": 35.0, "std": 3.0, "noise_std": 1.8},
}


def compute_gateway_workload(res_offered: float, crew_offered: float, bag_offered: float, noise: float) -> float:
    """Aggregates downstream services with routing and protocol framing overhead."""
    aggregated = (res_offered * 0.95) + (crew_offered * 0.85) + (bag_offered * 0.70)
    overhead = 6.0  # health probes, heartbeat, TLS renegotiation
    return max(5.0, aggregated + overhead + noise)


def simulate_single_incident(
    incident_id: str,
    scenario_name: str,
    scenario_intensity: float,
    seed: int,
    total_steps: int = 90,
    sampling_interval_sec: float = 20.0,
    seasonal_period_steps: int = 15,
) -> IncidentTrajectory:
    """Deterministically simulates a 90-step trajectory for all 5 services."""
    rng = np.random.RandomState(seed)

    records_by_service: Dict[str, List[WorkloadRecord]] = {
        svc: [] for svc in [s.value for s in ServiceIdentifier]
    }

    # Pre-generate common temporal profile multipliers for step t in [0, total_steps)
    temporal_profile = np.ones(total_steps, dtype=np.float64)

    if scenario_name == "steady_nominal":
        # Slow gentle drift via low-order autoregressive modulation
        ar = 0.0
        for t in range(total_steps):
            ar = 0.85 * ar + rng.normal(0, 0.03)
            temporal_profile[t] = 1.0 + ar

    elif scenario_name == "periodic_wave":
        # 5-minute cyclic wave (15 steps) with minor second harmonic
        for t in range(total_steps):
            angle = 2.0 * math.pi * (t % seasonal_period_steps) / seasonal_period_steps
            wave = 0.22 * math.sin(angle) + 0.08 * math.cos(2.0 * angle)
            temporal_profile[t] = 1.0 + wave

    elif scenario_name == "morning_ramp":
        # Smooth sigmoidal ramp up across first 45 steps (+50%), then stabilizes
        ramp_center = total_steps * 0.35
        ramp_width = total_steps * 0.15
        for t in range(total_steps):
            sigmoid = 1.0 / (1.0 + math.exp(- (t - ramp_center) / max(1.0, ramp_width)))
            temporal_profile[t] = 0.85 + (0.55 * sigmoid)

    elif scenario_name == "burst_surge":
        # Abrupt flash surge at random onset t0 in [28, 44] lasting 6-11 steps
        burst_start = int(rng.uniform(28, 44))
        burst_duration = int(rng.uniform(6, 11))
        burst_peak = rng.uniform(0.70, 1.10)
        for t in range(total_steps):
            if burst_start <= t < burst_start + burst_duration:
                progress = (t - burst_start) / max(1, burst_duration)
                temporal_profile[t] = 1.0 + burst_peak * math.sin(math.pi * progress)
            elif t >= burst_start + burst_duration:
                steps_after = t - (burst_start + burst_duration)
                decay = math.exp(-steps_after / 8.0)
                temporal_profile[t] = 1.0 + (burst_peak * 0.3 * decay)
            else:
                temporal_profile[t] = 1.0

    elif scenario_name == "heldout_extreme_surge":
        # Extreme compound regime: rapid double burst with heavy saturation
        burst1_start = 22
        burst2_start = 52
        for t in range(total_steps):
            mult = 1.0
            if burst1_start <= t < burst1_start + 14:
                mult += 1.40 * math.sin(math.pi * (t - burst1_start) / 14.0)
            if burst2_start <= t < burst2_start + 18:
                mult += 1.65 * math.sin(math.pi * (t - burst2_start) / 18.0)
            temporal_profile[t] = mult

    # Apply scenario intensity scaling
    temporal_profile *= scenario_intensity

    # Step-by-step synthesis across all 5 services
    for t in range(total_steps):
        elapsed_sec = t * sampling_interval_sec
        profile_mult = max(0.2, float(temporal_profile[t]))

        # 1. Cloud microservices: Reservations, Crew, Baggage
        res_cfg = SERVICE_BASE_DEMAND[ServiceIdentifier.RESERVATIONS.value]
        res_noise = rng.normal(0, res_cfg["noise_std"])
        res_offered = max(2.0, (res_cfg["mean"] * profile_mult) + res_noise)

        crew_cfg = SERVICE_BASE_DEMAND[ServiceIdentifier.CREW.value]
        crew_noise = rng.normal(0, crew_cfg["noise_std"])
        crew_offered = max(1.5, (crew_cfg["mean"] * profile_mult) + crew_noise)

        bag_cfg = SERVICE_BASE_DEMAND[ServiceIdentifier.BAGGAGE.value]
        bag_noise = rng.normal(0, bag_cfg["noise_std"])
        bag_offered = max(2.0, (bag_cfg["mean"] * profile_mult) + bag_noise)

        # 2. Boundary Gateway
        gw_noise = rng.normal(0, 3.5)
        gw_offered = compute_gateway_workload(res_offered, crew_offered, bag_offered, gw_noise)

        # 3. Legacy Core
        legacy_offered = gw_offered * 0.98 + rng.normal(0, 1.2)
        legacy_offered = max(2.0, legacy_offered)

        offered_by_svc = {
            ServiceIdentifier.RESERVATIONS.value: res_offered,
            ServiceIdentifier.CREW.value: crew_offered,
            ServiceIdentifier.BAGGAGE.value: bag_offered,
            ServiceIdentifier.BOUNDARY_GATEWAY.value: gw_offered,
            ServiceIdentifier.LEGACY_CORE.value: legacy_offered,
        }

        for svc_id, offered_rps in offered_by_svc.items():
            cap_rps = CONFIGURED_SERVICE_CAPACITIES[svc_id]
            gen_type = SERVICE_GENERATION_MAPPING[svc_id].value

            if cap_rps is not None:
                # Throughput is bounded by capacity with graceful saturation curve
                if offered_rps <= cap_rps:
                    completed_rps = offered_rps * (1.0 - 0.005 * rng.uniform(0, 1))
                else:
                    excess = offered_rps - cap_rps
                    completed_rps = cap_rps + (excess * 0.12)  # severely throttled
                
                margin_rps = cap_rps - offered_rps
                headroom_rps = max(0.0, margin_rps)
                overload_rps = max(0.0, -margin_rps)
                utilization = offered_rps / cap_rps
                capacity_type = "configured_simulation_assumption"
            else:
                completed_rps = offered_rps * 0.99
                margin_rps = None
                headroom_rps = None
                overload_rps = None
                utilization = None
                capacity_type = "unknown_uninstrumented"

            record = WorkloadRecord(
                incident_id=incident_id,
                service_id=svc_id,
                generation_type=gen_type,
                step_index=t,
                elapsed_time_sec=round(elapsed_sec, 1),
                sampling_interval_sec=sampling_interval_sec,
                offered_workload_rps=round(offered_rps, 3),
                completed_throughput_rps=round(completed_rps, 3),
                configured_capacity_rps=cap_rps,
                capacity_margin_rps=round(margin_rps, 3) if margin_rps is not None else None,
                available_headroom_rps=round(headroom_rps, 3) if headroom_rps is not None else None,
                overload_rps=round(overload_rps, 3) if overload_rps is not None else None,
                capacity_utilization=round(utilization, 4) if utilization is not None else None,
                capacity_type=capacity_type,
                data_provenance="synthetic",
                scenario_name=scenario_name,
                scenario_intensity=round(scenario_intensity, 3),
            )
            validate_workload_record(record)
            records_by_service[svc_id].append(record)

    return IncidentTrajectory(
        incident_id=incident_id,
        scenario_name=scenario_name,
        scenario_intensity=round(scenario_intensity, 3),
        seed=seed,
        total_steps=total_steps,
        sampling_interval_sec=sampling_interval_sec,
        duration_seconds=total_steps * sampling_interval_sec,
        records_by_service=records_by_service,
    )


def generate_workload_dataset(
    num_standard_incidents: int = 200,
    num_diagnostic_surge_incidents: int = 25,
    num_locked_surge_incidents: int = 25,
    master_seed: int = 42,
    total_steps: int = 90,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generates the full workload dataset and split manifest."""
    random.seed(master_seed)
    np.random.seed(master_seed)

    standard_scenarios = list(SCENARIOS.keys())
    standard_weights = [SCENARIOS[s]["weight"] for s in standard_scenarios]

    all_trajectories: List[IncidentTrajectory] = []

    # 1. Standard incidents (200 incidents)
    for i in range(num_standard_incidents):
        inc_id = f"inc_{i:04d}"
        chosen_scenario = random.choices(standard_scenarios, weights=standard_weights, k=1)[0]
        min_int, max_int = SCENARIOS[chosen_scenario]["intensity_range"]
        intensity = random.uniform(min_int, max_int)
        inc_seed = master_seed + 1000 + i

        traj = simulate_single_incident(
            incident_id=inc_id,
            scenario_name=chosen_scenario,
            scenario_intensity=intensity,
            seed=inc_seed,
            total_steps=total_steps,
        )
        all_trajectories.append(traj)

    # 2. Diagnostic unseen extreme surge incidents (25 incidents, original Task 1 surge set)
    for j in range(num_diagnostic_surge_incidents):
        inc_id = f"inc_unseen_{j:04d}"
        intensity = random.uniform(*UNSEEN_INTENSITY_RANGE)
        inc_seed = master_seed + 5000 + j

        traj = simulate_single_incident(
            incident_id=inc_id,
            scenario_name=UNSEEN_SCENARIO,
            scenario_intensity=intensity,
            seed=inc_seed,
            total_steps=total_steps,
        )
        all_trajectories.append(traj)

    # 3. Fresh, independently seeded, locked unseen extreme surge incidents (25 incidents)
    for k in range(num_locked_surge_incidents):
        inc_id = f"inc_surge_locked_{k:04d}"
        intensity = random.uniform(*UNSEEN_INTENSITY_RANGE)
        inc_seed = master_seed + 9000 + k

        traj = simulate_single_incident(
            incident_id=inc_id,
            scenario_name=UNSEEN_SCENARIO,
            scenario_intensity=intensity,
            seed=inc_seed,
            total_steps=total_steps,
        )
        all_trajectories.append(traj)

    standard_ids = [t.incident_id for t in all_trajectories[:num_standard_incidents]]
    diagnostic_ids = [
        t.incident_id for t in all_trajectories[num_standard_incidents : num_standard_incidents + num_diagnostic_surge_incidents]
    ]
    locked_ids = [
        t.incident_id for t in all_trajectories[num_standard_incidents + num_diagnostic_surge_incidents :]
    ]

    train_ids = standard_ids[:120]
    val_ids = standard_ids[120:150]
    cal_ids = standard_ids[150:175]
    test_std_ids = standard_ids[175:200]

    split_manifest = {
        "dataset_name": "BACCP-Workload-Forecasting-Dataset",
        "contract_version": CONTRACT_VERSION,
        "generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "master_seed": master_seed,
        "data_provenance": "synthetic",
        "simulation_assumptions": {
            "micro_cycle_period_steps": 15,
            "micro_cycle_period_seconds": 300.0,
            "micro_cycle_note": "5-minute periodic micro-cycle assumption representing ticketing/batch sync patterns; simulation assumption.",
            "configured_capacities_note": "Provisioned capacity assumptions for cloud services; uninstrumented for legacy.",
        },
        "sampling_interval_seconds": 20.0,
        "trajectory_steps": total_steps,
        "trajectory_duration_seconds": total_steps * 20.0,
        "counts": {
            "total_incidents": len(all_trajectories),
            "train": len(train_ids),
            "validation": len(val_ids),
            "calibration": len(cal_ids),
            "test_standard": len(test_std_ids),
            "test_unseen_surge_diagnostic": len(diagnostic_ids),
            "test_unseen_surge_locked": len(locked_ids),
        },
        "splits": {
            "train": train_ids,
            "validation": val_ids,
            "calibration": cal_ids,
            "test_standard": test_std_ids,
            "test_unseen_surge_diagnostic": diagnostic_ids,
            "test_unseen_surge_locked": locked_ids,
        },
        "services": [s.value for s in ServiceIdentifier],
        "configured_capacities_rps": CONFIGURED_SERVICE_CAPACITIES,
    }

    dataset_dict = {
        "metadata": split_manifest,
        "trajectories": [t.to_dict() for t in all_trajectories],
    }

    # Compute SHA-256 hash of dataset payload
    serialized_bytes = json.dumps(dataset_dict, sort_keys=True).encode("utf-8")
    dataset_sha256 = hashlib.sha256(serialized_bytes).hexdigest()
    split_manifest["dataset_sha256"] = dataset_sha256
    dataset_dict["metadata"]["dataset_sha256"] = dataset_sha256

    return dataset_dict, split_manifest


def main():
    parser = argparse.ArgumentParser(description="BACCP Workload Dataset Generator")
    parser.add_argument("--standard-incidents", type=int, default=200, help="Number of standard incidents")
    parser.add_argument("--diagnostic-surge-incidents", type=int, default=25, help="Number of diagnostic surge incidents")
    parser.add_argument("--locked-surge-incidents", type=int, default=25, help="Number of locked surge evaluation incidents")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--steps", type=int, default=90, help="Steps per trajectory (90 * 20s = 1800s)")
    args = parser.parse_args()

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Generating BACCP Workload Forecasting Dataset (seed={args.seed})...")
    dataset, manifest = generate_workload_dataset(
        num_standard_incidents=args.standard_incidents,
        num_diagnostic_surge_incidents=args.diagnostic_surge_incidents,
        num_locked_surge_incidents=args.locked_surge_incidents,
        master_seed=args.seed,
        total_steps=args.steps,
    )

    data_file = OUTPUT_DATA_DIR / "workload_dataset.json"
    data_file.write_text(json.dumps(dataset, indent=2))
    print(f"[+] Saved workload dataset to {data_file} ({data_file.stat().st_size / 1024:.1f} KB)")

    manifest_file = OUTPUT_RESULTS_DIR / "split_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"[+] Saved split manifest to {manifest_file}")

    print("\n[✓] Data Generation Complete:")
    print(f"    Total Incidents: {manifest['counts']['total_incidents']}")
    print(f"    Train: {manifest['counts']['train']}")
    print(f"    Validation: {manifest['counts']['validation']}")
    print(f"    Calibration: {manifest['counts']['calibration']}")
    print(f"    Test Standard: {manifest['counts']['test_standard']}")
    print(f"    Diagnostic Surge: {manifest['counts']['test_unseen_surge_diagnostic']}")
    print(f"    Locked Surge: {manifest['counts']['test_unseen_surge_locked']}")
    print(f"    SHA-256: {manifest['dataset_sha256']}")


if __name__ == "__main__":
    main()

"""Generator for diagnostic localization fixture.

Creates 25 controlled single-node injection incidents with explicit ground-truth
origin annotations across all 5 candidate topology nodes:
- reservations (cloud)
- crew (cloud)
- baggage (cloud)
- boundary-gateway (gateway)
- legacy-core (legacy)

This fixture is used SOLELY to benchmark suspected-origin localization accuracy
in an isolated diagnostic setting, since degradation_dataset.json does not record
individual injected origin nodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "ai-models" / "data" / "degradation_prediction" / "diagnostic_localization_fixture.json"

SERVICES = ["reservations", "crew", "baggage", "boundary-gateway", "legacy-core"]
SERVICE_GENERATION = {
    "reservations": "cloud",
    "crew": "cloud",
    "baggage": "cloud",
    "boundary-gateway": "gateway",
    "legacy-core": "legacy",
}

NOMINAL_VALS = {
    "reservations": {"offered": 48.0, "latency": 35.0, "backlog": 0.0, "cap": 80.0},
    "crew": {"offered": 28.0, "latency": 25.0, "backlog": 0.0, "cap": 50.0},
    "baggage": {"offered": 36.0, "latency": 30.0, "backlog": 0.0, "cap": 60.0},
    "boundary-gateway": {"offered": 42.0, "latency": 20.0, "backlog": 0.0, "cap": 100.0},
    "legacy-core": {"offered": 42.0, "latency": 50.0, "backlog": 0.0, "cap": None},
}


def generate_diagnostic_fixture() -> None:
    incidents = []
    dt = 20.0
    steps = 10  # 200s

    case_id = 1
    for target_node in SERVICES:
        for variation in range(5):
            inc_id = f"diag_{target_node.replace('-', '_')}_{variation + 1:02d}"
            svc_telemetry: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SERVICES}
            bnd_telemetry: List[Dict[str, Any]] = []

            # Multiplier for severity variation
            sev_mult = 1.0 + (variation * 0.15)

            for step in range(steps):
                ts = float(step * dt)

                # Boundary step
                if target_node == "boundary-gateway" and step >= 3:
                    drift = 12.0 + 25.0 * sev_mult * ((step - 2) / 7.0)
                    gw_q = 10.0 + 40.0 * sev_mult * ((step - 2) / 7.0)
                    b_lat = 20.0 + 45.0 * sev_mult * ((step - 2) / 7.0)
                else:
                    drift = 12.0 + (0.5 * variation)
                    gw_q = 2.0
                    b_lat = 20.0

                bnd_telemetry.append({
                    "timestamp_sec": ts,
                    "step_index": step,
                    "sync_drift_score": round(drift, 2),
                    "gateway_queue_depth": round(gw_q, 2),
                    "boundary_latency_ms": round(b_lat, 2),
                    "boundary_error_rate": 0.01 if drift > 25.0 else 0.0,
                    "is_stale": False,
                })

                # Service steps
                for s in SERVICES:
                    base = NOMINAL_VALS[s]
                    is_injected = (s == target_node) and (step >= 3)

                    if is_injected:
                        if s == "legacy-core":
                            offered = base["offered"] * (1.1 + 0.1 * variation)
                            latency = base["latency"] + 70.0 * sev_mult * ((step - 2) / 7.0)
                            backlog = 15.0 * sev_mult * (step - 2)
                            completed = base["offered"]
                            error = 0.02 * ((step - 2) / 7.0)
                            drops = 0.0
                        elif s == "boundary-gateway":
                            offered = base["offered"] * (1.2 + 0.1 * variation)
                            latency = base["latency"] + 45.0 * sev_mult * ((step - 2) / 7.0)
                            backlog = gw_q
                            completed = min(offered, base["offered"])
                            error = 0.01
                            drops = 0.0
                        else:
                            offered = base["offered"] * (1.6 + 0.2 * variation)
                            cap = base["cap"]
                            backlog = max(0.0, (offered - cap) * dt * (step - 2) * 0.2)
                            latency = base["latency"] + 55.0 * sev_mult * ((step - 2) / 7.0)
                            completed = min(offered, cap)
                            error = 0.02 * ((step - 2) / 7.0)
                            drops = max(0.0, offered - cap) if backlog > 50.0 else 0.0
                    else:
                        # Nominal with small normal variation
                        offered = base["offered"] + (variation * 0.5)
                        latency = base["latency"] + (variation * 0.2)
                        backlog = 0.0
                        completed = offered
                        error = 0.0
                        drops = 0.0

                    svc_telemetry[s].append({
                        "timestamp_sec": ts,
                        "step_index": step,
                        "service_id": s,
                        "offered_rate": round(offered, 2),
                        "completed_rate": round(completed, 2),
                        "backlog_requests": round(backlog, 2),
                        "latency_ms": round(latency, 2),
                        "error_rate": round(error, 4),
                        "dropped_requests": round(drops, 2),
                        "effective_capacity": base["cap"],
                    })

            incidents.append({
                "incident_id": inc_id,
                "regime": "controlled_single_node_injection",
                "injected_origin_node": target_node,
                "injected_origin_generation": SERVICE_GENERATION[target_node],
                "injected_at_step": 3,
                "duration_sec": float(steps * dt),
                "step_count": steps,
                "service_telemetry": svc_telemetry,
                "boundary_telemetry": bnd_telemetry,
                "metadata": {
                    "synthetic_diagnostic_purpose": "suspected_origin_localization_evaluation",
                    "disclaimer": "Diagnostic synthetic fixture only; does not establish real-world causality.",
                },
                "split": "diagnostic_fixture",
            })
            case_id += 1

    payload = {
        "metadata": {
            "version": "baccp-diagnostic-localization-v1.0.0",
            "total_incidents": len(incidents),
            "candidate_nodes": SERVICES,
            "candidate_count": len(SERVICES),
            "cases_per_candidate": 5,
            "description": "Controlled synthetic single-node injection benchmark for graph localization audit.",
        },
        "incidents": incidents,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Generated {len(incidents)} diagnostic incidents to {FIXTURE_PATH}")


if __name__ == "__main__":
    generate_diagnostic_fixture()

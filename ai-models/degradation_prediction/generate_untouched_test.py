"""Generate Final Untouched Assessment Dataset for BACCP Degradation Prediction.

Version: baccp-untouched-test-v1.0.0

Generates 25 untouched test incidents with independently defined operational regimes
and stress conditions (abrupt bursts, capacity reductions, boundary slowdowns, gradual ramps, and nominal load).
Independent evaluation seed: 9999.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from .data_contract import (
        CONTRACT_VERSION,
        DegradationRegime,
        SAMPLING_INTERVAL_SECONDS,
    )
    from .generate_data import simulate_single_incident
except ImportError:
    from degradation_prediction.data_contract import (
        CONTRACT_VERSION,
        DegradationRegime,
        SAMPLING_INTERVAL_SECONDS,
    )
    from degradation_prediction.generate_data import simulate_single_incident

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_MODELS_DIR = REPO_ROOT / "ai-models"
OUTPUT_PATH = AI_MODELS_DIR / "data" / "degradation_prediction" / "untouched_final_test.json"


def generate_untouched_dataset(
    output_path: Path = OUTPUT_PATH,
    seed: int = 9999,
) -> Dict[str, Any]:
    """Generates 25 disjoint, untouched assessment incidents."""
    rng = np.random.RandomState(seed)

    # Balanced operational regime distribution with stress conditions
    regime_spec = {
        DegradationRegime.NOMINAL: 7,
        DegradationRegime.GRADUAL_RAMP: 5,
        DegradationRegime.ABRUPT_BURST: 5,
        DegradationRegime.CAPACITY_REDUCTION: 4,
        DegradationRegime.BOUNDARY_SLOWDOWN: 4,
    }

    incidents: List[Dict[str, Any]] = []
    split_manifest: Dict[str, List[str]] = {"untouched_final_test": []}
    regime_manifest: Dict[str, List[str]] = {r.value: [] for r in DegradationRegime}

    counter = 0
    for regime, count in regime_spec.items():
        for _ in range(count):
            inc_id = f"inc_untouched_{counter:04d}"
            counter += 1
            inc = simulate_single_incident(
                incident_id=inc_id,
                regime=regime,
                rng=rng,
                step_count=30,
            )
            inc_dict = inc.to_dict()
            inc_dict["split"] = "untouched_final_test"
            incidents.append(inc_dict)

            split_manifest["untouched_final_test"].append(inc_id)
            regime_manifest[regime.value].append(inc_id)

    serialized = json.dumps(incidents, sort_keys=True).encode("utf-8")
    dataset_sha256 = hashlib.sha256(serialized).hexdigest()

    dataset_payload = {
        "metadata": {
            "contract_version": CONTRACT_VERSION,
            "dataset_sha256": dataset_sha256,
            "evaluation_seed": seed,
            "description": "Final untouched assessment set generated after locking code, models, and thresholds.",
            "sampling_interval_seconds": SAMPLING_INTERVAL_SECONDS,
            "total_incidents": len(incidents),
            "step_count_per_incident": 30,
            "total_timesteps": len(incidents) * 30,
            "splits": {"untouched_final_test": len(incidents)},
            "regimes": {k: len(v) for k, v in regime_manifest.items()},
        },
        "split_manifest": split_manifest,
        "incidents": incidents,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset_payload, f, indent=2)

    print(f"Generated {len(incidents)} untouched incidents saved to {output_path}")
    print(f"Dataset SHA-256: {dataset_sha256}")
    return dataset_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate untouched final evaluation dataset.")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    generate_untouched_dataset(Path(args.output), seed=args.seed)

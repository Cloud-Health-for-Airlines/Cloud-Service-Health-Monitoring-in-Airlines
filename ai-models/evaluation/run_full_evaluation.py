"""Master Reproducibility Entrypoint for BACCP Full Evaluation Pipeline.

Executes:
1. Dataset validation
2. Checkpoint validation and model loading
3. Predictive models benchmark (Flat GCN vs Domain GNN vs BACCP HeteroRGCN)
4. Circuit breaker policies benchmark (No Mitigation vs Rule Baseline vs PPO)
5. Metric computation and verification
6. Results serialization (JSON, CSV, MD)
7. Matplotlib plot generation
8. Exit code 0 on success, non-zero on failure
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_DIR = ROOT / "ai-models/cascade_predictor"
CB_DIR = ROOT / "ai-models/circuit-breaker"
EVAL_DIR = ROOT / "ai-models/evaluation"

for p in (ROOT, PREDICTOR_DIR, CB_DIR, EVAL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baccp.eval_runner")


def validate_dataset(data_path: Path) -> None:
    logger.info(f"Step 1/7: Validating dataset at {data_path}...")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file missing: {data_path}")

    data = json.loads(data_path.read_text())
    if "metadata" not in data or "samples" not in data:
        raise ValueError(f"Dataset malformed: missing metadata or samples in {data_path}")

    samples = data["samples"]
    if len(samples) < 1000:
        raise ValueError(f"Dataset sample count ({len(samples)}) below minimum 1000")

    sample_0 = samples[0]
    required_keys = {"sample_id", "cascade_label", "lead_time_seconds", "root_cause_node", "severity_level", "snapshots"}
    missing = required_keys - set(sample_0.keys())
    if missing:
        raise ValueError(f"Sample missing required keys: {missing}")

    logger.info(f"Dataset validated successfully: {len(samples)} samples (hash: {data['metadata'].get('version', '1.0')})")


def validate_checkpoints() -> None:
    logger.info("Step 2/7: Validating trained model checkpoints...")
    weights_dir = ROOT / "ai-models/weights"

    required_checkpoints = [
        weights_dir / "cascade_predictor_tier1a.pt",
        weights_dir / "baseline_flat_graph.pt",
        weights_dir / "baseline_domain_typed.pt",
        weights_dir / "circuit_breaker_ppo.pt",
    ]

    for ckpt in required_checkpoints:
        if not ckpt.is_file():
            raise FileNotFoundError(f"Required checkpoint missing: {ckpt.name}")
        logger.info(f"  Found checkpoint: {ckpt.name} ({ckpt.stat().st_size} bytes)")


def main() -> int:
    logger.info("=================================================================")
    logger.info("Starting BACCP Full Evaluation Pipeline")
    logger.info("=================================================================")

    data_path = ROOT / "ai-models/data/dataset.json"
    results_dir = ROOT / "results"
    plots_dir = results_dir / "plots"

    try:
        # Step 1: Validate Dataset
        validate_dataset(data_path)

        # Step 2: Validate Checkpoints
        validate_checkpoints()

        # Step 3: Run Full Comparative Experiment
        logger.info("Step 3/7: Running predictive models benchmark...")
        logger.info("Step 4/7: Running circuit breaker policies benchmark...")
        logger.info("Step 5/7: Generating publication-quality plots...")
        from compare_baselines import run_full_experiment
        res = run_full_experiment(
            data_path=str(data_path),
            seed=42,
            episodes=50,
            output_dir=str(results_dir),
        )

        # Step 6: Validate Result Files
        logger.info("Step 6/7: Validating generated result artifacts...")
        csv_path = results_dir / "baseline_comparison.csv"
        json_path = results_dir / "baseline_comparison.json"
        md_path = results_dir / "baseline_comparison.md"

        for f in (csv_path, json_path, md_path):
            if not f.is_file() or f.stat().st_size == 0:
                raise RuntimeError(f"Generated result artifact missing or empty: {f}")
            logger.info(f"  Verified artifact: {f.name} ({f.stat().st_size} bytes)")

        # Step 7: Validate Plots
        logger.info("Step 7/7: Validating generated matplotlib figures...")
        expected_plots = [
            "model_performance_comparison.png",
            "lead_time_distribution.png",
            "circuit_breaker_throughput_tradeoff.png",
            "service_priority_retention.png",
        ]
        for p_name in expected_plots:
            p_file = plots_dir / p_name
            if not p_file.is_file() or p_file.stat().st_size == 0:
                raise RuntimeError(f"Expected plot missing or empty: {p_name}")
            logger.info(f"  Verified figure: {p_name} ({p_file.stat().st_size} bytes)")

        logger.info("=================================================================")
        logger.info("BACCP Full Evaluation Pipeline Completed Successfully (Exit Code 0)")
        logger.info("=================================================================")
        return 0

    except Exception as exc:
        logger.error(f"BACCP Full Evaluation Pipeline FAILED: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

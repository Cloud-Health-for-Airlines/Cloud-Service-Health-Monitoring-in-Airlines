"""Package Task 2C Review Bundle ZIP for BACCP Degradation Prediction.

Version: baccp-package-review-v1.0.0
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_MODELS_DIR = REPO_ROOT / "ai-models"
RESULTS_DIR = REPO_ROOT / "results" / "degradation_prediction"
OUTPUT_ZIP = RESULTS_DIR / "baccp_task2c_review_bundle.zip"

FILES_TO_PACKAGE = [
    # Data contracts and generators
    AI_MODELS_DIR / "degradation_prediction" / "data_contract.py",
    AI_MODELS_DIR / "degradation_prediction" / "generate_data.py",
    AI_MODELS_DIR / "degradation_prediction" / "generate_untouched_test.py",
    # Features and schema
    AI_MODELS_DIR / "degradation_prediction" / "features.py",
    # Models, training, inference
    AI_MODELS_DIR / "degradation_prediction" / "models.py",
    AI_MODELS_DIR / "degradation_prediction" / "train.py",
    AI_MODELS_DIR / "degradation_prediction" / "inference.py",
    # Evaluation and lock manifest
    AI_MODELS_DIR / "degradation_prediction" / "evaluate.py",
    AI_MODELS_DIR / "degradation_prediction" / "lock_and_evaluate_untouched.py",
    # Tests
    AI_MODELS_DIR / "tests" / "test_degradation_prediction.py",
    AI_MODELS_DIR / "tests" / "test_workload_forecasting.py",
    # Package bridge
    REPO_ROOT / "ai_models" / "degradation_prediction.py",
    # Fixtures
    AI_MODELS_DIR / "data" / "degradation_prediction" / "fixture_test_incidents.json",
    # Machine-readable results and reports
    RESULTS_DIR / "metrics_summary.json",
    RESULTS_DIR / "evaluation_report.md",
    RESULTS_DIR / "lock_manifest.json",
    RESULTS_DIR / "untouched_test_metrics.json",
]


def create_review_zip():
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in FILES_TO_PACKAGE:
            if file_path.exists():
                arcname = file_path.relative_to(REPO_ROOT)
                zf.write(file_path, arcname=str(arcname))
                print(f"Added: {arcname}")
            else:
                print(f"WARNING: File not found: {file_path}")

    zip_bytes = OUTPUT_ZIP.read_bytes()
    zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()
    print(f"\nCreated review bundle: {OUTPUT_ZIP}")
    print(f"Size: {len(zip_bytes) / 1024:.1f} KB")
    print(f"SHA-256: {zip_sha256}")
    return OUTPUT_ZIP, zip_sha256


if __name__ == "__main__":
    create_review_zip()

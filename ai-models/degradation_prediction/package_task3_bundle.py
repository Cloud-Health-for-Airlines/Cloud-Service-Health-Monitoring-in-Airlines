"""Package Task 3A Review Bundle ZIP, Manifest, and Checksums for BACCP Degradation Prediction.

Prerequisite compliance:
- Strictly excludes the manifest itself from its own files list.
- Packages frozen prediction models, workload forecaster bundle, saved background fixtures, code, tests, and evidence.
- Writes archive and manifest hashes to a separate checksum file (checksums_sha256.txt) avoiding circular self-hashing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
AI_MODELS_DIR = REPO_ROOT / "ai-models"
RESULTS_DIR = REPO_ROOT / "results" / "degradation_prediction"
OUTPUT_ZIP = RESULTS_DIR / "baccp_task3_review_bundle.zip"
MANIFEST_PATH = RESULTS_DIR / "task3_manifest.json"
PATCH_PATH = RESULTS_DIR / "task3_diff.patch"
CHECKSUMS_PATH = RESULTS_DIR / "checksums_sha256.txt"

FILES_TO_PACKAGE = [
    # Data contracts, features, models, inference, explanation engine
    AI_MODELS_DIR / "degradation_prediction" / "data_contract.py",
    AI_MODELS_DIR / "degradation_prediction" / "features.py",
    AI_MODELS_DIR / "degradation_prediction" / "models.py",
    AI_MODELS_DIR / "degradation_prediction" / "inference.py",
    AI_MODELS_DIR / "degradation_prediction" / "explanation.py",
    AI_MODELS_DIR / "degradation_prediction" / "evaluate_explanations.py",
    AI_MODELS_DIR / "degradation_prediction" / "package_task3_bundle.py",
    AI_MODELS_DIR / "degradation_prediction" / "generate_diagnostic_fixture.py",
    AI_MODELS_DIR / "degradation_prediction" / "__init__.py",
    # Workload forecasting source dependencies (upstream forecaster)
    AI_MODELS_DIR / "workload_forecasting" / "__init__.py",
    AI_MODELS_DIR / "workload_forecasting" / "data_contract.py",
    AI_MODELS_DIR / "workload_forecasting" / "features.py",
    AI_MODELS_DIR / "workload_forecasting" / "models.py",
    AI_MODELS_DIR / "workload_forecasting" / "inference.py",
    # Package bridges
    REPO_ROOT / "ai_models" / "degradation_prediction.py",
    REPO_ROOT / "ai_models" / "workload_forecasting.py",
    # Requirements
    AI_MODELS_DIR / "requirements.txt",
    # Tests
    AI_MODELS_DIR / "tests" / "test_degradation_prediction.py",
    AI_MODELS_DIR / "tests" / "test_degradation_explanations.py",
    # Fixtures & Saved Training Background
    AI_MODELS_DIR / "data" / "degradation_prediction" / "saved_training_background.json",
    AI_MODELS_DIR / "data" / "degradation_prediction" / "diagnostic_localization_fixture.json",
    AI_MODELS_DIR / "data" / "degradation_prediction" / "fixture_test_incidents.json",
    # Model Weights Bundles
    AI_MODELS_DIR / "weights" / "degradation_prediction" / "degradation_models_bundle.pkl",
    AI_MODELS_DIR / "weights" / "workload_forecasting" / "workload_models_bundle.pkl",
    # Evaluation Artifacts and Plots
    RESULTS_DIR / "explanation_metrics.json",
    RESULTS_DIR / "representative_incident_briefs.md",
    RESULTS_DIR / "plots" / "attribution_summary.png",
]


def generate_patch() -> None:
    """Generates git diff patch against HEAD or previous commit."""
    try:
        res = subprocess.run(
            ["git", "diff", "HEAD~1"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        patch_text = res.stdout if res.stdout else ""
        if not patch_text:
            res = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            patch_text = res.stdout
        with open(PATCH_PATH, "w", encoding="utf-8") as f:
            f.write(patch_text)
        print(f"Generated patch at {PATCH_PATH}")
    except Exception as e:
        print(f"Error generating patch: {e}")


def package_task3_bundle() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_patch()
    if PATCH_PATH.exists() and PATCH_PATH not in FILES_TO_PACKAGE:
        FILES_TO_PACKAGE.append(PATCH_PATH)

    # 1. Build Manifest (strictly excluding manifest itself and checksums file)
    files_dict: dict = {}
    for fpath in FILES_TO_PACKAGE:
        if fpath.exists():
            rel_path = str(fpath.relative_to(REPO_ROOT))
            if "task3_manifest.json" in rel_path or "checksums_sha256.txt" in rel_path:
                continue
            data = fpath.read_bytes()
            files_dict[rel_path] = {
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        else:
            print(f"WARNING: File not found: {fpath}")

    # Compute evaluation dataset hash for input provenance
    eval_dataset_path = AI_MODELS_DIR / "data" / "degradation_prediction" / "degradation_dataset.json"
    eval_dataset_sha256 = hashlib.sha256(eval_dataset_path.read_bytes()).hexdigest() if eval_dataset_path.exists() else None

    manifest_payload = {
        "manifest_version": "baccp-task3b-manifest-v1.0.0",
        "description": "Task 3B Explanation Consistency & Evaluation Review Bundle (manifest strictly excluded from files list)",
        "file_count": len(files_dict),
        "files": files_dict,
        "archive": {
            "name": OUTPUT_ZIP.name,
            "sha256": None,
        },
        "reproducibility": {
            "execution_command": "PYTHONPATH=ai-models python ai-models/degradation_prediction/evaluate_explanations.py",
            "test_command": "PYTHONPATH=ai-models pytest ai-models/tests/test_degradation_prediction.py ai-models/tests/test_degradation_explanations.py",
            "python_environment": "Python 3.9+, numpy>=1.24.0, scipy>=1.11.0, scikit-learn>=1.3.0, shap>=0.42.0, matplotlib>=3.7.0, pytest>=7.4.0",
            "evaluation_input_dataset": {
                "relative_path": "ai-models/data/degradation_prediction/degradation_dataset.json",
                "sha256": eval_dataset_sha256,
                "reconstruction_note": "Production evaluation split (130 incidents) deterministically processed from degradation_dataset.json",
            },
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    # 2. Build ZIP bundle including manifest and all files
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(MANIFEST_PATH, arcname=str(MANIFEST_PATH.relative_to(REPO_ROOT)))
        for fpath in FILES_TO_PACKAGE:
            if fpath.exists():
                arcname = str(fpath.relative_to(REPO_ROOT))
                zf.write(fpath, arcname=arcname)

    zip_bytes = OUTPUT_ZIP.read_bytes()
    zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()

    # Update manifest archive sha256
    manifest_payload["archive"]["sha256"] = zip_sha256
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    # 3. Write separate checksum file avoiding circular self-hashing
    checksum_lines = [
        f"{zip_sha256}  baccp_task3_review_bundle.zip",
        f"{manifest_sha256}  task3_manifest.json",
    ]
    if PATCH_PATH.exists():
        patch_sha256 = hashlib.sha256(PATCH_PATH.read_bytes()).hexdigest()
        checksum_lines.append(f"{patch_sha256}  task3_diff.patch")

    # Add core artifact checksums
    core_artifacts = [
        AI_MODELS_DIR / "weights" / "degradation_prediction" / "degradation_models_bundle.pkl",
        AI_MODELS_DIR / "weights" / "workload_forecasting" / "workload_models_bundle.pkl",
        AI_MODELS_DIR / "data" / "degradation_prediction" / "saved_training_background.json",
        AI_MODELS_DIR / "data" / "degradation_prediction" / "diagnostic_localization_fixture.json",
        RESULTS_DIR / "explanation_metrics.json",
        RESULTS_DIR / "representative_incident_briefs.md",
    ]
    for art in core_artifacts:
        if art.exists():
            art_sha = hashlib.sha256(art.read_bytes()).hexdigest()
            checksum_lines.append(f"{art_sha}  {art.relative_to(REPO_ROOT)}")

    with open(CHECKSUMS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")

    print("\n" + "=" * 60)
    print("TASK 3A REVIEW BUNDLE PACKAGING COMPLETE")
    print("=" * 60)
    print(f"Archive:       {OUTPUT_ZIP}")
    print(f"Archive Size:  {len(zip_bytes) / (1024 * 1024):.2f} MB")
    print(f"Archive SHA:   {zip_sha256}")
    print(f"Manifest:      {MANIFEST_PATH}")
    print(f"Manifest SHA:  {manifest_sha256}")
    print(f"Checksums:     {CHECKSUMS_PATH}")
    print(f"Files Count:   {len(files_dict)} (manifest excluded from files list)")


if __name__ == "__main__":
    package_task3_bundle()

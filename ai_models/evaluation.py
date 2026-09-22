"""Standard Python package bridge for ai-models/evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from importlib import import_module

REPO_ROOT = Path(__file__).resolve().parent.parent
AI_MODELS_DIR = REPO_ROOT / "ai-models"
if str(AI_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(AI_MODELS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_mod = import_module("evaluation")

run_comprehensive_evaluation = getattr(_mod, "run_comprehensive_evaluation")
train_cascade_predictor_models = getattr(_mod, "train_cascade_predictor_models")
train_circuit_breaker_models = getattr(_mod, "train_circuit_breaker_models")
run_baseline_comparison = getattr(_mod, "run_baseline_comparison")

__all__ = [
    "run_comprehensive_evaluation",
    "train_cascade_predictor_models",
    "train_circuit_breaker_models",
    "run_baseline_comparison",
]

"""Evaluation, training pipelines, and baseline comparison harness for BACCP."""

from __future__ import annotations

import sys
from pathlib import Path

_curr_dir = Path(__file__).resolve().parent
if str(_curr_dir) not in sys.path:
    sys.path.insert(0, str(_curr_dir))

from evaluate import run_comprehensive_evaluation
from train_cascade_predictor import train_cascade_predictor_models
from train_circuit_breaker import train_circuit_breaker_models
from baselines import run_baseline_comparison

__all__ = [
    "run_comprehensive_evaluation",
    "train_cascade_predictor_models",
    "train_circuit_breaker_models",
    "run_baseline_comparison",
]

"""Standard Python package bridge for ai-models/data."""

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

_mod = import_module("data")

DatasetGenerator = getattr(_mod, "DatasetGenerator")
load_chaos_profiles = getattr(_mod, "load_chaos_profiles")
DATASET_DIR = getattr(_mod, "DATASET_DIR")

__all__ = [
    "DatasetGenerator",
    "load_chaos_profiles",
    "DATASET_DIR",
]

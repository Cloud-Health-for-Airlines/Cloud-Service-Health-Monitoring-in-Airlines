"""Dataset generation and telemetry harvesting module for BACCP."""

from __future__ import annotations

import sys
from pathlib import Path

_curr_dir = Path(__file__).resolve().parent
if str(_curr_dir) not in sys.path:
    sys.path.insert(0, str(_curr_dir))

from generate_dataset import DatasetGenerator, load_chaos_profiles, DATASET_DIR

__all__ = [
    "DatasetGenerator",
    "load_chaos_profiles",
    "DATASET_DIR",
]

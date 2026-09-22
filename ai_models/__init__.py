"""BACCP AI Models Python Package (Standard Python Identifier Bridge).

Provides standard import access to components in `ai-models/`:
- `ai_models.graph_builder`
- `ai_models.cascade_predictor`
- `ai_models.circuit_breaker`
- `ai_models.data`
- `ai_models.evaluation`
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
AI_MODELS_DIR = REPO_ROOT / "ai-models"

if str(AI_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(AI_MODELS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

__version__ = "1.0.0"

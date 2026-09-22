"""Standard Python package bridge for ai-models/cascade-predictor."""

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

_mod = import_module("cascade-predictor")

BaselineGATGRU = getattr(_mod, "BaselineGATGRU")
EdgeFeaturedGATLayer = getattr(_mod, "EdgeFeaturedGATLayer")
HeteroCascadePredictor = getattr(_mod, "HeteroCascadePredictor")
RelationalConvLayer = getattr(_mod, "RelationalConvLayer")
HeteroGNN = getattr(_mod, "HeteroGNN")
CascadePredictor = getattr(_mod, "CascadePredictor")
MultiTaskHead = getattr(_mod, "MultiTaskHead")
MultiTaskCascadeHead = getattr(_mod, "MultiTaskCascadeHead")
NODE_NAMES = getattr(_mod, "NODE_NAMES")
SEVERITY_LEVELS = getattr(_mod, "SEVERITY_LEVELS")
TemperatureScaler = getattr(_mod, "TemperatureScaler")
SplitConformalCalibrator = getattr(_mod, "SplitConformalCalibrator")
format_incident_explanation = getattr(_mod, "format_incident_explanation")
predict = getattr(_mod, "predict")
get_inference_engine = getattr(_mod, "get_inference_engine")

__all__ = [
    "BaselineGATGRU",
    "EdgeFeaturedGATLayer",
    "HeteroCascadePredictor",
    "RelationalConvLayer",
    "HeteroGNN",
    "CascadePredictor",
    "MultiTaskHead",
    "MultiTaskCascadeHead",
    "NODE_NAMES",
    "SEVERITY_LEVELS",
    "TemperatureScaler",
    "SplitConformalCalibrator",
    "format_incident_explanation",
    "predict",
    "get_inference_engine",
]

"""Cascade Predictor module: Tier 0 Baseline GAT-GRU, Tier 1A Heterogeneous RGCN, Multi-Task Head, and Explainability."""

from __future__ import annotations

import sys
from pathlib import Path

_curr_dir = Path(__file__).resolve().parent
if str(_curr_dir) not in sys.path:
    sys.path.insert(0, str(_curr_dir))

from baseline_gat_gru import BaselineGATGRU, EdgeFeaturedGATLayer
from hetero_gnn import HeteroCascadePredictor, RelationalConvLayer
from multi_task_head import MultiTaskHead, NODE_NAMES, SEVERITY_LEVELS
from explainability import TemperatureScaler, SplitConformalCalibrator, format_incident_explanation

# Convenience Aliases
HeteroGNN = HeteroCascadePredictor
CascadePredictor = BaselineGATGRU
MultiTaskCascadeHead = MultiTaskHead


def predict(*args, **kwargs):
    """Lazy-loaded inference entrypoint preventing circular imports."""
    from inference import predict as _predict
    return _predict(*args, **kwargs)


def get_inference_engine():
    """Lazy-loaded inference engine singleton."""
    from inference import CascadePredictorInferenceEngine
    return CascadePredictorInferenceEngine()


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

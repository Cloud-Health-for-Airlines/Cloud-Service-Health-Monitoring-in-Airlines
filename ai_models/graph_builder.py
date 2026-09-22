"""Standard Python package bridge for ai-models/graph-builder."""

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

_mod = import_module("graph-builder")

GenerationTypedGraph = getattr(_mod, "GenerationTypedGraph")
HeteroGraphData = getattr(_mod, "HeteroGraphData")
NODE_TYPES = getattr(_mod, "NODE_TYPES")
EDGE_RELATIONS = getattr(_mod, "EDGE_RELATIONS")
compute_sync_drift_score = getattr(_mod, "compute_sync_drift_score")
extract_graph_features = getattr(_mod, "extract_graph_features")

__all__ = [
    "GenerationTypedGraph",
    "HeteroGraphData",
    "NODE_TYPES",
    "EDGE_RELATIONS",
    "compute_sync_drift_score",
    "extract_graph_features",
]

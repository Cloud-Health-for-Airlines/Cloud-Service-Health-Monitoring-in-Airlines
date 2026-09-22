"""Graph Builder module for generation-typed dependency graphs and topological features."""

from __future__ import annotations

import sys
from pathlib import Path

_curr_dir = Path(__file__).resolve().parent
if str(_curr_dir) not in sys.path:
    sys.path.insert(0, str(_curr_dir))

from schema import GenerationTypedGraph, HeteroGraphData, NODE_TYPES, EDGE_RELATIONS
from features import compute_sync_drift_score, extract_graph_features

__all__ = [
    "GenerationTypedGraph",
    "HeteroGraphData",
    "NODE_TYPES",
    "EDGE_RELATIONS",
    "compute_sync_drift_score",
    "extract_graph_features",
]

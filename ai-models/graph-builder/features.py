"""Feature engineering and mathematical boundary sync-drift computation for BACCP.

Implements:
1. Digital-Twin State Synchronization Drift score:
   ε(t) = (||Φ(t) - Ψ(t)||_2 / (||Φ(t)||_2 + δ)) * 100
   augmented with boundary queue pressure and latency variance.
2. Generation-aware topological centrality and boundary bottleneck index.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch


def compute_sync_drift_score(
    gateway_submission_rate: float,
    legacy_completion_rate: float,
    gateway_queue_depth: float = 0.0,
    boundary_latency_ms: float = 15.0,
    nominal_latency_ms: float = 15.0,
    boundary_error_rate: float = 0.0,
    delta: float = 1e-3,
) -> float:
    """Compute real mathematical State Synchronization Drift score ε(t) in [0.0, 100.0]%.
    
    Parameters:
        gateway_submission_rate (Φ): requests submitted from cloud to boundary gateway (req/s)
        legacy_completion_rate (Ψ): transactions completed by legacy core (req/s)
        gateway_queue_depth: pending backlog at boundary gateway
        boundary_latency_ms: round-trip latency on gateway-legacy TCP link
        nominal_latency_ms: baseline round-trip latency (nominal = 15ms)
        boundary_error_rate: ratio of failed/timed-out legacy requests [0.0, 1.0]
        delta: regularization constant preventing division by zero
        
    Returns:
        drift_score: float in [0.0, 100.0] matching frontend gauge expectations:
            - Nominal healthy: < 45.0%
            - Warning threshold: >= 45.0%
            - Critical boundary hazard: >= 70.0%
    """
    phi = max(0.0, float(gateway_submission_rate))
    psi = max(0.0, float(legacy_completion_rate))
    
    # Base rate divergence ||Φ - Ψ||_2 / (||Φ||_2 + δ) * 100
    rate_diff = abs(phi - psi)
    base_drift = (rate_diff / (phi + delta)) * 100.0

    # Queue pressure component: queue backlog directly indicates unserviced legacy demand
    # A queue depth of 50 requests contributes +20% drift
    queue_drift = min(40.0, (float(gateway_queue_depth) / 50.0) * 20.0)

    # Latency anomaly ratio: delay above nominal indicates legacy bufferbloat or stall
    lat_ratio = max(1.0, float(boundary_latency_ms) / max(1.0, float(nominal_latency_ms)))
    lat_drift = min(35.0, math.log2(lat_ratio) * 10.0) if lat_ratio > 1.0 else 0.0

    # Error divergence component: 10% errors adds 25% drift
    error_drift = min(50.0, float(boundary_error_rate) * 250.0)

    # Combined drift with smooth saturation
    total_drift = base_drift + queue_drift + lat_drift + error_drift
    
    # Minimum baseline ambient noise ~8-14% (normal distributed clock skew)
    drift_clamped = max(8.5, min(99.5, total_drift))
    return round(drift_clamped, 2)


def compute_topological_features(
    node_names: List[str],
    node_types: Dict[str, str],
    edges: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute generation-aware centrality and boundary bottleneck index.
    
    Unlike standard degree or betweenness centrality which treats all edges uniformly,
    generation-crossing paths (crossing legacy <-> gateway or gateway <-> cloud)
    are weighted 3x more heavily than intra-cloud edges.
    """
    N = len(node_names)
    idx_map = {name: i for i, name in enumerate(node_names)}
    
    # Weight adjacency matrix: generation-crossing edges have higher weight
    dist_matrix = np.full((N, N), np.inf)
    np.fill_diagonal(dist_matrix, 0.0)

    for edge in edges:
        s = edge.get("source")
        d = edge.get("destination") or edge.get("target")
        if s and d and s in idx_map and d in idx_map:
            u, v = idx_map[s], idx_map[d]
            s_type = node_types.get(s, "cloud-native")
            d_type = node_types.get(d, "cloud-native")
            
            # Distance cost: generation boundary crossing is critical (higher cost/weight)
            cost = 1.0 if s_type == d_type else 3.0
            dist_matrix[u, v] = min(dist_matrix[u, v], cost)

    # Floyd-Warshall for all-pairs shortest paths
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]

    # Closeness centrality and boundary bridge index
    features: Dict[str, Dict[str, float]] = {}
    for name in node_names:
        i = idx_map[name]
        finite_dists = dist_matrix[i][np.isfinite(dist_matrix[i])]
        sum_dist = float(np.sum(finite_dists))
        closeness = (len(finite_dists) - 1) / (sum_dist + 1e-6) if sum_dist > 0 else 0.0
        
        # Generation boundary bridge index: 1.0 for boundary-gateway, 0.5 for legacy, 0.2 for cloud
        nt = node_types.get(name, "cloud-native")
        if nt == "boundary-gateway":
            bridge_index = 1.0
        elif nt == "legacy":
            bridge_index = 0.6
        else:
            bridge_index = 0.2

        features[name] = {
            "generation_closeness": round(float(closeness), 4),
            "bridge_index": bridge_index,
        }

    return features


def extract_graph_features(
    node_metrics: Dict[str, Dict[str, float]],
    node_types: Dict[str, str],
    edges: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Derive full boundary telemetry feature vector and per-node topological features."""
    gw_metrics = node_metrics.get("boundary-gateway", {})
    legacy_metrics = node_metrics.get("legacy-core", {})

    sub_rate = float(gw_metrics.get("request_rate", 45.0))
    comp_rate = float(legacy_metrics.get("request_rate", sub_rate))
    q_depth = float(gw_metrics.get("queue_depth", 0.0))
    lat_ms = float(gw_metrics.get("latency_ms", 15.0))
    err_rate = float(gw_metrics.get("error_rate", 0.0))

    drift = compute_sync_drift_score(
        gateway_submission_rate=sub_rate,
        legacy_completion_rate=comp_rate,
        gateway_queue_depth=q_depth,
        boundary_latency_ms=lat_ms,
        boundary_error_rate=err_rate,
    )

    topo = compute_topological_features(
        node_names=list(node_types.keys()),
        node_types=node_types,
        edges=edges,
    )

    boundary_features = {
        "sync_drift_score": drift,
        "gateway_submission_rate": sub_rate,
        "legacy_completion_rate": comp_rate,
        "gateway_queue_depth": q_depth,
        "boundary_latency_ms": lat_ms,
        "boundary_error_rate": err_rate,
    }

    return boundary_features, topo

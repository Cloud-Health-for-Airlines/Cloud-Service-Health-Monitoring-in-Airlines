"""Inference entry point for BACCP Cascade Predictor.

Provides a module-level singleton interface for SageMaker / Backend consumers,
returning the exact non-negotiable dictionary structure required by frontend/src/api/adapter.js.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import torch

logger = logging.getLogger("baccp.inference.predictor")

AI_MODELS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = AI_MODELS_DIR.parent
if str(AI_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(AI_MODELS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from importlib import import_module
schema_mod = import_module("graph-builder.schema")
GenerationTypedGraph = schema_mod.GenerationTypedGraph
CANONICAL_SERVICES = schema_mod.CANONICAL_SERVICES

features_mod = import_module("graph-builder.features")
compute_sync_drift_score = features_mod.compute_sync_drift_score

hetero_mod = import_module("cascade-predictor.hetero_gnn")
HeteroCascadePredictor = hetero_mod.HeteroCascadePredictor

multi_task_mod = import_module("cascade-predictor.multi_task_head")
MultiTaskHead = multi_task_mod.MultiTaskHead
NODE_NAMES = multi_task_mod.NODE_NAMES
SEVERITY_LEVELS = multi_task_mod.SEVERITY_LEVELS

explain_mod = import_module("cascade-predictor.explainability")
SplitConformalCalibrator = explain_mod.SplitConformalCalibrator
format_incident_explanation = explain_mod.format_incident_explanation

WEIGHTS_PATH = AI_MODELS_DIR / "weights" / "cascade_predictor_tier1a.pt"


class CascadePredictorInferenceEngine:
    """Singleton model loader and inference engine for Tier 1A Hetero-RGCN."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.graph_builder = GenerationTypedGraph()
        self.encoder = HeteroCascadePredictor(node_in_dim=6, hidden_dim=48, gru_hidden_dim=32, num_layers=2)
        self.head = MultiTaskHead(in_dim=32, num_nodes=5, num_severities=4, hidden_dim=32)
        self.conformal = SplitConformalCalibrator(alpha=0.10)
        self.temperature = 1.0
        self.is_loaded = False

        if WEIGHTS_PATH.exists():
            try:
                pkg = torch.load(WEIGHTS_PATH, map_location="cpu")
                self.encoder.load_state_dict(pkg["encoder"])
                self.head.load_state_dict(pkg["head"])
                self.temperature = pkg.get("temperature", 1.0)
                self.conformal.quantile = pkg.get("conformal_quantile", 0.08)
                self.encoder.eval()
                self.head.eval()
                self.is_loaded = True
                logger.info(f"Successfully loaded trained CascadePredictor weights from {WEIGHTS_PATH}")
            except Exception as exc:
                logger.warning(f"Could not load trained model weights ({exc}). Inference engine uninitialized.")
        else:
            logger.warning(f"No checkpoint found at {WEIGHTS_PATH}.")

        self._initialized = True

    def predict(
        self,
        graph_data: Optional[Dict[str, Any]] = None,
        telemetry_features: Optional[Dict[str, Any]] = None,
        boundary_features: Optional[Dict[str, Any]] = None,
        sync_drift_score: Optional[float] = None,
        active_fault: Optional[str] = None,
        fault_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute inference matching the exact non-negotiable contract."""
        if not self.is_loaded:
            raise RuntimeError("Trained weights not loaded.")

        # Determine drift score
        drift = 14.2
        if sync_drift_score is not None:
            drift = float(sync_drift_score)
        elif boundary_features and "sync_drift_score" in boundary_features:
            drift = float(boundary_features["sync_drift_score"])

        # Construct input features
        edges = []
        if graph_data and "edges" in graph_data:
            edges = graph_data["edges"]
        else:
            # Default canonical connectivity
            edges = [
                {"source": "boundary-gateway", "destination": "legacy-core", "protocol": "tcp", "observation_count": 100},
                {"source": "legacy-core", "destination": "boundary-gateway", "protocol": "tcp", "observation_count": 98},
                {"source": "boundary-gateway", "destination": "reservations", "protocol": "http", "observation_count": 50},
                {"source": "reservations", "destination": "boundary-gateway", "protocol": "http", "observation_count": 50},
                {"source": "reservations", "destination": "crew", "protocol": "http", "observation_count": 20},
                {"source": "reservations", "destination": "baggage", "protocol": "http", "observation_count": 15},
            ]

        # Extract per-node metrics
        gw_lat = float(boundary_features.get("boundary_latency_ms", 15.0)) if boundary_features else 15.0
        gw_queue = float(boundary_features.get("gateway_queue_depth", 0.0)) if boundary_features else 0.0
        gw_err = float(boundary_features.get("boundary_error_rate", 0.0)) if boundary_features else 0.0

        # Modulate metrics when fault or elevated drift is indicated
        if active_fault:
            multipliers = {"low": 1.2, "medium": 1.6, "high": 2.2}
            mult = multipliers.get(fault_level or "medium", 1.5)
            if active_fault == "network-delay":
                gw_lat = max(gw_lat, 500.0 * mult)
                gw_queue = max(gw_queue, 30.0 * mult)
            elif active_fault == "connection-drop":
                gw_err = max(gw_err, 0.40 * mult)
                gw_queue = max(gw_queue, 35.0 * mult)
                gw_lat = max(gw_lat, 100.0 * mult)
            elif active_fault == "batch-job-stall":
                gw_lat = max(gw_lat, 900.0 * mult)
                gw_queue = max(gw_queue, 50.0 * mult)
                gw_err = max(gw_err, 0.30 * mult)
        elif drift >= 70.0:
            gw_lat = max(gw_lat, 650.0)
            gw_queue = max(gw_queue, 50.0)
            gw_err = max(gw_err, 0.20)
        elif drift >= 45.0:
            gw_lat = max(gw_lat, 200.0)
            gw_queue = max(gw_queue, 20.0)

        node_metrics = {
            "legacy-core": {"request_rate": 45.0, "latency_ms": gw_lat * 0.8, "error_rate": gw_err, "queue_depth": 0.0, "cpu_load": min(1.0, 0.3 + (gw_lat / 1500.0) * 0.5), "sync_drift_score": drift},
            "boundary-gateway": {"request_rate": 50.0, "latency_ms": gw_lat, "error_rate": gw_err, "queue_depth": gw_queue, "cpu_load": min(1.0, 0.4 + (gw_queue / 150.0) * 0.5), "sync_drift_score": drift},
            "reservations": {"request_rate": 20.0, "latency_ms": 12.0 + gw_lat * 0.2, "error_rate": gw_err * 0.5, "queue_depth": max(0.0, gw_queue * 0.2), "cpu_load": 0.2, "sync_drift_score": drift},
            "crew": {"request_rate": 15.0, "latency_ms": 10.0, "error_rate": 0.0, "queue_depth": 0.0, "cpu_load": 0.15, "sync_drift_score": drift},
            "baggage": {"request_rate": 10.0, "latency_ms": 14.0, "error_rate": 0.0, "queue_depth": 0.0, "cpu_load": 0.1, "sync_drift_score": drift},
        }

        # Build progressive temporal sequence of 5 snapshots
        seq = []
        for t in range(5):
            progress = (t + 1) / 5.0
            cur_metrics = {k: dict(v) for k, v in node_metrics.items()}
            cur_drift = 12.0 * (1.0 - progress) + drift * progress

            for node_name in cur_metrics:
                cur_metrics[node_name]["sync_drift_score"] = cur_drift

            cur_metrics["legacy-core"]["latency_ms"] = 15.0 * (1.0 - progress) + gw_lat * progress
            cur_metrics["legacy-core"]["error_rate"] = gw_err * progress
            cur_metrics["boundary-gateway"]["latency_ms"] = 15.0 * (1.0 - progress) + gw_lat * progress
            cur_metrics["boundary-gateway"]["queue_depth"] = gw_queue * progress
            cur_metrics["boundary-gateway"]["error_rate"] = gw_err * progress
            cur_metrics["reservations"]["latency_ms"] = 12.0 + (gw_lat * 0.35) * progress
            cur_metrics["reservations"]["error_rate"] = (gw_err * 0.4) * progress

            cur_snap = self.graph_builder.build_snapshot(cur_metrics, edges)
            seq.append((cur_snap.x_dict, cur_snap.edge_index_dict))

        # Forward pass
        with torch.no_grad():
            _, shared_repr, attributions = self.encoder(seq)
            preds = self.head(shared_repr)

        prob = float(preds["cascade_probability"].squeeze().item())
        lead_time_sec = float(preds["lead_time_seconds"].squeeze().item())
        
        # Conformal interval
        conf_lower, conf_upper = self.conformal.predict_interval(prob)

        # Most likely failure node
        rc_idx = int(preds["root_cause_probs"].argmax(dim=-1).item())
        root_cause = NODE_NAMES[rc_idx]

        # Severity
        sev_idx = int(preds["severity_probs"].argmax(dim=-1).item())
        severity = SEVERITY_LEVELS[sev_idx]

        affected_nodes = [root_cause]
        if prob > 0.5:
            affected_nodes.extend(["reservations", "crew", "baggage"])
        elif prob > 0.3:
            affected_nodes.append("reservations")
        affected_nodes = list(dict.fromkeys(affected_nodes))

        # Natural language brief
        explanation = format_incident_explanation(
            cascade_prob=prob,
            root_cause_node=root_cause,
            affected_nodes=affected_nodes,
            lead_time_sec=lead_time_sec,
            sync_drift_score=drift,
            conf_lower=conf_lower,
            conf_upper=conf_upper,
            relation_attributions=attributions,
        )

        return {
            "cascade_probability": round(prob, 4),
            "predicted_failure_location": root_cause,
            "predicted_root_cause_node": root_cause,
            "predicted_affected_nodes": affected_nodes,
            "severity": severity,
            "lead_time": round(lead_time_sec, 1),
            "estimated_lead_time_seconds": round(lead_time_sec, 1),
            "confidence": 0.90,
            "conformal_bounds": {
                "confidence_level": 0.90,
                "lower": conf_lower,
                "upper": conf_upper,
            },
            "boundary_sync_drift": round(drift, 2),
            "explanation": explanation,
            "model_metadata": {
                "engine": "RGCN-Hetero-CascadePredictor-v1",
                "calibration": "Split-Conformal-Sliding-Window",
                "mode": "trained-model-local",
            },
        }


_engine = None


def predict(
    graph_data: Optional[Dict[str, Any]] = None,
    sync_drift_score: Optional[float] = None,
    telemetry_features: Optional[Dict[str, Any]] = None,
    boundary_features: Optional[Dict[str, Any]] = None,
    active_fault: Optional[str] = None,
    fault_level: Optional[str] = None,
) -> Dict[str, Any]:
    """Single entry point for backend/cloud/sagemaker.py."""
    global _engine
    if _engine is None:
        _engine = CascadePredictorInferenceEngine()
    return _engine.predict(
        graph_data=graph_data,
        sync_drift_score=sync_drift_score,
        telemetry_features=telemetry_features,
        boundary_features=boundary_features,
        active_fault=active_fault,
        fault_level=fault_level,
    )

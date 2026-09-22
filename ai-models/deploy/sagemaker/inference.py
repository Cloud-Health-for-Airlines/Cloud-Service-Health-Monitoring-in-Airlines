"""Amazon SageMaker PyTorch Container Inference Handler for BACCP Cascade Predictor.

Implements the official SageMaker PyTorch model serving contract:
- model_fn(model_dir): Loads HeteroCascadePredictor from trained checkpoint.
- input_fn(request_body, request_content_type): Deserializes and validates JSON request.
- predict_fn(input_data, model): Executes multi-task inference and computes conformal intervals.
- output_fn(prediction, accept): Serializes prediction result to JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import torch

# Ensure local cascade predictor model is importable
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
PREDICTOR_DIR = PROJECT_ROOT / "ai-models/cascade_predictor"

for p in (PROJECT_ROOT, PREDICTOR_DIR, FILE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from model import HeteroCascadePredictor
except ImportError:
    # If in packaged tarball structure
    from .model import HeteroCascadePredictor  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baccp.sagemaker_inference")


def model_fn(model_dir: str) -> HeteroCascadePredictor:
    """Load trained HeteroCascadePredictor checkpoint into memory."""
    logger.info(f"Loading BACCP Cascade Predictor from model_dir: {model_dir}")
    model_path = Path(model_dir)

    candidate_files = [
        model_path / "cascade_predictor_tier1a.pt",
        model_path / "model.pt",
        model_path / "model_weights.pt",
    ]

    weights_file = None
    for cand in candidate_files:
        if cand.is_file():
            weights_file = cand
            break

    if weights_file is None:
        # Fall back to root repository weights if available during local testing
        fallback = PROJECT_ROOT / "ai-models/weights/cascade_predictor_tier1a.pt"
        if fallback.is_file():
            weights_file = fallback
        else:
            raise FileNotFoundError(f"No trained model checkpoint found in {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeteroCascadePredictor(
        node_in_dim=7,
        hidden_dim=64,
        num_relations=5,
        num_nodes=5,
        num_severities=5,
    ).to(device)

    state_dict = torch.load(weights_file, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Successfully loaded BACCP Cascade Predictor on device: {device}")
    return model


def input_fn(request_body: str, request_content_type: str = "application/json") -> Dict[str, Any]:
    """Deserialize and validate incoming JSON request."""
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}. Expected application/json")

    payload = json.loads(request_body) if isinstance(request_body, (str, bytes)) else request_body

    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    # Validate sync_drift_score if present
    if "sync_drift_score" in payload:
        drift = payload["sync_drift_score"]
        if not isinstance(drift, (int, float)) or drift < 0.0 or drift > 100.0:
            raise ValueError(f"Invalid sync_drift_score: {drift}. Must be numeric in [0.0, 100.0]")

    return payload


def predict_fn(input_data: Dict[str, Any], model: HeteroCascadePredictor) -> Dict[str, Any]:
    """Execute PyTorch forward inference and return normalized prediction."""
    drift = float(input_data.get("sync_drift_score", 12.4))
    active_fault = input_data.get("active_fault")
    fault_level = input_data.get("fault_level")

    snapshots = input_data.get("snapshots")
    if snapshots is None:
        snapshots = _synthesize_snapshots(drift, active_fault, fault_level)

    with torch.no_grad():
        result = model.predict_inference(snapshots)

    # Conformal bounds (90% prediction interval)
    conf_interval = result.get("confidence_interval_90", [0.0, 1.0])
    result["conformal_bounds"] = {
        "confidence_level": 0.90,
        "lower": conf_interval[0],
        "upper": conf_interval[1],
    }
    result["boundary_sync_drift"] = drift

    return result


def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> str:
    """Format and serialize prediction output to JSON."""
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept header: {accept}. Expected application/json")
    return json.dumps(prediction, default=str)


def _synthesize_snapshots(
    drift: float,
    active_fault: Optional[str] = None,
    fault_level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Synthesize 5 temporal snapshots matching canonical 5-node schema."""
    snapshots = []
    for t in range(5):
        prog = t / 4.0
        t_drift = drift * (0.8 + 0.2 * prog)
        node_features = []
        for i in range(5):
            type_idx = 0 if i < 3 else (1 if i == 3 else 2)
            one_hot = [1.0 if j == type_idx else 0.0 for j in range(3)]
            lat = 15.0 if i < 3 else (20.0 if i == 3 else 65.0)
            rate = 400.0 if i < 3 else (1200.0 if i == 3 else 350.0)
            err = 0.001
            cpu = 0.3

            if active_fault and t >= 2:
                mult = {"low": 1.5, "medium": 3.0, "high": 6.0}.get(fault_level, 2.0)
                lat *= mult
                err = min(0.3, err * mult * 10)
                cpu = min(0.95, cpu * mult)

            norm_lat = min(10.0, max(0.0, lat / 100.0))
            norm_rate = min(5.0, max(0.0, rate / 1000.0))
            norm_err = min(1.0, max(0.0, err))
            norm_cpu = min(1.0, max(0.0, cpu))
            node_features.append([norm_lat, norm_rate, norm_err, norm_cpu] + one_hot)

        snapshots.append({
            "t": t,
            "sync_drift_score": t_drift,
            "node_features": node_features,
        })
    return snapshots

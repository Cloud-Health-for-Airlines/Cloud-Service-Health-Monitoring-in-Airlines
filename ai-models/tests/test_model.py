"""Unit and Integration Tests for BACCP Cascade Predictor & Checkpoint Verification."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_DIR = ROOT / "ai-models/cascade_predictor"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PREDICTOR_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICTOR_DIR))

try:
    from model import HeteroCascadePredictor
except (ImportError, ModuleNotFoundError):
    from ai_models.cascade_predictor.model import HeteroCascadePredictor


class TestCascadePredictor(unittest.TestCase):
    """Test model architecture, checkpoint serialization, and deterministic inference."""

    @classmethod
    def setUpClass(cls):
        cls.weights_path = ROOT / "ai-models/weights/cascade_predictor_tier1a.pt"
        cls.meta_path = ROOT / "ai-models/weights/cascade_predictor_metadata.json"
        cls.data_path = ROOT / "ai-models/data/dataset.json"

    def test_checkpoint_files_exist(self):
        """Verify saved weights and metadata files exist."""
        self.assertTrue(self.weights_path.is_file(), f"Missing weights: {self.weights_path}")
        self.assertTrue(self.meta_path.is_file(), f"Missing metadata: {self.meta_path}")

        meta = json.loads(self.meta_path.read_text())
        self.assertEqual(meta["model_architecture"], "HeteroRGCN-GRU-MultiTask (Tier 1A)")
        self.assertIn("test_metrics", meta)
        self.assertGreaterEqual(meta["test_metrics"]["f1"], 0.85)

    def test_model_initialization_and_forward(self):
        """Verify model initialization and forward tensor pass."""
        model = HeteroCascadePredictor(
            node_in_dim=7,
            hidden_dim=64,
            num_relations=5,
            num_nodes=5,
            num_severities=5,
        )
        batch_size = 4
        T = 5
        x = torch.randn(batch_size, T, 5, 7)
        drift = torch.rand(batch_size, T)

        outputs = model.forward_tensors(x, drift)
        self.assertIn("cascade_prob", outputs)
        self.assertIn("lead_time", outputs)
        self.assertIn("root_cause_logits", outputs)
        self.assertIn("severity_logits", outputs)

        self.assertEqual(outputs["cascade_prob"].shape, (batch_size,))
        self.assertEqual(outputs["lead_time"].shape, (batch_size,))
        self.assertEqual(outputs["root_cause_logits"].shape, (batch_size, 5))
        self.assertEqual(outputs["severity_logits"].shape, (batch_size, 5))

        # Check values are in valid mathematical domains
        self.assertTrue((outputs["cascade_prob"] >= 0.0).all() and (outputs["cascade_prob"] <= 1.0).all())
        self.assertTrue((outputs["lead_time"] >= 0.0).all())

    def test_checkpoint_loading_clean_process(self):
        """Verify that weights load into a newly constructed model instance without error."""
        device = torch.device("cpu")
        model = HeteroCascadePredictor(
            node_in_dim=7,
            hidden_dim=64,
            num_relations=5,
            num_nodes=5,
            num_severities=5,
        ).to(device)

        state_dict = torch.load(self.weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()

    def test_deterministic_inference(self):
        """Verify that inference is completely deterministic across repeated invocations."""
        device = torch.device("cpu")
        model = HeteroCascadePredictor().to(device)
        model.load_state_dict(torch.load(self.weights_path, map_location=device, weights_only=False))
        model.eval()

        # Dummy sample snapshot sequence (5 snapshots)
        snapshots = [
            {
                "sync_drift_score": 55.0,
                "node_features": [[0.5, 0.4, 0.01, 0.6, 1.0, 0.0, 0.0] for _ in range(5)],
            }
            for _ in range(5)
        ]

        res1 = model.predict_inference(snapshots)
        res2 = model.predict_inference(snapshots)

        self.assertEqual(res1["cascade_probability"], res2["cascade_probability"])
        self.assertEqual(res1["estimated_lead_time_seconds"], res2["estimated_lead_time_seconds"])
        self.assertEqual(res1["predicted_root_cause_node"], res2["predicted_root_cause_node"])
        self.assertEqual(res1["severity"], res2["severity"])
        self.assertEqual(res1["confidence_interval_90"], res2["confidence_interval_90"])

    def test_dataset_schema(self):
        """Verify generated dataset adheres to schema."""
        self.assertTrue(self.data_path.is_file(), f"Dataset missing: {self.data_path}")
        data = json.loads(self.data_path.read_text())
        self.assertIn("metadata", data)
        self.assertIn("samples", data)
        self.assertGreaterEqual(len(data["samples"]), 1000)

        sample = data["samples"][0]
        self.assertIn("cascade_label", sample)
        self.assertIn("lead_time_seconds", sample)
        self.assertIn("root_cause_node", sample)
        self.assertIn("severity_level", sample)
        self.assertIn("snapshots", sample)
        self.assertEqual(len(sample["snapshots"]), 5)


if __name__ == "__main__":
    unittest.main()

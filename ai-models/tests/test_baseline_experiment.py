"""Unit and Integration Tests for BACCP Baseline Models and Evaluation Artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_DIR = ROOT / "ai-models/cascade_predictor"
CB_DIR = ROOT / "ai-models/circuit-breaker"
EVAL_DIR = ROOT / "ai-models/evaluation"

for p in (ROOT, PREDICTOR_DIR, CB_DIR, EVAL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from baselines import DomainTypedCascadePredictor, FlatGraphCascadePredictor
    from model import HeteroCascadePredictor
    from ppo_agent import PPOAgent
    from run_full_evaluation import validate_checkpoints, validate_dataset
except ImportError:
    pass


class TestBaselineModels(unittest.TestCase):
    """Test baseline model architectures and forward passes."""

    def test_flat_graph_predictor_forward(self):
        """Verify FlatGraphCascadePredictor forward pass and output shapes."""
        model = FlatGraphCascadePredictor()
        batch_size = 4
        T = 5
        x = torch.randn(batch_size, T, 5, 4)

        outputs = model.forward_tensors(x)
        self.assertIn("cascade_prob", outputs)
        self.assertIn("lead_time", outputs)
        self.assertIn("root_cause_logits", outputs)
        self.assertIn("severity_logits", outputs)

        self.assertEqual(outputs["cascade_prob"].shape, (batch_size,))
        self.assertEqual(outputs["lead_time"].shape, (batch_size,))
        self.assertEqual(outputs["root_cause_logits"].shape, (batch_size, 5))
        self.assertEqual(outputs["severity_logits"].shape, (batch_size, 5))

    def test_domain_typed_predictor_forward(self):
        """Verify DomainTypedCascadePredictor forward pass and output shapes."""
        model = DomainTypedCascadePredictor()
        batch_size = 4
        T = 5
        x = torch.randn(batch_size, T, 5, 4)

        outputs = model.forward_tensors(x)
        self.assertIn("cascade_prob", outputs)
        self.assertIn("lead_time", outputs)
        self.assertIn("root_cause_logits", outputs)
        self.assertIn("severity_logits", outputs)

        self.assertEqual(outputs["cascade_prob"].shape, (batch_size,))
        self.assertEqual(outputs["lead_time"].shape, (batch_size,))
        self.assertEqual(outputs["root_cause_logits"].shape, (batch_size, 5))
        self.assertEqual(outputs["severity_logits"].shape, (batch_size, 5))


class TestCheckpoints(unittest.TestCase):
    """Verify all 4 trained checkpoints exist, are non-empty, and load cleanly."""

    def setUp(self):
        self.weights_dir = ROOT / "ai-models/weights"

    def test_checkpoint_files_exist_and_load(self):
        checkpoints = {
            "cascade_predictor_tier1a.pt": HeteroCascadePredictor(),
            "baseline_flat_graph.pt": FlatGraphCascadePredictor(),
            "baseline_domain_typed.pt": DomainTypedCascadePredictor(),
        }

        for ckpt_name, model in checkpoints.items():
            ckpt_path = self.weights_dir / ckpt_name
            self.assertTrue(ckpt_path.is_file(), f"Missing checkpoint: {ckpt_name}")
            self.assertGreater(ckpt_path.stat().st_size, 1000, f"Checkpoint too small: {ckpt_name}")

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt)
            model.eval()

    def test_ppo_checkpoint_exists_and_loads(self):
        ppo_path = self.weights_dir / "circuit_breaker_ppo.pt"
        self.assertTrue(ppo_path.is_file(), "Missing PPO checkpoint")
        self.assertGreater(ppo_path.stat().st_size, 1000)

        agent = PPOAgent(state_dim=6, action_dim=1)
        agent.load_checkpoint(ppo_path)
        self.assertIsNotNone(agent.ac)


class TestEvaluationArtifacts(unittest.TestCase):
    """Verify generated baseline comparison reports, tables, and figures."""

    def setUp(self):
        self.results_dir = ROOT / "results"
        self.plots_dir = self.results_dir / "plots"

    def test_evaluation_data_files(self):
        json_file = self.results_dir / "baseline_comparison.json"
        csv_file = self.results_dir / "baseline_comparison.csv"
        md_file = self.results_dir / "baseline_comparison.md"

        self.assertTrue(json_file.is_file(), f"Missing JSON: {json_file}")
        self.assertTrue(csv_file.is_file(), f"Missing CSV: {csv_file}")
        self.assertTrue(md_file.is_file(), f"Missing MD: {md_file}")

        # JSON content validation
        with open(json_file) as f:
            data = json.load(f)
        self.assertIn("predictive_models", data)
        self.assertIn("circuit_breaker_policies", data)
        self.assertIn("plots_generated", data)

        self.assertIn("baseline_1_flat_gcn", data["predictive_models"])
        self.assertIn("baseline_2_domain_typed", data["predictive_models"])
        self.assertIn("model_3_baccp_hetero_rgcn", data["predictive_models"])

        self.assertIn("no_mitigation", data["circuit_breaker_policies"])
        self.assertIn("rule_based", data["circuit_breaker_policies"])
        self.assertIn("ppo", data["circuit_breaker_policies"])

        # CSV content validation
        with open(csv_file) as f:
            reader = csv.reader(f)
            rows = [r for r in reader if r]
        self.assertGreaterEqual(len(rows), 4)  # Header + 3 models

        # MD content validation
        md_text = md_file.read_text()
        self.assertIn("BACCP Experimental Baseline Comparison Report", md_text)
        self.assertIn("BACCP HeteroRGCN", md_text)
        self.assertIn("BACCP Trained PPO Policy", md_text)

    def test_evaluation_plots(self):
        expected_plots = [
            "model_performance_comparison.png",
            "lead_time_distribution.png",
            "circuit_breaker_throughput_tradeoff.png",
            "service_priority_retention.png",
        ]
        for plot_name in expected_plots:
            plot_path = self.plots_dir / plot_name
            self.assertTrue(plot_path.is_file(), f"Missing plot: {plot_name}")
            self.assertGreater(plot_path.stat().st_size, 10000, f"Plot appears corrupted/empty: {plot_name}")


class TestEvaluationPipelineValidation(unittest.TestCase):
    """Test validation steps in the evaluation pipeline runner."""

    def test_pipeline_validators(self):
        data_path = ROOT / "ai-models/data/dataset.json"
        # Should not raise any exception
        validate_dataset(data_path)
        validate_checkpoints()


if __name__ == "__main__":
    unittest.main()

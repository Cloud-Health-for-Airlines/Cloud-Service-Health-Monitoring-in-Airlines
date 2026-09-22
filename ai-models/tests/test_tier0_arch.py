"""Architecture unit tests for Tier 0 BACCP baseline components.

Verifies:
1. Generation-typed heterogeneous graph schema & tensor conversions
2. Boundary sync-drift score computation & saturation boundaries
3. GAT-GRU forward & backward gradient passes without NaNs
4. DQN agent action selection and training step stability
"""

from __future__ import annotations

import unittest
import torch
import numpy as np

import sys
from pathlib import Path
AI_MODELS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_MODELS_DIR))

# Also allow direct package imports
sys.path.insert(0, str(AI_MODELS_DIR.parent))

from importlib import import_module
schema_mod = import_module("ai-models.graph-builder.schema")
GenerationTypedGraph = schema_mod.GenerationTypedGraph

features_mod = import_module("ai-models.graph-builder.features")
compute_sync_drift_score = features_mod.compute_sync_drift_score
compute_topological_features = features_mod.compute_topological_features

gat_gru_mod = import_module("ai-models.cascade-predictor.baseline_gat_gru")
BaselineGATGRU = gat_gru_mod.BaselineGATGRU
EdgeFeaturedGATLayer = gat_gru_mod.EdgeFeaturedGATLayer

dqn_mod = import_module("ai-models.circuit-breaker.baseline_dqn")
BaselineDQNAgent = dqn_mod.BaselineDQNAgent


class TestTier0Architecture(unittest.TestCase):
    """Test suite ensuring Tier 0 architectural integrity."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.builder = GenerationTypedGraph()

        # Canonical testbed 5-node topology
        self.node_metrics = {
            "legacy-core": {"request_rate": 30.0, "latency_ms": 20.0, "error_rate": 0.0, "queue_depth": 0.0, "cpu_load": 0.4},
            "boundary-gateway": {"request_rate": 45.0, "latency_ms": 25.0, "error_rate": 0.02, "queue_depth": 5.0, "cpu_load": 0.3},
            "reservations": {"request_rate": 20.0, "latency_ms": 12.0, "error_rate": 0.0, "queue_depth": 0.0, "cpu_load": 0.2},
            "crew": {"request_rate": 15.0, "latency_ms": 10.0, "error_rate": 0.0, "queue_depth": 0.0, "cpu_load": 0.15},
            "baggage": {"request_rate": 10.0, "latency_ms": 14.0, "error_rate": 0.0, "queue_depth": 0.0, "cpu_load": 0.1},
        }
        self.edges = [
            {"source": "boundary-gateway", "destination": "legacy-core", "protocol": "tcp", "observation_count": 100},
            {"source": "legacy-core", "destination": "boundary-gateway", "protocol": "tcp", "observation_count": 98},
            {"source": "boundary-gateway", "destination": "reservations", "protocol": "http", "observation_count": 50},
            {"source": "reservations", "destination": "boundary-gateway", "protocol": "http", "observation_count": 50},
            {"source": "reservations", "destination": "crew", "protocol": "http", "observation_count": 20},
        ]

    def test_schema_snapshot_and_flat_conversion(self):
        """Verify heterogeneous graph creation and homogeneous flattening."""
        snapshot = self.builder.build_snapshot(self.node_metrics, self.edges)

        # Check generation types exist
        self.assertIn("legacy", snapshot.x_dict)
        self.assertIn("boundary-gateway", snapshot.x_dict)
        self.assertIn("cloud-native", snapshot.x_dict)
        self.assertEqual(snapshot.x_dict["legacy"].shape[0], 1)
        self.assertEqual(snapshot.x_dict["boundary-gateway"].shape[0], 1)
        self.assertEqual(snapshot.x_dict["cloud-native"].shape[0], 3)

        # Check flattened homogeneous graph
        flat_x, flat_edges, flat_attrs = self.builder.to_flat_tensors(snapshot)
        self.assertEqual(flat_x.shape, (5, 6))
        self.assertEqual(flat_edges.shape[0], 2)
        self.assertEqual(flat_attrs.shape[1], 3)
        self.assertFalse(torch.isnan(flat_x).any())

    def test_sync_drift_score_dynamics(self):
        """Verify mathematical sync drift conforms to nominal/critical thresholds."""
        # 1. Nominal healthy state
        nominal_drift = compute_sync_drift_score(
            gateway_submission_rate=50.0,
            legacy_completion_rate=50.0,
            gateway_queue_depth=0.0,
            boundary_latency_ms=15.0,
        )
        self.assertLess(nominal_drift, 45.0)

        # 2. Elevated queue & latency stall
        elevated_drift = compute_sync_drift_score(
            gateway_submission_rate=60.0,
            legacy_completion_rate=30.0,
            gateway_queue_depth=35.0,
            boundary_latency_ms=250.0,
        )
        self.assertGreaterEqual(elevated_drift, 45.0)

        # 3. Severe failure state
        critical_drift = compute_sync_drift_score(
            gateway_submission_rate=80.0,
            legacy_completion_rate=5.0,
            gateway_queue_depth=120.0,
            boundary_latency_ms=1500.0,
            boundary_error_rate=0.40,
        )
        self.assertGreaterEqual(critical_drift, 70.0)
        self.assertLessEqual(critical_drift, 100.0)

    def test_gat_gru_forward_and_backward_pass(self):
        """Verify baseline GAT-GRU forward pass, sigmoid bounds, and gradient backpropagation."""
        model = BaselineGATGRU(in_node_feats=6, in_edge_feats=3, gnn_hidden_dim=32, gru_hidden_dim=24)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Build a sequence of 4 snapshots
        snapshots = []
        for t in range(4):
            # Mutate metrics slightly to simulate time series
            metrics = {k: dict(v) for k, v in self.node_metrics.items()}
            metrics["boundary-gateway"]["latency_ms"] += t * 10.0
            snap = self.builder.build_snapshot(metrics, self.edges, timestamp=float(t))
            flat_x, flat_edges, flat_attrs = self.builder.to_flat_tensors(snap)
            snapshots.append((flat_x, flat_edges, flat_attrs))

        # Forward pass
        prob = model(snapshots)
        self.assertTrue(torch.is_tensor(prob))
        self.assertEqual(prob.shape, torch.Size([]))
        self.assertGreaterEqual(prob.item(), 0.0)
        self.assertLessEqual(prob.item(), 1.0)
        self.assertFalse(torch.isnan(prob))

        # Backward pass
        target = torch.tensor(1.0, dtype=torch.float32)
        loss = torch.nn.functional.binary_cross_entropy(prob, target)
        optimizer.zero_grad()
        loss.backward()

        # Ensure all parameters have valid non-NaN gradients
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Param {name} has None grad")
                self.assertFalse(torch.isnan(param.grad).any(), f"Param {name} has NaN grad")
                grad_count += 1
        self.assertGreater(grad_count, 0)

    def test_baseline_dqn_agent(self):
        """Verify single-agent DQN action selection, mitigation formatting, and training step."""
        agent = BaselineDQNAgent(state_dim=5, action_dim=5)
        
        # Test state extraction
        state = agent.extract_state(cascade_probability=0.85, sync_drift_score=75.0, gateway_latency_ms=600.0)
        self.assertEqual(state.shape, (5,))

        # Test action formatting
        action_name, throttle_rate, reason = agent.choose_mitigation(
            cascade_probability=0.85,
            boundary_sync_drift=75.0,
            explore=False,
        )
        self.assertIn(action_name, ["CLOSED", "THROTTLED", "OPEN"])
        self.assertGreaterEqual(throttle_rate, 0.0)
        self.assertLessEqual(throttle_rate, 1.0)
        self.assertIn("DQN Policy", reason)

        # Test experience storage and train step
        for _ in range(40):
            s = torch.randn(5)
            a = np.random.randint(0, 5)
            r = float(np.random.randn())
            s_next = torch.randn(5)
            d = False
            agent.store_transition(s, a, r, s_next, d)

        loss = agent.train_step(batch_size=16)
        self.assertIsNotNone(loss)
        self.assertFalse(np.isnan(loss))


if __name__ == "__main__":
    unittest.main()

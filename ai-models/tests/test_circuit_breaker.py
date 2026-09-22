"""Unit and Integration Tests for BACCP PPO Circuit Breaker & Checkpoint Verification."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CB_DIR = ROOT / "ai-models/circuit-breaker"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CB_DIR) not in sys.path:
    sys.path.insert(0, str(CB_DIR))

try:
    from env import BoundaryMitigationEnv, map_action_to_state
    from ppo_agent import PPOAgent
    from reward import compute_reward, compute_service_throughput
except (ImportError, ModuleNotFoundError):
    from ai_models.circuit_breaker.env import BoundaryMitigationEnv, map_action_to_state
    from ai_models.circuit_breaker.ppo_agent import PPOAgent
    from ai_models.circuit_breaker.reward import compute_reward, compute_service_throughput


class TestCircuitBreakerPPO(unittest.TestCase):
    """Test PPO checkpoint loading, action bounds, mapping, determinism, and safety."""

    @classmethod
    def setUpClass(cls):
        cls.ckpt_path = ROOT / "ai-models/weights/circuit_breaker_ppo.pt"

    def test_checkpoint_exists_and_loads(self):
        """Verify checkpoint file exists and loads into PPOAgent cleanly."""
        self.assertTrue(self.ckpt_path.is_file(), f"Missing checkpoint: {self.ckpt_path}")
        agent = PPOAgent(state_dim=6, action_dim=1)
        agent.load_checkpoint(self.ckpt_path)
        self.assertIsNotNone(agent.ac)

    def test_action_mapping_boundaries(self):
        """Verify CLOSED / THROTTLED / OPEN mapping across action space [-1.0, 1.0]."""
        # CLOSED domain: a < -0.33
        state, throttle = map_action_to_state(-0.95)
        self.assertEqual(state, "CLOSED")
        self.assertEqual(throttle, 0.0)

        state, throttle = map_action_to_state(-0.35)
        self.assertEqual(state, "CLOSED")
        self.assertEqual(throttle, 0.0)

        # THROTTLED domain: -0.33 <= a <= 0.33
        state, throttle = map_action_to_state(-0.33)
        self.assertEqual(state, "THROTTLED")
        self.assertAlmostEqual(throttle, 0.10, places=2)

        state, throttle = map_action_to_state(0.0)
        self.assertEqual(state, "THROTTLED")
        self.assertAlmostEqual(throttle, 0.50, places=2)

        state, throttle = map_action_to_state(0.33)
        self.assertEqual(state, "THROTTLED")
        self.assertAlmostEqual(throttle, 0.90, places=2)

        # OPEN domain: a > 0.33
        state, throttle = map_action_to_state(0.35)
        self.assertEqual(state, "OPEN")
        self.assertEqual(throttle, 1.0)

        state, throttle = map_action_to_state(0.99)
        self.assertEqual(state, "OPEN")
        self.assertEqual(throttle, 1.0)

    def test_actions_remain_within_valid_range(self):
        """Verify policy actions remain strictly within [-1.0, 1.0] and throttle in [0.0, 1.0]."""
        agent = PPOAgent(state_dim=6, action_dim=1)
        agent.load_checkpoint(self.ckpt_path)

        for _ in range(30):
            dummy_state = np.random.uniform(0.0, 1.0, size=6).astype(np.float32)
            raw_a, state_name, throttle = agent.act(dummy_state, deterministic=False)
            self.assertGreaterEqual(raw_a, -1.0)
            self.assertLessEqual(raw_a, 1.0)
            self.assertIn(state_name, ("CLOSED", "THROTTLED", "OPEN"))
            self.assertGreaterEqual(throttle, 0.0)
            self.assertLessEqual(throttle, 1.0)

    def test_deterministic_inference_reproducibility(self):
        """Verify deterministic=True yields identical action on same input."""
        agent = PPOAgent(state_dim=6, action_dim=1)
        agent.load_checkpoint(self.ckpt_path)

        dummy_state = np.array([0.65, 0.55, 0.35, 0.02, 0.0, 0.40], dtype=np.float32)
        raw_a1, state1, th1 = agent.act(dummy_state, deterministic=True)
        raw_a2, state2, th2 = agent.act(dummy_state, deterministic=True)

        self.assertEqual(raw_a1, raw_a2)
        self.assertEqual(state1, state2)
        self.assertEqual(th1, th2)

    def test_unsafe_actions_rejected(self):
        """Verify invalid or extreme float actions are caught and rejected."""
        with self.assertRaises(ValueError):
            map_action_to_state(float("nan"))
        with self.assertRaises(ValueError):
            map_action_to_state(float("inf"))
        with self.assertRaises(ValueError):
            map_action_to_state(float("-inf"))

    def test_service_prioritization_logic(self):
        """Verify reservations > crew > baggage shedding order."""
        # Low throttle (0.10) sheds only baggage
        tp_low = compute_service_throughput(0.10)
        self.assertEqual(tp_low["reservations"], 1.0)
        self.assertEqual(tp_low["crew"], 1.0)
        self.assertEqual(tp_low["baggage"], 0.50)

        # Medium throttle (0.35) sheds all baggage and some crew
        tp_med = compute_service_throughput(0.35)
        self.assertEqual(tp_med["reservations"], 1.0)
        self.assertLess(tp_med["crew"], 1.0)
        self.assertEqual(tp_med["baggage"], 0.0)

        # High throttle (0.75) sheds baggage, crew, and some reservations
        tp_high = compute_service_throughput(0.75)
        self.assertLess(tp_high["reservations"], 1.0)
        self.assertEqual(tp_high["crew"], 0.0)
        self.assertEqual(tp_high["baggage"], 0.0)


if __name__ == "__main__":
    unittest.main()

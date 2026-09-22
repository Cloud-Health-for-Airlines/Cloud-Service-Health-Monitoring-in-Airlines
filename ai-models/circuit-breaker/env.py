"""Gym-style Environment for Boundary Circuit Breaker RL Policy Training."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

try:
    from .reward import MultiObjectiveReward, CRITICALITY_WEIGHTS
except (ImportError, ValueError):
    from reward import MultiObjectiveReward, CRITICALITY_WEIGHTS


class BoundaryCircuitBreakerEnv:
    """Environment simulating airline boundary load shedding and circuit breaking."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.reward_fn = MultiObjectiveReward()
        self.state_dim = 5
        self.reset()

    def reset(self, hazard_mode: Optional[bool] = None) -> np.ndarray:
        """Reset environment to initial state."""
        # Randomly choose if episode is under hazard/cascade stress (60% probability)
        self.is_hazard = (self.rng.rand() < 0.60) if hazard_mode is None else hazard_mode
        self.step_count = 0
        self.max_steps = 25

        self.cascade_prob = self.rng.uniform(0.6, 0.95) if self.is_hazard else self.rng.uniform(0.02, 0.25)
        self.sync_drift = self.rng.uniform(50.0, 85.0) if self.is_hazard else self.rng.uniform(10.0, 30.0)
        self.gw_queue = self.rng.uniform(30.0, 80.0) if self.is_hazard else 0.0
        self.gw_latency = 15.0 + self.gw_queue * 3.0
        self.gw_error = 0.0
        self.throttle_rate = 0.0

        self.service_latencies = {
            "reservations": 12.0,
            "crew": 10.0,
            "baggage": 14.0,
        }
        self.nominal_latencies = dict(self.service_latencies)

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.array([
            self.cascade_prob,
            self.sync_drift / 100.0,
            min(1.0, self.gw_latency / 1500.0),
            self.gw_error,
            self.throttle_rate,
        ], dtype=np.float32)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Apply throttle action and advance simulation by 1 step.
        
        action: continuous float in [0.0, 1.0] representing throttle rate.
        """
        self.step_count += 1
        self.throttle_rate = float(np.clip(action, 0.0, 1.0))

        # Effect of throttle on gateway queue:
        # Throttle rate reduces incoming demand, draining queue
        drain = self.throttle_rate * 25.0
        inflow = 18.0 if self.is_hazard else 3.0
        self.gw_queue = max(0.0, min(150.0, self.gw_queue + inflow - drain))
        self.gw_latency = max(15.0, 15.0 + self.gw_queue * 3.0)

        # Gateway error rate
        if self.gw_queue > 60.0:
            self.gw_error = min(0.9, (self.gw_queue - 60.0) / 90.0)
        else:
            self.gw_error = max(0.0, self.gw_error - 0.1)

        # Downstream propagation if unmitigated
        cascade_occurred = False
        if self.is_hazard and self.throttle_rate < 0.35 and self.gw_queue > 50.0:
            cascade_occurred = True
            self.service_latencies["reservations"] = self.gw_latency * 0.5
            self.service_latencies["crew"] = self.gw_latency * 0.3
            self.service_latencies["baggage"] = self.gw_latency * 0.2
        else:
            # Protected by throttle or healthy
            for s in self.service_latencies:
                self.service_latencies[s] = self.nominal_latencies[s] + self.rng.normal(0, 1.0)

        # Update drift
        if self.is_hazard and not cascade_occurred and self.throttle_rate >= 0.4:
            self.sync_drift = max(20.0, self.sync_drift - 5.0)
            self.cascade_prob = max(0.2, self.cascade_prob - 0.08)

        # Compute reward
        reward, reward_info = self.reward_fn.calculate_reward(
            throttle_rate=self.throttle_rate,
            cascade_risk=self.cascade_prob,
            cascade_occurred=cascade_occurred,
            nominal_traffic=50.0,
            service_latencies=self.service_latencies,
            nominal_latencies=self.nominal_latencies,
        )

        done = (self.step_count >= self.max_steps)
        info = {
            "cascade_occurred": cascade_occurred,
            "reward_components": reward_info,
            "sync_drift": self.sync_drift,
        }

        return self._get_state(), reward, done, info

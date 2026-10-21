"""Gym-style Environment for Boundary Circuit Breaker RL Policy Training."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    from .reward import CRITICALITY_WEIGHTS, MultiObjectiveReward, compute_reward, compute_service_throughput
except (ImportError, ValueError):
    from reward import CRITICALITY_WEIGHTS, MultiObjectiveReward, compute_reward, compute_service_throughput


def map_action_to_state(action: float) -> Tuple[str, float]:
    """Map continuous action in [-1.0, 1.0] to discrete mitigation state and continuous throttle rate.

    - a < -0.33: CLOSED (0.0 throttle)
    - -0.33 <= a <= 0.33: THROTTLED (linearly mapped to [0.10, 0.90])
    - a > 0.33: OPEN (1.0 isolation)
    """
    if isinstance(action, (int, float)):
        if math.isnan(action) or math.isinf(action):
            raise ValueError(f"Invalid non-finite action: {action}")
    else:
        raise ValueError(f"Action must be numeric, got {type(action)}")

    if action < -0.33:
        return "CLOSED", 0.0
    elif action > 0.33:
        return "OPEN", 1.0
    else:
        # Linear mapping: a = -0.33 -> 0.10; a = 0.0 -> 0.50; a = 0.33 -> 0.90
        throttle = 0.50 + action * (0.80 / 0.66)
        return "THROTTLED", round(max(0.10, min(0.90, throttle)), 2)


class BoundaryMitigationEnv:
    """Environment simulating airline boundary load shedding and circuit breaking with 6D telemetry."""

    def __init__(self, max_steps: int = 20, seed: Optional[int] = 42):
        self.max_steps = max_steps
        self.state_dim = 6
        self.action_dim = 1
        self.rng = np.random.RandomState(seed or 42)
        self.reward_fn = MultiObjectiveReward()
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # 60% probability of elevated hazard stress
        self.is_hazard = bool(self.rng.rand() < 0.60)
        self.step_idx = 0

        self.cascade_prob = self.rng.uniform(0.65, 0.95) if self.is_hazard else self.rng.uniform(0.01, 0.20)
        self.boundary_drift = self.rng.uniform(50.0, 90.0) if self.is_hazard else self.rng.uniform(10.0, 30.0)
        self.gw_queue = self.rng.uniform(30.0, 75.0) if self.is_hazard else 0.0
        self.gw_latency = 15.0 + self.gw_queue * 3.0
        self.gw_error = min(0.3, self.gw_queue / 200.0) if self.is_hazard else 0.0
        self.throttle_rate = 0.0

        info = {
            "is_fault_active": self.is_hazard,
            "boundary_drift": round(self.boundary_drift, 2),
            "step": self.step_idx,
        }
        return self._get_obs(), info

    def _get_obs(self) -> np.ndarray:
        """Construct 6D observation vector."""
        return np.array([
            self.cascade_prob,
            self.boundary_drift / 100.0,
            min(1.0, self.gw_latency / 1500.0),
            min(1.0, self.gw_queue / 100.0),
            self.gw_error,
            self.throttle_rate,
        ], dtype=np.float32)

    def step(self, raw_action: float) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance simulation by 1 step applying continuous or mapped action."""
        self.step_idx += 1
        action_name, throttle = map_action_to_state(raw_action)
        self.throttle_rate = float(throttle)

        # Gateway queue dynamics
        drain = self.throttle_rate * 30.0
        inflow = 22.0 if self.is_hazard else 4.0
        self.gw_queue = max(0.0, min(150.0, self.gw_queue + inflow - drain))
        self.gw_latency = max(15.0, 15.0 + self.gw_queue * 3.0)

        if self.gw_queue > 50.0:
            self.gw_error = min(0.5, (self.gw_queue - 50.0) / 100.0)
        else:
            self.gw_error = max(0.0, self.gw_error - 0.05)

        # Cascade occurrence logic
        cascade_occurred = False
        if self.is_hazard and self.throttle_rate < 0.25 and self.gw_queue > 60.0:
            cascade_occurred = True

        # Recovery dynamics when mitigating effectively
        if self.is_hazard and not cascade_occurred and self.throttle_rate >= 0.30:
            self.boundary_drift = max(15.0, self.boundary_drift - 8.0)
            self.cascade_prob = max(0.05, self.cascade_prob - 0.12)

        # Service prioritization throughput
        tp_info = compute_service_throughput(self.throttle_rate)

        # Compute multi-objective reward
        reward = compute_reward(
            throttle_rate=self.throttle_rate,
            cascade_risk=self.cascade_prob,
            cascade_occurred=cascade_occurred,
            throughput_info=tp_info,
        )

        terminated = cascade_occurred
        truncated = (self.step_idx >= self.max_steps)

        step_info = {
            "step": self.step_idx,
            "throttle_rate": self.throttle_rate,
            "action": action_name,
            "throughput_info": tp_info,
            "gateway_latency_ms": round(self.gw_latency, 2),
            "boundary_drift": round(self.boundary_drift, 2),
            "cascade_occurred": cascade_occurred,
        }

        return self._get_obs(), reward, terminated, truncated, step_info


# Backward-compatible alias
BoundaryCircuitBreakerEnv = BoundaryMitigationEnv

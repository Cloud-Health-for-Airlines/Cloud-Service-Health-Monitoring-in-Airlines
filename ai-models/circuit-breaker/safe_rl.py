"""Tier 2B: Constrained MDP / Safe Reinforcement Learning for Airline Mitigation.

Enforces critical airline business and safety constraints:
1. Hard constraint: Never completely sever passenger reservation ticketing (max throttle <= 85%).
2. Lagrangian relaxation: Learns dual variable lambda to bound false-alarm throttling to <= 5%.
3. Safety projection filter: Intercepts raw policy actions to ensure operational compliance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class LagrangianSafetyFilter:
    """Constrained MDP Lagrangian safety filter for circuit breaking policies."""

    def __init__(
        self,
        max_allowed_throttle: float = 0.85,
        max_nominal_throttle: float = 0.08,
        cost_limit: float = 0.05,
        lr_lagrangian: float = 0.01,
        init_lambda: float = 1.0,
    ):
        self.max_allowed_throttle = max_allowed_throttle
        self.max_nominal_throttle = max_nominal_throttle
        self.cost_limit = cost_limit
        self.lr_lagrangian = lr_lagrangian
        self.lagrange_multiplier = float(init_lambda)

    def filter_action(
        self,
        raw_throttle_rate: float,
        cascade_probability: float,
        boundary_sync_drift: float,
    ) -> Tuple[float, bool, str]:
        """Project raw policy action into certified safe operational bounds.
        
        Returns:
            safe_throttle_rate: float in [0.0, max_allowed_throttle]
            was_intervened: bool indicating if safety filter overrode action
            safety_reason: explanation of intervention
        """
        throttle = float(raw_throttle_rate)
        was_intervened = False
        reason = "Passed certified safety constraints"

        # Hard Constraint 1: Airline Revenue Continuity
        # Under no circumstance may the reservation gateway be severed 100%
        if throttle > self.max_allowed_throttle:
            throttle = self.max_allowed_throttle
            was_intervened = True
            reason = f"Safety Barrier: Throttling clamped to {self.max_allowed_throttle:.0%} to preserve passenger booking continuity."

        # Hard Constraint 2: False-Alarm Protection in Nominal Operations
        # If drift is nominal (< 25%) and cascade probability is low (< 0.25), limit throttle
        if cascade_probability < 0.25 and boundary_sync_drift < 25.0:
            if throttle > self.max_nominal_throttle:
                throttle = 0.0
                was_intervened = True
                reason = "Safety Barrier: False-alarm throttling suppressed during verified nominal flight operations."

        return throttle, was_intervened, reason

    def update_lagrangian(self, episode_mean_constraint_cost: float) -> float:
        """Dual gradient ascent on Lagrange multiplier: lambda <- max(0, lambda + lr * (cost - limit))."""
        violation = episode_mean_constraint_cost - self.cost_limit
        self.lagrange_multiplier = max(0.0, self.lagrange_multiplier + self.lr_lagrangian * violation)
        return self.lagrange_multiplier

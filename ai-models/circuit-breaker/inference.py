"""Inference entry point for BACCP Circuit Breaker.

Provides a singleton interface for AWS Lambda / CircuitBreakerManager integration:
choose_action(cascade_probability, boundary_sync_drift, state=None) -> (action, throttle_rate, reason)
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("baccp.inference.breaker")

AI_MODELS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = AI_MODELS_DIR.parent
if str(AI_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(AI_MODELS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from importlib import import_module
ppo_mod = import_module("circuit-breaker.ppo_agent")
PPOAgent = ppo_mod.PPOAgent

WEIGHTS_PATH = AI_MODELS_DIR / "weights" / "circuit_breaker_ppo.pt"


class CircuitBreakerInferenceEngine:
    """Singleton policy loader for Tier 1F PPO circuit breaker."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.agent = PPOAgent(state_dim=5, hidden_dim=64)
        self.is_loaded = False

        if WEIGHTS_PATH.exists():
            try:
                self.agent.load_checkpoint(WEIGHTS_PATH)
                self.is_loaded = True
                logger.info(f"Successfully loaded trained PPO policy weights from {WEIGHTS_PATH}")
            except Exception as exc:
                logger.warning(f"Could not load circuit breaker weights ({exc}). Engine uninitialized.")
        else:
            logger.warning(f"No checkpoint found at {WEIGHTS_PATH}.")

        self._initialized = True

    def choose_action(
        self,
        cascade_probability: float,
        boundary_sync_drift: float,
        state: Optional[Dict[str, Any]] = None,
        gateway_latency_ms: Optional[float] = None,
        gateway_error_rate: Optional[float] = None,
        current_throttle_rate: Optional[float] = None,
        **kwargs,
    ) -> Tuple[str, float, str]:
        """Query policy and return (action, throttle_rate, reason)."""
        if not self.is_loaded:
            raise RuntimeError("Circuit breaker policy weights not loaded.")

        state = state or {}
        gw_lat = float(gateway_latency_ms if gateway_latency_ms is not None else state.get("gateway_latency_ms", 15.0))
        gw_err = float(gateway_error_rate if gateway_error_rate is not None else state.get("gateway_error_rate", 0.0))
        curr_rate = float(current_throttle_rate if current_throttle_rate is not None else state.get("current_throttle_rate", 0.0))

        return self.agent.choose_mitigation(
            cascade_probability=cascade_probability,
            boundary_sync_drift=boundary_sync_drift,
            gateway_latency_ms=gw_lat,
            gateway_error_rate=gw_err,
            current_throttle_rate=curr_rate,
        )


_engine = None


def choose_action(
    cascade_probability: float,
    boundary_sync_drift: float,
    state: Optional[Dict[str, Any]] = None,
    gateway_latency_ms: Optional[float] = None,
    gateway_error_rate: Optional[float] = None,
    current_throttle_rate: Optional[float] = None,
    **kwargs,
) -> Tuple[str, float, str]:
    """Single entry point for backend/cloud/lambda_handler.py."""
    global _engine
    if _engine is None:
        _engine = CircuitBreakerInferenceEngine()
    return _engine.choose_action(
        cascade_probability=cascade_probability,
        boundary_sync_drift=boundary_sync_drift,
        state=state,
        gateway_latency_ms=gateway_latency_ms,
        gateway_error_rate=gateway_error_rate,
        current_throttle_rate=current_throttle_rate,
        **kwargs,
    )


# Alias
recommend_action = choose_action

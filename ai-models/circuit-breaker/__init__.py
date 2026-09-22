"""Circuit Breaker module: Baseline DQN, Tier 1 PPO Agent, Multi-Objective Reward, MAPPO, and Safe RL."""

from __future__ import annotations

import sys
from pathlib import Path

_curr_dir = Path(__file__).resolve().parent
if str(_curr_dir) not in sys.path:
    sys.path.insert(0, str(_curr_dir))

from ppo_agent import PPOAgent, ActorCritic
from baseline_dqn import BaselineDQNAgent
from env import BoundaryCircuitBreakerEnv
from reward import MultiObjectiveReward, CRITICALITY_WEIGHTS
from mappo_agent import MAPPOAgent, MAPPOActor, MAPPOCentralizedCritic
from safe_rl import LagrangianSafetyFilter


def choose_action(*args, **kwargs):
    """Lazy-loaded inference entrypoint preventing circular imports."""
    from inference import choose_action as _act
    return _act(*args, **kwargs)


def recommend_action(*args, **kwargs):
    """Lazy-loaded inference entrypoint preventing circular imports."""
    from inference import recommend_action as _act
    return _act(*args, **kwargs)


def get_inference_engine():
    """Lazy-loaded circuit breaker inference engine singleton."""
    from inference import CircuitBreakerInferenceEngine
    return CircuitBreakerInferenceEngine()


__all__ = [
    "PPOAgent",
    "ActorCritic",
    "BaselineDQNAgent",
    "BoundaryCircuitBreakerEnv",
    "MultiObjectiveReward",
    "CRITICALITY_WEIGHTS",
    "MAPPOAgent",
    "MAPPOActor",
    "MAPPOCentralizedCritic",
    "LagrangianSafetyFilter",
    "choose_action",
    "recommend_action",
    "get_inference_engine",
]

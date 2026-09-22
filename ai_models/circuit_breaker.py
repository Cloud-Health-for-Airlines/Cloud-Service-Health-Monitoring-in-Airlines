"""Standard Python package bridge for ai-models/circuit-breaker."""

from __future__ import annotations

import sys
from pathlib import Path
from importlib import import_module

REPO_ROOT = Path(__file__).resolve().parent.parent
AI_MODELS_DIR = REPO_ROOT / "ai-models"
if str(AI_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(AI_MODELS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_mod = import_module("circuit-breaker")

PPOAgent = getattr(_mod, "PPOAgent")
ActorCritic = getattr(_mod, "ActorCritic")
BaselineDQNAgent = getattr(_mod, "BaselineDQNAgent")
BoundaryCircuitBreakerEnv = getattr(_mod, "BoundaryCircuitBreakerEnv")
MultiObjectiveReward = getattr(_mod, "MultiObjectiveReward")
CRITICALITY_WEIGHTS = getattr(_mod, "CRITICALITY_WEIGHTS")
choose_action = getattr(_mod, "choose_action")
recommend_action = getattr(_mod, "recommend_action")
MAPPOAgent = getattr(_mod, "MAPPOAgent")
MAPPOActor = getattr(_mod, "MAPPOActor")
MAPPOCentralizedCritic = getattr(_mod, "MAPPOCentralizedCritic")
LagrangianSafetyFilter = getattr(_mod, "LagrangianSafetyFilter")
get_inference_engine = getattr(_mod, "get_inference_engine")

__all__ = [
    "PPOAgent",
    "ActorCritic",
    "BaselineDQNAgent",
    "BoundaryCircuitBreakerEnv",
    "MultiObjectiveReward",
    "CRITICALITY_WEIGHTS",
    "choose_action",
    "recommend_action",
    "MAPPOAgent",
    "MAPPOActor",
    "MAPPOCentralizedCritic",
    "LagrangianSafetyFilter",
    "get_inference_engine",
]

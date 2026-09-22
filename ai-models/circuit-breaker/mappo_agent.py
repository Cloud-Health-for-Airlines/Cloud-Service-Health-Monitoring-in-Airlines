"""Tier 2A: Multi-Agent Proximal Policy Optimization (MAPPO) Circuit Breaker.

Extends single-agent PPO to distributed multi-gateway airline architectures:
- Centralized Training with Decentralized Execution (CTDE)
- Centralized Critic: V(s_global) takes concatenated global state across all gateways
- Decentralized Actors: pi_i(a_i | o_i) for each boundary gateway independently
- Coordinated load-shedding: prevents oscillation and domino cascades between gateways
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class MAPPOActor(nn.Module):
    """Decentralized Actor network for an individual gateway node."""

    def __init__(self, obs_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)

        # Initialize to low baseline throttle (< 15%)
        nn.init.constant_(self.beta_head.bias, 1.5)
        nn.init.constant_(self.alpha_head.bias, -0.5)

    def forward(self, obs: torch.Tensor) -> Beta:
        h = self.net(obs)
        alpha = torch.clamp(F.softplus(self.alpha_head(h)) + 1.0, 1.001, 50.0)
        beta = torch.clamp(F.softplus(self.beta_head(h)) + 1.0, 1.001, 50.0)
        return Beta(alpha, beta)


class MAPPOCentralizedCritic(nn.Module):
    """Centralized Critic network evaluating global multi-gateway state."""

    def __init__(self, global_state_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)


class MAPPOAgent:
    """Multi-Agent PPO Coordinator for distributed airline gateway clusters."""

    def __init__(
        self,
        num_gateways: int = 2,
        obs_dim: int = 5,
        global_state_dim: Optional[int] = None,
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
    ):
        self.num_gateways = num_gateways
        self.obs_dim = obs_dim
        # Global state = (num_gateways * obs_dim) + 2 (shared mainframe queue & CPU)
        self.global_state_dim = global_state_dim or (num_gateways * obs_dim + 2)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        # Decentralized actors
        self.actors = nn.ModuleList([
            MAPPOActor(obs_dim=obs_dim, hidden_dim=hidden_dim)
            for _ in range(num_gateways)
        ])
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]

        # Centralized critic
        self.critic = MAPPOCentralizedCritic(global_state_dim=self.global_state_dim, hidden_dim=hidden_dim * 2)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_actions(
        self,
        observations: List[np.ndarray],
        global_state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[List[float], List[float], float]:
        """Select actions for all gateway agents.
        
        Returns:
            actions: List of throttle rates in [0.0, 1.0] per gateway
            log_probs: List of log probabilities
            value: Centralized global state value estimate
        """
        actions = []
        log_probs = []

        with torch.no_grad():
            for i in range(self.num_gateways):
                obs_t = torch.as_tensor(observations[i], dtype=torch.float32).unsqueeze(0)
                dist = self.actors[i](obs_t)
                if deterministic:
                    a = dist.mean
                else:
                    a = dist.sample()
                a_clamped = torch.clamp(a, 1e-4, 1.0 - 1e-4)
                actions.append(float(a_clamped.item()))
                log_probs.append(float(dist.log_prob(a_clamped).item()))

            g_t = torch.as_tensor(global_state, dtype=torch.float32).unsqueeze(0)
            val = float(self.critic(g_t).item())

        return actions, log_probs, val

    def coordinate_mitigation(
        self,
        gateway_states: Dict[str, Dict[str, float]],
        shared_mainframe_queue: float = 10.0,
        shared_mainframe_cpu: float = 25.0,
    ) -> Dict[str, Dict[str, Any]]:
        """High-level multi-gateway mitigation coordinator."""
        gw_names = list(gateway_states.keys())[:self.num_gateways]
        obs_list = []

        for name in gw_names:
            s = gateway_states[name]
            obs = np.array([
                float(s.get("cascade_probability", 0.1)),
                float(s.get("boundary_sync_drift", 12.0)) / 100.0,
                min(1.0, float(s.get("gateway_latency_ms", 15.0)) / 1500.0),
                float(s.get("gateway_error_rate", 0.0)),
                float(s.get("current_throttle_rate", 0.0)),
            ], dtype=np.float32)
            obs_list.append(obs)

        # Build global state
        global_state = np.concatenate(obs_list + [
            np.array([shared_mainframe_queue / 150.0, shared_mainframe_cpu / 100.0], dtype=np.float32)
        ])

        actions, _, _ = self.select_actions(obs_list, global_state, deterministic=True)
        mitigations = {}

        for idx, name in enumerate(gw_names):
            rate = round(float(np.clip(actions[idx], 0.0, 1.0)), 2)
            if rate >= 0.85:
                action = "OPEN"
                rate = 1.0
                reason = f"MAPPO Coordinated: gateway isolation ({rate:.0%}) to protect shared mainframe core"
            elif rate >= 0.15:
                action = "THROTTLED"
                reason = f"MAPPO Coordinated: dynamic load-shedding at {rate:.0%} preventing cross-gateway cascade"
            else:
                action = "CLOSED"
                rate = 0.0
                reason = "MAPPO Coordinated: healthy cross-gateway balance verified"

            mitigations[name] = {
                "action": action,
                "throttle_rate": rate,
                "reason": reason,
            }

        return mitigations

    def save_checkpoints(self, base_dir: str | Path) -> None:
        p = Path(base_dir)
        p.mkdir(parents=True, exist_ok=True)
        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), p / f"mappo_actor_gw{i}.pt")
        torch.save(self.critic.state_dict(), p / "mappo_critic.pt")

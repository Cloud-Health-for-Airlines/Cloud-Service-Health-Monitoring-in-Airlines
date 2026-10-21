"""Tier 1F: Proximal Policy Optimization (PPO) Actor-Critic Circuit Breaker.

Continuous/hybrid action space PPO:
- Continuous policy raw action in [-1.0, 1.0] mapped to continuous throttle rate in [0.0, 1.0]
- Discrete mitigation mode: CLOSED (0.0), THROTTLED (0.10 - 0.90), OPEN (1.0)
- Gaussian Actor-Critic with Generalized Advantage Estimation (GAE)
- Direct load/save compatibility with ai-models/weights/circuit_breaker_ppo.pt
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .env import map_action_to_state
except (ImportError, ValueError):
    from env import map_action_to_state


class ActorCritic(nn.Module):
    """Actor-Critic network matching the trained checkpoint circuit_breaker_ppo.pt."""

    def __init__(self, state_dim: int = 6, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(state)
        val = self.critic(state)
        return mean, self.actor_logstd, val

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, calculate log probability and state value."""
        mean, logstd, val = self.forward(state)
        std = torch.exp(torch.clamp(logstd, -20.0, 2.0))

        if deterministic:
            action = mean
        else:
            action = mean + std * torch.randn_like(mean)

        action_clamped = torch.clamp(action, -1.0, 1.0)
        var = std ** 2
        log_prob = -((action_clamped - mean) ** 2) / (2 * var) - logstd - math.log(math.sqrt(2 * math.pi))
        log_prob = log_prob.sum(dim=-1)

        return action_clamped, log_prob, val.squeeze(-1)

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probability, entropy and value of given actions."""
        mean, logstd, val = self.forward(state)
        std = torch.exp(torch.clamp(logstd, -20.0, 2.0))
        var = std ** 2

        action_clamped = torch.clamp(action, -1.0, 1.0)
        log_prob = -((action_clamped - mean) ** 2) / (2 * var) - logstd - math.log(math.sqrt(2 * math.pi))
        log_prob = log_prob.sum(dim=-1)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + logstd).sum(dim=-1)

        return log_prob, val.squeeze(-1), entropy


class PPOAgent:
    """PPO policy optimizer with Generalized Advantage Estimation (GAE)."""

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 1,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ac = ActorCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.model = self.ac  # Backward-compatible alias
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[float, str, float]:
        """Compute action from observation vector.

        Returns:
            raw_action: float in [-1.0, 1.0]
            state_name: str in ("CLOSED", "THROTTLED", "OPEN")
            throttle: float in [0.0, 1.0]
        """
        if any(np.isnan(obs)) or any(np.isinf(obs)):
            raise ValueError(f"Observation contains non-finite values: {obs}")

        s_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, _, _ = self.ac.get_action(s_tensor, deterministic=deterministic)

        raw_a = float(action_t.squeeze().cpu().item())
        raw_a = max(-1.0, min(1.0, raw_a))
        state_name, throttle = map_action_to_state(raw_a)
        return raw_a, state_name, throttle

    def choose_mitigation(
        self,
        cascade_probability: float,
        boundary_sync_drift: float,
        gateway_latency_ms: float = 15.0,
        gateway_error_rate: float = 0.0,
        current_throttle_rate: float = 0.0,
        queue_saturation: float = 0.0,
        **kwargs: Any,
    ) -> Tuple[str, float, str]:
        """Inference entrypoint used by backend integration: returns (action, rate, reason)."""
        obs = np.array([
            float(cascade_probability),
            float(boundary_sync_drift) / 100.0,
            min(1.0, float(gateway_latency_ms) / 1500.0),
            float(queue_saturation),
            float(gateway_error_rate),
            float(current_throttle_rate),
        ], dtype=np.float32)

        raw_a, state_name, throttle = self.act(obs, deterministic=True)
        reason = (
            f"Learned PPO Policy: {state_name} actuated at rate {throttle:.0%} "
            f"[P={cascade_probability:.2f}, drift={boundary_sync_drift:.1f}%]"
        )
        return state_name, throttle, reason

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation in O(N)."""
        advantages = []
        gae = 0.0
        n_steps = len(rewards)

        vals = values + [next_value]
        for t in reversed(range(n_steps)):
            delta = rewards[t] + self.gamma * vals[t + 1] * (1.0 - float(dones[t])) - vals[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = advantages_t + torch.tensor(values, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        return advantages_t, returns_t

    def update(
        self,
        states: List[np.ndarray],
        actions: List[float],
        log_probs: List[float],
        returns: torch.Tensor,
        advantages: torch.Tensor,
        ppo_epochs: int = 4,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """PPO update with clipped surrogate objective and value function loss."""
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device).unsqueeze(-1)
        old_log_probs_t = torch.tensor(np.array(log_probs), dtype=torch.float32, device=self.device)

        total_loss = 0.0
        n_samples = len(states)
        indices = np.arange(n_samples)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                idx = indices[start:start + batch_size]
                b_states = states_t[idx]
                b_actions = actions_t[idx]
                b_old_log_probs = old_log_probs_t[idx]
                b_returns = returns[idx]
                b_advantages = advantages[idx]

                new_log_probs, val, entropy = self.ac.evaluate_actions(b_states, b_actions)
                ratio = torch.exp(torch.clamp(new_log_probs - b_old_log_probs, -20.0, 20.0))

                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                val_loss = F.mse_loss(val, b_returns)

                loss = policy_loss + self.value_coef * val_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()
                total_loss += float(loss.item())

        return {"loss_total": round(total_loss / max(1, ppo_epochs * math.ceil(n_samples / batch_size)), 4)}

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model state dictionary to disk."""
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.ac.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            target,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model state dictionary from disk."""
        target = Path(path).resolve()
        ckpt = torch.load(target, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        self.ac.load_state_dict(state_dict)
        self.ac.eval()

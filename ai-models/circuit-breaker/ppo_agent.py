"""Tier 1F: Proximal Policy Optimization (PPO) Actor-Critic Circuit Breaker.

Replaces discrete DQN with continuous/hybrid action space PPO:
- Continuous throttle rate in [0.0, 1.0]
- Discrete mitigation mode: CLOSED (0.0), THROTTLED (0.1 - 0.8), OPEN (1.0)
- Critic estimating baseline state value V(s)
- Generalized Advantage Estimation (GAE) with clipped surrogate objective
- Numerically bounded Beta policy with nominal-state bias initialization
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class ActorCritic(nn.Module):
    """Dual-head Actor-Critic network with Beta distribution for [0, 1] actions."""

    def __init__(self, state_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor: alpha and beta parameters for Beta distribution
        self.actor_alpha = nn.Linear(hidden_dim, 1)
        self.actor_beta = nn.Linear(hidden_dim, 1)

        # Bias initialization: beta > alpha so initial policy leans towards low throttle (< 15%)
        # preventing false-alarm throttling of healthy airline booking traffic
        nn.init.constant_(self.actor_beta.bias, 1.5)
        nn.init.constant_(self.actor_alpha.bias, -0.5)

        # Critic: outputs state value V(s)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[Beta, torch.Tensor]:
        features = self.shared(state)
        # alpha and beta must be > 1.0 for unimodal density; clamp to 50.0 to prevent steep gradients
        alpha = torch.clamp(F.softplus(self.actor_alpha(features)) + 1.0, 1.001, 50.0)
        beta = torch.clamp(F.softplus(self.actor_beta(features)) + 1.0, 1.001, 50.0)
        dist = Beta(alpha, beta)
        val = self.critic(features)
        return dist, val


class PPOAgent:
    """PPO policy optimizer with Generalized Advantage Estimation (GAE)."""

    def __init__(
        self,
        state_dim: int = 5,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.model = ActorCritic(state_dim=state_dim, hidden_dim=hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, float, float]:
        """Sample action from policy.
        
        Returns:
            action: float in [0.0, 1.0]
            log_prob: float
            value: float
        """
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, val = self.model(state_t)
            if deterministic:
                action_t = dist.mean
            else:
                action_t = dist.sample()
            action_clamped = torch.clamp(action_t, 1e-4, 1.0 - 1e-4)
            log_prob = dist.log_prob(action_clamped)

        return float(action_clamped.item()), float(log_prob.item()), float(val.item())

    def choose_mitigation(
        self,
        cascade_probability: float,
        boundary_sync_drift: float,
        gateway_latency_ms: float = 15.0,
        gateway_error_rate: float = 0.0,
        current_throttle_rate: float = 0.0,
    ) -> Tuple[str, float, str]:
        """Inference entrypoint used by backend integration: returns (action, rate, reason)."""
        state = np.array([
            float(cascade_probability),
            float(boundary_sync_drift) / 100.0,
            min(1.0, float(gateway_latency_ms) / 1500.0),
            float(gateway_error_rate),
            float(current_throttle_rate),
        ], dtype=np.float32)

        raw_action, _, _ = self.select_action(state, deterministic=True)
        throttle_rate = round(float(np.clip(raw_action, 0.0, 1.0)), 2)

        # Map continuous rate to standard airline circuit breaker actions
        if throttle_rate >= 0.85:
            action_name = "OPEN"
            throttle_rate = 1.0
            reason = f"Learned PPO Policy: boundary isolation actuated (rate=100%) due to critical cascade hazard [P={cascade_probability:.2f}, drift={boundary_sync_drift:.1f}%]"
        elif throttle_rate >= 0.15:
            action_name = "THROTTLED"
            reason = f"Learned PPO Policy: dynamic continuous rate-limiting at {throttle_rate:.0%} [P={cascade_probability:.2f}, drift={boundary_sync_drift:.1f}%]"
        else:
            action_name = "CLOSED"
            throttle_rate = 0.0
            reason = f"Learned PPO Policy: nominal health verified, circuit breaker CLOSED [P={cascade_probability:.2f}, drift={boundary_sync_drift:.1f}%]"

        return action_name, throttle_rate, reason

    def train_on_trajectories(
        self,
        states: List[np.ndarray],
        actions: List[float],
        log_probs: List[float],
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        epochs: int = 4,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """PPO update with Generalized Advantage Estimation."""
        n_steps = len(rewards)
        if n_steps == 0:
            return {"ppo_loss": 0.0}

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(-1)
        old_log_probs_t = torch.tensor(np.array(log_probs), dtype=torch.float32).unsqueeze(-1)

        # 1. Compute GAE in O(N) using append + reverse
        advantages = []
        returns = []
        gae = 0.0
        
        for t in reversed(range(n_steps)):
            next_val = values[t + 1] if t + 1 < n_steps and not dones[t] else 0.0
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * (0.0 if dones[t] else 1.0) * gae
            advantages.append(gae)
            returns.append(gae + values[t])

        advantages.reverse()
        returns.reverse()

        advantages_t = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1)
        returns_t = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)

        # Normalize advantages with variance guard
        adv_std = advantages_t.std()
        if not torch.isnan(adv_std) and adv_std > 1e-6:
            advantages_t = (advantages_t - advantages_t.mean()) / (adv_std + 1e-8)
        else:
            advantages_t = advantages_t - advantages_t.mean()

        total_loss = 0.0
        num_updates = 0

        for _ in range(epochs):
            indices = list(range(n_steps))
            np.random.shuffle(indices)

            for start in range(0, n_steps, batch_size):
                b_idx = indices[start:start + batch_size]
                b_states = states_t[b_idx]
                b_actions = actions_t[b_idx]
                b_old_log_probs = old_log_probs_t[b_idx]
                b_adv = advantages_t[b_idx]
                b_ret = returns_t[b_idx]

                dist, val = self.model(b_states)
                new_log_prob = dist.log_prob(b_actions.clamp(1e-4, 1.0 - 1e-4))
                entropy = dist.entropy().mean()

                # Ratio r(theta) with exponent clamping to prevent numerical explosion
                log_ratio = torch.clamp(new_log_prob - b_old_log_probs, -20.0, 20.0)
                ratio = torch.exp(log_ratio)

                # Clipped surrogate objective
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                val_loss = F.mse_loss(val, b_ret)

                loss = policy_loss + self.value_coef * val_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss += float(loss.item())
                num_updates += 1

        avg_loss = total_loss / max(1, num_updates)
        return {"ppo_loss": round(avg_loss, 4)}

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

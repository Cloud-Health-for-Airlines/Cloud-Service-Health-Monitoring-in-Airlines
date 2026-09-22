"""Tier 0 Baseline: Single-agent Deep Q-Network (DQN) circuit breaker.

Scoped to the boundary-gateway node.
Action space:
  0: CLOSED (throttle 0%)
  1: THROTTLE_LOW (throttle 25%)
  2: THROTTLE_MED (throttle 50%)
  3: THROTTLE_HIGH (throttle 75%)
  4: OPEN (isolate 100%)
"""

from __future__ import annotations

import collections
import random
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_MAP = {
    0: ("CLOSED", 0.0, "Nominal traffic allowed"),
    1: ("THROTTLED", 0.25, "Low mitigation throttle (25%)"),
    2: ("THROTTLED", 0.50, "Medium mitigation throttle (50%)"),
    3: ("THROTTLED", 0.75, "High mitigation throttle (75%)"),
    4: ("OPEN", 1.00, "Full boundary isolation actuated"),
}


class QNetwork(nn.Module):
    """Deep Q-Network estimating action-values Q(s, a)."""

    def __init__(self, state_dim: int = 5, action_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class BaselineDQNAgent:
    """Single-agent DQN policy for automated boundary circuit breaking."""

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 5,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.02,
        buffer_size: int = 5000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = collections.deque(maxlen=buffer_size)

    def extract_state(
        self,
        cascade_probability: float,
        sync_drift_score: float,
        gateway_latency_ms: float = 15.0,
        gateway_error_rate: float = 0.0,
        current_throttle_rate: float = 0.0,
    ) -> torch.Tensor:
        """Normalize system observables into standard state vector of dim=5."""
        state = [
            float(cascade_probability),
            float(sync_drift_score) / 100.0,
            min(1.0, float(gateway_latency_ms) / 1500.0),
            float(gateway_error_rate),
            float(current_throttle_rate),
        ]
        return torch.tensor(state, dtype=torch.float32)

    def select_action(self, state: torch.Tensor, explore: bool = True) -> int:
        """Select action via epsilon-greedy policy."""
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.policy_net(state)
            return int(q_values.argmax(dim=-1).item())

    def choose_mitigation(
        self,
        cascade_probability: float,
        boundary_sync_drift: float,
        gateway_latency_ms: float = 15.0,
        gateway_error_rate: float = 0.0,
        current_throttle_rate: float = 0.0,
        explore: bool = False,
    ) -> Tuple[str, float, str]:
        """Inference entrypoint: returns (action_name, throttle_rate, reason)."""
        state = self.extract_state(
            cascade_probability=cascade_probability,
            sync_drift_score=boundary_sync_drift,
            gateway_latency_ms=gateway_latency_ms,
            gateway_error_rate=gateway_error_rate,
            current_throttle_rate=current_throttle_rate,
        )
        action_idx = self.select_action(state, explore=explore)
        action_name, throttle_rate, desc = ACTION_MAP[action_idx]
        reason = f"DQN Policy (action={action_name}, rate={throttle_rate:.0%}): {desc} [P={cascade_probability:.2f}, drift={boundary_sync_drift:.1f}%]"
        return action_name, throttle_rate, reason

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Store experience tuple in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Perform one step of DQN gradient descent."""
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.stack(states)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.stack(next_states)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Q(s, a)
        curr_q = self.policy_net(states_t).gather(1, actions_t)

        # Target Q = r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = F.mse_loss(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    def update_target_network(self) -> None:
        """Sync target network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

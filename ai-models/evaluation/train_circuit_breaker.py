"""Training pipeline for BACCP Circuit Breaker RL agents (Tier 0 DQN & Tier 1F PPO).

Trains:
1. Tier 0 DQN agent on BoundaryCircuitBreakerEnv
2. Tier 1F PPO Actor-Critic agent on BoundaryCircuitBreakerEnv
3. Saves checkpoints to weights/circuit_breaker_dqn.pt and weights/circuit_breaker_ppo.pt
4. Compares policy performance against naive threshold ladder (from backend/cloud/lambda_handler.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

AI_MODELS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = AI_MODELS_DIR.parent
sys.path.insert(0, str(AI_MODELS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from importlib import import_module
dqn_mod = import_module("circuit-breaker.baseline_dqn")
BaselineDQNAgent = dqn_mod.BaselineDQNAgent

ppo_mod = import_module("circuit-breaker.ppo_agent")
PPOAgent = ppo_mod.PPOAgent

env_mod = import_module("circuit-breaker.env")
BoundaryCircuitBreakerEnv = env_mod.BoundaryCircuitBreakerEnv

WEIGHTS_DIR = AI_MODELS_DIR / "weights"


def evaluate_policy(env: BoundaryCircuitBreakerEnv, policy_fn, episodes: int = 50) -> Dict[str, float]:
    total_reward = 0.0
    cascades_avoided = 0
    total_hazards = 0
    false_alarms = 0
    total_nominals = 0

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = policy_fn(state)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            state = next_state

        total_reward += ep_reward
        if env.is_hazard:
            total_hazards += 1
            if not info["cascade_occurred"]:
                cascades_avoided += 1
        else:
            total_nominals += 1
            # False alarm = throttled heavily (> 20%) during healthy nominal operation
            if env.throttle_rate > 0.20:
                false_alarms += 1

    avoid_rate = (cascades_avoided / max(1, total_hazards)) * 100.0
    false_alarm_rate = (false_alarms / max(1, total_nominals)) * 100.0
    mean_reward = total_reward / episodes

    return {
        "mean_reward": round(float(mean_reward), 2),
        "cascade_avoidance_pct": round(float(avoid_rate), 1),
        "false_alarm_pct": round(float(false_alarm_rate), 1),
    }


def train_circuit_breaker_models(episodes: int = 200) -> Dict[str, Any]:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    env = BoundaryCircuitBreakerEnv(seed=42)

    # -------------------------------------------------------------
    # 1. Train Tier 0 DQN Agent
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" TRAINING TIER 0 BASELINE DQN CIRCUIT BREAKER")
    print("=" * 60)
    dqn_agent = BaselineDQNAgent(state_dim=5, action_dim=5, learning_rate=0.002)

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        while not done:
            action_idx = dqn_agent.select_action(torch.tensor(state, dtype=torch.float32), explore=True)
            # Map index to continuous throttle for env step
            throttle = action_idx * 0.25
            next_state, reward, done, _ = env.step(throttle)
            dqn_agent.store_transition(
                torch.tensor(state, dtype=torch.float32),
                action_idx,
                reward,
                torch.tensor(next_state, dtype=torch.float32),
                done,
            )
            dqn_agent.train_step(batch_size=32)
            state = next_state

        if ep % 20 == 0:
            dqn_agent.update_target_network()
        if ep % 50 == 0 or ep == episodes:
            print(f"DQN Episode {ep:03d}/{episodes:03d} | Epsilon: {dqn_agent.epsilon:.3f}")

    dqn_path = WEIGHTS_DIR / "circuit_breaker_dqn.pt"
    torch.save(dqn_agent.policy_net.state_dict(), dqn_path)
    print(f"[+] Saved Tier 0 DQN checkpoint to {dqn_path}")

    # -------------------------------------------------------------
    # 2. Train Tier 1F PPO Agent
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" TRAINING TIER 1F PPO ACTOR-CRITIC CIRCUIT BREAKER")
    print("=" * 60)
    ppo_agent = PPOAgent(state_dim=5, hidden_dim=64, lr=3e-4)

    rollout_states = []
    rollout_actions = []
    rollout_log_probs = []
    rollout_rewards = []
    rollout_values = []
    rollout_dones = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False

        while not done:
            action, log_prob, val = ppo_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            rollout_states.append(state)
            rollout_actions.append(action)
            rollout_log_probs.append(log_prob)
            rollout_rewards.append(reward)
            rollout_values.append(val)
            rollout_dones.append(done)

            state = next_state

        # Update PPO policy every 20 episodes
        if ep % 20 == 0:
            ppo_agent.train_on_trajectories(
                rollout_states,
                rollout_actions,
                rollout_log_probs,
                rollout_rewards,
                rollout_values,
                rollout_dones,
                epochs=4,
                batch_size=32,
            )
            rollout_states.clear()
            rollout_actions.clear()
            rollout_log_probs.clear()
            rollout_rewards.clear()
            rollout_values.clear()
            rollout_dones.clear()

        if ep % 50 == 0 or ep == episodes:
            print(f"PPO Episode {ep:03d}/{episodes:03d}")

    ppo_path = WEIGHTS_DIR / "circuit_breaker_ppo.pt"
    ppo_agent.save_checkpoint(ppo_path)
    print(f"[+] Saved Tier 1F PPO checkpoint to {ppo_path}")

    # -------------------------------------------------------------
    # 3. Policy Benchmark: Naive Rule vs DQN vs PPO
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" BENCHMARKING MITIGATION POLICIES")
    print("=" * 60)

    # Naive hard-coded threshold rule (from backend/cloud/lambda_handler.py):
    def rule_policy(state: np.ndarray) -> float:
        cascade_prob = state[0]
        sync_drift = state[1] * 100.0
        if cascade_prob >= 0.80 or sync_drift >= 70.0:
            return 1.0  # OPEN
        elif cascade_prob >= 0.55 or sync_drift >= 45.0:
            return min(0.75, 0.30 + (cascade_prob - 0.55) * 2.0)
        return 0.0

    def dqn_policy(state: np.ndarray) -> float:
        a_idx = dqn_agent.select_action(torch.tensor(state, dtype=torch.float32), explore=False)
        return a_idx * 0.25

    def ppo_policy(state: np.ndarray) -> float:
        act, _, _ = ppo_agent.select_action(state, deterministic=True)
        return act

    eval_rule = evaluate_policy(env, rule_policy, episodes=60)
    eval_dqn = evaluate_policy(env, dqn_policy, episodes=60)
    eval_ppo = evaluate_policy(env, ppo_policy, episodes=60)

    print(f"Naive Rule Ladder -> Reward: {eval_rule['mean_reward']} | Avoided: {eval_rule['cascade_avoidance_pct']}% | False Alarms: {eval_rule['false_alarm_pct']}%")
    print(f"Tier 0 DQN        -> Reward: {eval_dqn['mean_reward']} | Avoided: {eval_dqn['cascade_avoidance_pct']}% | False Alarms: {eval_dqn['false_alarm_pct']}%")
    print(f"Tier 1F PPO       -> Reward: {eval_ppo['mean_reward']} | Avoided: {eval_ppo['cascade_avoidance_pct']}% | False Alarms: {eval_ppo['false_alarm_pct']}%")

    return {
        "rule": eval_rule,
        "dqn": eval_dqn,
        "ppo": eval_ppo,
    }


if __name__ == "__main__":
    train_circuit_breaker_models(episodes=200)

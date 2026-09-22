"""Training pipeline for BACCP PPO Circuit Breaker.

Trains PPO policy on boundary mitigation environment with validation checkpointing.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CB_DIR = ROOT / "ai-models/circuit-breaker"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CB_DIR) not in sys.path:
    sys.path.insert(0, str(CB_DIR))

try:
    from env import BoundaryMitigationEnv
    from ppo_agent import PPOAgent
except (ImportError, ModuleNotFoundError):
    from .env import BoundaryMitigationEnv
    from .ppo_agent import PPOAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baccp.ppo.train")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_policy(agent: PPOAgent, env: BoundaryMitigationEnv, num_episodes: int = 20) -> Dict[str, float]:
    """Evaluate current policy deterministically across validation episodes."""
    total_rewards = []
    cascades = 0
    throughputs = []
    throttles = []
    latencies = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep + 1000)
        done = False
        ep_reward = 0.0

        while not done:
            _, _, throttle = agent.act(obs, deterministic=True)
            obs, reward, term, trunc, step_info = env.step(throttle)
            ep_reward += reward
            throughputs.append(step_info["throughput_info"]["weighted"])
            throttles.append(throttle)
            latencies.append(step_info["gateway_latency_ms"])

            if term:
                cascades += 1
            done = term or trunc

        total_rewards.append(ep_reward)

    return {
        "mean_reward": round(float(np.mean(total_rewards)), 4),
        "cascade_rate": round(cascades / float(num_episodes), 4),
        "mean_throughput": round(float(np.mean(throughputs)), 4),
        "mean_throttle": round(float(np.mean(throttles)), 4),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
    }


def train_ppo(
    total_episodes: int = 200,
    steps_per_rollout: int = 400,
    seed: int = 42,
    lr: float = 3e-4,
    output_checkpoint: str = "ai-models/weights/circuit_breaker_ppo.pt",
) -> Dict[str, Any]:
    set_seed(seed)
    env = BoundaryMitigationEnv(max_steps=20, seed=seed)
    val_env = BoundaryMitigationEnv(max_steps=20, seed=seed + 500)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=lr)

    logger.info(f"Initialized PPO training ({total_episodes} episodes, seed={seed})...")

    best_reward = -float("inf")
    best_weights = None
    history = []
    start_time = time.time()

    episodes_done = 0
    obs, _ = env.reset(seed=seed)

    while episodes_done < total_episodes:
        rollout_states = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []

        # Collect rollout trajectory
        for _ in range(steps_per_rollout):
            s_tensor = torch.tensor(obs, dtype=torch.float, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                action_t, log_prob_t, val_t = agent.ac.get_action(s_tensor, deterministic=False)

            action_val = float(action_t.cpu().item())
            log_prob_val = float(log_prob_t.cpu().item())
            val_val = float(val_t.cpu().item())

            next_obs, reward, term, trunc, _ = env.step(action_val)
            done = term or trunc

            rollout_states.append(obs)
            rollout_actions.append(action_val)
            rollout_log_probs.append(log_prob_val)
            rollout_rewards.append(reward)
            rollout_dones.append(done)
            rollout_values.append(val_val)

            obs = next_obs
            if done:
                episodes_done += 1
                obs, _ = env.reset()
                if episodes_done >= total_episodes:
                    break

        # Compute next state value for GAE
        s_tensor = torch.tensor(obs, dtype=torch.float, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            _, _, next_val = agent.ac.forward(s_tensor)
            next_val_val = float(next_val.cpu().item())

        advantages, returns = agent.compute_gae(
            rollout_rewards, rollout_values, rollout_dones, next_val_val
        )

        # Update PPO policy
        losses = agent.update(
            rollout_states, rollout_actions, rollout_log_probs, returns, advantages, ppo_epochs=4, batch_size=32
        )

        # Evaluate periodically
        val_metrics = evaluate_policy(agent, val_env, num_episodes=15)
        history.append({
            "episodes": episodes_done,
            "loss_total": losses["loss_total"],
            "loss_policy": losses["loss_policy"],
            "loss_value": losses["loss_value"],
            "val_reward": val_metrics["mean_reward"],
            "val_cascade_rate": val_metrics["cascade_rate"],
            "val_throughput": val_metrics["mean_throughput"],
        })

        if val_metrics["mean_reward"] > best_reward:
            best_reward = val_metrics["mean_reward"]
            best_weights = copy.deepcopy(agent.ac.state_dict())

        logger.info(
            f"Episodes: {episodes_done:03d}/{total_episodes} | "
            f"Loss: {losses['loss_total']:.4f} | "
            f"Val Reward: {val_metrics['mean_reward']:.2f} | "
            f"Cascade Rate: {val_metrics['cascade_rate'] * 100:.1f}% | "
            f"Throughput: {val_metrics['mean_throughput'] * 100:.1f}%"
        )

    # Restore best weights
    if best_weights is not None:
        agent.ac.load_state_dict(best_weights)

    # Save trained checkpoint
    out_path = Path(output_checkpoint).resolve()
    agent.save_checkpoint(out_path)
    logger.info(f"Trained PPO checkpoint successfully saved to: {out_path}")

    duration = round(time.time() - start_time, 2)
    return {
        "episodes": total_episodes,
        "duration_seconds": duration,
        "best_val_reward": best_reward,
        "history": history,
        "checkpoint_path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train BACCP PPO Circuit Breaker.")
    parser.add_argument("--episodes", type=int, default=200, help="Total training episodes")
    parser.add_argument("--steps", type=int, default=400, help="Steps per rollout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--output", type=str, default="ai-models/weights/circuit_breaker_ppo.pt", help="Checkpoint output path")
    args = parser.parse_args()

    train_ppo(
        total_episodes=args.episodes,
        steps_per_rollout=args.steps,
        seed=args.seed,
        lr=args.lr,
        output_checkpoint=args.output,
    )


if __name__ == "__main__":
    main()

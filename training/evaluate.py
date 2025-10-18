"""Offline evaluation harness for trained policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import torch

from agent import Agent
from training.train import make_default_env


def evaluate(checkpoint: Path, episodes: int = 20, team_size: int = 1, tick_skip: int = 8) -> None:  # pragma: no cover - depends on RLGym
    env, _ = make_default_env(team_size, tick_skip)
    agent = Agent()
    agent.policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    agent.policy.eval()

    wins = 0
    rewards = []

    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action_idx, _ = agent.policy.get_action(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action_idx))
            episode_reward += float(reward)

        rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1

        print(f"[eval] Episode {episode + 1}/{episodes} reward={episode_reward:.2f}")

    print("====================================")
    print(f"Average reward: {mean(rewards):.3f}")
    print(f"Win rate: {wins / episodes:.0%}")
    print("====================================")


def main() -> None:  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to the PPO checkpoint")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation matches")
    parser.add_argument("--team-size", type=int, default=1)
    parser.add_argument("--tick-skip", type=int, default=8)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.episodes, args.team_size, args.tick_skip)


if __name__ == "__main__":  # pragma: no cover - CLI utility
    main()


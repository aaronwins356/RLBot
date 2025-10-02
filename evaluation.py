"""Evaluation script for HybridBot PPO checkpoints.

The evaluation pipeline loads a saved PPO model and runs matches against different
opponents: self-play mirrors, the scripted HybridBot baseline, and the built-in Psyonix
bots.  Metrics such as goals scored, saves, ball touches and win rate are logged to provide
insight into progress over time.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from stable_baselines3 import PPO

from training import TrainingConfig, build_vec_env


@dataclass
class EvaluationResult:
    """Container summarizing metrics from evaluation matches."""

    goals_scored: int = 0
    goals_conceded: int = 0
    saves: int = 0
    touches: int = 0
    wins: int = 0
    matches: int = 0

    def record(self, info: Dict[str, float]) -> None:
        self.goals_scored += int(info.get("goals_scored", 0))
        self.goals_conceded += int(info.get("goals_conceded", 0))
        self.saves += int(info.get("saves", 0))
        self.touches += int(info.get("touches", 0))
        self.wins += int(info.get("win", 0))
        self.matches += 1

    def summary(self) -> str:
        if self.matches == 0:
            return "No matches played"
        win_rate = 100.0 * self.wins / self.matches
        return (
            f"Matches: {self.matches}\n"
            f"Goals scored: {self.goals_scored}\n"
            f"Goals conceded: {self.goals_conceded}\n"
            f"Saves: {self.saves}\n"
            f"Touches: {self.touches}\n"
            f"Win rate: {win_rate:.1f}%"
        )


def evaluate_against(model: PPO, opponent_mode: str, episodes: int) -> EvaluationResult:
    """Run evaluation episodes against a specified opponent mode."""

    config = TrainingConfig(num_envs=1, opponent_mode=opponent_mode)
    env = build_vec_env(config)
    result = EvaluationResult()
    for _ in range(episodes):
        obs = env.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        if isinstance(info, list):
            info = info[0]
        result.record(info if isinstance(info, dict) else {})
    env.close()
    return result


def print_section(title: str, result: EvaluationResult) -> None:
    print("=" * 60)
    print(title)
    print(result.summary())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HybridBot PPO checkpoints")
    parser.add_argument("--model", type=Path, required=True, help="Path to PPO checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model = PPO.load(args.model)

    print_section("Self-play mirror", evaluate_against(model, "self", args.episodes))
    print_section("Scripted HybridBot", evaluate_against(model, "scripted", args.episodes))
    print_section("Psyonix baseline", evaluate_against(model, "none", args.episodes))


if __name__ == "__main__":
    main()

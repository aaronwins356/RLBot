"""Training entry point for the HybridBot reinforcement learning project.

Running ``python training.py`` launches the PPO self-play pipeline.  The code is heavily
commented so readers understand each component of the RL stack:

* Why Proximal Policy Optimization (PPO) is chosen for Rocket League.
* How self-play accelerates the emergence of coordinated strategies.
* Why curriculum learning and scripted mechanics ease the cold start problem.
* How to scale training to millions of steps using vectorized environments.
* Why checkpointing and evaluation loops are essential for long-term experiments.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rl_components import (
    CurriculumStateSetter,
    HybridActionParser,
    HybridObsBuilder,
    HybridRewardFunction,
    common_terminal_conditions,
)

# RLGym imports are deferred inside functions so that developers without the game installed
# can still import this file for documentation purposes.

# -----------------------------------------------------------------------------
# Configuration data classes controlling the training pipeline.
# -----------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for PPO self-play training."""

    total_timesteps: int = 30_000_000
    learning_rate: float = 3e-4
    n_steps: int = 4096
    batch_size: int = 512
    gamma: float = 0.993
    gae_lambda: float = 0.9
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    tensorboard_log: str = "./tb_logs"
    checkpoint_dir: str = "./models"
    checkpoint_interval: int = 500_000
    eval_interval: int = 500_000
    num_envs: int = 1
    use_subprocess: bool = False
    opponent_mode: str = "self"
    net_arch: Optional[List[int]] = None


# -----------------------------------------------------------------------------
# Callbacks for checkpointing and logging.
# -----------------------------------------------------------------------------


class CheckpointCallback(BaseCallback):
    """Callback saving periodic checkpoints and tracking the best model."""

    def __init__(self, save_dir: Path, save_freq: int, eval_fn: Callable[[PPO], float]) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.eval_fn = eval_fn
        self.best_score: float = float("-inf")

    def _init_callback(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = self.save_dir / f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(path)
            score = self.eval_fn(self.model)
            if score > self.best_score:
                self.best_score = score
                self.model.save(self.save_dir / "best_model.zip")
            self.logger.record("checkpoint/score", score)
        return True


class RollingStatsCallback(BaseCallback):
    """Callback printing rolling averages to keep long runs observable."""

    def __init__(self, log_interval: int = 10_000) -> None:
        super().__init__()
        self.log_interval = log_interval
        self.episode_rewards: List[float] = []
        self.last_log_step: int = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            for done, reward in zip(self.locals["dones"], self.locals["rewards"]):
                if done:
                    self.episode_rewards.append(float(reward))
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            if self.episode_rewards:
                avg_reward = float(np.mean(self.episode_rewards[-100:]))
            else:
                avg_reward = 0.0
            print(f"Step {self.num_timesteps}: 100-episode avg reward {avg_reward:.3f}")
            self.last_log_step = self.num_timesteps
        return True


# -----------------------------------------------------------------------------
# Environment creation utilities.
# -----------------------------------------------------------------------------


def make_env_fn(config: TrainingConfig, seed: Optional[int] = None) -> Callable[[], VecEnv]:
    """Return a function that instantiates a self-play environment."""

    def _init() -> "Gym":
        from rlgym.envs import Match
        from rlgym.gym import Gym

        state_setter = CurriculumStateSetter()
        obs_builder = HybridObsBuilder()
        action_parser = HybridActionParser()
        reward_function = HybridRewardFunction()
        terminal_conditions = common_terminal_conditions()

        scripted_opponent = config.opponent_mode == "scripted"

        match = Match(
            team_size=1,
            state_setter=state_setter,
            obs_builder=obs_builder,
            action_parser=action_parser,
            reward_function=reward_function,
            terminal_conditions=terminal_conditions,
            enable_state_transitions=True,
            # When ``scripted_opponent`` is true, RLGym will call into ``opponent_policy`` for
            # the orange car.  We provide a lambda that mirrors the scripted heuristics.
            opponent_policy=None,
        )

        env = Gym(
            match=match,
            self_play=config.opponent_mode == "self",
            spawn_opponents=True,
            team_size=1,
        )

        if scripted_opponent:
            env.opponent_policy = ScriptedOpponentPolicy()

        if seed is not None:
            env.seed(seed)
        else:
            env.seed(random.randint(0, 2**32 - 1))
        return env

    return _init


class ScriptedOpponentPolicy:
    """Simple opponent that mirrors the HybridBot heuristics.

    The policy returns discrete action indices representing offensive or defensive intents.
    This stabilizes the early phases of training by providing a predictable adversary before
    the PPO policy becomes competitive.
    """

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # Very small heuristic: attack when ball is close, otherwise defend.
        ball_rel = obs[..., :3]
        distance = np.linalg.norm(ball_rel, axis=-1)
        actions = np.where(distance < 0.2, 0, 1)
        return actions.astype(np.int64)


def build_vec_env(config: TrainingConfig) -> VecEnv:
    """Create vectorized environment supporting parallel workers."""

    env_fns = [make_env_fn(config, seed=i) for i in range(config.num_envs)]
    if config.use_subprocess and config.num_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


# -----------------------------------------------------------------------------
# Evaluation utility used during training and by evaluation.py.
# -----------------------------------------------------------------------------


def evaluate_model(model: PPO, episodes: int = 5) -> float:
    """Run quick evaluation episodes returning average reward."""

    config = TrainingConfig(num_envs=1)
    env = build_vec_env(config)
    rewards: List[float] = []
    for _ in range(episodes):
        obs = env.reset()
        done = [False]
        episode_reward = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += float(reward[0])
        rewards.append(episode_reward)
    env.close()
    return float(np.mean(rewards))


# -----------------------------------------------------------------------------
# Training routine
# -----------------------------------------------------------------------------


def run_training(config: TrainingConfig) -> None:
    """Configure PPO and execute the long-term training loop."""

    env = build_vec_env(config)

    net_arch = config.net_arch or [256, 256, 128]
    policy_kwargs = dict(net_arch=net_arch)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        tensorboard_log=config.tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    callback = CheckpointCallback(
        save_dir=checkpoint_dir,
        save_freq=config.checkpoint_interval,
        eval_fn=lambda mdl: evaluate_model(mdl, episodes=3),
    )
    stats_callback = RollingStatsCallback()

    start_time = time.time()
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[callback, stats_callback],
        progress_bar=True,
    )
    duration = time.time() - start_time
    print(f"Training completed in {duration/3600:.2f} hours")

    final_path = checkpoint_dir / "final_model.zip"
    model.save(final_path)
    env.close()


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train HybridBot with PPO self-play")
    parser.add_argument("--timesteps", type=int, default=30_000_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--subprocess", action="store_true")
    parser.add_argument("--opponent", choices=["self", "scripted", "none"], default="self")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--net", nargs="*", type=int, default=[256, 256, 128])
    args = parser.parse_args()

    return TrainingConfig(
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        use_subprocess=args.subprocess,
        opponent_mode=args.opponent,
        learning_rate=args.lr,
        net_arch=args.net,
    )


if __name__ == "__main__":
    config = parse_args()
    run_training(config)

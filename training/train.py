"""End-to-end PPO training loop tailored for the hybrid RLBot agent."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agent import POLICY_LAYER_SIZES
from discrete_policy import DiscreteFF
from training.rewards import HybridReward
from your_act import YourActionParser
from your_obs import OBS_SIZE, YourOBS


try:  # pragma: no cover - dependencies are only required when training
    from rlgym.utils.action_parsers import ActionParser as RLGymActionParser
    from rlgym.utils.obs_builders import ObsBuilder as RLGymObsBuilder
    from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
    from rlgym.utils.state_setters import RandomState
    from rlgym_compat import make as make_rlgym_env
except ImportError:  # pragma: no cover - gracefully handle missing training deps
    RLGymActionParser = object  # type: ignore
    RLGymObsBuilder = object  # type: ignore
    GoalScoredCondition = TimeoutCondition = RandomState = None  # type: ignore
    make_rlgym_env = None  # type: ignore


def _ensure_dependency(name: str) -> None:
    if make_rlgym_env is None:
        raise ImportError(
            f"{name} is required for training. Install rlgym and rlgym-compat in a Python 3.9 environment."
        )


class TrainingObservationBuilder(RLGymObsBuilder):
    """Wrap :class:`YourOBS` so the training environment matches inference."""

    def __init__(self) -> None:
        super().__init__()
        self._builder = YourOBS()
        self._previous_actions: List[np.ndarray] = []

    def reset(self, initial_state) -> None:  # pragma: no cover - depends on RLGym
        self._previous_actions = [np.zeros(8, dtype=np.float32) for _ in initial_state.players]

    def pre_step(self, state, actions) -> None:  # pragma: no cover - depends on RLGym
        if actions is None:
            return
        if not self._previous_actions:
            self._previous_actions = [np.zeros(8, dtype=np.float32) for _ in range(len(actions))]
        for idx, action in enumerate(actions):
            self._previous_actions[idx] = np.asarray(action, dtype=np.float32)

    def build_obs(self, player, state, previous_action):  # pragma: no cover - depends on RLGym
        if previous_action is None:
            previous_action = np.zeros(8, dtype=np.float32)
        return self._builder.build_obs(player, state, previous_action)


class TrainingActionParser(RLGymActionParser):
    """Expose the hybrid action space to the training environment."""

    def __init__(self) -> None:
        super().__init__()
        self._parser = YourActionParser()

    @property
    def action_space(self):  # pragma: no cover - gym dependency
        return self._parser.action_space

    def parse_actions(self, actions, state) -> np.ndarray:  # pragma: no cover - depends on RLGym
        decoded = []
        for idx, action in enumerate(actions):
            player = state.players[idx]
            context = {"state": state, "player": player}
            decoded.append(self._parser.parse_actions([int(action)], context))
        return np.asarray(decoded, dtype=np.float32)


def make_default_env(team_size: int, tick_skip: int):  # pragma: no cover - depends on RLGym
    _ensure_dependency("rlgym-compat")
    reward = HybridReward()
    obs_builder = TrainingObservationBuilder()
    action_parser = TrainingActionParser()
    terminal_conditions = [GoalScoredCondition(), TimeoutCondition(4800 // tick_skip)]
    state_setter = RandomState()
    env = make_rlgym_env(
        reward_function=reward,
        observation_builder=obs_builder,
        action_parser=action_parser,
        terminal_conditions=terminal_conditions,
        state_setter=state_setter,
        spawn_opponents=True,
        team_size=team_size,
        tick_skip=tick_skip,
    )
    return env, action_parser


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, layer_sizes: Iterable[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class PPOTrainer:
    """Minimal PPO implementation matching the inference policy architecture."""

    def __init__(
        self,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.1,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = DiscreteFF(OBS_SIZE, action_dim, POLICY_LAYER_SIZES, self.device)
        self.value_fn = ValueNet(OBS_SIZE, POLICY_LAYER_SIZES).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_fn.parameters()), lr=learning_rate
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        probs = self.policy.get_output(obs_tensor)
        dist = Categorical(probs)
        if deterministic:
            action = torch.argmax(probs).item()
            log_prob = torch.log(probs[action] + 1e-11).item()
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
            action = action.item()
        value = self.value_fn(obs_tensor).item()
        return action, log_prob, value

    def compute_returns(
        self, rewards: List[float], dones: List[bool], values: List[float], last_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        returns = np.zeros(len(rewards) + 1, dtype=np.float32)
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns[-1] = last_value
        gae = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[step])
            delta = rewards[step] + self.gamma * returns[step + 1] * mask - values[step]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]
        return returns[:-1], advantages

    def update(self, batch: RolloutBatch, epochs: int = 4, batch_size: int = 1024) -> None:
        dataset_size = batch.observations.size(0)
        indices = np.arange(dataset_size)

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mini_idx = indices[start:end]

                obs = batch.observations[mini_idx].to(self.device)
                actions = batch.actions[mini_idx].to(self.device)
                old_log_probs = batch.log_probs[mini_idx].to(self.device)
                advantages = batch.advantages[mini_idx].to(self.device)
                returns = batch.returns[mini_idx].to(self.device)

                probs = self.policy.get_output(obs)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.value_fn(obs).squeeze(-1)
                value_loss = nn.functional.mse_loss(values, returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()


def collect_rollout(env, trainer: PPOTrainer, horizon: int) -> RolloutBatch:  # pragma: no cover - depends on RLGym
    observations: List[np.ndarray] = []
    actions: List[int] = []
    log_probs: List[float] = []
    rewards: List[float] = []
    dones: List[bool] = []
    values: List[float] = []

    obs = env.reset()

    for _ in range(horizon):
        action, log_prob, value = trainer.act(obs)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)

        obs, reward, done, _ = env.step(int(action))
        rewards.append(float(reward))
        dones.append(bool(done))

        if done:
            obs = env.reset()

    _, _, last_value = trainer.act(obs, deterministic=True)
    returns, advantages = trainer.compute_returns(rewards, dones, values, last_value)

    adv_tensor = torch.as_tensor(advantages, dtype=torch.float32)
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    batch = RolloutBatch(
        observations=torch.as_tensor(np.asarray(observations), dtype=torch.float32),
        actions=torch.as_tensor(np.asarray(actions), dtype=torch.int64),
        log_probs=torch.as_tensor(np.asarray(log_probs), dtype=torch.float32),
        returns=torch.as_tensor(returns, dtype=torch.float32),
        advantages=adv_tensor,
    )
    return batch


def train(
    total_steps: int,
    rollout_horizon: int,
    save_path: Path,
    team_size: int = 1,
    tick_skip: int = 8,
    learning_rate: float = 3e-4,
) -> None:  # pragma: no cover - depends on RLGym
    env, action_parser = make_default_env(team_size, tick_skip)
    trainer = PPOTrainer(action_dim=len(action_parser._parser.lookup_table), learning_rate=learning_rate)

    steps_collected = 0
    iteration = 0

    while steps_collected < total_steps:
        batch = collect_rollout(env, trainer, rollout_horizon)
        trainer.update(batch)
        steps_collected += rollout_horizon
        iteration += 1

        if iteration % 10 == 0:
            print(f"[train] Iteration {iteration}, steps={steps_collected}")

        if steps_collected >= total_steps:
            torch.save(trainer.policy.state_dict(), save_path)
            break


def main() -> None:  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser(description="Train the hybrid PPO Rocket League bot")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total environment steps")
    parser.add_argument("--rollout", type=int, default=8192, help="Rollout horizon per PPO update")
    parser.add_argument("--team-size", type=int, default=1, help="Number of players per team")
    parser.add_argument("--tick-skip", type=int, default=8, help="Physics tick skip used during training")
    parser.add_argument("--checkpoint", type=Path, default=Path("../PPO_POLICY.pt"), help="Path to save the trained weights")
    args = parser.parse_args()

    train(
        total_steps=args.steps,
        rollout_horizon=args.rollout,
        save_path=args.checkpoint,
        team_size=args.team_size,
        tick_skip=args.tick_skip,
    )


if __name__ == "__main__":  # pragma: no cover - CLI utility
    main()


"""Reusable reinforcement learning components for SuperBot."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Simple FIFO replay buffer used by DQN training."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.storage: List[Transition] = []
        self.position = 0

    def push(self, transition: Transition) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        indices = np.random.choice(len(self.storage), size=batch_size, replace=False)
        return [self.storage[i] for i in indices]

    def __len__(self) -> int:
        return len(self.storage)


class DQNetwork(nn.Module):
    """MLP used for discrete action-value estimation."""

    def __init__(self, input_dim: int, output_dim: int, hidden: Iterable[int] = (256, 256)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for width in hidden:
            layers.append(nn.Linear(last, width))
            layers.append(nn.ReLU())
            last = width
        layers.append(nn.Linear(last, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor)


class PolicyManager:
    """Abstraction switching between scripted and learned policies."""

    def __init__(self, state_dim: int, action_dim: int, device: Optional[str] = None) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = DQNetwork(state_dim, action_dim).to(self.device)
        self.target = DQNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.update_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.action_dim))
        with torch.no_grad():
            tensor = torch.from_numpy(state).float().to(self.device)
            q_values = self.policy(tensor.unsqueeze(0))
            return int(torch.argmax(q_values, dim=1).item())

    def optimize(self, buffer: ReplayBuffer, batch_size: int = 64, gamma: float = 0.99, tau: float = 0.005) -> float:
        if len(buffer) < batch_size:
            return 0.0

        transitions = buffer.sample(batch_size)
        states = torch.from_numpy(np.stack([t.state for t in transitions])).float().to(self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in transitions])).float().to(self.device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=self.device)

        q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            target = rewards + gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.update_steps += 1
        for target_param, param in zip(self.target.parameters(), self.policy.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * param.data)

        return float(loss.item())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.state_dict(), "target": self.target.state_dict()}, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.target.load_state_dict(checkpoint.get("target", checkpoint["policy"]))


# ----------------------------------------------------------------------------
# Imitation learning utilities
# ----------------------------------------------------------------------------


@dataclass
class ImitationExample:
    state: np.ndarray
    action: int


class ImitationDataset(torch.utils.data.Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset generated from human replays."""

    def __init__(self, examples: List[ImitationExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        state = torch.from_numpy(example.state).float()
        action = torch.tensor(example.action, dtype=torch.int64)
        return state, action


def train_imitation(policy: PolicyManager, dataset: ImitationDataset, epochs: int = 20, batch_size: int = 256) -> None:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for states, actions in loader:
            states = states.to(policy.device)
            actions = actions.to(policy.device)
            logits = policy.policy(states)
            loss = loss_fn(logits, actions)
            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()


# ----------------------------------------------------------------------------
# PPO / A3C placeholders (extendable points)
# ----------------------------------------------------------------------------


@dataclass
class PolicyFactory:
    """Factory class for building policy objects."""

    builder: Callable[[int, int], PolicyManager]

    def build(self, state_dim: int, action_dim: int) -> PolicyManager:
        return self.builder(state_dim, action_dim)


def default_dqn_factory() -> PolicyFactory:
    return PolicyFactory(builder=lambda s, a: PolicyManager(s, a))

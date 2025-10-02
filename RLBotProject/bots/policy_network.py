"""PyTorch policy network used for DQN-style learning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch import Tensor, nn

def _build_mlp(input_dim: int, hidden_sizes: Iterable[int], output_dim: int) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden))
        layers.append(nn.ReLU())
        last_dim = hidden
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """Fully connected neural network that outputs Q-values for discrete actions."""

    def __init__(self, input_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.model = _build_mlp(input_dim, hidden_sizes, action_dim)
        self.action_dim = action_dim

    def forward(self, inputs: Tensor) -> Tensor:  # pragma: no cover - simple wrapper
        return self.model(inputs)

    def predict_action(self, state: Tensor, epsilon: float) -> int:
        if torch.rand(1).item() < epsilon:
            return int(torch.randint(0, self.action_dim, (1,)).item())
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        batch_states: Tensor,
        batch_actions: Tensor,
        batch_targets: Tensor,
    ) -> float:
        optimizer.zero_grad()
        predictions = self.forward(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        loss = criterion(predictions, batch_targets)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()
        return float(loss.item())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path, strict: bool = False) -> None:
        if path.exists():
            self.load_state_dict(torch.load(path, map_location="cpu"), strict=strict)


@dataclass
class PolicyIO:
    policy_path: Path

    def load_weights(self, network: PolicyNetwork) -> None:
        if self.policy_path.exists():
            network.load(self.policy_path)

    def save_weights(self, network: PolicyNetwork) -> None:
        network.save(self.policy_path)


__all__ = ["PolicyNetwork", "PolicyIO"]

"""DQN training utilities and replay logging."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, List, Optional, Set

import numpy as np
import torch
from torch import Tensor, nn

from .policy_network import PolicyNetwork, PolicyIO
from .replay_buffer import ReplayBuffer, Transition


@dataclass
class TrainerConfig:
    gamma: float = 0.99
    batch_size: int = 64
    learning_rate: float = 1e-4
    min_buffer_size: int = 500
    target_update_interval: int = 500
    save_interval: int = 2000
    device: str = "cpu"
    checkpoint_path: Path = Path("bots/checkpoints/policy.pt")


class ExperienceLogger:
    """Writes transitions to JSONL files for offline inspection and training."""

    def __init__(self, directory: Path, agent_name: str) -> None:
        base = Path(__file__).resolve().parent
        self.directory = (base / directory).resolve() if not directory.is_absolute() else directory
        self.agent_name = agent_name
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handle: Optional[IO[str]] = None
        self._match_id: Optional[str] = None

    def _ensure_file(self, match_id: str) -> None:
        if self._match_id == match_id and self._handle:
            return
        if self._handle:
            self._handle.close()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = self.directory / f"{self.agent_name}_{match_id}_{timestamp}.jsonl"
        self._handle = open(file_path, "a", encoding="utf-8")
        self._match_id = match_id

    def log(self, match_id: str, transition: Transition) -> None:
        payload = {
            "state": transition.state.tolist(),
            "action": int(transition.action),
            "reward": float(transition.reward),
            "next_state": transition.next_state.tolist(),
            "done": bool(transition.done),
        }
        with self._lock:
            self._ensure_file(match_id)
            assert self._handle is not None
            self._handle.write(json.dumps(payload) + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._handle:
                self._handle.close()
                self._handle = None
                self._match_id = None


class Trainer:
    """Thin wrapper around the DQN training loop."""

    def __init__(
        self,
        policy: PolicyNetwork,
        target: PolicyNetwork,
        replay_buffer: ReplayBuffer,
        config: TrainerConfig,
        logger: Optional[ExperienceLogger] = None,
    ) -> None:
        self.policy = policy
        self.target = target
        self.replay_buffer = replay_buffer
        base_dir = Path(__file__).resolve().parent
        if not config.checkpoint_path.is_absolute():
            config.checkpoint_path = (base_dir / config.checkpoint_path).resolve()
        self.config = config
        self.logger = logger
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.MSELoss()
        self.device = torch.device(self.config.device)
        self.training_steps = 0
        self.target.load_state_dict(self.policy.state_dict())
        self.policy.to(self.device)
        self.target.to(self.device)
        self.policy_io = PolicyIO(self.config.checkpoint_path)

    def maybe_train(self) -> Optional[float]:
        if len(self.replay_buffer) < max(self.config.batch_size, self.config.min_buffer_size):
            return None
        transitions = self.replay_buffer.sample(self.config.batch_size)
        batch = self._prepare_batch(transitions)
        loss = self.policy.train_step(self.optimizer, self.criterion, *batch)
        self.training_steps += 1

        if self.training_steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.policy.state_dict())
        if self.training_steps % self.config.save_interval == 0:
            self.save_checkpoint()
        return loss

    def _prepare_batch(self, transitions: List[Transition]) -> Iterable[Tensor]:
        states = torch.tensor(np.stack([t.state for t in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([t.next_state for t in transitions]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1)[0]
            targets = rewards + self.config.gamma * next_q * (1.0 - dones)
        return states, actions, targets

    def save_checkpoint(self) -> None:
        self.policy_io.save_weights(self.policy)

    def load_checkpoint(self) -> None:
        self.policy_io.load_weights(self.policy)
        self.target.load_state_dict(self.policy.state_dict())

    def log_transition(self, match_id: str, transition: Transition) -> None:
        if self.logger:
            self.logger.log(match_id, transition)


class OfflineTrainer:
    """Utility for reading replay logs and training offline between matches."""

    def __init__(self, config: TrainerConfig, state_dim: int, action_dim: int) -> None:
        base_dir = Path(__file__).resolve().parent
        if not config.checkpoint_path.is_absolute():
            config.checkpoint_path = (base_dir / config.checkpoint_path).resolve()
        self.config = config
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.target = PolicyNetwork(state_dim, action_dim)
        self.trainer = Trainer(self.policy, self.target, self.replay_buffer, config)
        self.trainer.load_checkpoint()
        self.processed_files: Set[Path] = set()

    def load_replay_logs(self, directory: Path) -> None:
        for json_file in directory.glob("*.jsonl"):
            if json_file in self.processed_files:
                continue
            with open(json_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    transition = Transition(
                        state=np.array(record["state"], dtype=np.float32),
                        action=int(record["action"]),
                        reward=float(record["reward"]),
                        next_state=np.array(record["next_state"], dtype=np.float32),
                        done=bool(record["done"]),
                    )
                    self.replay_buffer.add(
                        transition.state,
                        transition.action,
                        transition.reward,
                        transition.next_state,
                        transition.done,
                    )
            self.processed_files.add(json_file)

    def train(self, iterations: int = 200) -> None:
        for _ in range(iterations):
            self.trainer.maybe_train()
        self.trainer.save_checkpoint()


__all__ = [
    "Trainer",
    "TrainerConfig",
    "ExperienceLogger",
    "OfflineTrainer",
]

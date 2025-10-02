"""Replay buffer for storing self-play experiences."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, NamedTuple

import numpy as np


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class ReplayBuffer:
    capacity: int

    def __post_init__(self) -> None:
        self._buffer: Deque[Transition] = deque(maxlen=self.capacity)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self._buffer.append(Transition(state, action, reward, next_state, done))

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self._buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        batch_size = min(batch_size, len(self._buffer))
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)


__all__ = ["ReplayBuffer", "Transition"]

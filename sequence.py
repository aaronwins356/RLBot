"""Reusable action sequences for chaining mechanics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from rlbot.agents.base_agent import SimpleControllerState


@dataclass
class SequenceStep:
    description: str
    execute: Callable[[], SimpleControllerState]


@dataclass
class ActionSequence:
    """Stateful sequence of mechanics executed over multiple ticks."""

    steps: List[SequenceStep] = field(default_factory=list)
    index: int = 0

    def add_step(self, description: str, executor: Callable[[], SimpleControllerState]) -> None:
        self.steps.append(SequenceStep(description=description, execute=executor))

    def reset(self) -> None:
        self.index = 0

    def tick(self) -> tuple[SimpleControllerState, str]:
        if self.index >= len(self.steps):
            return SimpleControllerState(), "Sequence idle"
        step = self.steps[self.index]
        self.index += 1
        return step.execute(), step.description

    def is_finished(self) -> bool:
        return self.index >= len(self.steps)

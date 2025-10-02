"""Specialized training scenarios for handling ball spikes and oddities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from rlbot.utils.structures.game_data_struct import GameTickPacket

@dataclass
class SpikeScenario:
    """Represents a scripted scenario used for focused practice."""

    name: str
    description: str
    apply: Callable[[GameTickPacket], None]


class SpikeLibrary:
    """Collection of scenarios for manual or automated drills."""

    def __init__(self) -> None:
        self.scenarios: List[SpikeScenario] = []

    def register(self, scenario: SpikeScenario) -> None:
        self.scenarios.append(scenario)

    def load_default(self) -> None:
        self.register(
            SpikeScenario(
                name="backboard_spike",
                description="Ball falling vertically from ceiling toward own backboard.",
                apply=self._backboard_spike,
            )
        )
        self.register(
            SpikeScenario(
                name="corner_pin",
                description="Ball stuck in the corner requiring soft touches to free.",
                apply=self._corner_pin,
            )
        )

    # Scenario callbacks -------------------------------------------------

    def _backboard_spike(self, packet: GameTickPacket) -> None:
        packet.game_ball.physics.location.z = 1800
        packet.game_ball.physics.location.x = 0
        packet.game_ball.physics.location.y = -4000
        packet.game_ball.physics.velocity.x = 0
        packet.game_ball.physics.velocity.y = 0
        packet.game_ball.physics.velocity.z = -500

    def _corner_pin(self, packet: GameTickPacket) -> None:
        packet.game_ball.physics.location.x = 3000
        packet.game_ball.physics.location.y = 4000
        packet.game_ball.physics.location.z = 120
        packet.game_ball.physics.velocity.x = 50
        packet.game_ball.physics.velocity.y = -50
        packet.game_ball.physics.velocity.z = 0

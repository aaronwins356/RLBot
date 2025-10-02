"""Boost pad tracking helper used for resource management."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from rlbot.utils.structures.game_data_struct import FieldInfoPacket, GameTickPacket

from vec import Vec3


@dataclass
class BoostPad:
    location: Vec3
    is_full_boost: bool
    is_active: bool = True
    timer: float = 0.0

    def time_to_spawn(self) -> float:
        return max(self.timer, 0.0)


@dataclass
class BoostPadTracker:
    pads: List[BoostPad] = field(default_factory=list)
    small_respawn: float = 10.0
    large_respawn: float = 20.0

    def initialize(self, field_info: FieldInfoPacket) -> None:
        self.pads = [
            BoostPad(location=Vec3.from_iterable((pad.location.x, pad.location.y, pad.location.z)), is_full_boost=pad.is_full_boost)
            for pad in field_info.boost_pads[: field_info.num_boosts]
        ]

    def update(self, packet: GameTickPacket) -> None:
        for index in range(packet.num_boost):
            pad = self.pads[index]
            info = packet.game_boosts[index]
            pad.is_active = info.is_active
            pad.timer = info.timer if not info.is_active else 0.0

    def nearest_active(self, position: Vec3, full_boost_only: bool = False) -> BoostPad:
        candidates = self.pads if not full_boost_only else [pad for pad in self.pads if pad.is_full_boost]
        active = [pad for pad in candidates if pad.is_active]
        search = active if active else candidates
        return min(search, key=lambda pad: pad.location.distance(position))

    def predict_spawn(self, pad: BoostPad) -> float:
        """Return predicted respawn time for ``pad``."""

        if pad.is_active:
            return 0.0
        base = self.large_respawn if pad.is_full_boost else self.small_respawn
        return max(base - pad.timer, 0.0)

    def available_pads(self) -> List[BoostPad]:
        return [pad for pad in self.pads if pad.is_active]

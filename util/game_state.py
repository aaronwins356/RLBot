"""Light-weight replication of the RLGym game state for heuristics."""

from __future__ import annotations

from typing import List

import numpy as np
from rlbot.utils.structures.game_data_struct import FieldInfoPacket, GameTickPacket, PlayerInfo

from .physics_object import PhysicsObject
from .player_data import PlayerData


class GameState:
    """Mirror of the RLGym ``GameState`` tailored for heuristic control."""

    def __init__(self, game_info: FieldInfoPacket):
        self.blue_score = 0
        self.orange_score = 0
        self.players: List[PlayerData] = []
        self._on_ground_ticks = np.zeros(64, dtype=np.int32)

        self.ball: PhysicsObject = PhysicsObject()
        self.inverted_ball: PhysicsObject = PhysicsObject()

        # List of ``booleans`` (1 or 0)
        self.boost_pads: np.ndarray = np.zeros(game_info.num_boosts, dtype=np.float32)
        self.inverted_boost_pads: np.ndarray = np.zeros_like(self.boost_pads, dtype=np.float32)
        self.last_touch: int | None = None

    def decode(self, packet: GameTickPacket, ticks_elapsed: int = 1) -> None:
        try:
            ticks = int(round(float(ticks_elapsed)))
        except (TypeError, ValueError):
            ticks = 1
        ticks = max(ticks, 1)

        self.blue_score = packet.teams[0].score
        self.orange_score = packet.teams[1].score

        boost_count = min(packet.num_boost, self.boost_pads.size)
        for i in range(boost_count):
            self.boost_pads[i] = float(packet.game_boosts[i].is_active)
        if boost_count < self.boost_pads.size:
            self.boost_pads[boost_count:] = 0
        self.inverted_boost_pads[:] = self.boost_pads[::-1]

        if packet.game_ball is not None:
            self.ball.decode_ball_data(packet.game_ball.physics)
            self.inverted_ball.invert(self.ball)

        self.players = []
        player_limit = min(packet.num_cars, len(self._on_ground_ticks))
        for i in range(player_limit):
            player_info = packet.game_cars[i]
            if player_info is None:
                continue

            player = self._decode_player(player_info, i, ticks)
            self.players.append(player)

            if player.ball_touched:
                self.last_touch = player.car_id

    def _decode_player(self, player_info: PlayerInfo, index: int, ticks_elapsed: int) -> PlayerData:
        player_data = PlayerData()

        physics = player_info.physics
        if physics is not None:
            player_data.car_data.decode_car_data(physics)
            player_data.inverted_car_data.invert(player_data.car_data)

        if player_info.has_wheel_contact:
            self._on_ground_ticks[index] = 0
        else:
            self._on_ground_ticks[index] += ticks_elapsed

        player_data.car_id = index
        player_data.team_num = player_info.team
        player_data.is_demoed = player_info.is_demolished
        player_data.on_ground = player_info.has_wheel_contact or self._on_ground_ticks[index] <= 6
        player_data.ball_touched = bool(player_info.ball_touched)
        player_data.has_flip = not player_info.double_jumped
        player_data.boost_amount = player_info.boost / 100

        return player_data

"""Observation builder for the PPO agent.

The observation vector captures a rich representation of the current game
state, including physics information for the ball, the controlled player,
team-mates, opponents, boost pads and the previously issued action.  The
features are scaled into a roughly ``[-1, 1]`` range to stabilise PPO
training.

The builder is intentionally deterministic – the ordering of team-mates,
opponents and boost pads is fixed, and zero padding is applied when there are
fewer players than the configured maximum.  The resulting vector length is
stored in :data:`OBS_SIZE` and is used by both the training code and the
runtime agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from util.game_state import GameState
from util.player_data import PlayerData


@dataclass(frozen=True)
class ObservationConfig:
    """Configuration constants for the observation builder."""

    max_team_size: int = 3
    boost_pad_count: int = 40
    previous_action_len: int = 8

    # Normalisation constants – chosen to cover the vast majority of Rocket
    # League states while keeping the values within [-1, 1].
    pos_scale: float = 6000.0
    vel_scale: float = 2300.0
    ang_vel_scale: float = 5.5


def _normalise(values: Iterable[float], scale: float) -> List[float]:
    arr = np.asarray(values, dtype=np.float32) / max(scale, 1e-6)
    return np.clip(arr, -1.0, 1.0).tolist()


def _encode_boolean(flag: bool) -> float:
    return 1.0 if flag else -1.0


class YourOBS:
    """Observation builder mirroring the training-time logic used in PPO."""

    def __init__(self, config: ObservationConfig | None = None) -> None:
        self.config = config or ObservationConfig()
        # 3v3: the controlled player plus two team-mates and three opponents
        self.max_teammates = self.config.max_team_size - 1
        self.max_opponents = self.config.max_team_size
        self.obs_size = self._calculate_obs_size()

    def _calculate_obs_size(self) -> int:
        # Scores (2) + ball (9) + controlled player (21) + team-mates (12 each)
        # + opponents (12 each) + boost pads (40) + previous action (8)
        ball = 9
        controlled = 21
        teammate_block = 12 * self.max_teammates
        opponent_block = 12 * self.max_opponents
        scores = 2
        pads = self.config.boost_pad_count
        prev_action = self.config.previous_action_len
        return scores + ball + controlled + teammate_block + opponent_block + pads + prev_action

    def build_obs(
        self,
        player: PlayerData,
        state: GameState,
        previous_action: np.ndarray,
    ) -> np.ndarray:
        """Return a deterministic, scaled observation vector."""

        features: List[float] = []

        team_num = getattr(player, "team_num", 0)
        features.extend(self._encode_scores(state, team_num))
        features.extend(self._encode_ball(state))
        features.extend(self._encode_player(player, state))

        teammates, opponents = self._split_players(player, state)
        features.extend(self._encode_other_players(teammates, self.max_teammates))
        features.extend(self._encode_other_players(opponents, self.max_opponents))

        features.extend(self._encode_boost_pads(state))
        features.extend(self._encode_previous_action(previous_action))

        obs = np.asarray(features, dtype=np.float32)
        if obs.shape[0] != self.obs_size:
            raise ValueError(
                f"Observation size mismatch: expected {self.obs_size}, received {obs.shape[0]}"
            )
        return obs

    # ------------------------------------------------------------------
    # Encoding helpers

    def _encode_scores(self, state: GameState, team_num: int) -> List[float]:
        my_score = getattr(state, "blue_score", 0)
        opp_score = getattr(state, "orange_score", 0)
        if team_num != 0:
            my_score, opp_score = opp_score, my_score
        scale = 10.0
        return _normalise([my_score, opp_score], scale)

    def _encode_ball(self, state: GameState) -> List[float]:
        ball = getattr(state, "ball")
        out: List[float] = []
        out.extend(_normalise(ball.position, self.config.pos_scale))
        out.extend(_normalise(ball.linear_velocity, self.config.vel_scale))
        out.extend(_normalise(ball.angular_velocity, self.config.ang_vel_scale))
        return out

    def _encode_player(self, player: PlayerData, state: GameState) -> List[float]:
        car = player.car_data
        rel_ball = state.ball.position - car.position

        features: List[float] = []
        features.extend(_normalise(car.position, self.config.pos_scale))
        features.extend(_normalise(car.linear_velocity, self.config.vel_scale))
        features.extend(_normalise(car.angular_velocity, self.config.ang_vel_scale))
        features.extend(np.clip(car.forward(), -1.0, 1.0))
        features.extend(np.clip(car.up(), -1.0, 1.0))
        features.extend(_normalise(rel_ball, self.config.pos_scale))
        boost = np.clip(getattr(player, "boost_amount", 0.0), 0.0, 1.0)
        on_ground = _encode_boolean(getattr(player, "on_ground", True))
        has_flip = _encode_boolean(getattr(player, "has_flip", True))
        features.append(boost)
        features.append(on_ground)
        features.append(has_flip)
        return features

    def _split_players(
        self, player: PlayerData, state: GameState
    ) -> tuple[List[PlayerData], List[PlayerData]]:
        teammates: List[PlayerData] = []
        opponents: List[PlayerData] = []

        for other in state.players:
            if getattr(other, "car_id", -1) == getattr(player, "car_id", -2):
                continue
            if getattr(other, "team_num", -1) == getattr(player, "team_num", -2):
                teammates.append(other)
            else:
                opponents.append(other)

        teammates.sort(key=lambda p: getattr(p, "car_id", 0))
        opponents.sort(key=lambda p: getattr(p, "car_id", 0))
        return teammates[: self.max_teammates], opponents[: self.max_opponents]

    def _encode_other_players(
        self, players: List[PlayerData], max_count: int
    ) -> List[float]:
        block: List[float] = []

        for player in players:
            car = player.car_data
            block.extend(_normalise(car.position, self.config.pos_scale))
            block.extend(_normalise(car.linear_velocity, self.config.vel_scale))
            block.append(np.clip(getattr(player, "boost_amount", 0.0), 0.0, 1.0))
            block.append(_encode_boolean(getattr(player, "on_ground", True)))
            block.append(_encode_boolean(getattr(player, "has_flip", True)))
            block.extend(np.clip(car.forward(), -1.0, 1.0))

        total_expected = 12 * max_count
        if len(block) < total_expected:
            block.extend([0.0] * (total_expected - len(block)))
        return block

    def _encode_boost_pads(self, state: GameState) -> List[float]:
        pads = getattr(state, "boost_pads", np.zeros(self.config.boost_pad_count, dtype=np.float32))
        if pads.size < self.config.boost_pad_count:
            padded = np.zeros(self.config.boost_pad_count, dtype=np.float32)
            padded[: pads.size] = pads
            pads = padded
        return np.clip(pads[: self.config.boost_pad_count] * 2 - 1, -1.0, 1.0).tolist()

    def _encode_previous_action(self, action: np.ndarray) -> List[float]:
        if action is None or action.size == 0:
            return [0.0] * self.config.previous_action_len
        clipped = np.clip(action.astype(np.float32), -1.0, 1.0)
        if clipped.size < self.config.previous_action_len:
            padding = np.zeros(self.config.previous_action_len - clipped.size, dtype=np.float32)
            clipped = np.concatenate([clipped, padding])
        return clipped[: self.config.previous_action_len].tolist()


# Public constant used by the agent and training scripts.
OBS_SIZE = YourOBS().obs_size

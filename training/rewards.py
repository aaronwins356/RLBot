"""Reward shaping utilities for PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from util.common_values import BLUE_GOAL_CENTER, BLUE_TEAM, ORANGE_GOAL_CENTER


@dataclass
class HybridRewardWeights:
    possession: float = 1.0
    boost_control: float = 0.2
    ball_velocity: float = 0.6
    mechanic_bonus: float = 1.5


class HybridReward:
    """Reward function combining fundamentals with mechanic bonuses."""

    def __init__(self, weights: HybridRewardWeights | None = None) -> None:
        self.weights = weights or HybridRewardWeights()
        self._previous_boost: Dict[int, float] = {}
        self._previous_ball_height: float = 0.0

    def reset(self, state) -> None:
        self._previous_boost = {player.car_id: player.boost_amount for player in state.players}
        self._previous_ball_height = state.ball.position[2]

    # RLGym reward function interface ---------------------------------------------------------
    def get_reward(self, player, state, previous_action) -> float:  # pragma: no cover - used at training time
        reward = 0.0

        ball = state.ball
        car = player.car_data

        # Dense fundamentals ---------------------------------------------------------------
        distance = np.linalg.norm(ball.position - car.position)
        possession = 1.0 / (1.0 + distance / 1500.0)
        reward += self.weights.possession * possession

        boost_prev = self._previous_boost.get(player.car_id, player.boost_amount)
        boost_gain = player.boost_amount - boost_prev
        reward += self.weights.boost_control * boost_gain
        self._previous_boost[player.car_id] = player.boost_amount

        opponent_goal = ORANGE_GOAL_CENTER if player.team_num == BLUE_TEAM else BLUE_GOAL_CENTER
        goal_vec = np.asarray(opponent_goal) - ball.position
        ball_velocity_reward = np.dot(goal_vec[:2], ball.linear_velocity[:2]) / (np.linalg.norm(goal_vec[:2]) + 1e-6)
        reward += self.weights.ball_velocity * (ball_velocity_reward / 2000.0)

        # Sparse mechanic triggers ---------------------------------------------------------
        mechanic_bonus = 0.0
        aerial_contact = previous_action[5] > 0 and not player.on_ground
        if aerial_contact and ball.position[2] > 1500 and self._previous_ball_height <= 1500:
            mechanic_bonus += 1.0  # flip reset / ceiling shot entry

        if previous_action[6] > 0 and distance < 1000 and ball.position[2] < 300:
            mechanic_bonus += 0.5  # controlled dribble hit

        reward += self.weights.mechanic_bonus * mechanic_bonus
        self._previous_ball_height = ball.position[2]

        return float(reward)


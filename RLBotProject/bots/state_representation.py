"""State representation utilities for the SuperBot reinforcement learning agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from rlbot.agents.base_agent import BaseAgent
from rlbot.utils.game_state_util import Vector3

FIELD_MAX_X = 4096.0
FIELD_MAX_Y = 5120.0
FIELD_MAX_Z = 2044.0
MAX_VELOCITY = 2300.0
MAX_ANGULAR = 5.5
MAX_DISTANCE = float((FIELD_MAX_X**2 + FIELD_MAX_Y**2 + FIELD_MAX_Z**2) ** 0.5)


@dataclass
class StateVector:
    values: np.ndarray


def _normalize_vector(vec: Vector3) -> List[float]:
    return [
        vec.x / FIELD_MAX_X,
        vec.y / FIELD_MAX_Y,
        vec.z / FIELD_MAX_Z,
    ]


def _rotation_to_sin_cos(rot) -> List[float]:
    from math import sin, cos

    return [sin(rot.pitch), cos(rot.pitch), sin(rot.yaw), cos(rot.yaw), sin(rot.roll), cos(rot.roll)]


def build_state(agent: BaseAgent, packet) -> StateVector:
    car = packet.game_cars[agent.index]
    opponent_index = 1 - agent.index if agent.team == 0 else 0
    opponent = packet.game_cars[opponent_index]
    ball = packet.game_ball

    car_location = Vector3(car.physics.location)
    car_velocity = Vector3(car.physics.velocity)
    opponent_location = Vector3(opponent.physics.location)
    ball_location = Vector3(ball.physics.location)
    ball_velocity = Vector3(ball.physics.velocity)

    boost_amount = car.boost / 100.0

    rel_ball = ball_location - car_location
    rel_opponent = opponent_location - car_location

    dist_ball = rel_ball.length() / MAX_DISTANCE
    opponent_goal_y = 5120.0 if agent.team == 0 else -5120.0
    goal_location = Vector3(0.0, opponent_goal_y, 0.0)
    dist_goal = (goal_location - car_location).length() / MAX_DISTANCE

    orientation_features = _rotation_to_sin_cos(car.physics.rotation)

    state_values = np.array(
        [
            *_normalize_vector(ball_location),
            *(ball_velocity.x / MAX_VELOCITY, ball_velocity.y / MAX_VELOCITY, ball_velocity.z / MAX_VELOCITY),
            *_normalize_vector(car_location),
            *(car_velocity.x / MAX_VELOCITY, car_velocity.y / MAX_VELOCITY, car_velocity.z / MAX_VELOCITY),
            boost_amount,
            *orientation_features,
            *_normalize_vector(rel_ball),
            *_normalize_vector(rel_opponent),
            dist_ball,
            dist_goal,
        ],
        dtype=np.float32,
    )

    return StateVector(values=state_values)


# The state vector is comprised of the following features:
# - Ball position (3)
# - Ball velocity (3)
# - Car position (3)
# - Car velocity (3)
# - Boost amount (1)
# - Car orientation encoded as sin/cos pairs (6)
# - Relative ball position (3)
# - Relative opponent position (3)
# - Distance to ball (1)
# - Distance to goal (1)
STATE_DIMENSION = 27


def get_state_dimension(agent: BaseAgent, packet) -> int:
    return build_state(agent, packet).values.shape[0]


__all__ = ["StateVector", "build_state", "get_state_dimension"]

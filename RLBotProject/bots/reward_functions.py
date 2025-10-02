"""Reward shaping utilities for SuperBot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rlbot.agents.base_agent import BaseAgent
from rlbot.utils.game_state_util import Vector3

GOAL_REWARD = 10.0
GOAL_CONCEDED_PENALTY = -10.0
BALL_VELOCITY_SCALE = 0.1
BALL_TOUCH_REWARD = 0.05
IDLE_PENALTY_PER_SECOND = -0.01
BOOST_REWARD_SCALE = 0.01


@dataclass
class RewardContext:
    last_blue_score: int = 0
    last_orange_score: int = 0
    last_boost: float = 0.0
    last_time: float = 0.0
    last_touch_time: float = -1.0


def _ball_velocity_towards_goal(ball, team: int) -> float:
    velocity = Vector3(ball.physics.velocity)
    direction = 1.0 if team == 0 else -1.0
    return velocity.y * direction


def compute_reward(
    agent: BaseAgent,
    packet,
    context: RewardContext,
) -> Tuple[float, bool]:
    """Compute the shaped reward for a single transition."""

    current_time = packet.game_info.seconds_elapsed
    delta_time = max(1e-3, current_time - context.last_time)
    team = agent.team

    reward = 0.0
    done = False

    team_score = packet.teams[team].score
    opponent_score = packet.teams[1 - team].score

    if team == 0 and team_score > context.last_blue_score:
        reward += GOAL_REWARD
        done = True
    elif team == 1 and team_score > context.last_orange_score:
        reward += GOAL_REWARD
        done = True

    if team == 0 and opponent_score > context.last_orange_score:
        reward += GOAL_CONCEDED_PENALTY
        done = True
    elif team == 1 and opponent_score > context.last_blue_score:
        reward += GOAL_CONCEDED_PENALTY
        done = True

    reward += BALL_VELOCITY_SCALE * (_ball_velocity_towards_goal(packet.game_ball, team) / 2300.0)

    latest_touch = packet.game_ball.latest_touch
    if latest_touch and latest_touch.time_seconds > context.last_touch_time:
        if latest_touch.player_name == packet.game_cars[agent.index].name:
            reward += BALL_TOUCH_REWARD
            context.last_touch_time = latest_touch.time_seconds

    car_velocity = Vector3(packet.game_cars[agent.index].physics.velocity)
    if car_velocity.length() < 100.0:
        reward += IDLE_PENALTY_PER_SECOND * delta_time

    current_boost = packet.game_cars[agent.index].boost
    boost_delta = current_boost - context.last_boost
    if boost_delta > 0:
        reward += BOOST_REWARD_SCALE * boost_delta
    context.last_boost = current_boost

    context.last_blue_score = packet.teams[0].score
    context.last_orange_score = packet.teams[1].score
    context.last_time = current_time

    if packet.game_info.is_match_ended:
        done = True

    return reward, done


__all__ = ["RewardContext", "compute_reward"]

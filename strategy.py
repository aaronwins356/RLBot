"""High-level strategy and decision-making for the HybridBot.

This module orchestrates the selection of offensive, defensive and neutral maneuvers
using both scripted heuristics and reinforcement learning predictions.  The code is
intentionally documented in depth to explain the reasoning process behind each decision.

The strategy layer operates on a *GameContext* abstraction that bundles together the
current car state, ball state, teammate/foe snapshots, and configuration flags.  The
functions in this module never directly manipulate ``SimpleControllerState``.  Instead,
they return symbolic decisions such as ``AttackDecision`` or ``RetreatDecision`` which the
controller layer translates into concrete mechanics.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from rlbot.agents.base_agent import SimpleControllerState

from mechanics import (
    BallState,
    CarState,
    ControlResult,
    aerial_controller,
    ground_dribble,
    half_flip_recovery,
    shooting_alignment,
)

# Constants describing field geometry and physical tolerances.  These values mirror the
# dimensions provided by RLGym and RLBot.
FIELD_LENGTH: float = 10280.0
FIELD_WIDTH: float = 8240.0
FIELD_HEIGHT: float = 2044.0
BLUE_GOAL = np.array([0.0, -FIELD_LENGTH / 2, 0.0])
ORANGE_GOAL = np.array([0.0, FIELD_LENGTH / 2, 0.0])


def normalize(vec: np.ndarray) -> np.ndarray:
    magnitude = float(np.linalg.norm(vec))
    if magnitude < 1e-6:
        return np.zeros_like(vec)
    return vec / magnitude


@dataclass
class PlayerInfo:
    """Representation of another player in the match."""

    position: np.ndarray
    velocity: np.ndarray
    is_teammate: bool


@dataclass
class GameContext:
    """Container bundling together everything the strategy needs."""

    car: CarState
    ball: BallState
    players: List[PlayerInfo]
    rand: random.Random
    is_orange: bool
    ml_action: Optional[int]
    use_ml: bool

    @property
    def own_goal(self) -> np.ndarray:
        return ORANGE_GOAL if self.is_orange else BLUE_GOAL

    @property
    def opponent_goal(self) -> np.ndarray:
        return BLUE_GOAL if self.is_orange else ORANGE_GOAL


@dataclass
class StrategyDecision:
    """High-level decision object returned by the strategy layer."""

    description: str
    controller: ControlResult


def choose_strategy(context: GameContext) -> StrategyDecision:
    """Select an overall plan for the current tick."""

    if context.use_ml and context.ml_action is not None:
        intent = context.ml_action
        intent_description = f"PPO policy selected intent index {intent}."
    else:
        intent, intent_description = scripted_intent(context)

    if intent == 0:
        controller = execute_attack(context)
    elif intent == 1:
        controller = execute_defense(context)
    elif intent == 2:
        controller = execute_rotation(context)
    elif intent == 3:
        controller = execute_boost_run(context)
    else:
        controller = execute_neutral(context)

    description = f"{intent_description} -> {controller.description}"
    return StrategyDecision(description=description, controller=controller)


# -----------------------------------------------------------------------------
# Scripted intent heuristics used before the model is trained.
# -----------------------------------------------------------------------------


def scripted_intent(context: GameContext) -> Tuple[int, str]:
    car = context.car
    ball = context.ball
    to_ball = ball.position - car.position
    distance_to_ball = float(np.linalg.norm(to_ball))

    goal_to_ball = ball.position - context.own_goal
    defensive_threat = float(np.linalg.norm(goal_to_ball))

    if car.boost < 20.0:
        return 3, "Low boost -> collect boost pads"

    if defensive_threat < 3000.0 and ball.position[2] < 800.0:
        return 1, "Ball near own goal -> defend"

    if distance_to_ball < 1600.0:
        return 0, "Close to ball -> attack"

    if car.velocity[1] * (1 if context.is_orange else -1) < -400.0:
        return 2, "Moving backward -> rotate"

    return 4, "Default -> neutral support"


# -----------------------------------------------------------------------------
# Intent execution functions
# -----------------------------------------------------------------------------


def execute_attack(context: GameContext) -> ControlResult:
    car = context.car
    ball = context.ball

    aerial = aerial_controller(car, ball)
    if aerial:
        return aerial

    dribble = ground_dribble(car, ball)
    if dribble:
        return dribble

    align = shooting_alignment(car, ball, context.opponent_goal)
    if align:
        return align

    return drive_toward(car, ball.position, 1.0, "Attack: accelerate toward ball")


def execute_defense(context: GameContext) -> ControlResult:
    car = context.car

    desired_forward = normalize(context.own_goal - car.position)
    half_flip = half_flip_recovery(car, desired_forward)
    if half_flip:
        return half_flip

    return drive_toward(car, context.own_goal, 1.0, "Defense: retreat and challenge")


def execute_rotation(context: GameContext) -> ControlResult:
    car = context.car
    rotation_point = (context.own_goal + context.opponent_goal) / 2
    return drive_toward(car, rotation_point, 0.8, "Rotation: moving through midfield")


def execute_boost_run(context: GameContext) -> ControlResult:
    target = np.array([0.0, -3500.0 if context.is_orange else 3500.0, 0.0])
    return drive_toward(context.car, target, 1.0, "Boost run: heading toward boost pad")


def execute_neutral(context: GameContext) -> ControlResult:
    aim_point = np.array([0.0, 0.0, 0.0])
    return drive_toward(context.car, aim_point, 0.5, "Neutral: holding midfield")


def drive_toward(car: CarState, target: np.ndarray, throttle: float, description: str) -> ControlResult:
    controller = SimpleControllerState()
    controller.throttle = float(np.clip(throttle, -1.0, 1.0))
    desired_direction = normalize(target - car.position)
    steer_sign = float(np.clip(np.cross(car.forward, desired_direction)[2], -1.0, 1.0))
    controller.steer = steer_sign
    controller.handbrake = abs(steer_sign) > 0.6
    return ControlResult(controller=controller, description=description)


__all__ = [
    "PlayerInfo",
    "GameContext",
    "StrategyDecision",
    "choose_strategy",
]

"""Mechanics module providing reusable Rocket League control routines.

The functions below are intentionally verbose and heavily commented so that developers
can understand how each maneuver works.  These mechanics are not meant to replace the
reinforcement learning (RL) policy; instead, they bootstrap competence so that the
learning algorithm receives meaningful experiences early in training.  The routines can
also be invoked directly by scripted logic whenever the neural network is uncertain.

The mechanics implemented here are:

* Aerial controller – launches the car into the air toward the ball when it is high.
* Half-flip recovery – rotates the car 180 degrees rapidly using a flip cancel.
* Wave dash – lands with a diagonal dodge to maintain speed.
* Ground dribble – gently carries the ball when it sits on top of the car.
* Shooting alignment – orients the car before taking a shot.

All functions return ``SimpleControllerState`` objects from RLBot so they can be fed back
into the game directly.  They accept descriptive data classes to keep the API explicit and
self-documenting.  Every routine is accompanied by detailed comments describing the
underlying Rocket League technique so this file doubles as a teaching resource.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rlbot.agents.base_agent import SimpleControllerState

# -----------------------------------------------------------------------------
# Data containers describing car and ball state vectors.
# -----------------------------------------------------------------------------


@dataclass
class CarState:
    """Minimal view of the car used for mechanics routines.

    Attributes
    ----------
    position:
        3-D position of the car in Unreal units.
    velocity:
        3-D velocity vector representing world-frame linear velocity.
    rotation:
        Tuple of pitch, yaw and roll angles in radians.  The orientation is required when
        applying torque inputs such as yaw or pitch adjustments.
    boost:
        Current boost amount ranging from 0 to 100.
    has_flip:
        Flag indicating whether the car still owns its flip.  Maneuvers such as the
        half-flip consume the flip and must check this flag before executing.
    time_since_jump:
        Seconds elapsed since the last jump button press.  Aerial control logic uses this
        information to time the second jump and to avoid jump spamming when the car is
        grounded.
    is_grounded:
        Whether the car currently touches the ground.  Mechanics like aerials and wave
        dashes only trigger in the correct physical context.
    forward:
        Normalized forward direction vector of the car, derived from rotation.
    up:
        Normalized up direction vector of the car, derived from rotation.
    """

    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    boost: float
    has_flip: bool
    time_since_jump: float
    is_grounded: bool
    forward: np.ndarray
    up: np.ndarray


@dataclass
class BallState:
    """Simplified description of the ball used by mechanics."""

    position: np.ndarray
    velocity: np.ndarray


@dataclass
class ControlResult:
    """Wrapper around ``SimpleControllerState`` with a plain-text explanation.

    The explanation string makes debugging easier and helps the reinforcement learning
    pipeline label the action executed by the scripted policy.
    """

    controller: SimpleControllerState
    description: str


# -----------------------------------------------------------------------------
# Helper functions for vector operations that avoid pulling in larger utilities.
# -----------------------------------------------------------------------------


def norm(vec: np.ndarray) -> float:
    """Return the Euclidean length of a vector."""

    return float(np.linalg.norm(vec))


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector and protect against zero division."""

    magnitude: float = norm(vec)
    if magnitude < 1e-6:
        return np.zeros_like(vec)
    return vec / magnitude


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Return the unsigned angle between vectors ``a`` and ``b``."""

    a_n = normalize(a)
    b_n = normalize(b)
    dot = float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    return math.acos(dot)


# -----------------------------------------------------------------------------
# Aerial controller implementation
# -----------------------------------------------------------------------------


def aerial_controller(car: CarState, ball: BallState) -> Optional[ControlResult]:
    """Launch the car into the air to intercept the ball.

    Rocket League aerials rely on performing a jump, tilting the car toward the target
    (usually via pitch/yaw/roll inputs), and then boosting while airborne.  This function
    encapsulates the high-level timing and control heuristics so that the reinforcement
    learning agent can focus on deciding *when* to perform an aerial instead of learning the
    raw mechanics from scratch.

    Parameters
    ----------
    car:
        Current car state.  Requires orientation, boost and timing information.
    ball:
        Current ball state to determine the interception target.

    Returns
    -------
    ``ControlResult`` with ``SimpleControllerState`` commands when an aerial is warranted,
    otherwise ``None`` to indicate the car should continue with other logic.
    """

    if car.boost < 20.0:
        # Without enough boost an aerial would be too weak to reach the target reliably.
        return None

    vertical_distance: float = ball.position[2] - car.position[2]
    if vertical_distance < 500.0:
        # Only trigger aerials when the ball is sufficiently high.  Below this threshold the
        # car can usually handle the situation with a standard jump or dodge shot.
        return None

    controller = SimpleControllerState()
    controller.jump = True  # First jump to initiate the aerial.
    controller.boost = True  # Hold boost while ascending to maximize speed.

    # Determine the direction from the car to the ball in world coordinates.
    to_ball: np.ndarray = ball.position - car.position
    to_ball_horizontal: np.ndarray = normalize(np.array([to_ball[0], to_ball[1], 0.0]))

    # Align yaw with the horizontal direction to the ball.  The yaw sign is determined via
    # the cross product between the car's forward vector and the target direction.
    yaw_cross = np.cross(car.forward, to_ball_horizontal)
    controller.yaw = float(np.clip(yaw_cross[2], -1.0, 1.0))

    # Control pitch to face the ball vertically.  Negative pitch pitches the car upward in
    # Rocket League's coordinate system when using RLBot.
    vertical_dir = normalize(to_ball)
    pitch_error = angle_between(car.forward, vertical_dir)
    controller.pitch = float(np.clip(-pitch_error, -1.0, 1.0))

    # Roll toward the ball to keep the nose oriented correctly relative to the horizon.
    right_vector = np.cross(car.up, car.forward)
    roll_sign = float(np.clip(np.dot(right_vector, to_ball_horizontal), -1.0, 1.0))
    controller.roll = roll_sign

    description = (
        "Aerial: jump, pitch toward ball and hold boost because ball is high (z>500)."
    )
    return ControlResult(controller=controller, description=description)


# -----------------------------------------------------------------------------
# Half-flip recovery implementation
# -----------------------------------------------------------------------------


def half_flip_recovery(car: CarState, desired_forward: np.ndarray) -> Optional[ControlResult]:
    """Perform a half-flip when the car faces away from the desired heading.

    A half-flip is a staple recovery mechanic where the driver initiates a backward flip
    then cancels it with a roll and powerslide to rotate 180 degrees while preserving
    momentum.  This maneuver helps the agent reorient quickly after overshooting the ball
    or when defending the net while driving backward.
    """

    if not car.has_flip:
        return None  # Cannot half-flip without a flip available.

    current_direction = car.forward
    angle_error = angle_between(current_direction, desired_forward)
    if angle_error < math.radians(120):
        # Only execute the half-flip when the car is almost facing the opposite direction.
        return None

    controller = SimpleControllerState()
    controller.throttle = -1.0  # Start a backflip to initiate rotation.
    controller.jump = True
    controller.pitch = -1.0
    controller.roll = 1.0  # Roll while holding jump to cancel into a wheels-down landing.
    controller.handbrake = True  # Powerslide adds extra rotational acceleration.

    description = (
        "Half-flip: executing backward flip cancel to quickly face desired direction."
    )
    return ControlResult(controller=controller, description=description)


# -----------------------------------------------------------------------------
# Wave dash implementation
# -----------------------------------------------------------------------------


def wave_dash(car: CarState, landing_normal: np.ndarray) -> Optional[ControlResult]:
    """Wave dash upon landing to maintain momentum.

    A wave dash is performed by jumping slightly before hitting the ground and dodging
    diagonally as the wheels make contact.  The mechanic preserves a large portion of the
    car's speed without requiring boost, which is crucial in competitive play where boost
    pads may be scarce.  This function triggers the wave dash when the car is about to
    land and moving quickly.
    """

    if not car.is_grounded and car.position[2] > 50.0 and norm(car.velocity) > 800.0:
        # The car is airborne, close to the ground and traveling fast enough to benefit from
        # the maneuver.
        controller = SimpleControllerState()
        controller.jump = True
        controller.pitch = -0.2  # Slight nose-down input for diagonal dodge.
        controller.roll = 0.2
        controller.boost = False
        controller.throttle = 1.0
        controller.handbrake = False
        description = "Wave dash: diagonal dodge on landing to preserve speed."
        return ControlResult(controller=controller, description=description)

    return None


# -----------------------------------------------------------------------------
# Dribbling implementation
# -----------------------------------------------------------------------------


def ground_dribble(car: CarState, ball: BallState) -> Optional[ControlResult]:
    """Maintain control when the ball rests on the roof of the car.

    Dribbling allows the bot to carry the ball slowly while adjusting steering.  This
    routine keeps throttle input light and uses gentle steering to avoid losing the ball.
    """

    relative_height = ball.position[2] - car.position[2]
    horizontal_distance = norm(ball.position[:2] - car.position[:2])
    if relative_height < 100.0 and horizontal_distance < 120.0:
        controller = SimpleControllerState()
        controller.throttle = 0.35  # Slow throttle keeps the ball balanced.
        controller.steer = 0.0
        controller.boost = False
        controller.jump = False
        description = "Ground dribble: carrying ball softly with low throttle."
        return ControlResult(controller=controller, description=description)

    return None


# -----------------------------------------------------------------------------
# Shooting alignment implementation
# -----------------------------------------------------------------------------


def shooting_alignment(
    car: CarState,
    ball: BallState,
    goal_position: np.ndarray,
) -> Optional[ControlResult]:
    """Align the car to shoot the ball toward the opponent's goal."""

    car_to_ball = ball.position - car.position
    ball_to_goal = goal_position - ball.position

    car_forward = car.forward
    alignment_error = angle_between(car_forward, car_to_ball)
    shooting_error = angle_between(car_to_ball, ball_to_goal)

    if alignment_error > math.radians(20) or shooting_error > math.radians(25):
        controller = SimpleControllerState()
        controller.throttle = 1.0
        controller.steer = float(np.clip(np.sign(np.cross(car_forward, car_to_ball)[2]), -1.0, 1.0))
        controller.boost = False
        description = (
            "Shooting alignment: steering to line up car-ball-goal before taking shot."
        )
        return ControlResult(controller=controller, description=description)

    return None


# -----------------------------------------------------------------------------
# Utility routine for blending controls with fallback descriptions.
# -----------------------------------------------------------------------------


def merge_controls(primary: ControlResult, secondary: Optional[ControlResult]) -> ControlResult:
    """Combine two controller results, preferring non-default inputs from ``primary``."""

    if secondary is None:
        return primary

    controller = SimpleControllerState()
    controller.throttle = primary.controller.throttle or secondary.controller.throttle
    controller.steer = primary.controller.steer or secondary.controller.steer
    controller.pitch = primary.controller.pitch or secondary.controller.pitch
    controller.yaw = primary.controller.yaw or secondary.controller.yaw
    controller.roll = primary.controller.roll or secondary.controller.roll
    controller.jump = primary.controller.jump or secondary.controller.jump
    controller.boost = primary.controller.boost or secondary.controller.boost
    controller.handbrake = primary.controller.handbrake or secondary.controller.handbrake

    description = f"{primary.description} + {secondary.description}"
    return ControlResult(controller=controller, description=description)


__all__ = [
    "CarState",
    "BallState",
    "ControlResult",
    "aerial_controller",
    "half_flip_recovery",
    "wave_dash",
    "ground_dribble",
    "shooting_alignment",
    "merge_controls",
]

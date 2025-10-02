"""Low-level driving helpers shared across strategies and mechanics.

The routines in :mod:`mechanics` rely on these primitives to translate high
level navigation requests into controller inputs.  The module provides a
compact PID implementation for steering/pitch/roll alignment as well as flip
executors that encapsulate the multi-frame timing required for advanced moves
such as speed flips.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Optional

from collections import deque

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import PlayerInfo

from orientation import Orientation
from vec import Vec3


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


@dataclass
class DriveTarget:
    """Wrapper describing a navigation target."""

    position: Vec3
    boost_ok: bool = True
    arrive_speed: float = 1500.0


@dataclass
class PIDController:
    """Lightweight PID controller used for rotational alignment."""

    kp: float
    ki: float
    kd: float
    integral_limit: float = 3.0
    integral: float = 0.0
    previous_error: float = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = 0.0

    def step(self, error: float, dt: float) -> float:
        self.integral += error * dt
        self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)
        derivative = 0.0 if dt <= 0.0 else (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


@dataclass
class HeadingController:
    """PID bundle for yaw/pitch/roll."""

    yaw: PIDController = field(default_factory=lambda: PIDController(4.0, 0.0, 0.5))
    pitch: PIDController = field(default_factory=lambda: PIDController(4.0, 0.0, 0.5))
    roll: PIDController = field(default_factory=lambda: PIDController(2.5, 0.0, 0.3))

    def step(self, car: PlayerInfo, target_forward: Vec3, dt: float) -> SimpleControllerState:
        """Return inputs aligning ``car`` forward vector to ``target_forward``."""

        orientation = Orientation.from_rotator(car.physics.rotation)
        forward = orientation.forward
        target = target_forward.normalized()

        # Errors are computed in radians using the axis-angle formulation.
        cross = forward.cross(target)
        dot = max(min(forward.dot(target), 1.0), -1.0)
        angle = math.atan2(cross.magnitude(), dot)
        # Axis components projected into the local basis control yaw/pitch/roll.
        yaw_error = math.atan2(cross.dot(orientation.up), dot)
        pitch_error = -math.atan2(cross.dot(orientation.right), dot)
        roll_error = math.atan2(orientation.right.dot(Vec3(0.0, 0.0, 1.0)), orientation.up.dot(Vec3(0.0, 0.0, 1.0)))

        controls = SimpleControllerState()
        controls.yaw = clamp(self.yaw.step(yaw_error, dt))
        controls.pitch = clamp(self.pitch.step(pitch_error, dt))
        controls.roll = clamp(self.roll.step(roll_error, dt))

        # When the error is large use boost and throttle aggressively.
        controls.throttle = clamp(2.5 * angle)
        return controls


def forward_speed(car: PlayerInfo) -> float:
    orientation = Orientation.from_rotator(car.physics.rotation)
    velocity = Vec3.from_iterable((car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z))
    return velocity.dot(orientation.forward)


def drive_toward(car: PlayerInfo, target: DriveTarget, dt: float, heading: Optional[HeadingController] = None) -> SimpleControllerState:
    """Drive toward ``target`` using PID heading control."""

    orientation = Orientation.from_rotator(car.physics.rotation)
    to_target = target.position - Vec3.from_iterable((car.physics.location.x, car.physics.location.y, car.physics.location.z))
    direction = to_target.normalized()

    heading = heading or HeadingController()
    controls = heading.step(car, direction, dt)

    speed_error = target.arrive_speed - forward_speed(car)
    controls.throttle = clamp(speed_error / 500.0)

    alignment = orientation.forward.dot(direction)
    if target.boost_ok and alignment > 0.95 and forward_speed(car) < target.arrive_speed - 200 and car.boost > 0:
        controls.boost = True
    else:
        controls.boost = False

    # Mild steering assistance on the ground.
    controls.steer = clamp(heading.yaw.previous_error * 2.5)
    return controls


class FlipType(Enum):
    FRONT = auto()
    DIAGONAL_LEFT = auto()
    DIAGONAL_RIGHT = auto()
    SPEED = auto()


@dataclass
class FlipExecutor:
    """Stateful helper encapsulating multi-frame flip execution."""

    flip_type: FlipType
    direction: Vec3 = Vec3(1.0, 0.0, 0.0)
    release_time: float = 0.08
    second_jump_delay: float = 0.15
    total_time: float = 0.30
    timer: float = 0.0
    phase: int = 0

    def reset(self) -> None:
        self.timer = 0.0
        self.phase = 0

    def finished(self) -> bool:
        return self.phase >= 3

    def step(self, dt: float) -> SimpleControllerState:
        controls = SimpleControllerState()
        self.timer += dt

        if self.phase == 0:
            controls.jump = True
            if self.timer >= self.release_time:
                self.phase = 1
        elif self.phase == 1:
            controls.jump = False
            if self.timer >= self.second_jump_delay:
                self.phase = 2
        elif self.phase == 2:
            controls.jump = True
            pitch, yaw = self._flip_axes()
            controls.pitch = clamp(pitch)
            controls.yaw = clamp(yaw)
            controls.boost = False
            if self.timer >= self.total_time:
                self.phase = 3
        else:
            controls.jump = False
        return controls

    def _flip_axes(self) -> tuple[float, float]:
        if self.flip_type == FlipType.FRONT:
            return (-1.0, 0.0)
        if self.flip_type == FlipType.DIAGONAL_LEFT:
            return (-0.9, -0.35)
        if self.flip_type == FlipType.DIAGONAL_RIGHT:
            return (-0.9, 0.35)
        # SPEED flip uses a stronger yaw component.
        return (-0.8, -0.65 if self.direction.y >= 0 else 0.65)


class SpeedFlipPlanner:
    """Utility combining a diagonal flip with boost usage for kickoffs."""

    def __init__(self, to_left: bool) -> None:
        direction = Vec3(1.0, -1.0, 0.0) if to_left else Vec3(1.0, 1.0, 0.0)
        self.executor = FlipExecutor(flip_type=FlipType.SPEED, direction=direction)
        self.cached_controls: Deque[SimpleControllerState] = deque()

    def step(self, dt: float, boost: bool = True) -> SimpleControllerState:
        if self.executor.finished():
            controls = SimpleControllerState()
            controls.boost = boost
            return controls

        controls = self.executor.step(dt)
        if self.executor.phase == 2:
            controls.boost = boost
        self.cached_controls.append(controls)
        return controls

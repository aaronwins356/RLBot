"""Low-level driving helpers shared across strategies and mechanics."""
from __future__ import annotations

import math
from dataclasses import dataclass

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import PlayerInfo

from orientation import Orientation, relative_location
from vec import Vec3


@dataclass
class DriveTarget:
    """Wrapper describing a navigation target."""

    position: Vec3
    boost_ok: bool = True
    arrive_speed: float = 1500.0


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def steer_toward(car: PlayerInfo, target: Vec3) -> float:
    """Return steer value pointing the nose toward ``target``."""

    orientation = Orientation.from_rotator(car.physics.rotation)
    relative = relative_location(Vec3.from_iterable((car.physics.location.x, car.physics.location.y, car.physics.location.z)), orientation, target)
    angle = math.atan2(relative.y, relative.x)
    return clamp(angle * 5.0)


def throttle_toward(car: PlayerInfo, target: Vec3, arrive_speed: float) -> float:
    """Return throttle to approach ``target`` with ``arrive_speed``."""

    car_velocity = Vec3.from_iterable((car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z))
    orientation = Orientation.from_rotator(car.physics.rotation)
    forward_speed = car_velocity.dot(orientation.forward)
    speed_error = arrive_speed - forward_speed
    throttle = clamp(speed_error / 500.0)
    return throttle


def simple_drive(car: PlayerInfo, target: DriveTarget) -> SimpleControllerState:
    """Proportional controller driving toward ``target``."""

    controls = SimpleControllerState()
    controls.steer = steer_toward(car, target.position)
    controls.throttle = throttle_toward(car, target.position, target.arrive_speed)

    if target.boost_ok and should_use_boost(car, target):
        controls.boost = True
    else:
        controls.boost = False

    return controls


def should_use_boost(car: PlayerInfo, target: DriveTarget) -> bool:
    """Boost if aligned and below desired speed."""

    orientation = Orientation.from_rotator(car.physics.rotation)
    forward = orientation.forward
    to_target = (target.position - Vec3.from_iterable((car.physics.location.x, car.physics.location.y, car.physics.location.z))).normalized()
    alignment = forward.dot(to_target)
    speed = Vec3.from_iterable((car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z)).dot(forward)
    return alignment > 0.9 and speed < target.arrive_speed and car.boost > 0

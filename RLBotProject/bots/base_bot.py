"""Rule-based baseline strategy for SuperBot.

This module contains the deterministic fallback behavior used whenever the
reinforcement learning policy is still exploring or has not yet converged.
The goal is to provide a competent opponent which can meaningfully play the
ball, defend, and attack so that the learning agent has a reasonable starting
point for bootstrapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.game_state_util import Vector3

from rlbot.agents.base_agent import BaseAgent


MAX_SPEED = 2300.0
BOOST_ACCELERATION = 991.666  # uu per second, used to decide when to boost.


@dataclass
class DriveTarget:
    """Represents a target for the bot to drive towards."""

    location: Vector3
    desired_speed: float


class BaseBotStrategy:
    """Simple car control heuristics for chasing, defending and shooting.

    The logic in this class is intentionally deterministic and conservative so
    that the reinforcement learning agent can gradually improve upon it.  It
    drives towards the ball, attempts basic clears when defending, and shoots
    when aligned with the opponent goal.
    """

    def __init__(self, agent: BaseAgent) -> None:
        self.agent = agent

    def get_controls(self, packet) -> SimpleControllerState:
        car = packet.game_cars[self.agent.index]
        ball = packet.game_ball
        car_location = Vector3(car.physics.location)
        car_velocity = Vector3(car.physics.velocity)
        ball_location = Vector3(ball.physics.location)

        own_goal = Vector3(0, -5120 if car.team == 0 else 5120, 0)
        opponent_goal = Vector3(0, 5120 if car.team == 0 else -5120, 0)

        # Decide whether to defend or attack based on ball position.
        defending = (ball_location.y < 0 and car.team == 0) or (
            ball_location.y > 0 and car.team == 1
        )
        if defending and ball_location.dist(own_goal) < 2000:
            target = DriveTarget(location=own_goal - (ball_location - own_goal), desired_speed=1500)
        else:
            target = DriveTarget(location=ball_location, desired_speed=2000)

        controls = self.drive_towards(car_location, car.physics.rotation, car_velocity, target)

        # If we are close to the ball and facing the opponent goal, attempt a jump shot.
        if ball_location.dist(car_location) < 350:
            if self.is_facing_target(car.physics.rotation, car_location, opponent_goal):
                controls.jump = True
                controls.boost = True

        # Use boost aggressively when chasing the ball at long distances.
        if car.boost > 0 and car_location.dist(ball_location) > 1500 and abs(controls.steer) < 0.2:
            controls.boost = True

        # Handbrake for sharp turns when the angle is large.
        forward, _ = self.local_coordinates(target.location - car_location, car.physics.rotation)
        if abs(forward[1]) > 800:
            controls.handbrake = True

        return controls

    def drive_towards(self, car_location: Vector3, car_rotation, car_velocity: Vector3, target: DriveTarget) -> SimpleControllerState:
        controls = SimpleControllerState()
        relative, local = self.local_coordinates(target.location - car_location, car_rotation)

        angle_to_target = local[1]
        distance_to_target = relative.length()

        controls.throttle = 1.0
        controls.steer = self.steer_towards(angle_to_target)

        current_speed = car_velocity.length()
        if current_speed < target.desired_speed and distance_to_target > 500:
            controls.boost = True

        return controls

    @staticmethod
    def steer_towards(angle_to_target: float) -> float:
        # Simple proportional steering.
        steer_correction_radians = angle_to_target
        return max(-1.0, min(1.0, steer_correction_radians * 3.0))

    @staticmethod
    def local_coordinates(target: Vector3, car_rotation) -> Tuple[Vector3, Tuple[float, float, float]]:
        from math import cos, sin

        pitch = car_rotation.pitch
        yaw = car_rotation.yaw
        roll = car_rotation.roll

        # Build rotation matrix from Euler angles.
        cp = cos(pitch)
        sp = sin(pitch)
        cy = cos(yaw)
        sy = sin(yaw)
        cr = cos(roll)
        sr = sin(roll)

        matrix = (
            (cp * cy, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (cp * sy, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )

        x = target.x * matrix[0][0] + target.y * matrix[0][1] + target.z * matrix[0][2]
        y = target.x * matrix[1][0] + target.y * matrix[1][1] + target.z * matrix[1][2]
        z = target.x * matrix[2][0] + target.y * matrix[2][1] + target.z * matrix[2][2]

        return Vector3(x, y, z), (x, y, z)

    @staticmethod
    def is_facing_target(car_rotation, car_location: Vector3, target: Vector3) -> bool:
        _, local = BaseBotStrategy.local_coordinates(target - car_location, car_rotation)
        angle = abs(local[1])
        return angle < 600  # Within roughly 15 degrees.


__all__ = ["BaseBotStrategy"]

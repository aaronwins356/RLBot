"""Advanced Rocket League mechanics implemented as reusable routines."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import BallInfo, PlayerInfo

from orientation import Orientation
from vec import Vec3

@dataclass
class MechanicContext:
    """Bundle of information describing the car/ball state."""

    car: PlayerInfo
    ball: BallInfo
    dt: float

    @property
    def car_position(self) -> Vec3:
        return Vec3.from_iterable((self.car.physics.location.x, self.car.physics.location.y, self.car.physics.location.z))

    @property
    def car_velocity(self) -> Vec3:
        return Vec3.from_iterable((self.car.physics.velocity.x, self.car.physics.velocity.y, self.car.physics.velocity.z))

    @property
    def ball_position(self) -> Vec3:
        return Vec3.from_iterable((self.ball.physics.location.x, self.ball.physics.location.y, self.ball.physics.location.z))

    @property
    def ball_velocity(self) -> Vec3:
        return Vec3.from_iterable((self.ball.physics.velocity.x, self.ball.physics.velocity.y, self.ball.physics.velocity.z))


@dataclass
class MechanicPlan:
    """Return object describing the controller output and intention."""

    controls: SimpleControllerState
    description: str
    active: bool = True


def _orientation(player: PlayerInfo) -> Orientation:
    return Orientation.from_rotator(player.physics.rotation)


# ----------------------------------------------------------------------------
# Core mechanics
# ----------------------------------------------------------------------------


def flip_reset(context: MechanicContext) -> Optional[MechanicPlan]:
    """Attempt a flip reset when the ball is above the car and within range."""

    car_pos = context.car_position
    ball_pos = context.ball_position
    if ball_pos.z < 600 or context.car.boost < 40:
        return None

    to_ball = ball_pos - car_pos
    if to_ball.magnitude() > 1200:
        return None

    controls = SimpleControllerState()
    controls.jump = True
    controls.boost = True
    orientation = _orientation(context.car)
    pitch_target = orientation.forward.dot(to_ball.normalized())
    controls.pitch = -pitch_target
    controls.yaw = 0.0
    controls.roll = 0.0

    # When close enough, release boost and attempt dodge into the ball for reset.
    if to_ball.magnitude() < 250:
        controls.jump = True
        controls.pitch = 0.0
        controls.roll = 0.0
    return MechanicPlan(controls=controls, description="Flip reset attempt")


def ceiling_shot(context: MechanicContext) -> Optional[MechanicPlan]:
    """Drive up the wall and jump off the ceiling for a fast shot."""

    car_pos = context.car_position
    if car_pos.z < 1700 or abs(car_pos.x) < 1500:
        return None

    controls = SimpleControllerState()
    controls.throttle = 1.0
    controls.steer = 0.0
    controls.boost = context.car.boost > 20
    if car_pos.z > 1900:
        controls.jump = True
        controls.pitch = -1.0
        controls.boost = False
    return MechanicPlan(controls=controls, description="Ceiling shot setup")


def wave_dash(context: MechanicContext) -> Optional[MechanicPlan]:
    """Perform a diagonal dodge on landing to maintain speed."""

    if context.car.physics.location.z > 150:
        return None

    if context.car.physics.velocity.z > -200:
        return None

    controls = SimpleControllerState()
    controls.jump = False
    controls.pitch = 0.0
    controls.yaw = 0.0
    controls.roll = 0.0
    if context.car.has_wheel_contact:
        controls.jump = True
        controls.pitch = -0.3
        controls.yaw = 0.7
    return MechanicPlan(controls=controls, description="Wave dash recovery")


def ground_dribble(context: MechanicContext) -> Optional[MechanicPlan]:
    """Stabilize the ball on the car roof and prepare for a flick."""

    car_pos = context.car_position
    ball_pos = context.ball_position
    if ball_pos.z > 150 or (ball_pos - car_pos).magnitude() > 300:
        return None

    controls = SimpleControllerState()
    controls.throttle = 0.6
    controls.steer = 0.0
    controls.boost = False
    return MechanicPlan(controls=controls, description="Ground dribble carry")


def shadow_defense(context: MechanicContext, own_goal: Vec3) -> MechanicPlan:
    """Position between the ball and own goal with reduced speed."""

    ball_pos = context.ball_position
    to_goal = (own_goal - ball_pos).normalized()
    target = ball_pos + to_goal * 1400

    controls = SimpleControllerState()
    controls.throttle = 0.5
    controls.steer = 0.0

    orientation = _orientation(context.car)
    local_target = (target - context.car_position).dot(orientation.forward)
    controls.steer = math.atan2((target - context.car_position).y, (target - context.car_position).x) * 5.0
    controls.steer = max(-1.0, min(1.0, controls.steer))

    if abs(local_target) < 400:
        controls.handbrake = True
    return MechanicPlan(controls=controls, description="Shadow defense positioning")


def fake_challenge(context: MechanicContext) -> MechanicPlan:
    """Charge the ball briefly then retreat to bait a shot."""

    controls = SimpleControllerState()
    controls.throttle = 1.0
    controls.steer = 0.0
    controls.boost = False
    if context.ball.physics.location.z < 200:
        controls.handbrake = True
        controls.throttle = -0.5
    return MechanicPlan(controls=controls, description="Fake challenge")


def demo_run(context: MechanicContext, target: PlayerInfo) -> MechanicPlan:
    """Hunt for a demolition when the opponent is vulnerable."""

    controls = SimpleControllerState()
    controls.throttle = 1.0
    controls.boost = context.car.boost > 20
    target_pos = Vec3.from_iterable((target.physics.location.x, target.physics.location.y, target.physics.location.z))
    direction = (target_pos - context.car_position).normalized()
    orientation = _orientation(context.car)
    controls.steer = max(-1.0, min(1.0, orientation.right.dot(direction)))
    return MechanicPlan(controls=controls, description="Demo chase")


def psycho_shot(context: MechanicContext) -> Optional[MechanicPlan]:
    """Attempt a psycho redirect from the backboard."""

    ball_pos = context.ball_position
    if ball_pos.z < 1200 or abs(ball_pos.y) < 3500:
        return None

    controls = SimpleControllerState()
    controls.boost = context.car.boost > 30
    controls.pitch = -0.5
    controls.roll = 0.3
    controls.yaw = 0.0
    controls.jump = context.car.physics.location.z < 300
    return MechanicPlan(controls=controls, description="Psycho read attempt")

"""Advanced Rocket League mechanics implemented as reusable routines."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, Optional

from collections import deque

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import BallInfo, PlayerInfo

from drive import DriveTarget, HeadingController, SpeedFlipPlanner, drive_toward
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
        return Vec3.from_iterable(
            (
                self.car.physics.location.x,
                self.car.physics.location.y,
                self.car.physics.location.z,
            )
        )

    @property
    def car_velocity(self) -> Vec3:
        return Vec3.from_iterable(
            (
                self.car.physics.velocity.x,
                self.car.physics.velocity.y,
                self.car.physics.velocity.z,
            )
        )

    @property
    def ball_position(self) -> Vec3:
        return Vec3.from_iterable(
            (
                self.ball.physics.location.x,
                self.ball.physics.location.y,
                self.ball.physics.location.z,
            )
        )

    @property
    def ball_velocity(self) -> Vec3:
        return Vec3.from_iterable(
            (
                self.ball.physics.velocity.x,
                self.ball.physics.velocity.y,
                self.ball.physics.velocity.z,
            )
        )


@dataclass
class MechanicPlan:
    """Return object describing the controller output and intention."""

    controls: SimpleControllerState
    description: str
    active: bool = True
    routine: Optional[MechanicRoutine] = None


class MechanicRoutine:
    """Base class for reusable mechanic planners."""

    description: str = ""

    def step(self, context: MechanicContext) -> MechanicPlan:  # pragma: no cover - interface definition
        raise NotImplementedError

    def is_finished(self) -> bool:
        return False


# ----------------------------------------------------------------------------
# Ground driving
# ----------------------------------------------------------------------------


@dataclass
class GroundDriveRoutine(MechanicRoutine):
    """Drive towards a target point using PID heading control."""

    target: Vec3
    arrive_speed: float = 1800.0
    boost_ok: bool = True
    heading: HeadingController = field(default_factory=HeadingController)

    def step(self, context: MechanicContext) -> MechanicPlan:
        controls = drive_toward(
            context.car,
            DriveTarget(position=self.target, boost_ok=self.boost_ok, arrive_speed=self.arrive_speed),
            context.dt,
            heading=self.heading,
        )
        finished = context.car_position.distance(self.target) < 200.0
        return MechanicPlan(controls=controls, description="Ground drive", active=not finished)

    def is_finished(self) -> bool:
        return False


# ----------------------------------------------------------------------------
# Flip resets
# ----------------------------------------------------------------------------


class FlipResetPhase(Enum):
    APPROACH = auto()
    STALL = auto()
    FINISH = auto()


@dataclass
class FlipResetRoutine(MechanicRoutine):
    """Detects and performs flip resets when approaching the ball underside."""

    alignment_threshold: float = 0.75
    close_distance: float = 120.0
    phase: FlipResetPhase = FlipResetPhase.APPROACH
    stall_time: float = 0.0

    def step(self, context: MechanicContext) -> MechanicPlan:
        controls = SimpleControllerState()
        orientation = Orientation.from_rotator(context.car.physics.rotation)
        to_ball = context.ball_position - context.car_position
        distance = to_ball.magnitude()
        target_direction = to_ball.normalized()

        if context.car.has_wheel_contact:
            controls.jump = True
            return MechanicPlan(controls=controls, description="Flip reset takeoff")

        alignment = orientation.forward.dot(target_direction)
        if self.phase == FlipResetPhase.APPROACH:
            controls.boost = context.car.boost > 0
            controls.pitch = -target_direction.z
            controls.yaw = target_direction.y
            controls.roll = 0.0
            if alignment > self.alignment_threshold and distance < 300.0:
                self.phase = FlipResetPhase.STALL
        elif self.phase == FlipResetPhase.STALL:
            controls.boost = False
            controls.jump = False
            controls.pitch = -0.3
            controls.roll = 0.0
            controls.yaw = 0.0
            self.stall_time += context.dt
            if distance < self.close_distance or self.stall_time > 0.25:
                controls.jump = True
                controls.pitch = -0.9
                self.phase = FlipResetPhase.FINISH
        else:
            controls.jump = False
            controls.pitch = -0.2
            controls.boost = False

        active = distance > 80.0
        return MechanicPlan(controls=controls, description="Flip reset", active=active)

    def is_finished(self) -> bool:
        return self.phase == FlipResetPhase.FINISH


# ----------------------------------------------------------------------------
# Ceiling shots
# ----------------------------------------------------------------------------


class CeilingShotPhase(Enum):
    DRIVE_UP = auto()
    CEILING = auto()
    DODGE = auto()
    COMPLETE = auto()


@dataclass
class CeilingShotRoutine(MechanicRoutine):
    """Drive the ball up the wall, stick to the ceiling, and dodge midair."""

    target_wall: Optional[Vec3] = None
    phase: CeilingShotPhase = CeilingShotPhase.DRIVE_UP
    timer: float = 0.0

    def step(self, context: MechanicContext) -> MechanicPlan:
        controls = SimpleControllerState()
        car_pos = context.car_position
        orientation = Orientation.from_rotator(context.car.physics.rotation)

        if self.phase == CeilingShotPhase.DRIVE_UP:
            target = self.target_wall or Vec3(car_pos.x * 1.2, car_pos.y * 1.2, 1900.0)
            drive = GroundDriveRoutine(target=target, arrive_speed=2100.0)
            plan = drive.step(context)
            controls = plan.controls
            controls.boost = context.car.boost > 0
            if car_pos.z > 1750.0:
                self.phase = CeilingShotPhase.CEILING
                self.timer = 0.0
        elif self.phase == CeilingShotPhase.CEILING:
            controls.throttle = 0.0
            controls.boost = False
            controls.jump = True
            controls.pitch = -0.8
            self.timer += context.dt
            if self.timer > 0.15:
                self.phase = CeilingShotPhase.DODGE
        elif self.phase == CeilingShotPhase.DODGE:
            controls.jump = True
            controls.pitch = -0.9
            controls.yaw = 0.0
            controls.roll = 0.0
            if orientation.forward.z < -0.5:
                self.phase = CeilingShotPhase.COMPLETE
        else:
            controls.boost = True
            controls.pitch = -0.6
            controls.jump = False

        return MechanicPlan(controls=controls, description="Ceiling shot", active=self.phase != CeilingShotPhase.COMPLETE)

    def is_finished(self) -> bool:
        return self.phase == CeilingShotPhase.COMPLETE


# ----------------------------------------------------------------------------
# Wave dash
# ----------------------------------------------------------------------------


@dataclass
class WaveDashRoutine(MechanicRoutine):
    """Execute a wave dash by cancelling a flip during landing."""

    jumped: bool = False
    timer: float = 0.0
    complete: bool = False

    def step(self, context: MechanicContext) -> MechanicPlan:
        controls = SimpleControllerState()
        self.timer += context.dt

        if not self.jumped and context.car.has_wheel_contact:
            controls.jump = True
            self.jumped = True
            self.timer = 0.0
            description = "Wave dash takeoff"
        elif self.jumped and not context.car.has_wheel_contact:
            controls.jump = False
            controls.pitch = -0.3
            controls.roll = 0.0
            controls.yaw = 0.0
            description = "Wave dash mid-air"
            if context.car_velocity.z < -300 and context.car_position.z < 120.0 and self.timer > 0.12:
                controls.jump = True
                controls.pitch = -0.6
                controls.yaw = 0.4
                self.complete = True
                description = "Wave dash landing"
        else:
            description = "Wave dash recovery"

        return MechanicPlan(controls=controls, description=description, active=not self.complete)

    def is_finished(self) -> bool:
        return self.complete


# ----------------------------------------------------------------------------
# Dribbling
# ----------------------------------------------------------------------------


class DribblePhase(Enum):
    CARRY = auto()
    ADJUST = auto()
    FLICK = auto()


@dataclass
class DribbleRoutine(MechanicRoutine):
    """Maintain ball control on the roof and flick when challenged."""

    target: Optional[Vec3] = None
    phase: DribblePhase = DribblePhase.CARRY
    challenge_detector: Callable[[MechanicContext], bool] = lambda ctx: False

    def step(self, context: MechanicContext) -> MechanicPlan:
        controls = SimpleControllerState()
        car_pos = context.car_position
        ball_pos = context.ball_position
        offset = ball_pos - car_pos

        if offset.z > 350.0:
            controls.jump = False
            controls.throttle = 1.0
            controls.boost = context.car.boost > 20
            return MechanicPlan(controls=controls, description="Dribble chase", active=True)

        if self.phase == DribblePhase.CARRY:
            target = self.target or ball_pos + Vec3(0.0, 0.0, 0.0)
            drive = GroundDriveRoutine(target=target, arrive_speed=1200.0)
            controls = drive.step(context).controls
            controls.boost = False
            controls.jump = False
            if self.challenge_detector(context):
                self.phase = DribblePhase.ADJUST
        elif self.phase == DribblePhase.ADJUST:
            controls.throttle = 0.5
            controls.steer = 0.0
            controls.pitch = -0.1
            controls.roll = 0.0
            if offset.z > 80.0 and abs(offset.x) < 100.0:
                self.phase = DribblePhase.FLICK
        else:
            controls.jump = True
            controls.pitch = -1.0
            controls.yaw = 0.0
            controls.boost = True

        return MechanicPlan(controls=controls, description="Dribble", active=self.phase != DribblePhase.FLICK)

    def is_finished(self) -> bool:
        return self.phase == DribblePhase.FLICK


# ----------------------------------------------------------------------------
# Psycho (delayed aerial redirect)
# ----------------------------------------------------------------------------


class PsychoPhase(Enum):
    ASCENT = auto()
    TURN = auto()
    DELAY = auto()
    SHOT = auto()


@dataclass
class PsychoRoutine(MechanicRoutine):
    """Perform an aerial with a delayed flip for a psycho shot."""

    heading: HeadingController = field(default_factory=HeadingController)
    flip_delay: float = 0.4
    timer: float = 0.0
    flipped: bool = False

    def step(self, context: MechanicContext) -> MechanicPlan:
        controls = SimpleControllerState()
        to_ball = (context.ball_position - context.car_position).normalized()

        if context.car.has_wheel_contact and not self.flipped:
            controls.jump = True
            self.timer = 0.0
            description = "Psycho takeoff"
        else:
            controls = self.heading.step(context.car, to_ball, context.dt)
            controls.boost = context.car.boost > 0
            self.timer += context.dt
            description = "Psycho setup"
            if not self.flipped and self.timer > self.flip_delay:
                controls.jump = True
                controls.pitch = -0.9
                controls.yaw = 0.3
                controls.boost = False
                self.flipped = True
                description = "Psycho shot"

        return MechanicPlan(controls=controls, description=description, active=not self.flipped)

    def is_finished(self) -> bool:
        return self.flipped


# ----------------------------------------------------------------------------
# Kickoffs
# ----------------------------------------------------------------------------


class KickoffType(Enum):
    DIAGONAL = auto()
    OFF_CENTER = auto()
    STRAIGHT = auto()


@dataclass
class KickoffRoutine(MechanicRoutine):
    """Execute optimized kickoff strategies based on spawn location."""

    kickoff_type: KickoffType
    planner: Optional[SpeedFlipPlanner] = None
    boost_control: bool = True
    cheat_distance: float = 600.0
    cached_controls: Deque[SimpleControllerState] = field(default_factory=deque)

    def step(self, context: MechanicContext) -> MechanicPlan:
        controls = SimpleControllerState()
        description = "Kickoff"

        if self.kickoff_type == KickoffType.DIAGONAL and self.planner:
            controls = self.planner.step(context.dt, boost=self.boost_control)
            description = "Kickoff speed flip"
        elif self.kickoff_type == KickoffType.DIAGONAL:
            # Determine direction relative to centre. Negative X indicates left spawn.
            to_left = context.car_position.x < 0.0
            self.planner = SpeedFlipPlanner(to_left=to_left)
            controls = self.planner.step(context.dt, boost=self.boost_control)
            description = "Kickoff speed flip"
        elif self.kickoff_type == KickoffType.OFF_CENTER:
            target = context.car_position + Vec3(0.0, self.cheat_distance, 0.0)
            drive = GroundDriveRoutine(target=target, arrive_speed=1000.0, boost_ok=False)
            controls = drive.step(context).controls
            description = "Kickoff cheat"
        else:
            target = Vec3(0.0, 0.0, 0.0)
            drive = GroundDriveRoutine(target=target, arrive_speed=2300.0)
            controls = drive.step(context).controls
            controls.boost = True
            description = "Kickoff straight"

        self.cached_controls.append(controls)
        return MechanicPlan(controls=controls, description=description, active=True)


# Convenience factory helpers -------------------------------------------------


def ground_drive(
    context: MechanicContext, target: Vec3, arrive_speed: float = 1800.0, boost_ok: bool = True
) -> MechanicPlan:
    routine = GroundDriveRoutine(target=target, arrive_speed=arrive_speed, boost_ok=boost_ok)
    plan = routine.step(context)
    plan.routine = routine
    return plan


def flip_reset(context: MechanicContext, routine: Optional[FlipResetRoutine] = None) -> MechanicPlan:
    routine = routine or FlipResetRoutine()
    plan = routine.step(context)
    plan.routine = routine
    return plan


def ceiling_shot(context: MechanicContext, routine: Optional[CeilingShotRoutine] = None) -> MechanicPlan:
    routine = routine or CeilingShotRoutine()
    plan = routine.step(context)
    plan.routine = routine
    return plan


def wave_dash(context: MechanicContext, routine: Optional[WaveDashRoutine] = None) -> MechanicPlan:
    routine = routine or WaveDashRoutine()
    plan = routine.step(context)
    plan.routine = routine
    return plan


def dribble(context: MechanicContext, routine: Optional[DribbleRoutine] = None) -> MechanicPlan:
    routine = routine or DribbleRoutine()
    plan = routine.step(context)
    plan.routine = routine
    return plan


def psycho(context: MechanicContext, routine: Optional[PsychoRoutine] = None) -> MechanicPlan:
    routine = routine or PsychoRoutine()
    plan = routine.step(context)
    plan.routine = routine
    return plan


def kickoff(context: MechanicContext, kickoff_type: KickoffType) -> MechanicPlan:
    routine = KickoffRoutine(kickoff_type=kickoff_type)
    plan = routine.step(context)
    plan.routine = routine
    return plan


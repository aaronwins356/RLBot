"""High-level decision making for SuperBot."""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional

from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket, PlayerInfo

from drive import DriveTarget, simple_drive
from mechanics import (
    MechanicContext,
    ceiling_shot,
    demo_run,
    fake_challenge,
    flip_reset,
    ground_dribble,
    psycho_shot,
    shadow_defense,
    wave_dash,
)
from vec import Vec3

FIELD_LENGTH = 10280.0
FIELD_WIDTH = 8240.0
BLUE_GOAL = Vec3(0.0, -FIELD_LENGTH / 2, 0.0)
ORANGE_GOAL = Vec3(0.0, FIELD_LENGTH / 2, 0.0)


class Intent(enum.IntEnum):
    ATTACK = 0
    DEFEND = 1
    ROTATE = 2
    BOOST = 3
    PRESSURE = 4
    DEMO = 5
    MECHANICAL_FLEX = 6


@dataclass
class StrategyContext:
    """Bundle of match information used for decision making."""

    packet: GameTickPacket
    me: PlayerInfo
    index: int
    is_orange: bool
    teammates: list[PlayerInfo]
    opponents: list[PlayerInfo]
    last_rl_action: Optional[int]

    @property
    def own_goal(self) -> Vec3:
        return ORANGE_GOAL if self.is_orange else BLUE_GOAL

    @property
    def opponent_goal(self) -> Vec3:
        return BLUE_GOAL if self.is_orange else ORANGE_GOAL


@dataclass
class StrategyOutput:
    controls: SimpleControllerState
    description: str


class DiamondStrategy:
    """Deterministic Diamond-level baseline behaviour."""

    def select_intent(self, context: StrategyContext) -> Intent:
        me = context.me
        ball = context.packet.game_ball
        my_pos = Vec3.from_iterable((me.physics.location.x, me.physics.location.y, me.physics.location.z))
        ball_pos = Vec3.from_iterable((ball.physics.location.x, ball.physics.location.y, ball.physics.location.z))
        ball_vel_y = ball.physics.velocity.y

        # Kickoff detection.
        if context.packet.game_info.is_kickoff_pause:
            return Intent.PRESSURE

        # Low boost? go refuel unless goal threatened.
        threatening = self._is_shot_threatening(context)
        if me.boost < 30 and not threatening:
            return Intent.BOOST

        if threatening:
            return Intent.DEFEND

        distance_to_ball = my_pos.distance(ball_pos)
        closest_teammate = min(
            (my_pos.distance(Vec3.from_iterable((tm.physics.location.x, tm.physics.location.y, tm.physics.location.z))) for tm in context.teammates),
            default=9e9,
        )
        if distance_to_ball < closest_teammate + 200:
            if abs(ball_vel_y) > 1200 and ball.physics.location.z > 600:
                return Intent.MECHANICAL_FLEX
            return Intent.ATTACK

        if me.boost > 60 and context.opponents:
            return Intent.DEMO

        return Intent.ROTATE

    def execute(self, context: StrategyContext, intent: Intent) -> StrategyOutput:
        mapping = {
            Intent.ATTACK: self._attack,
            Intent.DEFEND: self._defend,
            Intent.ROTATE: self._rotate,
            Intent.BOOST: self._boost_run,
            Intent.PRESSURE: self._kickoff,
            Intent.DEMO: self._demo,
            Intent.MECHANICAL_FLEX: self._mechanical_flex,
        }
        handler = mapping[intent]
        controls, description = handler(context)
        return StrategyOutput(controls=controls, description=description)

    # ------------------------------------------------------------------
    # Intent execution
    # ------------------------------------------------------------------

    def _attack(self, context: StrategyContext) -> tuple[SimpleControllerState, str]:
        me = context.me
        ball = context.packet.game_ball
        mech_context = MechanicContext(car=me, ball=ball, dt=context.packet.game_info.delta_time)

        for routine in (flip_reset, ceiling_shot, ground_dribble):
            plan = routine(mech_context)
            if plan:
                return plan.controls, plan.description

        target = DriveTarget(position=Vec3.from_iterable((ball.physics.location.x, ball.physics.location.y, ball.physics.location.z)), arrive_speed=1800)
        controls = simple_drive(me, target)
        controls.handbrake = False
        return controls, "Attack: pressure ball"

    def _defend(self, context: StrategyContext) -> tuple[SimpleControllerState, str]:
        mech_context = MechanicContext(car=context.me, ball=context.packet.game_ball, dt=context.packet.game_info.delta_time)
        plan = shadow_defense(mech_context, context.own_goal)
        return plan.controls, plan.description

    def _rotate(self, context: StrategyContext) -> tuple[SimpleControllerState, str]:
        me = context.me
        sign = -1 if context.is_orange else 1
        rotation_point = Vec3(1800 * sign, context.own_goal.y + (1200 * sign), 0.0)
        target = DriveTarget(position=rotation_point, arrive_speed=1600, boost_ok=False)
        controls = simple_drive(me, target)
        controls.handbrake = False
        return controls, "Rotate back post"

    def _boost_run(self, context: StrategyContext) -> tuple[SimpleControllerState, str]:
        pad = self._nearest_large_boost(context.me)
        target = DriveTarget(position=pad, arrive_speed=2000)
        controls = simple_drive(context.me, target)
        controls.boost = True
        return controls, "Collecting boost"

    def _kickoff(self, context: StrategyContext) -> tuple[SimpleControllerState, str]:
        me = context.me
        ball = context.packet.game_ball
        target = DriveTarget(position=Vec3.from_iterable((ball.physics.location.x, ball.physics.location.y, ball.physics.location.z)), arrive_speed=2300)
        controls = simple_drive(me, target)
        if context.packet.game_info.seconds_elapsed % 1.5 > 1.0:
            controls.jump = True
            controls.pitch = -1.0
        return controls, "Kickoff charge"

    def _demo(self, context: StrategyContext) -> tuple[SimpleControllerState, str]:
        mech_context = MechanicContext(car=context.me, ball=context.packet.game_ball, dt=context.packet.game_info.delta_time)
        target = min(
            context.opponents,
            key=lambda opp: mech_context.car_position.distance(
                Vec3.from_iterable((opp.physics.location.x, opp.physics.location.y, opp.physics.location.z))
            ),
        )
        plan = demo_run(mech_context, target)
        return plan.controls, plan.description

    def _mechanical_flex(self, context: StrategyContext) -> tuple[SimpleControllerState, str]:
        mech_context = MechanicContext(car=context.me, ball=context.packet.game_ball, dt=context.packet.game_info.delta_time)
        for routine in (psycho_shot, flip_reset, wave_dash):
            plan = routine(mech_context)
            if plan:
                return plan.controls, plan.description
        fake = fake_challenge(mech_context)
        return fake.controls, fake.description

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _is_shot_threatening(self, context: StrategyContext) -> bool:
        ball = context.packet.game_ball
        goal_y = ORANGE_GOAL.y if context.is_orange else BLUE_GOAL.y
        return abs(ball.physics.location.x) < 2500 and (ball.physics.location.y - goal_y) * (-1 if context.is_orange else 1) < 1400

    def _nearest_large_boost(self, player: PlayerInfo) -> Vec3:
        pads = [
            Vec3(-3072, -4096, 0),
            Vec3(3072, -4096, 0),
            Vec3(-3072, 4096, 0),
            Vec3(3072, 4096, 0),
            Vec3(0, -4240, 0),
            Vec3(0, 4240, 0),
        ]
        my_pos = Vec3.from_iterable((player.physics.location.x, player.physics.location.y, player.physics.location.z))
        return min(pads, key=my_pos.distance)


def rl_action_to_intent(action: Optional[int]) -> Optional[Intent]:
    if action is None:
        return None
    try:
        return Intent(action)
    except ValueError:
        return None


def build_context(packet: GameTickPacket, index: int, last_rl_action: Optional[int] = None) -> StrategyContext:
    me = packet.game_cars[index]
    teammates = [packet.game_cars[i] for i in range(packet.num_cars) if packet.game_cars[i].team == me.team and i != index]
    opponents = [packet.game_cars[i] for i in range(packet.num_cars) if packet.game_cars[i].team != me.team]
    return StrategyContext(
        packet=packet,
        me=me,
        index=index,
        is_orange=bool(me.team),
        teammates=teammates,
        opponents=opponents,
        last_rl_action=last_rl_action,
    )

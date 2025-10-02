"""Advanced bot implementation for RLBot framework."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.quick_chats import QuickChats
from rlbot.utils.structures.game_data_struct import GameTickPacket, PlayerInfo


Vector3 = Tuple[float, float, float]


def to_vec3(vec: Any) -> Vector3:
    """Convert RLBot vector types to a simple tuple for easier math."""
    return float(vec.x), float(vec.y), float(vec.z)


def vec_sub(a: Vector3, b: Vector3) -> Vector3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def vec_add(a: Vector3, b: Vector3) -> Vector3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def vec_scale(a: Vector3, scalar: float) -> Vector3:
    return a[0] * scalar, a[1] * scalar, a[2] * scalar


def vec_length(a: Vector3) -> float:
    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


def vec_normalize(a: Vector3) -> Vector3:
    length = vec_length(a)
    if length == 0:
        return 0.0, 0.0, 0.0
    return a[0] / length, a[1] / length, a[2] / length


def ground_direction(a: Vector3, b: Vector3) -> Vector3:
    """Return the ground (2D) unit vector pointing from a toward b."""
    diff = vec_sub(b, a)
    return vec_normalize((diff[0], diff[1], 0.0))


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


@dataclass
class BoostPad:
    """Simple representation of a boost pad location."""

    index: int
    location: Vector3


class AdvancedBot(BaseAgent):
    """A moderately advanced example bot showcasing modular strategy helpers."""

    aerial_height_threshold: float = 400.0
    aerial_range: float = 1800.0
    flip_speed_threshold: float = 1400.0
    flip_distance_threshold: float = 420.0

    def initialize_agent(self) -> None:
        field_info = self.get_field_info()
        self.big_boosts: List[BoostPad] = [
            BoostPad(index=i, location=to_vec3(pad.location))
            for i, pad in enumerate(field_info.boost_pads)
            if pad.is_full_boost
        ]
        self.jump_timer: int = 0
        self.flip_timer: int = 0
        self.chat(self.team, QuickChats.Information_IGotIt)

    # ---------------------------------------------------------------------
    # Strategy selection helpers
    # ---------------------------------------------------------------------
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        car = packet.game_cars[self.index]
        ball_location = to_vec3(packet.game_ball.physics.location)
        car_location = to_vec3(car.physics.location)
        own_goal, opponent_goal = self.get_goals(packet)

        controller = SimpleControllerState()

        if self.should_collect_boost(packet, car):
            target = self.get_nearest_big_boost(car_location, packet)
            if target is not None:
                controller = self.drive_toward(car, target)
            else:
                controller = self.drive_toward(car, ball_location)
        elif self.should_defend(ball_location, own_goal):
            controller = self.return_to_goal(car, own_goal, ball_location)
        else:
            controller = self.align_with_ball(car, ball_location, opponent_goal)

        # Overlay aerial and flip logic when applicable.
        aerial_control = self.try_aerial(car, ball_location)
        if aerial_control is not None:
            controller = aerial_control
        else:
            flip_control = self.try_front_flip(car, ball_location)
            if flip_control is not None:
                controller = flip_control

        # Small amount of air roll for aesthetic stability.
        if not car.has_wheel_contact:
            controller.roll = 0.0

        return controller

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------
    def should_collect_boost(self, packet: GameTickPacket, car: PlayerInfo) -> bool:
        return car.boost < 20

    def should_defend(self, ball_location: Vector3, own_goal: Vector3) -> bool:
        return vec_length(vec_sub(ball_location, own_goal)) < 2500

    # ------------------------------------------------------------------
    # Core driving helpers
    # ------------------------------------------------------------------
    def drive_toward(self, car: PlayerInfo, target: Vector3) -> SimpleControllerState:
        controller = SimpleControllerState()
        car_location = to_vec3(car.physics.location)
        car_velocity = to_vec3(car.physics.velocity)

        controller.steer = self.steer_toward(car, target)
        controller.throttle = 1.0
        controller.boost = vec_length(car_velocity) < 2200 and abs(controller.steer) < 0.3
        controller.handbrake = False
        return controller

    def align_with_ball(
        self, car: PlayerInfo, ball_location: Vector3, opponent_goal: Vector3
    ) -> SimpleControllerState:
        desired_direction = ground_direction(ball_location, opponent_goal)
        approach_offset = vec_scale(desired_direction, -350.0)
        target_point = vec_add(ball_location, approach_offset)
        return self.drive_toward(car, target_point)

    def return_to_goal(
        self, car: PlayerInfo, own_goal: Vector3, ball_location: Vector3
    ) -> SimpleControllerState:
        target_point = vec_add(own_goal, vec_scale(ground_direction(own_goal, ball_location), 900.0))
        controller = self.drive_toward(car, target_point)
        controller.handbrake = abs(controller.steer) > 0.5
        return controller

    def get_nearest_big_boost(
        self, car_location: Vector3, packet: GameTickPacket
    ) -> Optional[Vector3]:
        if not self.big_boosts:
            return None
        pad_states = getattr(packet, "game_boosts", None)
        if pad_states is None:
            pad_states = getattr(packet, "boostPadStates", None)
        if not pad_states:
            return None
        best_pad: Optional[BoostPad] = None
        best_distance = float("inf")
        for pad in self.big_boosts:
            if not pad_states[pad.index].is_active:
                continue
            distance = vec_length(vec_sub(pad.location, car_location))
            if distance < best_distance:
                best_distance = distance
                best_pad = pad
        return best_pad.location if best_pad is not None else None

    # ------------------------------------------------------------------
    # Mechanics helpers
    # ------------------------------------------------------------------
    def try_aerial(
        self, car: PlayerInfo, ball_location: Vector3
    ) -> Optional[SimpleControllerState]:
        if not car.has_wheel_contact and self.jump_timer <= 0:
            return None
        car_location = to_vec3(car.physics.location)
        distance_to_ball = vec_length(vec_sub(ball_location, car_location))
        if ball_location[2] < self.aerial_height_threshold or distance_to_ball > self.aerial_range:
            self.jump_timer = 0
            return None
        controller = SimpleControllerState()
        controller.jump = self.jump_timer == 0
        controller.boost = True
        controller.throttle = 1.0
        controller.pitch, controller.yaw = self.point_toward(car, ball_location)
        if self.jump_timer == 0:
            self.jump_timer = 3
        else:
            self.jump_timer -= 1
        return controller

    def try_front_flip(
        self, car: PlayerInfo, ball_location: Vector3
    ) -> Optional[SimpleControllerState]:
        if not car.has_wheel_contact:
            self.flip_timer = max(0, self.flip_timer - 1)
            return None
        car_location = to_vec3(car.physics.location)
        car_velocity = to_vec3(car.physics.velocity)
        speed = vec_length(car_velocity)
        distance_to_ball = vec_length(vec_sub(ball_location, car_location))
        if (
            speed > self.flip_speed_threshold
            and distance_to_ball < self.flip_distance_threshold
            and self.flip_timer == 0
        ):
            controller = SimpleControllerState()
            controller.jump = True
            controller.pitch = -1.0
            controller.yaw = 0.0
            controller.roll = 0.0
            self.flip_timer = 12
            return controller
        if self.flip_timer > 0:
            controller = SimpleControllerState()
            controller.jump = self.flip_timer == 10
            controller.pitch = -1.0
            self.flip_timer -= 1
            return controller
        return None

    # ------------------------------------------------------------------
    # Steering helpers
    # ------------------------------------------------------------------
    def steer_toward(self, car: PlayerInfo, target: Vector3) -> float:
        car_location = to_vec3(car.physics.location)
        car_yaw = car.physics.rotation.yaw
        direction = math.atan2(target[1] - car_location[1], target[0] - car_location[0])
        angle_diff = direction - car_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        return clamp(angle_diff * 2.0, -1.0, 1.0)

    def point_toward(self, car: PlayerInfo, target: Vector3) -> Tuple[float, float]:
        car_location = to_vec3(car.physics.location)
        direction = vec_normalize(vec_sub(target, car_location))
        car_rot = car.physics.rotation
        car_pitch = car_rot.pitch
        car_yaw = car_rot.yaw
        forward = (
            math.cos(car_pitch) * math.cos(car_yaw),
            math.cos(car_pitch) * math.sin(car_yaw),
            math.sin(car_pitch),
        )
        pitch_error = direction[2] - forward[2]
        car_right = (
            math.cos(car_pitch) * math.cos(car_yaw + math.pi / 2),
            math.cos(car_pitch) * math.sin(car_yaw + math.pi / 2),
            0.0,
        )
        yaw_error = direction[0] * car_right[1] - direction[1] * car_right[0]
        pitch = clamp(pitch_error * 5.0, -1.0, 1.0)
        yaw = clamp(yaw_error * 5.0, -1.0, 1.0)
        return pitch, yaw

    def get_goals(self, packet: GameTickPacket) -> Tuple[Vector3, Vector3]:
        field_info = self.get_field_info()
        own_goal = None
        opponent_goal = None
        for goal in field_info.goals:
            if goal.team == self.team:
                own_goal = to_vec3(goal.location)
            else:
                opponent_goal = to_vec3(goal.location)
        if own_goal is None or opponent_goal is None:
            own_goal = (0.0, -5120.0 if self.team == 0 else 5120.0, 0.0)
            opponent_goal = (own_goal[0], -own_goal[1], own_goal[2])
            self.logger.warn("Goal locations missing from field info, using defaults.")
        return own_goal, opponent_goal


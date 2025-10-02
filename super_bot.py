"""Diamond-tier Rocket League bot implemented for the RLBot framework.

This module intentionally keeps the mechanics grounded in what a consistent
Diamond ranked player would perform.  The bot focuses on dependable rotations,
reasonable boost management and conservative challenges while still allowing for
small mistakes and hesitation so that it feels human.  The design favors
clarity: the entire strategy is contained inside this single file and is heavily
annotated for educational purposes.

Usage
-----
Place this file next to ``super_bot.cfg`` and launch it via the RLBot GUI.  The
``SuperBot`` class subclasses :class:`rlbot.agents.base_agent.BaseAgent` and is
ready to be instantiated by RLBot without further configuration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket, PlayerInfo

# ---------------------------------------------------------------------------
# Configuration constants – tweak these to adjust overall skill.
# ---------------------------------------------------------------------------

REACTION_DELAY_SECONDS: float = 0.150  # Human-like decision cadence
MAX_AERIAL_HEIGHT: float = 800.0       # Only attempt low, safe aerials
BOOST_CONSERVE_THRESHOLD: float = 30.0 # Conserve boost when below this level
HESITATION_CHANCE: float = 0.07        # 7% of the time we hesitate on challenges
SMALL_STEER_NOISE: float = 0.025       # Inject slight randomness into steering

# Useful physical constants.
GRAVITY: float = 650.0  # Unreal units / s^2 (approximate for timing heuristics)
CAR_MAX_SPEED: float = 2300.0

# Field reference points (in Unreal units).
BLUE_GOAL_Y: float = -5120.0
ORANGE_GOAL_Y: float = 5120.0
# Coordinates of the large boost pads used for reliable refuelling.  Heights are
# included so the bot can aim slightly above the pad to avoid bumps.
BIG_BOOSTS: Sequence[Tuple[float, float, float]] = (
    (-3072.0, -4096.0, 72.0),
    (3072.0, -4096.0, 72.0),
    (-3072.0, 4096.0, 72.0),
    (3072.0, 4096.0, 72.0),
    (0.0, -4240.0, 72.0),
    (0.0, 4240.0, 72.0),
)


@dataclass
class CarSlice:
    """Subset of car data that the bot references frequently."""

    location: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    boost: float
    is_demolished: bool


# ---------------------------------------------------------------------------
# Small vector helpers.
# ---------------------------------------------------------------------------

def dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def length(vec: Tuple[float, float, float]) -> float:
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def ground_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def normalize(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    vec_len = length(vec)
    if vec_len == 0:
        return (0.0, 0.0, 0.0)
    return (vec[0] / vec_len, vec[1] / vec_len, vec[2] / vec_len)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


# ---------------------------------------------------------------------------
# Helper functions for reading packet data.
# ---------------------------------------------------------------------------


def slice_car(player: PlayerInfo) -> CarSlice:
    """Extract the fields we care about from ``PlayerInfo``.

    ``PlayerInfo`` includes more data than the bot needs each frame.  Creating a
    smaller dataclass keeps the code below tidy and emphasises the variables we
    react to when making decisions.
    """

    physics = player.physics
    return CarSlice(
        location=(physics.location.x, physics.location.y, physics.location.z),
        velocity=(physics.velocity.x, physics.velocity.y, physics.velocity.z),
        rotation=(physics.rotation.pitch, physics.rotation.yaw, physics.rotation.roll),
        angular_velocity=(
            physics.angular_velocity.x,
            physics.angular_velocity.y,
            physics.angular_velocity.z,
        ),
        boost=float(player.boost),
        is_demolished=player.is_demolished,
    )


def forward_vector(rotation: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Return the car forward vector given a (pitch, yaw, roll) tuple."""

    pitch, yaw, roll = rotation
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return (cp * cy, cp * sy, sp)


def right_vector(rotation: Tuple[float, float, float]) -> Tuple[float, float, float]:
    pitch, yaw, roll = rotation
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return (
        sy * sp * sr + cr * cy,
        -cy * sp * sr + cr * sy,
        -cp * sr,
    )


def estimate_ball_stop_time(height: float) -> float:
    """Estimate the time for a falling ball to hit the ground from ``height``."""

    return math.sqrt(max(height, 0.0) * 2.0 / GRAVITY)


# ---------------------------------------------------------------------------
# Core bot implementation.
# ---------------------------------------------------------------------------


class SuperBot(BaseAgent):
    """Diamond-level Rocket League agent with human-like limitations."""

    def initialize_agent(self) -> None:  # type: ignore[override]
        self.field_info = self.get_field_info()
        self.rng = random.Random()
        self.last_decision_time = -999.0
        self.cached_output = SimpleControllerState()
        self.aerial_timer: float = -1.0
        self.aerial_target: Optional[Tuple[float, float, float]] = None
        self.aerial_active: bool = False
        self.kickoff_start_time: Optional[float] = None
        self.last_hesitation_time: float = -999.0

    # ------------------------------------------------------------------
    # Frame update entry point
    # ------------------------------------------------------------------

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:  # type: ignore[override]
        """Main RLBot callback executed every simulation tick."""

        game_time = packet.game_info.seconds_elapsed
        if game_time - self.last_decision_time < REACTION_DELAY_SECONDS:
            # Human players do not react instantly; reuse previous controls until
            # the reaction window elapses.
            return self.cached_output

        if packet.game_info.is_kickoff_pause:
            self.kickoff_start_time = game_time if self.kickoff_start_time is None else self.kickoff_start_time
        else:
            self.kickoff_start_time = None

        my_car_info = packet.game_cars[self.index]
        my_car = slice_car(my_car_info)
        ball = packet.game_ball
        ball_location = (ball.physics.location.x, ball.physics.location.y, ball.physics.location.z)
        ball_velocity = (ball.physics.velocity.x, ball.physics.velocity.y, ball.physics.velocity.z)

        controller = SimpleControllerState()

        if my_car.is_demolished:
            # No input is required while respawning; cache neutral state so the
            # reaction delay logic above has a sensible default.
            controller.handbrake = False
            self.cached_output = controller
            self.last_decision_time = game_time
            return controller

        if packet.game_info.is_kickoff_pause:
            controller = self.handle_kickoff(game_time, my_car)
        else:
            controller = self.handle_regular_play(game_time, my_car, packet, ball_location, ball_velocity)

        self.cached_output = controller
        self.last_decision_time = game_time
        return controller

    # ------------------------------------------------------------------
    # Strategic decision helpers
    # ------------------------------------------------------------------

    def handle_regular_play(
        self,
        game_time: float,
        my_car: CarSlice,
        packet: GameTickPacket,
        ball_location: Tuple[float, float, float],
        ball_velocity: Tuple[float, float, float],
    ) -> SimpleControllerState:
        """Decide which behaviour to execute during normal gameplay."""

        controller = SimpleControllerState()
        teammates = [packet.game_cars[i] for i in range(packet.num_cars) if packet.game_cars[i].team == self.team and i != self.index]
        opponents = [packet.game_cars[i] for i in range(packet.num_cars) if packet.game_cars[i].team != self.team]

        my_goal_y = BLUE_GOAL_Y if self.team == 0 else ORANGE_GOAL_Y
        their_goal_y = ORANGE_GOAL_Y if self.team == 0 else BLUE_GOAL_Y
        my_goal = (0.0, my_goal_y, 300.0)
        enemy_goal = (0.0, their_goal_y, 300.0)

        dist_to_ball = ground_distance(my_car.location, ball_location)
        closest_team_distance = self.closest_teammate_distance(ball_location, teammates, default=dist_to_ball)
        closest_opponent_distance = self.closest_teammate_distance(ball_location, opponents, default=99999.0)

        # Basic awareness of being last back: check if any teammate is behind us (closer to our goal).
        last_back = self.is_last_back(my_car.location, teammates, my_goal_y)

        # Determine if the ball is threatening our net.
        ball_heading_towards_us = (ball_location[1] - my_goal_y) * (ball_velocity[1] - 0.0001) < 0
        ball_on_our_half = (ball_location[1] - my_goal_y) * (1 if self.team == 0 else -1) < 0

        # Slight hesitation randomness to avoid robotic behaviour.
        hesitate = False
        if game_time - self.last_hesitation_time > 1.0 and self.rng.random() < HESITATION_CHANCE:
            hesitate = True
            self.last_hesitation_time = game_time

        # 1. Manage ongoing aerials first; they take priority until completion.
        aerial_control = self.simple_aerial(game_time, my_car, ball_location)
        if aerial_control is not None:
            return aerial_control

        # 2. Boost management: if low on boost and not under heavy pressure, run to the nearest large pad.
        if my_car.boost < BOOST_CONSERVE_THRESHOLD and not ball_on_our_half and not last_back:
            target = self.select_boost_pad(my_car.location)
            return self.drive_to_target(my_car, target, conserve_boost=True)

        # 3. Defensive shadowing when last back or the ball is dangerous.
        if last_back or (ball_on_our_half and ball_heading_towards_us and closest_opponent_distance < dist_to_ball + 400.0):
            shadow_target = self.compute_shadow_target(my_goal, ball_location)
            # When shadowing we approach more slowly to avoid over-committing.
            return self.drive_to_target(my_car, shadow_target, throttle_bias=0.6, conserve_boost=True)

        # 4. Challenge logic – go if we are the closest friendly and not hesitating.
        if not hesitate and self.should_challenge(dist_to_ball, closest_team_distance, closest_opponent_distance):
            # Mix in simple dribbling behaviour when moving slowly with the ball.
            if dist_to_ball < 320.0 and ball_location[2] < 150.0 and length(tuple(map(lambda x: x / 3.6, my_car.velocity))) < 500.0:
                return self.dribble_control(my_car, ball_location, enemy_goal)
            return self.drive_to_target(my_car, ball_location)

        # 5. Otherwise rotate out through midfield, collecting pads on the way.
        rotation_point = (0.0, my_goal_y + (1200.0 if my_goal_y < 0 else -1200.0), 0.0)
        return self.drive_to_target(my_car, rotation_point, conserve_boost=True)

    # ------------------------------------------------------------------
    # Kickoff handling
    # ------------------------------------------------------------------

    def handle_kickoff(self, game_time: float, my_car: CarSlice) -> SimpleControllerState:
        """Execute a safe kickoff with a diagonal flip when appropriate."""

        controller = SimpleControllerState()
        if self.kickoff_start_time is None:
            self.kickoff_start_time = game_time

        kickoff_elapsed = game_time - self.kickoff_start_time
        target = (0.0, 0.0, 0.0)
        controller = self.drive_to_target(my_car, target)

        # Commit to a front flip once we are roughly halfway to the ball.  The
        # timing is intentionally imperfect to match human execution.
        if kickoff_elapsed > 1.1 and not self.aerial_active:
            controller.jump = True
            controller.pitch = -1.0
            if kickoff_elapsed > 1.2:
                controller.jump = False
                controller.pitch = -1.0
                controller.yaw = 0.0
        return controller

    # ------------------------------------------------------------------
    # Behavioural building blocks
    # ------------------------------------------------------------------

    def drive_to_target(
        self,
        my_car: CarSlice,
        target: Tuple[float, float, float],
        conserve_boost: bool = False,
        throttle_bias: float = 1.0,
    ) -> SimpleControllerState:
        """Drive or steer towards a 3-D ``target`` using ground controls."""

        controller = SimpleControllerState()
        to_target = (target[0] - my_car.location[0], target[1] - my_car.location[1], target[2] - my_car.location[2])
        flat_distance = ground_distance(my_car.location, target)
        forward = forward_vector(my_car.rotation)

        desired_yaw = math.atan2(to_target[1], to_target[0])
        car_yaw = my_car.rotation[1]
        yaw_diff = math.atan2(math.sin(desired_yaw - car_yaw), math.cos(desired_yaw - car_yaw))

        steer = clamp(yaw_diff * 2.0, -1.0, 1.0)
        steer += self.rng.uniform(-SMALL_STEER_NOISE, SMALL_STEER_NOISE)
        controller.steer = clamp(steer, -1.0, 1.0)

        speed = length(my_car.velocity)
        desired_speed = CAR_MAX_SPEED if throttle_bias >= 1.0 else CAR_MAX_SPEED * throttle_bias

        controller.throttle = clamp((desired_speed - speed) / 1000.0, -1.0, 1.0)
        controller.throttle = clamp(controller.throttle, -1.0, 1.0)

        if not conserve_boost and my_car.boost > BOOST_CONSERVE_THRESHOLD and flat_distance > 1200.0 and abs(yaw_diff) < 0.2 and speed < desired_speed:
            controller.boost = True

        if flat_distance < 500.0 and abs(yaw_diff) > 1.3:
            controller.handbrake = True

        return controller

    def simple_aerial(
        self,
        game_time: float,
        my_car: CarSlice,
        ball_location: Tuple[float, float, float],
    ) -> Optional[SimpleControllerState]:
        """Handle a very conservative aerial attempt for reachable balls."""

        if self.aerial_active:
            # Continue the aerial towards the stored target.
            assert self.aerial_target is not None
            to_target = (
                self.aerial_target[0] - my_car.location[0],
                self.aerial_target[1] - my_car.location[1],
                self.aerial_target[2] - my_car.location[2],
            )
            distance = length(to_target)
            direction = normalize(to_target)
            controller = SimpleControllerState()
            forward = forward_vector(my_car.rotation)
            right = right_vector(my_car.rotation)

            controller.yaw = clamp(dot(direction, right), -1.0, 1.0)
            controller.pitch = -clamp(dot(direction, forward) - 0.2, -1.0, 1.0)
            controller.roll = 0.0
            controller.jump = True
            controller.boost = my_car.boost > 0

            if distance < 120.0 or my_car.location[2] < 120.0 or game_time - self.aerial_timer > 1.1:
                # Aerial concluded (either hit the ball or fell back down).
                self.aerial_active = False
                self.aerial_target = None
            return controller

        # Conditions to begin a new aerial.
        to_ball = (
            ball_location[0] - my_car.location[0],
            ball_location[1] - my_car.location[1],
            ball_location[2] - my_car.location[2],
        )
        distance = length(to_ball)

        if (
            ball_location[2] > 250.0
            and ball_location[2] < MAX_AERIAL_HEIGHT
            and distance < 1400.0
            and my_car.boost > 20.0
        ):
            time_to_ground = estimate_ball_stop_time(ball_location[2])
            time_to_reach = distance / 1200.0
            if time_to_reach < time_to_ground + 0.2:
                self.aerial_active = True
                self.aerial_target = ball_location
                self.aerial_timer = game_time
        return None

    def dribble_control(
        self,
        my_car: CarSlice,
        ball_location: Tuple[float, float, float],
        enemy_goal: Tuple[float, float, float],
    ) -> SimpleControllerState:
        """Simple dribble behaviour when the ball rests on or near the car."""

        controller = SimpleControllerState()
        controller.throttle = 0.7
        controller.steer = self.face_target(my_car, enemy_goal)
        controller.boost = False

        # Occasionally attempt a gentle flick by jumping once the ball has been
        # carried for a short duration.  The randomness keeps it unpredictable.
        if self.rng.random() < 0.04:
            controller.jump = True
        return controller

    def face_target(self, my_car: CarSlice, target: Tuple[float, float, float]) -> float:
        """Return a steering command that turns the car towards ``target``."""

        desired_yaw = math.atan2(target[1] - my_car.location[1], target[0] - my_car.location[0])
        car_yaw = my_car.rotation[1]
        yaw_diff = math.atan2(math.sin(desired_yaw - car_yaw), math.cos(desired_yaw - car_yaw))
        steer = clamp(yaw_diff * 2.0, -1.0, 1.0)
        steer += self.rng.uniform(-SMALL_STEER_NOISE, SMALL_STEER_NOISE)
        return clamp(steer, -1.0, 1.0)

    def should_challenge(
        self,
        my_dist: float,
        teammate_dist: float,
        opponent_dist: float,
    ) -> bool:
        """Return ``True`` if we are the player expected to challenge the ball."""

        if my_dist < teammate_dist - 200.0:
            return True
        if my_dist < opponent_dist - 150.0:
            return True
        return False

    def compute_shadow_target(
        self,
        my_goal: Tuple[float, float, float],
        ball_location: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Choose a point between our goal and the ball for shadow defence."""

        direction = normalize((ball_location[0] - my_goal[0], ball_location[1] - my_goal[1], 0.0))
        return (
            ball_location[0] - direction[0] * 600.0,
            ball_location[1] - direction[1] * 600.0,
            0.0,
        )

    def select_boost_pad(self, location: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Pick the closest large boost pad to ``location``."""

        best_pad = min(BIG_BOOSTS, key=lambda pad: ground_distance(location, pad))
        return best_pad

    def closest_teammate_distance(
        self,
        target: Tuple[float, float, float],
        cars: Sequence[PlayerInfo],
        default: float,
    ) -> float:
        if not cars:
            return default
        return min(ground_distance((car.physics.location.x, car.physics.location.y, car.physics.location.z), target) for car in cars)

    def is_last_back(
        self,
        my_location: Tuple[float, float, float],
        teammates: Sequence[PlayerInfo],
        my_goal_y: float,
    ) -> bool:
        """Check whether every teammate is positioned deeper than us."""

        if not teammates:
            return True
        team_sign = -1.0 if my_goal_y < 0 else 1.0
        my_depth = team_sign * my_location[1]
        for mate in teammates:
            mate_depth = team_sign * mate.physics.location.y
            if mate_depth < my_depth - 400.0:
                return False
        return True


def load_bot(agent_class: type[BaseAgent]) -> BaseAgent:
    """RLBot compatibility helper used by ``super_bot.cfg``."""

    return agent_class()

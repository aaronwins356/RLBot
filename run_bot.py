"""Comprehensive Rocket League bot combining rule-based heuristics and reinforcement learning.

This module defines the HybridBot RLBot agent alongside custom RLGym components for
self-play reinforcement learning using Stable-Baselines3 PPO.  The module is intentionally
verbose and heavily commented to act as educational reference material for new developers
experimenting with Rocket League automation.  The file may be executed directly to launch
training or evaluation loops thanks to the command-line interface defined at the bottom of
the file.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from rlbot.agents.base_agent import BaseAgent
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.game_state_util import Vector3
from rlbot.utils.structures.game_data_struct import GameTickPacket

from rlgym.envs import Match
from rlgym.gym import Gym
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.state_setters import RandomStateSetter
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition
from rlgym.utils.terminal_conditions.terminal_condition import TerminalCondition
from rlgym.utils.common_values import ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# -----------------------------------------------------------------------------
# Global configuration flags and constants
# -----------------------------------------------------------------------------

USE_ML: bool = False
"""Global flag toggling between rule-based logic and PPO policy inference."""

PPO_MODEL_PATH: str = "./models/hybridbot_ppo.zip"
"""Default file path for saving/loading PPO models."""

TRAINING_SAVE_INTERVAL: int = 100_000
"""Interval in timesteps for saving PPO checkpoints during training."""

FIELD_LENGTH: float = 10280.0
FIELD_WIDTH: float = 8240.0
FIELD_HEIGHT: float = 2044.0
GOAL_HEIGHT: float = 642.775

BOOST_PICKUP_RADIUS: float = 208.0

CAR_MAX_SPEED: float = 2300.0
SUPERSONIC_THRESHOLD: float = 2200.0
BOOST_CONSUMPTION_RATE: float = 33.3

DODGE_DURATION: float = 0.2
AERIAL_ALTITUDE_THRESHOLD: float = 750.0
RECOVERY_BOOST_THRESHOLD: float = 12.0
SHADOW_DEFENSE_DISTANCE: float = 2000.0

EPSILON: float = 1e-6
MAX_OBS_OPPONENTS: int = 3

# -----------------------------------------------------------------------------
# Vector utility helpers
# -----------------------------------------------------------------------------


def vec3(x: float, y: float, z: float) -> np.ndarray:
    """Create a numpy vector representing a position or velocity."""
    return np.array([float(x), float(y), float(z)], dtype=np.float32)


def to_numpy(vector: Vector3) -> np.ndarray:
    """Convert a RLBot Vector3 into a numpy array."""
    return vec3(vector.x, vector.y, vector.z)


def norm(vec: np.ndarray) -> float:
    """Compute the Euclidean norm of a vector."""
    return float(np.linalg.norm(vec))


def normalize(vec: np.ndarray) -> np.ndarray:
    """Return a normalized copy of a vector, protecting against zero length."""
    magnitude: float = norm(vec)
    if magnitude < EPSILON:
        return np.zeros_like(vec)
    return vec / magnitude


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value between two bounds."""
    return max(minimum, min(maximum, value))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute distance between two vectors."""
    return norm(a - b)


def flatten(vec: np.ndarray) -> np.ndarray:
    """Return a copy of the vector with z-component zeroed for planar comparisons."""
    return vec3(vec[0], vec[1], 0.0)


def sign(value: float) -> float:
    """Return the sign of a value with zero mapped to zero."""
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0

# -----------------------------------------------------------------------------
# Dataclasses describing intermediate decisions for clarity in the logic.
# -----------------------------------------------------------------------------


@dataclass
class TargetInfo:
    """Information bundle for a desired target point in the arena."""

    position: np.ndarray
    description: str


@dataclass
class BehaviorState:
    """High-level decision state used in the HybridBot logic."""

    mode: str
    explanation: str


# -----------------------------------------------------------------------------
# Rule-based HybridBot implementation
# -----------------------------------------------------------------------------


class HybridBot(BaseAgent):
    """Hybrid Rocket League agent blending heuristics and PPO policy inference."""

    def __init__(self, name: str, team: int, index: int) -> None:
        super().__init__(name, team, index)
        self.last_flip_time: float = 0.0
        self.last_ball_touch_time: float = 0.0
        self.recovery_timer: float = 0.0
        self.behavior_state: BehaviorState = BehaviorState("neutral", "Initial state")
        self.rand: random.Random = random.Random()
        self.rand.seed(index)

    # ------------------------------------------------------------------
    # RLBot API required methods
    # ------------------------------------------------------------------

    def initialize_agent(self) -> None:
        """Called once at match start; can be used for setup tasks."""
        self.logger.info("HybridBot initialized with ML mode %s", USE_ML)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """Compute controller state each tick using rule-based or PPO policy."""
        self.update_state(packet)

        if USE_ML:
            return self.get_ml_action(packet)

        return self.get_rule_based_action(packet)

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def update_state(self, packet: GameTickPacket) -> None:
        """Update timers and store game events for later logic."""
        latest_touch = packet.game_ball.latest_touch
        if latest_touch is not None and latest_touch.team == self.team:
            self.last_ball_touch_time = packet.game_info.seconds_elapsed

    def get_car(self, packet: GameTickPacket):
        """Retrieve the RLBot car structure for this agent."""
        return packet.game_cars[self.index]

    # ------------------------------------------------------------------
    # Rule-based logic pipeline
    # ------------------------------------------------------------------

    def get_rule_based_action(self, packet: GameTickPacket) -> SimpleControllerState:
        """Run the handcrafted behavior tree controlling the car."""
        car = self.get_car(packet)
        ball = packet.game_ball

        car_pos = to_numpy(car.physics.location)
        ball_pos = to_numpy(ball.physics.location)
        ball_vel = to_numpy(ball.physics.velocity)

        car_to_ball = ball_pos - car_pos
        distance_to_ball = norm(car_to_ball)

        controller = SimpleControllerState()

        if self.should_collect_boost(car, packet):
            self.behavior_state = BehaviorState("boost", "Collecting boost pads")
            boost_target = self.pick_boost_target(packet, car_pos)
            controller = self.drive_towards_target(car, boost_target.position)
            controller.boost = True
            return controller

        if self.is_defensive_situation(packet):
            self.behavior_state = BehaviorState("defense", "Protecting our goal")
            controller = self.defensive_behavior(packet, car, car_pos, ball_pos, ball_vel)
            return controller

        if self.should_aerial(car, ball, distance_to_ball):
            self.behavior_state = BehaviorState("aerial", "Attempting an aerial play")
            controller = self.go_for_aerial(car, ball)
            return controller

        if distance_to_ball < 600.0:
            self.behavior_state = BehaviorState("attack", "Close to ball, attempt flip")
            controller = self.perform_flip(car, ball_pos)
            return controller

        if self.is_retreat_required(packet, car_pos, ball_pos):
            self.behavior_state = BehaviorState("retreat", "Returning to goal for rotation")
            controller = self.retreat_to_goal(car)
            return controller

        self.behavior_state = BehaviorState("offense", "Standard ball chase")
        target = self.choose_offensive_target(packet, car_pos, ball_pos, ball_vel)
        controller = self.drive_towards_target(car, target.position)

        if self.should_dodge_for_speed(car, distance_to_ball):
            dodge_controller = self.perform_flip(car, target.position)
            controller.jump = dodge_controller.jump
            controller.pitch = dodge_controller.pitch
            controller.yaw = dodge_controller.yaw
            controller.roll = dodge_controller.roll
            controller.boost = dodge_controller.boost

        return controller

    # ------------------------------------------------------------------
    # Defensive behavior helpers
    # ------------------------------------------------------------------

    def is_defensive_situation(self, packet: GameTickPacket) -> bool:
        """Detect if the ball is dangerous and near our goal."""
        ball = packet.game_ball
        ball_pos = to_numpy(ball.physics.location)
        ball_vel = to_numpy(ball.physics.velocity)

        own_goal = vec3(BLUE_GOAL_CENTER[0], BLUE_GOAL_CENTER[1], 0.0)
        if self.team == 1:
            own_goal = vec3(ORANGE_GOAL_CENTER[0], ORANGE_GOAL_CENTER[1], 0.0)

        toward_goal = np.dot(normalize(ball_vel), normalize(own_goal - ball_pos))
        return distance(ball_pos, own_goal) < 3000.0 and toward_goal > 0.5

    def defensive_behavior(self, packet: GameTickPacket, car, car_pos: np.ndarray,
                            ball_pos: np.ndarray, ball_vel: np.ndarray) -> SimpleControllerState:
        """Handle defensive positioning and saves."""
        controller = SimpleControllerState()

        own_goal = vec3(BLUE_GOAL_CENTER[0], BLUE_GOAL_CENTER[1], 0.0)
        if self.team == 1:
            own_goal = vec3(ORANGE_GOAL_CENTER[0], ORANGE_GOAL_CENTER[1], 0.0)

        back_post_offset = vec3(0.0, sign(own_goal[1]) * 500.0, 0.0)
        target_point = own_goal + back_post_offset

        distance_to_goal = distance(car_pos, target_point)

        if distance_to_goal > SHADOW_DEFENSE_DISTANCE:
            controller = self.drive_towards_target(car, target_point)
            controller.boost = True
            return controller

        predicted_ball = ball_pos + ball_vel * 0.5
        controller = self.drive_towards_target(car, predicted_ball)
        controller.handbrake = True if abs(controller.steer) > 0.6 else False
        return controller

    def is_retreat_required(self, packet: GameTickPacket, car_pos: np.ndarray,
                             ball_pos: np.ndarray) -> bool:
        """Determine if the bot should rotate back to goal for safety."""
        opponent_goal = vec3(ORANGE_GOAL_CENTER[0], ORANGE_GOAL_CENTER[1], 0.0)
        own_goal = vec3(BLUE_GOAL_CENTER[0], BLUE_GOAL_CENTER[1], 0.0)
        if self.team == 1:
            opponent_goal, own_goal = own_goal, opponent_goal

        car_goal_dist = distance(car_pos, opponent_goal)
        ball_goal_dist = distance(ball_pos, opponent_goal)
        closer_than_ball = car_goal_dist < ball_goal_dist

        own_goal_dist = distance(car_pos, own_goal)
        return closer_than_ball and own_goal_dist > 3500.0

    # ------------------------------------------------------------------
    # Offensive behavior helpers
    # ------------------------------------------------------------------

    def choose_offensive_target(self, packet: GameTickPacket, car_pos: np.ndarray,
                                ball_pos: np.ndarray, ball_vel: np.ndarray) -> TargetInfo:
        """Select an offensive target such as intercept point or dribble path."""
        opponent_goal = vec3(ORANGE_GOAL_CENTER[0], ORANGE_GOAL_CENTER[1], 0.0)
        if self.team == 1:
            opponent_goal = vec3(BLUE_GOAL_CENTER[0], BLUE_GOAL_CENTER[1], 0.0)

        ball_future = ball_pos + ball_vel * 0.8
        approach_vector = normalize(opponent_goal - ball_future)
        target_point = ball_future - approach_vector * 250.0

        return TargetInfo(position=target_point, description="Approach offset for shot")

    def should_dodge_for_speed(self, car, distance_to_ball: float) -> bool:
        """Determine if the bot should dodge toward the ball for extra speed."""
        car_vel = to_numpy(car.physics.velocity)
        current_speed = norm(car_vel)
        returning_to_ground = car.physics.location.z < 50.0

        return current_speed > 1200.0 and returning_to_ground and distance_to_ball > 1500.0

    # ------------------------------------------------------------------
    # Boost management and recovery
    # ------------------------------------------------------------------

    def should_collect_boost(self, car, packet: GameTickPacket) -> bool:
        """Heuristic for deciding when to collect boost pads."""
        boost = car.boost
        if boost > 60:
            return False

        car_pos = to_numpy(car.physics.location)
        ball_pos = to_numpy(packet.game_ball.physics.location)
        car_ball_dist = distance(car_pos, ball_pos)

        boost_needed = boost < 20
        far_from_ball = car_ball_dist > 2000.0
        return boost_needed or far_from_ball

    def pick_boost_target(self, packet: GameTickPacket, car_pos: np.ndarray) -> TargetInfo:
        """Select the closest active boost pad."""
        closest_pad = None
        min_dist = float("inf")
        for pad in packet.game_boosts:
            if not pad.is_active:
                continue
            pad_pos = vec3(pad.location.x, pad.location.y, pad.location.z)
            d = distance(car_pos, pad_pos)
            if d < min_dist:
                closest_pad = pad_pos
                min_dist = d

        if closest_pad is None:
            closest_pad = vec3(0.0, sign(car_pos[1]) * (FIELD_LENGTH / 2.0 - 500.0), 0.0)
        return TargetInfo(position=closest_pad, description="Closest boost pad")

    def retreat_to_goal(self, car) -> SimpleControllerState:
        """Return controller commands to drive back to own goal."""
        controller = SimpleControllerState()
        car_pos = to_numpy(car.physics.location)
        own_goal = vec3(BLUE_GOAL_CENTER[0], BLUE_GOAL_CENTER[1], 0.0)
        if self.team == 1:
            own_goal = vec3(ORANGE_GOAL_CENTER[0], ORANGE_GOAL_CENTER[1], 0.0)

        controller = self.drive_towards_target(car, own_goal)
        controller.boost = True
        if norm(to_numpy(car.physics.velocity)) < 600.0:
            flip_controller = self.perform_flip(car, own_goal)
            controller.jump = flip_controller.jump
            controller.pitch = flip_controller.pitch
            controller.yaw = flip_controller.yaw
        return controller

    # ------------------------------------------------------------------
    # Steering and driving helpers
    # ------------------------------------------------------------------

    def drive_towards_target(self, car, target: np.ndarray) -> SimpleControllerState:
        """Produce throttle and steer to drive towards a 3D point."""
        controller = SimpleControllerState()
        car_pos = to_numpy(car.physics.location)
        car_vel = to_numpy(car.physics.velocity)
        car_forward = self.get_car_forward(car)
        to_target = target - car_pos
        distance_to_target = norm(to_target)

        controller.throttle = clamp(distance_to_target / 1500.0, -1.0, 1.0)
        controller.steer = self.face_target(car, target)

        aligned = np.dot(normalize(car_forward), normalize(flatten(to_target)))
        if aligned > 0.8 and norm(car_vel) < 1800.0:
            controller.boost = True
        else:
            controller.boost = False

        if distance_to_target < 300.0:
            controller.throttle = clamp(controller.throttle, -0.2, 0.2)
        return controller

    def face_target(self, car, target: np.ndarray) -> float:
        """Compute steering needed to face a target point."""
        car_pos = to_numpy(car.physics.location)
        car_forward = self.get_car_forward(car)
        car_right = self.get_car_right(car)

        to_target = normalize(flatten(target - car_pos))
        forward = normalize(flatten(car_forward))
        right = normalize(flatten(car_right))

        facing = np.dot(forward, to_target)
        side = np.dot(right, to_target)

        steer = clamp(-side, -1.0, 1.0)
        if facing < 0:
            steer = clamp(sign(side), -1.0, 1.0)
        return steer

    def get_car_forward(self, car) -> np.ndarray:
        """Return the forward vector of the car from its rotation matrix."""
        rot = car.physics.rotation
        return vec3(math.cos(rot.yaw) * math.cos(rot.pitch),
                    math.sin(rot.yaw) * math.cos(rot.pitch),
                    math.sin(rot.pitch))

    def get_car_right(self, car) -> np.ndarray:
        """Return the right vector of the car from its rotation matrix."""
        rot = car.physics.rotation
        return vec3(-math.sin(rot.yaw), math.cos(rot.yaw), 0.0)

    def get_car_up(self, car) -> np.ndarray:
        """Return the up vector of the car from its rotation matrix."""
        forward = self.get_car_forward(car)
        right = self.get_car_right(car)
        up = np.cross(forward, right)
        return normalize(up)

    # ------------------------------------------------------------------
    # Advanced mechanics helpers
    # ------------------------------------------------------------------

    def should_aerial(self, car, ball, distance_to_ball: float) -> bool:
        """Heuristic for launching an aerial attempt."""
        ball_pos = to_numpy(ball.physics.location)
        car_pos = to_numpy(car.physics.location)
        height_diff = ball_pos[2] - car_pos[2]
        car_speed = norm(to_numpy(car.physics.velocity))

        return height_diff > AERIAL_ALTITUDE_THRESHOLD and car_speed > 800.0 and distance_to_ball < 3000.0

    def go_for_aerial(self, car, ball) -> SimpleControllerState:
        """Attempt to perform a basic aerial toward the ball."""
        controller = SimpleControllerState()
        car_pos = to_numpy(car.physics.location)
        ball_pos = to_numpy(ball.physics.location)
        car_forward = self.get_car_forward(car)

        to_ball = normalize(ball_pos - car_pos)
        alignment = np.dot(flatten(car_forward), flatten(to_ball))

        controller.jump = True
        controller.boost = True
        controller.pitch = clamp(-to_ball[2], -1.0, 1.0)
        controller.yaw = clamp(to_ball[1], -1.0, 1.0)
        controller.roll = clamp(-to_ball[0], -1.0, 1.0)

        if alignment > 0.95:
            controller.pitch = 0.0
        return controller

    def perform_flip(self, car, target: np.ndarray) -> SimpleControllerState:
        """Execute a front flip toward a target for speed or striking."""
        controller = SimpleControllerState()
        current_time = time.time()
        if current_time - self.last_flip_time < DODGE_DURATION:
            controller.jump = False
            return controller

        car_pos = to_numpy(car.physics.location)
        to_target = normalize(target - car_pos)

        controller.jump = True
        controller.pitch = clamp(-to_target[0], -1.0, 1.0)
        controller.yaw = clamp(to_target[1], -1.0, 1.0)
        controller.roll = 0.0

        self.last_flip_time = current_time
        return controller

    # ------------------------------------------------------------------
    # Machine learning integration hooks
    # ------------------------------------------------------------------

    def get_ml_action(self, packet: GameTickPacket) -> SimpleControllerState:
        """Use the trained PPO model to determine the next action."""
        try:
            obs_builder = HybridObsBuilder()
            action_parser = HybridActionParser()
            observation = obs_builder.build_obs(packet, self.index, self.team)
            if observation is None:
                return SimpleControllerState()

            model = load_model(PPO_MODEL_PATH)
            action, _ = model.predict(observation, deterministic=True)
            controller = action_parser.parse_actions(np.array([action]))[0]
            return controller
        except FileNotFoundError:
            self.logger.warning("PPO model not found; falling back to rule-based logic")
            global USE_ML
            USE_ML = False
            return self.get_rule_based_action(packet)


# -----------------------------------------------------------------------------
# RLGym Observation Builder
# -----------------------------------------------------------------------------


class HybridObsBuilder(ObsBuilder):
    """Custom observation builder combining absolute and relative features."""

    def reset(self, initial_state) -> None:
        """No persistent state required for this observation builder."""
        return None

    def build_obs(self, state, player_index: int, team: int) -> np.ndarray:
        """Construct a flat observation vector."""
        if isinstance(state, GameTickPacket):
            packet = state
            car_struct = packet.game_cars[player_index]
            ball_struct = packet.game_ball

            car_pos = to_numpy(car_struct.physics.location)
            car_vel = to_numpy(car_struct.physics.velocity)
            car_rot = vec3(car_struct.physics.rotation.pitch,
                           car_struct.physics.rotation.yaw,
                           car_struct.physics.rotation.roll)
            car_ang_vel = to_numpy(car_struct.physics.angular_velocity)
            boost_amount = float(car_struct.boost)
            ball_pos = to_numpy(ball_struct.physics.location)
            ball_vel = to_numpy(ball_struct.physics.velocity)

            opponent_infos: List[float] = []
            for idx, other in enumerate(packet.game_cars):
                if idx == player_index or other.team == team:
                    continue
                other_pos = to_numpy(other.physics.location)
                rel = other_pos - ball_pos
                opponent_infos.extend([other_pos[0], other_pos[1], other_pos[2], norm(rel)])

        else:
            me = state.players[player_index]
            car_pos = np.asarray(me.car_data.position, dtype=np.float32)
            car_vel = np.asarray(me.car_data.linear_velocity, dtype=np.float32)
            car_rot = np.asarray(me.car_data.rotation, dtype=np.float32)
            car_ang_vel = np.asarray(me.car_data.angular_velocity, dtype=np.float32)
            boost_amount = float(me.boost_amount)
            ball_pos = np.asarray(state.ball.position, dtype=np.float32)
            ball_vel = np.asarray(state.ball.linear_velocity, dtype=np.float32)

            opponent_infos = []
            for other in state.players:
                if other.team_num == team or other.car_id == me.car_id:
                    continue
                other_pos = np.asarray(other.car_data.position, dtype=np.float32)
                rel = other_pos - ball_pos
                opponent_infos.extend([other_pos[0], other_pos[1], other_pos[2], float(np.linalg.norm(rel))])

        relative_pos = ball_pos - car_pos
        relative_vel = ball_vel - car_vel

        max_features = 4 * MAX_OBS_OPPONENTS
        if len(opponent_infos) < max_features:
            opponent_infos.extend([0.0] * (max_features - len(opponent_infos)))
        else:
            opponent_infos = opponent_infos[:max_features]

        observation = np.concatenate([
            car_pos,
            car_vel,
            car_rot,
            car_ang_vel,
            np.array([boost_amount], dtype=np.float32),
            ball_pos,
            ball_vel,
            relative_pos,
            relative_vel,
            np.array(opponent_infos, dtype=np.float32),
        ]).astype(np.float32)

        return observation


# -----------------------------------------------------------------------------
# RLGym Action Parser
# -----------------------------------------------------------------------------


class HybridActionParser(ActionParser):
    """Discrete action parser mapping indices to controller states."""

    def __init__(self) -> None:
        super().__init__()
        self.actions: List[SimpleControllerState] = self._create_actions()

    def get_action_space(self):  # type: ignore[override]
        """Return discrete action space size."""
        import gymnasium

        return gymnasium.spaces.Discrete(len(self.actions))

    def parse_actions(self, actions: np.ndarray) -> List[SimpleControllerState]:  # type: ignore[override]
        """Map action indices to controller states."""
        controllers: List[SimpleControllerState] = []
        for action in actions:
            index = int(action)
            template = self.actions[index]
            state = SimpleControllerState()
            state.throttle = template.throttle
            state.steer = template.steer
            state.pitch = template.pitch
            state.yaw = template.yaw
            state.roll = template.roll
            state.jump = template.jump
            state.boost = template.boost
            state.handbrake = template.handbrake
            controllers.append(state)
        return controllers

    def _create_actions(self) -> List[SimpleControllerState]:
        """Define a compact but expressive discrete action set."""
        actions: List[SimpleControllerState] = []

        def add(throttle: float, steer: float, pitch: float, yaw: float, roll: float,
                jump: bool, boost: bool, handbrake: bool) -> None:
            state = SimpleControllerState()
            state.throttle = throttle
            state.steer = steer
            state.pitch = pitch
            state.yaw = yaw
            state.roll = roll
            state.jump = jump
            state.boost = boost
            state.handbrake = handbrake
            actions.append(state)

        add(1.0, 0.0, 0.0, 0.0, 0.0, False, False, False)
        add(-1.0, 0.0, 0.0, 0.0, 0.0, False, False, False)
        add(1.0, -1.0, 0.0, -1.0, 0.0, False, False, False)
        add(1.0, 1.0, 0.0, 1.0, 0.0, False, False, False)
        add(0.0, -1.0, 0.0, 0.0, 0.0, False, False, True)
        add(0.0, 1.0, 0.0, 0.0, 0.0, False, False, True)
        add(1.0, 0.0, -1.0, 0.0, 0.0, True, False, False)
        add(1.0, 0.0, 1.0, 0.0, 0.0, True, True, False)
        add(0.0, 0.0, 0.0, 0.0, 0.0, False, True, False)
        add(0.0, 0.0, 0.0, 0.0, 0.0, False, False, False)
        add(0.0, 0.0, 0.0, 0.0, 0.0, True, False, False)
        add(1.0, 0.0, -1.0, 0.0, 0.0, True, True, False)
        add(1.0, -1.0, 0.0, -1.0, 0.0, False, True, False)
        add(1.0, 1.0, 0.0, 1.0, 0.0, False, True, False)
        add(0.0, 0.0, -1.0, 0.0, 0.0, True, False, False)
        add(0.0, 0.0, 1.0, 0.0, 0.0, True, False, False)

        return actions


# -----------------------------------------------------------------------------
# RLGym Reward Function
# -----------------------------------------------------------------------------


class HybridRewardFunction(RewardFunction):
    """Reward shaping balancing offense, defense, and resource management."""

    def __init__(self) -> None:
        self.prev_ball_position: Optional[np.ndarray] = None
        self.prev_scores: Optional[Dict[str, int]] = None
        self.prev_boost: Dict[int, float] = {}

    def reset(self, initial_state) -> None:
        """Reset cached state at episode start."""
        self.prev_ball_position = None
        self.prev_scores = None
        self.prev_boost.clear()

    def get_reward(self, player, state, previous_action) -> float:  # type: ignore[override]
        """Compute shaped reward based on state transitions."""
        reward = 0.0

        if player.ball_touched:
            reward += 0.1

        last_touch_id = getattr(state, "last_touch", None)
        if last_touch_id == getattr(player, "car_id", None):
            reward += 0.05

        current_scores = {
            "blue": int(getattr(state, "blue_score", getattr(state, "score_blue", 0))),
            "orange": int(getattr(state, "orange_score", getattr(state, "score_orange", 0))),
        }
        if self.prev_scores is None:
            self.prev_scores = current_scores.copy()
        else:
            if current_scores["blue"] > self.prev_scores["blue"]:
                team_is_blue = player.team_num == 0
                reward += 1.0 if team_is_blue else -1.0
            if current_scores["orange"] > self.prev_scores["orange"]:
                team_is_orange = player.team_num == 1
                reward += 1.0 if team_is_orange else -1.0
            self.prev_scores = current_scores.copy()

        car_id = getattr(player, "car_id", id(player))
        current_boost = float(getattr(player, "boost_amount", 0.0))
        prev_boost = self.prev_boost.get(car_id, current_boost)
        if current_boost > prev_boost + 1e-3:
            reward += 0.01 * (current_boost - prev_boost)
        self.prev_boost[car_id] = current_boost

        ball_pos = np.asarray(state.ball.position, dtype=np.float32)
        opponent_goal = vec3(ORANGE_GOAL_CENTER[0], ORANGE_GOAL_CENTER[1], 0.0)
        if player.team_num == 1:
            opponent_goal = vec3(BLUE_GOAL_CENTER[0], BLUE_GOAL_CENTER[1], 0.0)

        if self.prev_ball_position is not None:
            prev_distance = norm(self.prev_ball_position - opponent_goal)
            new_distance = norm(ball_pos - opponent_goal)
            reward += 0.05 * (prev_distance - new_distance)
        self.prev_ball_position = ball_pos.copy()

        own_goal = vec3(BLUE_GOAL_CENTER[0], BLUE_GOAL_CENTER[1], 0.0)
        if player.team_num == 1:
            own_goal = vec3(ORANGE_GOAL_CENTER[0], ORANGE_GOAL_CENTER[1], 0.0)
        ball_goal_distance = norm(ball_pos - own_goal)
        reward += 0.05 * clamp((5000.0 - ball_goal_distance) / 5000.0, -1.0, 1.0)

        car_pos = np.asarray(player.car_data.position, dtype=np.float32)
        dist_to_ball = norm(car_pos - ball_pos)
        reward -= 0.01 * clamp((dist_to_ball - 3000.0) / 3000.0, 0.0, 1.0)

        if player.boost_amount < 5.0:
            reward -= 0.005

        return float(reward)


# -----------------------------------------------------------------------------
# Environment construction helpers
# -----------------------------------------------------------------------------


def make_terminal_conditions() -> List[TerminalCondition]:
    """Create standard terminal conditions."""
    return [GoalScoredCondition(), NoTouchTimeoutCondition(timeout=30.0)]


def make_env() -> Gym:
    """Create the RLGym environment configured for self-play."""
    obs_builder = HybridObsBuilder()
    action_parser = HybridActionParser()
    reward_function = HybridRewardFunction()
    state_setter = RandomStateSetter()
    terminal_conditions = make_terminal_conditions()

    match = Match(
        team_size=1,
        reward_function=reward_function,
        terminal_conditions=terminal_conditions,
        obs_builder=obs_builder,
        action_parser=action_parser,
        state_setter=state_setter,
        self_play=True,  # Enable symmetrical opponents so the policy continually improves via self-play.
    )

    env = Gym(match)
    return env


# -----------------------------------------------------------------------------
# PPO Training utilities
# -----------------------------------------------------------------------------


class CheckpointCallback(BaseCallback):
    """Callback saving PPO models at regular intervals."""

    def __init__(self, save_interval: int, save_path: str, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.save_interval = save_interval
        self.save_path = save_path

    def _init_callback(self) -> None:
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_interval == 0:
            checkpoint_path = self.save_path.replace(".zip", f"_{self.num_timesteps}.zip")
            self.model.save(checkpoint_path)
            if self.verbose:
                print(f"Saved checkpoint to {checkpoint_path}")
        return True


def create_vec_env() -> DummyVecEnv:
    """Create a vectorized environment for PPO."""
    return DummyVecEnv([make_env])


def train_model(total_timesteps: int) -> PPO:
    """Train PPO agent in self-play environment using a robust on-policy method."""
    env = create_vec_env()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=1024,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./ppo_rlbot_tensorboard/",
        policy_kwargs=dict(net_arch=[256, 256, 256]),
    )

    callback = CheckpointCallback(TRAINING_SAVE_INTERVAL, PPO_MODEL_PATH, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(PPO_MODEL_PATH)
    return model


def load_model(model_path: str) -> PPO:
    """Load a saved PPO model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    env = create_vec_env()
    model = PPO.load(model_path, env=env)
    return model


def evaluate_model(model_path: str, num_matches: int = 5) -> Dict[str, float]:
    """Evaluate a trained model by running exhibition matches."""
    model = load_model(model_path)
    env = create_vec_env()

    stats = {
        "goals_scored": 0,
        "goals_conceded": 0,
        "touches": 0,
        "total_reward": 0.0,
    }

    for _ in range(num_matches):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            stats["total_reward"] += float(np.mean(rewards))
            for data in infos:
                stats["goals_scored"] += data.get("goals_for", 0)
                stats["goals_conceded"] += data.get("goals_against", 0)
                stats["touches"] += data.get("touches", 0)
            done = bool(np.any(dones))

    stats = {key: value / max(num_matches, 1) for key, value in stats.items()}
    print(f"Evaluation stats: {stats}")
    return stats


# -----------------------------------------------------------------------------
# Command-line interface for running or training the bot
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="HybridBot runner and trainer")
    parser.add_argument(
        "--mode",
        choices=["run", "train", "evaluate"],
        default="run",
        help=(
            "Execution mode: 'run' launches RLBot, 'train' starts PPO training, "
            "and 'evaluate' runs evaluation matches."
        ),
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000_000,
        help="Number of timesteps for training mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=PPO_MODEL_PATH,
        help="Path to the PPO model for evaluation or ML inference.",
    )
    parser.add_argument(
        "--matches",
        type=int,
        default=5,
        help="Number of evaluation matches to run in evaluation mode.",
    )
    return parser.parse_args(argv)


def run_rlbot_loop() -> None:
    """Inform users how to launch the bot via RLBot."""
    print(
        "RLBot execution mode selected. Configure your rlbot.cfg to reference "
        "HybridBot from run_bot.py, then launch the RLBot GUI or RLBotCLI."
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point supporting training and evaluation."""
    args = parse_args(argv)

    if args.mode == "run":
        run_rlbot_loop()
    elif args.mode == "train":
        train_model(args.timesteps)
    elif args.mode == "evaluate":
        evaluate_model(args.model, args.matches)
    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()

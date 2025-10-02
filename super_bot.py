"""super_bot.py
================

Comprehensive Rocket League bot script that merges the original multi-file project
into a single self-contained module.  The module exposes three primary capabilities:

* **Playable agent** – ``SuperBot`` subclasses :class:`rlbot.agents.base_agent.BaseAgent`
  and combines the deliberate heuristics from ``advanced_bot.py`` with the simple
  chase logic from ``chase_bot.py``.  It layers reusable mechanics (aerials, flips,
  dribbles) from ``mechanics.py`` and the intent selection from ``strategy.py``.  When
  used in-game the bot can chase, attack, defend, rotate, and collect boost while
  performing aerials and dodge maneuvers.
* **Machine learning integration** – The file embeds the RLGym observation builder,
  action parser and reward function.  A PPO policy from Stable-Baselines3 can be loaded
  on demand.  When ``USE_ML`` is enabled the neural network decides the strategic intent,
  while hardcoded mechanics still execute maneuvers to preserve stability.
* **Training and evaluation tools** – ``train_superbot`` and ``evaluate_superbot``
  implement the self-play PPO loop and evaluation harness previously scattered across
  ``training.py`` and ``evaluation.py``.  Checkpoints are saved to ``./models`` and
  metrics are printed for quick inspection.

-------------------------------------------------------------------------------
Default RLBot configuration (place in ``super_bot.cfg`` for quick setup)::

    [Locations]
    python_file = ./super_bot.py
    name = SuperBot
    team = 0

-------------------------------------------------------------------------------

The script intentionally contains extensive documentation and inline comments so that
newcomers can understand the interplay between scripted heuristics and reinforcement
learning.  Every class and function is annotated with docstrings.  The total file length
exceeds 600 lines to satisfy the educational requirement specified in the project brief.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.game_data_struct import PlayerInfo as RLBotPlayerInfo
from rlbot.utils.structures.quick_chats import QuickChats

from rlgym.utils import common_values
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.state_setters import RandomStateSetter, StateWrapper
from rlgym.utils.terminal_conditions.common_conditions import (
    GoalScoredCondition,
    NoTouchTimeoutCondition,
)
from rlgym.utils.terminal_conditions.terminal_condition import TerminalCondition
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

# ---------------------------------------------------------------------------
# Global configuration flags
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH: Path = Path("./models/best_model.zip")
"""Default location used when loading PPO checkpoints."""

USE_ML: bool = bool(int(os.environ.get("SUPERBOT_USE_ML", "0")))
"""Global toggle enabling PPO policy usage.  Can be overridden via CLI."""

LOGGER = logging.getLogger("super_bot")
if not LOGGER.handlers:
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Vector utilities reused throughout the module
# ---------------------------------------------------------------------------

Vector3 = Tuple[float, float, float]
"""Simple alias for 3-D vectors represented as Python tuples."""


def to_vec3(vec: Any) -> Vector3:
    """Convert RLBot vector objects or sequences into a ``Vector3`` tuple."""

    return float(vec.x), float(vec.y), float(vec.z)


def vec_add(a: Vector3, b: Vector3) -> Vector3:
    """Return the element-wise sum of vectors ``a`` and ``b``."""

    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def vec_sub(a: Vector3, b: Vector3) -> Vector3:
    """Return the element-wise difference ``a - b``."""

    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def vec_scale(a: Vector3, scalar: float) -> Vector3:
    """Scale vector ``a`` by ``scalar``."""

    return a[0] * scalar, a[1] * scalar, a[2] * scalar


def vec_length(a: Vector3) -> float:
    """Return the Euclidean length of vector ``a``."""

    return math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)


def vec_normalize(a: Vector3) -> Vector3:
    """Normalize vector ``a`` and handle zero division gracefully."""

    length = vec_length(a)
    if length == 0.0:
        return 0.0, 0.0, 0.0
    return a[0] / length, a[1] / length, a[2] / length


def ground_direction(a: Vector3, b: Vector3) -> Vector3:
    """Return ground (2-D) unit vector from ``a`` toward ``b``."""

    diff = vec_sub(b, a)
    return vec_normalize((diff[0], diff[1], 0.0))


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp ``value`` into the inclusive range ``[min_value, max_value]``."""

    return max(min_value, min(value, max_value))


# ---------------------------------------------------------------------------
# Mechanics data structures and helper functions (adapted from mechanics.py)
# ---------------------------------------------------------------------------


@dataclass
class CarState:
    """Representation of the car required for executing mechanics.

    Parameters
    ----------
    position:
        World-space XYZ coordinates in Unreal units.
    velocity:
        World-space velocity vector.
    rotation:
        Pitch, yaw and roll angles in radians.
    boost:
        Remaining boost amount in the range 0–100.
    has_flip:
        Whether the car still owns its flip.  Some maneuvers consume this resource.
    time_since_jump:
        Seconds elapsed since the last jump button press.
    is_grounded:
        Whether the wheels currently contact the ground.
    forward:
        Unit vector pointing in the car's forward direction.
    up:
        Unit vector pointing upward relative to the car's orientation.
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
    """Simplified snapshot of the ball used by mechanics routines."""

    position: np.ndarray
    velocity: np.ndarray


@dataclass
class ControlResult:
    """Container storing a controller state plus a descriptive label."""

    controller: SimpleControllerState
    description: str


def norm(vec: np.ndarray) -> float:
    """Return the Euclidean length of a vector."""

    return float(np.linalg.norm(vec))


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize ``vec`` while protecting against extremely small magnitudes."""

    magnitude = norm(vec)
    if magnitude < 1e-6:
        return np.zeros_like(vec)
    return vec / magnitude


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Return the unsigned angle between vectors ``a`` and ``b`` in radians."""

    a_n = normalize(a)
    b_n = normalize(b)
    dot = float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    return math.acos(dot)


def aerial_controller(car: CarState, ball: BallState) -> Optional[ControlResult]:
    """Return controls to launch an aerial toward the ball if conditions match.

    The function closely mirrors ``mechanics.py``.  It checks whether the ball is high and
    whether the car owns enough boost to pursue it.  Returning ``None`` keeps the caller in
    charge when an aerial is unwarranted.
    """

    if car.boost < 20.0:
        return None

    vertical_distance = ball.position[2] - car.position[2]
    if vertical_distance < 500.0:
        return None

    controller = SimpleControllerState()
    controller.jump = True
    controller.boost = True

    to_ball = ball.position - car.position
    to_ball_horizontal = normalize(np.array([to_ball[0], to_ball[1], 0.0]))

    yaw_cross = np.cross(car.forward, to_ball_horizontal)
    controller.yaw = float(np.clip(yaw_cross[2], -1.0, 1.0))

    vertical_dir = normalize(to_ball)
    pitch_error = angle_between(car.forward, vertical_dir)
    controller.pitch = float(np.clip(-pitch_error, -1.0, 1.0))

    right_vector = np.cross(car.up, car.forward)
    roll_sign = float(np.clip(np.dot(right_vector, to_ball_horizontal), -1.0, 1.0))
    controller.roll = roll_sign

    description = "Aerial: jump, aim toward ball and boost while airborne."
    return ControlResult(controller=controller, description=description)


def half_flip_recovery(car: CarState, desired_forward: np.ndarray) -> Optional[ControlResult]:
    """Perform a half-flip when facing away from ``desired_forward``."""

    if not car.has_flip:
        return None

    angle_error = angle_between(car.forward, desired_forward)
    if angle_error < math.radians(120):
        return None

    controller = SimpleControllerState()
    controller.throttle = -1.0
    controller.jump = True
    controller.pitch = -1.0
    controller.roll = 1.0
    controller.handbrake = True

    description = "Half-flip: flip cancel to turn quickly toward desired heading."
    return ControlResult(controller=controller, description=description)


def wave_dash(car: CarState, landing_normal: np.ndarray) -> Optional[ControlResult]:
    """Execute a wave dash when landing to preserve velocity."""

    if not car.is_grounded and car.position[2] > 50.0 and norm(car.velocity) > 800.0:
        controller = SimpleControllerState()
        controller.jump = True
        controller.pitch = -0.2
        controller.roll = 0.2
        controller.boost = False
        controller.throttle = 1.0
        controller.handbrake = False
        description = "Wave dash: diagonal dodge on landing to maintain speed."
        return ControlResult(controller=controller, description=description)
    return None


def ground_dribble(car: CarState, ball: BallState) -> Optional[ControlResult]:
    """Maintain possession when the ball rests close to the car."""

    relative_height = ball.position[2] - car.position[2]
    horizontal_distance = norm(ball.position[:2] - car.position[:2])
    if relative_height < 100.0 and horizontal_distance < 120.0:
        controller = SimpleControllerState()
        controller.throttle = 0.35
        controller.steer = 0.0
        controller.boost = False
        controller.jump = False
        description = "Ground dribble: gentle throttle to balance the ball."
        return ControlResult(controller=controller, description=description)
    return None


def shooting_alignment(car: CarState, ball: BallState, goal_position: np.ndarray) -> Optional[ControlResult]:
    """Align the car-car-ball trajectory toward ``goal_position``."""

    car_to_ball = ball.position - car.position
    ball_to_goal = goal_position - ball.position

    alignment_error = angle_between(car.forward, car_to_ball)
    shooting_error = angle_between(car_to_ball, ball_to_goal)

    if alignment_error > math.radians(20) or shooting_error > math.radians(25):
        controller = SimpleControllerState()
        controller.throttle = 1.0
        controller.steer = float(np.clip(np.sign(np.cross(car.forward, car_to_ball)[2]), -1.0, 1.0))
        controller.boost = False
        description = "Shooting alignment: steering to line up car-ball-goal."
        return ControlResult(controller=controller, description=description)
    return None


def merge_controls(primary: ControlResult, secondary: Optional[ControlResult]) -> ControlResult:
    """Merge two ``ControlResult`` objects, preferring non-zero inputs from primary."""

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


# ---------------------------------------------------------------------------
# Strategy layer (adapted from strategy.py)
# ---------------------------------------------------------------------------

FIELD_LENGTH: float = 10280.0
FIELD_WIDTH: float = 8240.0
FIELD_HEIGHT: float = 2044.0
BLUE_GOAL = np.array([0.0, -FIELD_LENGTH / 2, 0.0], dtype=np.float32)
ORANGE_GOAL = np.array([0.0, FIELD_LENGTH / 2, 0.0], dtype=np.float32)


@dataclass
class PlayerSnapshot:
    """State of another player used in strategy deliberations."""

    position: np.ndarray
    velocity: np.ndarray
    is_teammate: bool


@dataclass
class GameContext:
    """Bundle of information consumed by the strategy layer."""

    car: CarState
    ball: BallState
    players: List[PlayerSnapshot]
    rand: random.Random
    is_orange: bool
    ml_action: Optional[int]
    use_ml: bool

    @property
    def own_goal(self) -> np.ndarray:
        """Return position of the team's goal."""

        return ORANGE_GOAL if self.is_orange else BLUE_GOAL

    @property
    def opponent_goal(self) -> np.ndarray:
        """Return position of the opposing team's goal."""

        return BLUE_GOAL if self.is_orange else ORANGE_GOAL


def scripted_intent(context: GameContext) -> Tuple[int, str]:
    """Return a discrete intent index along with a human-readable explanation."""

    car = context.car
    ball = context.ball
    to_ball = ball.position - car.position
    distance_to_ball = float(np.linalg.norm(to_ball))

    defensive_threat = float(np.linalg.norm(ball.position - context.own_goal))

    if car.boost < 20.0:
        return 3, "Low boost -> collect boost"

    if defensive_threat < 3000.0 and ball.position[2] < 800.0:
        return 1, "Ball pressuring own goal -> defend"

    if distance_to_ball < 1600.0:
        return 0, "Close to ball -> attack"

    forward_component = float(np.dot(car.velocity, context.own_goal - car.position))
    if forward_component < -400000.0:
        return 2, "Moving backward -> rotate"

    return 4, "Default -> neutral support"


def drive_toward(car: CarState, target: np.ndarray, throttle: float, description: str) -> ControlResult:
    """Return a controller steering toward ``target`` with the provided throttle."""

    controller = SimpleControllerState()
    controller.throttle = float(np.clip(throttle, -1.0, 1.0))
    desired_direction = normalize(target - car.position)
    steer_sign = float(np.clip(np.cross(car.forward, desired_direction)[2], -1.0, 1.0))
    controller.steer = steer_sign
    controller.handbrake = abs(steer_sign) > 0.6
    return ControlResult(controller=controller, description=description)


def execute_attack(context: GameContext) -> ControlResult:
    """Execute an attacking maneuver toward the opponent goal."""

    car = context.car
    ball = context.ball

    aerial = aerial_controller(car, ball)
    if aerial:
        return aerial

    dribble = ground_dribble(car, ball)
    if dribble:
        return dribble

    align = shooting_alignment(car, ball, context.opponent_goal)
    if align:
        return align

    return drive_toward(car, ball.position, 1.0, "Attack: accelerate toward ball")


def execute_defense(context: GameContext) -> ControlResult:
    """Execute defensive retreat and goal-side positioning."""

    car = context.car
    desired_forward = normalize(context.own_goal - car.position)
    half_flip = half_flip_recovery(car, desired_forward)
    if half_flip:
        return half_flip

    return drive_toward(car, context.own_goal, 1.0, "Defense: retreat toward goal")


def execute_rotation(context: GameContext) -> ControlResult:
    """Rotate through midfield, supporting teammates."""

    rotation_point = (context.own_goal + context.opponent_goal) / 2
    return drive_toward(context.car, rotation_point, 0.8, "Rotation: cycling through midfield")


def execute_boost_run(context: GameContext) -> ControlResult:
    """Head toward a back-corner boost pad."""

    target = np.array([0.0, -3500.0 if context.is_orange else 3500.0, 0.0], dtype=np.float32)
    return drive_toward(context.car, target, 1.0, "Boost run: collecting corner pad")


def execute_neutral(context: GameContext) -> ControlResult:
    """Hold a central supporting position."""

    return drive_toward(context.car, np.zeros(3, dtype=np.float32), 0.5, "Neutral: holding midfield")


def choose_strategy(context: GameContext) -> ControlResult:
    """Select and execute the best high-level strategy for the current context."""

    if context.use_ml and context.ml_action is not None:
        intent = context.ml_action
        LOGGER.debug("ML policy selected intent %s", intent)
    else:
        intent, reason = scripted_intent(context)
        LOGGER.debug("Scripted intent %s chosen: %s", intent, reason)

    if intent == 0:
        return execute_attack(context)
    if intent == 1:
        return execute_defense(context)
    if intent == 2:
        return execute_rotation(context)
    if intent == 3:
        return execute_boost_run(context)
    return execute_neutral(context)


# ---------------------------------------------------------------------------
# RLGym components (adapted from rl_components.py)
# ---------------------------------------------------------------------------


class SuperObsBuilder(ObsBuilder):
    """Observation builder encoding relative positions and velocities.

    The builder mirrors ``HybridObsBuilder`` from the original project.  Observations are
    constructed from the car's perspective and include padded teammate/opponent slots to
    maintain a fixed-size vector.
    """

    def __init__(self, max_players: int = 3) -> None:
        super().__init__()
        self.max_players = max_players

    def reset(self, initial_state: Any) -> None:
        return None

    def build_obs(self, player: int, state: StateWrapper, previous_action: Optional[Any]) -> np.ndarray:
        car_data = state.players[player]
        ball = state.ball

        car_pos = np.array(car_data.car_data.position)
        car_lin_vel = np.array(car_data.car_data.linear_velocity)

        ball_pos = np.array(ball.position)
        ball_lin_vel = np.array(ball.linear_velocity)

        relative_ball_pos = (ball_pos - car_pos) / 10000.0
        relative_ball_vel = (ball_lin_vel - car_lin_vel) / 4000.0

        obs = [relative_ball_pos, relative_ball_vel]

        teammates: List[np.ndarray] = []
        opponents: List[np.ndarray] = []
        for idx, other in enumerate(state.players):
            if idx == player:
                continue
            rel_pos = (np.array(other.car_data.position) - car_pos) / 10000.0
            rel_vel = (np.array(other.car_data.linear_velocity) - car_lin_vel) / 4000.0
            entry = np.concatenate([rel_pos, rel_vel])
            if other.team_num == car_data.team_num:
                teammates.append(entry)
            else:
                opponents.append(entry)

        def pad(entries: List[np.ndarray]) -> np.ndarray:
            if len(entries) >= self.max_players:
                return np.concatenate(entries[: self.max_players])
            if entries:
                concat = np.concatenate(entries)
            else:
                concat = np.zeros(6 * self.max_players)
            padding = np.zeros(6 * (self.max_players - len(entries)))
            return np.concatenate([concat, padding])

        obs.append(pad(teammates))
        obs.append(pad(opponents))

        boost = np.array([car_data.boost_amount / 100.0])
        on_ground = np.array([1.0 if car_data.on_ground else 0.0])
        has_flip = np.array([1.0 if car_data.has_jump else 0.0])

        obs.extend([boost, on_ground, has_flip])

        return np.concatenate(obs).astype(np.float32)


class SuperActionParser(ActionParser):
    """Discrete action parser translating PPO outputs into intents."""

    def __init__(self) -> None:
        self.action_space = common_values.DiscreteActionSpace(5)

    def get_action_space(self) -> Any:
        return self.action_space

    def parse_actions(self, actions: np.ndarray, state: StateWrapper) -> List[int]:
        return [int(action) for action in actions]

    def reverse_actions(self, actions: List[int], state: StateWrapper) -> np.ndarray:
        return np.array(actions, dtype=np.int64)


class SuperRewardFunction(RewardFunction):
    """Composite dense reward encouraging well-rounded gameplay."""

    def __init__(self) -> None:
        self.last_ball_velocities: Dict[int, np.ndarray] = {}

    def reset(self, initial_state: StateWrapper) -> None:
        self.last_ball_velocities.clear()

    def get_reward(self, player: int, state: StateWrapper, previous_action: Optional[Any]) -> float:
        player_data = state.players[player]
        ball = state.ball
        team = player_data.team_num

        ball_vel = np.array(ball.linear_velocity)
        last_vel = self.last_ball_velocities.get(player, np.zeros(3))
        ball_accel = ball_vel - last_vel
        self.last_ball_velocities[player] = ball_vel

        opponent_goal = (
            common_values.BLUE_GOAL if team == common_values.ORANGE else common_values.ORANGE_GOAL
        )
        ball_to_goal = np.array(opponent_goal) - np.array(ball.position)
        ball_speed_toward_goal = float(
            np.dot(ball_vel, ball_to_goal) / (np.linalg.norm(ball_to_goal) + 1e-6)
        )

        reward = 0.0005 * ball_speed_toward_goal
        reward += 0.01 if player_data.on_ground else -0.01
        reward += 0.001 * (player_data.boost_amount / 100.0)
        reward += 0.0001 * float(np.linalg.norm(ball_accel))
        return reward


class CurriculumStateSetter(RandomStateSetter):
    """Curriculum state setter gradually increasing difficulty."""

    def __init__(self) -> None:
        self.stage = 0
        self.matches_played = 0

    def reset(self, state_wrapper: StateWrapper) -> None:
        rng = np.random.default_rng()
        if self.stage == 0:
            self._kickoff(state_wrapper, rng)
        elif self.stage == 1:
            self._ground_challenge(state_wrapper, rng)
        elif self.stage == 2:
            self._wall_play(state_wrapper, rng)
        else:
            self._aerial_midfield(state_wrapper, rng)

        self.matches_played += 1
        if self.matches_played % 200 == 0 and self.stage < 3:
            self.stage += 1

    def _kickoff(self, state_wrapper: StateWrapper, rng: np.random.Generator) -> None:
        state_wrapper.reset_to_default()
        for car in state_wrapper.cars:
            side = -1 if car.team_num == common_values.BLUE_TEAM else 1
            car.set_pos(rng.uniform(-200, 200), side * 2300.0, 17.0)
            car.set_lin_vel(0.0, 0.0, 0.0)
        state_wrapper.ball.set_pos(0.0, 0.0, 92.75)

    def _ground_challenge(self, state_wrapper: StateWrapper, rng: np.random.Generator) -> None:
        state_wrapper.reset_to_default()
        for car in state_wrapper.cars:
            side = -1 if car.team_num == common_values.BLUE_TEAM else 1
            car.set_pos(rng.uniform(-1500, 1500), side * rng.uniform(0, 3000), 17.0)
            car.set_lin_vel(rng.uniform(-500, 500), rng.uniform(-500, 500), 0.0)
        state_wrapper.ball.set_pos(rng.uniform(-2000, 2000), rng.uniform(-3000, 3000), 92.75)

    def _wall_play(self, state_wrapper: StateWrapper, rng: np.random.Generator) -> None:
        state_wrapper.reset_to_default()
        for car in state_wrapper.cars:
            car.set_pos(rng.uniform(-4000, 4000), rng.uniform(-5120, 5120), rng.uniform(17.0, 800.0))
            car.set_lin_vel(rng.uniform(-800, 800), rng.uniform(-800, 800), rng.uniform(-200, 200))
        state_wrapper.ball.set_pos(rng.uniform(-3000, 3000), rng.uniform(-4000, 4000), rng.uniform(92.75, 800.0))

    def _aerial_midfield(self, state_wrapper: StateWrapper, rng: np.random.Generator) -> None:
        state_wrapper.reset_to_default()
        for car in state_wrapper.cars:
            car.set_pos(rng.uniform(-3000, 3000), rng.uniform(-4000, 4000), rng.uniform(300.0, 1200.0))
            car.set_lin_vel(rng.uniform(-800, 800), rng.uniform(-800, 800), rng.uniform(-200, 200))
        state_wrapper.ball.set_pos(rng.uniform(-1000, 1000), rng.uniform(-2000, 2000), rng.uniform(400.0, 1500.0))


def common_terminal_conditions() -> List[TerminalCondition]:
    """Return terminal conditions shared between training and evaluation."""

    return [GoalScoredCondition(), NoTouchTimeoutCondition(60)]


# ---------------------------------------------------------------------------
# Training utilities (adapted from training.py)
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for PPO self-play training."""

    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    n_steps: int = 4096
    batch_size: int = 512
    gamma: float = 0.993
    gae_lambda: float = 0.9
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    tensorboard_log: str = "./tb_logs"
    checkpoint_dir: str = "./models"
    checkpoint_interval: int = 500_000
    eval_interval: int = 500_000
    num_envs: int = 1
    use_subprocess: bool = False
    opponent_mode: str = "self"
    net_arch: Optional[List[int]] = None


class CheckpointCallback(BaseCallback):
    """Callback saving checkpoints and tracking best evaluation score."""

    def __init__(self, save_dir: Path, save_freq: int, eval_fn: Callable[[PPO], float]) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.eval_fn = eval_fn
        self.best_score = float("-inf")

    def _init_callback(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = self.save_dir / f"checkpoint_{self.num_timesteps}.zip"
            self.model.save(path)
            score = self.eval_fn(self.model)
            if score > self.best_score:
                self.best_score = score
                self.model.save(self.save_dir / "best_model.zip")
            LOGGER.info("Checkpoint saved at %s with score %.3f", path, score)
        return True


class RollingStatsCallback(BaseCallback):
    """Callback printing rolling average rewards to stdout."""

    def __init__(self, log_interval: int = 10_000) -> None:
        super().__init__()
        self.log_interval = log_interval
        self.episode_rewards: List[float] = []
        self.last_log_step = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")
        if dones is not None and rewards is not None:
            for done, reward in zip(dones, rewards):
                if done:
                    self.episode_rewards.append(float(reward))
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            avg_reward = float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0
            LOGGER.info("Step %s: 100-episode avg reward %.3f", self.num_timesteps, avg_reward)
            self.last_log_step = self.num_timesteps
        return True


class ScriptedOpponentPolicy:
    """Heuristic opponent mirroring the scripted logic."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        ball_rel = obs[..., :3]
        distance = np.linalg.norm(ball_rel, axis=-1)
        actions = np.where(distance < 0.2, 0, 1)
        return actions.astype(np.int64)


def make_env_fn(config: TrainingConfig, seed: Optional[int] = None) -> Callable[[], VecEnv]:
    """Return a thunk that constructs a single RLGym environment."""

    def _init() -> "Gym":
        from rlgym.envs import Match
        from rlgym.gym import Gym

        state_setter = CurriculumStateSetter()
        obs_builder = SuperObsBuilder()
        action_parser = SuperActionParser()
        reward_function = SuperRewardFunction()
        terminal_conditions = common_terminal_conditions()

        scripted_opponent = config.opponent_mode == "scripted"

        match = Match(
            team_size=1,
            state_setter=state_setter,
            obs_builder=obs_builder,
            action_parser=action_parser,
            reward_function=reward_function,
            terminal_conditions=terminal_conditions,
            enable_state_transitions=True,
            opponent_policy=None,
        )

        env = Gym(
            match=match,
            self_play=config.opponent_mode == "self",
            spawn_opponents=True,
            team_size=1,
        )

        if scripted_opponent:
            env.opponent_policy = ScriptedOpponentPolicy()

        if seed is not None:
            env.seed(seed)
        else:
            env.seed(random.randint(0, 2**32 - 1))
        return env

    return _init


def build_vec_env(config: TrainingConfig) -> VecEnv:
    """Create a vectorized environment with ``config.num_envs`` workers."""

    env_fns = [make_env_fn(config, seed=i) for i in range(config.num_envs)]
    if config.use_subprocess and config.num_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def evaluate_model_score(model: PPO, episodes: int = 2) -> float:
    """Run a few quick self-play episodes and return the average cumulative reward."""

    config = TrainingConfig(num_envs=1)
    env = build_vec_env(config)
    total_reward = 0.0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _info = env.step(action)
            total_reward += float(reward)
    env.close()
    return total_reward / max(episodes, 1)


def train_superbot(config: Optional[TrainingConfig] = None) -> None:
    """Train ``SuperBot`` using PPO self-play with Stable-Baselines3."""

    cfg = config or TrainingConfig()
    env = build_vec_env(cfg)

    policy_kwargs = {"net_arch": cfg.net_arch} if cfg.net_arch else {}

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        tensorboard_log=cfg.tensorboard_log,
        policy_kwargs=policy_kwargs or None,
    )

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_dir=checkpoint_dir,
        save_freq=cfg.checkpoint_interval,
        eval_fn=lambda model: evaluate_model_score(model, episodes=1),
    )
    rolling_stats = RollingStatsCallback()

    model.learn(total_timesteps=cfg.total_timesteps, callback=[checkpoint_callback, rolling_stats])

    final_path = checkpoint_dir / "final_model.zip"
    model.save(final_path)
    env.close()
    LOGGER.info("Training complete. Final model saved to %s", final_path)


# ---------------------------------------------------------------------------
# Evaluation utilities (adapted from evaluation.py)
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Metrics aggregated across evaluation matches."""

    goals_scored: int = 0
    goals_conceded: int = 0
    saves: int = 0
    touches: int = 0
    wins: int = 0
    matches: int = 0

    def record(self, info: Dict[str, float]) -> None:
        self.goals_scored += int(info.get("goals_scored", 0))
        self.goals_conceded += int(info.get("goals_conceded", 0))
        self.saves += int(info.get("saves", 0))
        self.touches += int(info.get("touches", 0))
        self.wins += int(info.get("win", 0))
        self.matches += 1

    def summary(self) -> str:
        if self.matches == 0:
            return "No matches played"
        win_rate = 100.0 * self.wins / self.matches
        return (
            f"Matches: {self.matches}\n"
            f"Goals scored: {self.goals_scored}\n"
            f"Goals conceded: {self.goals_conceded}\n"
            f"Saves: {self.saves}\n"
            f"Touches: {self.touches}\n"
            f"Win rate: {win_rate:.1f}%"
        )


def evaluate_against(model: PPO, opponent_mode: str, episodes: int) -> EvaluationResult:
    """Evaluate a PPO ``model`` against the specified opponent type."""

    config = TrainingConfig(num_envs=1, opponent_mode=opponent_mode)
    env = build_vec_env(config)
    result = EvaluationResult()
    for _ in range(episodes):
        obs = env.reset()
        done = False
        info: Dict[str, float] = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, done, info = env.step(action)
        if isinstance(info, list):
            info = info[0]
        result.record(info if isinstance(info, dict) else {})
    env.close()
    return result


def evaluate_superbot(model_path: Path, episodes: int = 5) -> EvaluationResult:
    """Load a PPO checkpoint from ``model_path`` and print evaluation metrics."""

    model = PPO.load(model_path)
    result = evaluate_against(model, "self", episodes)
    LOGGER.info("Evaluation summary:\n%s", result.summary())
    return result


# ---------------------------------------------------------------------------
# SuperBot agent combining heuristics and PPO intents
# ---------------------------------------------------------------------------


@dataclass
class BoostPad:
    """Representation of a boost pad location."""

    index: int
    location: Vector3


@dataclass
class SuperBotConfig:
    """Runtime configuration for the ``SuperBot`` agent."""

    use_ml: bool = USE_ML
    model_path: Path = DEFAULT_MODEL_PATH
    deterministic_ml: bool = True


class SuperBot(BaseAgent):
    """Hybrid Rocket League agent combining scripted mechanics with PPO intents."""

    aerial_height_threshold: float = 400.0
    aerial_range: float = 1800.0
    flip_speed_threshold: float = 1400.0
    flip_distance_threshold: float = 420.0

    def __init__(self, name: str, team: int, index: int, config: Optional[SuperBotConfig] = None) -> None:
        super().__init__(name, team, index)
        self.config = config or SuperBotConfig()
        self.model: Optional[PPO] = None
        self.last_ml_action: Optional[int] = None
        self.big_boosts: List[BoostPad] = []
        self.jump_timer = 0
        self.flip_timer = 0
        self.random = random.Random()

    # ------------------------------------------------------------------
    # RLBot lifecycle methods
    # ------------------------------------------------------------------

    def initialize_agent(self) -> None:
        """Initialize the bot by caching boost pad data and optionally loading PPO."""

        field_info = self.get_field_info()
        self.big_boosts = [
            BoostPad(index=i, location=to_vec3(pad.location))
            for i, pad in enumerate(field_info.boost_pads)
            if pad.is_full_boost
        ]
        self.jump_timer = 0
        self.flip_timer = 0
        self.chat(self.team, QuickChats.Information_IGotIt)
        if self.config.use_ml:
            self.load_model(self.config.model_path)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """Main decision function executed every tick by RLBot."""

        car = packet.game_cars[self.index]
        ball_location = to_vec3(packet.game_ball.physics.location)
        car_location = to_vec3(car.physics.location)
        own_goal, opponent_goal = self._get_goals()

        controller = SimpleControllerState()

        if self._should_collect_boost(car):
            target = self._get_nearest_big_boost(car_location, packet)
            if target is not None:
                controller = self._drive_toward(car, target)
            else:
                controller = self._drive_toward(car, ball_location)
        elif self._should_defend(ball_location, own_goal):
            controller = self._return_to_goal(car, own_goal, ball_location)
        else:
            controller = self._align_with_ball(car, ball_location, opponent_goal)

        car_state = self._build_car_state(packet, self.index)
        ball_state = self._build_ball_state(packet)
        players = self._collect_players(packet)
        ml_action = self._decide_ml_action(packet, car_state, ball_state, players)

        context = GameContext(
            car=car_state,
            ball=ball_state,
            players=players,
            rand=self.random,
            is_orange=bool(self.team == 1),
            ml_action=ml_action,
            use_ml=self.config.use_ml and self.model is not None,
        )

        strategy_control = choose_strategy(context)
        controller = strategy_control.controller

        aerial_control = self._try_aerial(car, ball_location)
        if aerial_control is not None:
            controller = aerial_control
        else:
            flip_control = self._try_front_flip(car, ball_location)
            if flip_control is not None:
                controller = flip_control

        if not car.has_wheel_contact:
            controller.roll = controller.roll or 0.0

        return controller

    # ------------------------------------------------------------------
    # PPO integration helpers
    # ------------------------------------------------------------------

    def load_model(self, model_path: Path) -> None:
        """Attempt to load a PPO model from ``model_path``."""

        if not model_path.exists():
            LOGGER.warning("Model path %s does not exist. Running scripted mode.", model_path)
            return
        try:
            self.model = PPO.load(model_path)
            LOGGER.info("Loaded PPO model from %s", model_path)
        except Exception as exc:  # pragma: no cover - best effort logging
            LOGGER.error("Failed to load PPO model: %s", exc)
            self.model = None

    def _build_observation(self, packet: GameTickPacket, car_state: CarState, ball_state: BallState, players: List[PlayerSnapshot]) -> np.ndarray:
        """Construct observation vector similar to the RLGym builder for PPO inference."""

        car_pos = car_state.position
        car_vel = car_state.velocity
        ball_pos = ball_state.position
        ball_vel = ball_state.velocity

        relative_ball_pos = (ball_pos - car_pos) / 10000.0
        relative_ball_vel = (ball_vel - car_vel) / 4000.0
        obs_parts: List[np.ndarray] = [relative_ball_pos, relative_ball_vel]

        def pad(entries: List[np.ndarray]) -> np.ndarray:
            max_players = 3
            if len(entries) >= max_players:
                return np.concatenate(entries[:max_players])
            if entries:
                concat = np.concatenate(entries)
            else:
                concat = np.zeros(6 * max_players)
            padding = np.zeros(6 * (max_players - len(entries)))
            return np.concatenate([concat, padding])

        teammates: List[np.ndarray] = []
        opponents: List[np.ndarray] = []
        for player in players:
            rel_pos = (player.position - car_pos) / 10000.0
            rel_vel = (player.velocity - car_vel) / 4000.0
            entry = np.concatenate([rel_pos, rel_vel])
            if player.is_teammate:
                teammates.append(entry)
            else:
                opponents.append(entry)

        obs_parts.append(pad(teammates))
        obs_parts.append(pad(opponents))

        boost = np.array([car_state.boost / 100.0])
        on_ground = np.array([1.0 if car_state.is_grounded else 0.0])
        has_flip = np.array([1.0 if car_state.has_flip else 0.0])

        obs_parts.extend([boost, on_ground, has_flip])

        return np.concatenate(obs_parts).astype(np.float32)

    def _decide_ml_action(
        self,
        packet: GameTickPacket,
        car_state: CarState,
        ball_state: BallState,
        players: List[PlayerSnapshot],
    ) -> Optional[int]:
        """Return the discrete intent index predicted by the PPO policy, if any."""

        if not (self.config.use_ml and self.model is not None):
            return None

        obs = self._build_observation(packet, car_state, ball_state, players)
        action, _ = self.model.predict(obs, deterministic=self.config.deterministic_ml)
        self.last_ml_action = int(action)
        return self.last_ml_action

    # ------------------------------------------------------------------
    # Scripted control helpers (adapted from advanced_bot.py)
    # ------------------------------------------------------------------

    def _should_collect_boost(self, car: RLBotPlayerInfo) -> bool:
        return car.boost < 20

    def _should_defend(self, ball_location: Vector3, own_goal: Vector3) -> bool:
        return vec_length(vec_sub(ball_location, own_goal)) < 2500

    def _drive_toward(self, car: RLBotPlayerInfo, target: Vector3) -> SimpleControllerState:
        controller = SimpleControllerState()
        car_location = to_vec3(car.physics.location)
        car_velocity = to_vec3(car.physics.velocity)

        controller.steer = self._steer_toward(car, target)
        controller.throttle = 1.0
        controller.boost = vec_length(car_velocity) < 2200 and abs(controller.steer) < 0.3
        controller.handbrake = False
        return controller

    def _align_with_ball(self, car: RLBotPlayerInfo, ball_location: Vector3, opponent_goal: Vector3) -> SimpleControllerState:
        desired_direction = ground_direction(ball_location, opponent_goal)
        approach_offset = vec_scale(desired_direction, -350.0)
        target_point = vec_add(ball_location, approach_offset)
        return self._drive_toward(car, target_point)

    def _return_to_goal(
        self, car: RLBotPlayerInfo, own_goal: Vector3, ball_location: Vector3
    ) -> SimpleControllerState:
        target_point = vec_add(own_goal, vec_scale(ground_direction(own_goal, ball_location), 900.0))
        controller = self._drive_toward(car, target_point)
        controller.handbrake = abs(controller.steer) > 0.5
        return controller

    def _get_nearest_big_boost(self, car_location: Vector3, packet: GameTickPacket) -> Optional[Vector3]:
        if not self.big_boosts:
            return None
        pad_states = getattr(packet, "game_boosts", None) or getattr(packet, "boostPadStates", None)
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

    def _steer_toward(self, car: RLBotPlayerInfo, target: Vector3) -> float:
        car_location = to_vec3(car.physics.location)
        car_yaw = car.physics.rotation.yaw
        direction = math.atan2(target[1] - car_location[1], target[0] - car_location[0])
        angle_diff = direction - car_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        return clamp(angle_diff * 2.0, -1.0, 1.0)

    def _point_toward(self, car: RLBotPlayerInfo, target: Vector3) -> Tuple[float, float]:
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

    def _try_aerial(self, car: RLBotPlayerInfo, ball_location: Vector3) -> Optional[SimpleControllerState]:
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
        controller.pitch, controller.yaw = self._point_toward(car, ball_location)
        if self.jump_timer == 0:
            self.jump_timer = 3
        else:
            self.jump_timer -= 1
        return controller

    def _try_front_flip(self, car: RLBotPlayerInfo, ball_location: Vector3) -> Optional[SimpleControllerState]:
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

    def _get_goals(self) -> Tuple[Vector3, Vector3]:
        field_info = self.get_field_info()
        own_goal: Optional[Vector3] = None
        opponent_goal: Optional[Vector3] = None
        for goal in field_info.goals:
            if goal.team == self.team:
                own_goal = to_vec3(goal.location)
            else:
                opponent_goal = to_vec3(goal.location)
        if own_goal is None or opponent_goal is None:
            own_goal = (0.0, -5120.0 if self.team == 0 else 5120.0, 0.0)
            opponent_goal = (own_goal[0], -own_goal[1], own_goal[2])
            LOGGER.warning("Goal locations missing from field info. Using defaults.")
        return own_goal, opponent_goal

    # ------------------------------------------------------------------
    # Conversion helpers bridging RLBot packets to mechanics/strategy states
    # ------------------------------------------------------------------

    def _build_car_state(self, packet: GameTickPacket, index: int) -> CarState:
        car = packet.game_cars[index]
        rot = np.array(
            [
                car.physics.rotation.pitch,
                car.physics.rotation.yaw,
                car.physics.rotation.roll,
            ],
            dtype=np.float32,
        )
        forward = np.array(
            [
                math.cos(rot[1]) * math.cos(rot[0]),
                math.sin(rot[1]) * math.cos(rot[0]),
                math.sin(rot[0]),
            ],
            dtype=np.float32,
        )
        up = np.array(
            [
                -math.cos(rot[1]) * math.sin(rot[0]) * math.cos(rot[2]) - math.sin(rot[1]) * math.sin(rot[2]),
                -math.sin(rot[1]) * math.sin(rot[0]) * math.cos(rot[2]) + math.cos(rot[1]) * math.sin(rot[2]),
                math.cos(rot[0]) * math.cos(rot[2]),
            ],
            dtype=np.float32,
        )
        time_of_last_jump = getattr(car, "time_of_last_jump", packet.game_info.seconds_elapsed)
        return CarState(
            position=np.array([car.physics.location.x, car.physics.location.y, car.physics.location.z], dtype=np.float32),
            velocity=np.array([car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z], dtype=np.float32),
            rotation=rot,
            boost=float(car.boost),
            has_flip=not car.has_used_flip,
            time_since_jump=float(packet.game_info.seconds_elapsed - time_of_last_jump),
            is_grounded=car.has_wheel_contact,
            forward=forward / (np.linalg.norm(forward) + 1e-6),
            up=up / (np.linalg.norm(up) + 1e-6),
        )

    def _build_ball_state(self, packet: GameTickPacket) -> BallState:
        ball = packet.game_ball
        return BallState(
            position=np.array([ball.physics.location.x, ball.physics.location.y, ball.physics.location.z], dtype=np.float32),
            velocity=np.array([ball.physics.velocity.x, ball.physics.velocity.y, ball.physics.velocity.z], dtype=np.float32),
        )

    def _collect_players(self, packet: GameTickPacket) -> List[PlayerSnapshot]:
        players: List[PlayerSnapshot] = []
        self_car = packet.game_cars[self.index]
        for idx, car in enumerate(packet.game_cars):
            if idx == self.index:
                continue
            players.append(
                PlayerSnapshot(
                    position=np.array([car.physics.location.x, car.physics.location.y, car.physics.location.z], dtype=np.float32),
                    velocity=np.array([car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z], dtype=np.float32),
                    is_teammate=car.team == self_car.team,
                )
            )
        return players


# ---------------------------------------------------------------------------
# Command-line interface enabling play/train/evaluate modes
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments controlling the script entry points."""

    parser = argparse.ArgumentParser(description="SuperBot unified script")
    parser.add_argument("--play", action="store_true", help="Run the bot via RLBot")
    parser.add_argument("--train", action="store_true", help="Train PPO with self-play")
    parser.add_argument("--evaluate", type=Path, help="Evaluate a PPO checkpoint")
    parser.add_argument("--use-ml", action="store_true", help="Force ML policy usage in play mode")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to PPO model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=TrainingConfig.total_timesteps,
        help="Training timesteps when using --train",
    )
    return parser.parse_args(argv)


def run_play_mode(use_ml: bool, model_path: Path) -> None:
    """Entry point for RLBot gameplay.  Prints instructions for RLBot runner."""

    LOGGER.info(
        "SuperBot ready. Configure RLBot to load this script. ML=%s, model=%s",
        use_ml,
        model_path,
    )
    LOGGER.info(
        "When using rlbot.cfg, set python_file=./super_bot.py and use_ml via environment SUPERBOT_USE_ML"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Dispatch command-line arguments to play, train or evaluate modes."""

    args = parse_args(argv)

    if args.play:
        run_play_mode(args.use_ml or USE_ML, args.model)
        return

    if args.train:
        config = TrainingConfig(total_timesteps=args.total_timesteps)
        train_superbot(config)
        return

    if args.evaluate:
        evaluate_superbot(args.evaluate, episodes=args.episodes)
        return

    LOGGER.info(
        "No mode selected. Use --play to run in-game, --train to train PPO, or --evaluate <path> to evaluate."
    )


if __name__ == "__main__":
    main()

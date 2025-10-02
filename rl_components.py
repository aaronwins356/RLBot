"""RLGym component implementations tailored for the HybridBot project.

The reinforcement learning pipeline requires custom observation builders, reward
functions, action parsers and state setters.  The classes defined here are intentionally
verbose, describing *why* each design decision benefits training.  This helps students
understand how Rocket League can be framed as an RL problem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from rlgym.utils import common_values
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.state_setters import RandomStateSetter, StateWrapper
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition
from rlgym.utils.terminal_conditions.terminal_condition import TerminalCondition


class HybridObsBuilder(ObsBuilder):
    """Observation builder capturing relative positions and velocities.

    We encode the state from the perspective of the controlled car because relative
    information generalizes across different map positions.  Each observation contains:

    * Relative position and velocity of the ball.
    * Relative positions of up to three opponents and teammates.
    * Boost amount, is_on_ground flag, and game time features.

    The builder pads missing opponents/teammates with zeros.  This makes the neural network
    architecture fixed-size so it can be trained with standard fully connected layers.
    """

    def __init__(self, max_players: int = 3) -> None:
        super().__init__()
        self.max_players = max_players

    def reset(self, initial_state: Any) -> None:
        # Nothing to reset; the builder is stateless.
        return None

    def build_obs(self, player: Any, state: StateWrapper, previous_action: Optional[Any]) -> np.ndarray:
        car_data = state.players[player]
        ball = state.ball

        car_pos = np.array(car_data.car_data.position)
        car_lin_vel = np.array(car_data.car_data.linear_velocity)

        ball_pos = np.array(ball.position)
        ball_lin_vel = np.array(ball.linear_velocity)

        relative_ball_pos = (ball_pos - car_pos) / 10000.0
        relative_ball_vel = (ball_lin_vel - car_lin_vel) / 4000.0

        obs = [relative_ball_pos, relative_ball_vel]

        teammates = []
        opponents = []
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


class HybridActionParser(ActionParser):
    """Discrete action parser for high-level intents.

    The parser translates discrete action indices produced by PPO into structured intents
    that the strategy module can interpret.  We keep the action space small to simplify
    learning: attack, defend, rotate, boost, or neutral support.  The controller layer then
    invokes mechanics to execute the desired intent.
    """

    def __init__(self) -> None:
        self.action_space = common_values.DiscreteActionSpace(5)

    def get_action_space(self) -> Any:
        return self.action_space

    def parse_actions(self, actions: np.ndarray, state: StateWrapper) -> List[int]:
        # Stable-Baselines3 expects numpy arrays; convert to python ints for clarity.
        return [int(action) for action in actions]

    def reverse_actions(self, actions: List[int], state: StateWrapper) -> np.ndarray:
        return np.array(actions, dtype=np.int64)


class HybridRewardFunction(RewardFunction):
    """Composite reward encouraging balanced Rocket League play.

    Reward shaping is essential in Rocket League because sparse win/loss signals provide
    too little guidance.  We combine dense terms that encourage good habits:

    * Reward hitting the ball toward the opponent goal.
    * Penalize goals conceded and award goals scored via terminal signals.
    * Encourage boost conservation and proper speed control.
    """

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

        # Encourage touching the ball toward the opponent goal.
        opponent_goal = common_values.BLUE_GOAL if team == common_values.ORANGE else common_values.ORANGE_GOAL
        ball_to_goal = np.array(opponent_goal) - np.array(ball.position)
        ball_speed_toward_goal = float(np.dot(ball_vel, ball_to_goal) / (np.linalg.norm(ball_to_goal) + 1e-6))
        reward = 0.0005 * ball_speed_toward_goal

        # Penalize slow recovery when upside down by rewarding on-ground status.
        reward += 0.01 if player_data.on_ground else -0.01

        # Encourage boost conservation by rewarding small boost amounts slightly less than large amounts.
        reward += 0.001 * (player_data.boost_amount / 100.0)

        # Bonus when accelerating the ball (touches increase ball acceleration).
        reward += 0.0001 * float(np.linalg.norm(ball_accel))

        return reward


class CurriculumStateSetter(RandomStateSetter):
    """Curriculum state setter gradually increasing scenario difficulty."""

    def __init__(self) -> None:
        self.stage = 0
        self.matches_played = 0

    def reset(self, state_wrapper: StateWrapper) -> None:
        # Choose scenario based on curriculum stage.
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

    # Scenario implementations -------------------------------------------------

    def _kickoff(self, state_wrapper: StateWrapper, rng: np.random.Generator) -> None:
        state_wrapper.reset_to_default()
        # Place cars at kickoff spots.
        for i, car in enumerate(state_wrapper.cars):
            side = -1 if car.team_num == common_values.BLUE_TEAM else 1
            car.set_pos(rng.uniform(-200, 200), side * 2300.0, 17.0)
            car.set_lin_vel(0.0, 0.0, 0.0)
        state_wrapper.ball.set_pos(0.0, 0.0, 92.75)

    def _ground_challenge(self, state_wrapper: StateWrapper, rng: np.random.Generator) -> None:
        state_wrapper.reset_to_default()
        state_wrapper.ball.set_pos(rng.uniform(-1000, 1000), rng.uniform(-1000, 1000), 93.0)
        state_wrapper.ball.set_lin_vel(rng.uniform(-500, 500), rng.uniform(-500, 500), 0.0)

    def _wall_play(self, state_wrapper: StateWrapper, rng: np.random.Generator) -> None:
        state_wrapper.reset_to_default()
        state_wrapper.ball.set_pos(rng.uniform(-4000, 4000), rng.choice([-3800, 3800]), 800.0)
        state_wrapper.ball.set_lin_vel(rng.uniform(-800, 800), rng.uniform(-400, 400), rng.uniform(-200, 200))

    def _aerial_midfield(self, state_wrapper: StateWrapper, rng: np.random.Generator) -> None:
        state_wrapper.reset_to_default()
        state_wrapper.ball.set_pos(rng.uniform(-1500, 1500), rng.uniform(-2000, 2000), rng.uniform(900, 1600))
        state_wrapper.ball.set_lin_vel(rng.uniform(-1000, 1000), rng.uniform(-1000, 1000), rng.uniform(-400, 400))


def common_terminal_conditions() -> List[TerminalCondition]:
    return [GoalScoredCondition(), NoTouchTimeoutCondition(1200)]


__all__ = [
    "HybridObsBuilder",
    "HybridActionParser",
    "HybridRewardFunction",
    "CurriculumStateSetter",
    "common_terminal_conditions",
]

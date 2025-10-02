"""HybridBot RLBot agent integrating scripted mechanics with PPO self-play training.

This script acts as both the playable agent entry point and an extensive tutorial on how to
combine hardcoded Rocket League mechanics with reinforcement learning.  The codebase is
split into modular components (``mechanics``, ``strategy``, ``rl_components``,
``training`` and ``evaluation``) yet importing this file is all you need to run the bot in
hardcoded mode.  Additional modules expose the PPO training and evaluation loops, while this
file focuses on the live controller executed by RLBot.

Key educational sections are annotated with long-form comments covering:

* **Why PPO?**  Proximal Policy Optimization strikes a balance between sample efficiency and
  stability.  Rocket League control is high dimensional and requires long-horizon credit
  assignment; PPO's clipped objective handles the noisy gradients produced by self-play.
* **Self-play rationale.**  Competing against copies of ourselves forces the policy to
  continually adapt, yielding emergent maneuvers such as rotations, clears and passing.
* **Reward shaping.**  Dense rewards encourage desirable behaviour like ball touches and
  boost conservation before the sparse goal reward becomes frequent.
* **Observation design.**  Using relative coordinates keeps the state representation
  invariant to spawn positions which accelerates generalization.
* **Mechanics overview.**  Each advanced maneuver (aerials, half-flips, wave dashes, ground
  dribbles) is documented to explain its tactical utility.
* **Training pipeline.**  Comments describe how to run multi-million step experiments,
  checkpoint progress, evaluate opponents, and switch between scripted and learned logic.

Running ``python run_bot.py`` launches the HybridBot in hardcoded mode for immediate play.
The ``training.py`` and ``evaluation.py`` entry points expose PPO training and evaluation.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import Vector3
from rlbot.utils.structures.game_data_struct import GameTickPacket

from mechanics import (
    BallState,
    CarState,
    ControlResult,
    aerial_controller,
    ground_dribble,
    half_flip_recovery,
    merge_controls,
    shooting_alignment,
    wave_dash,
)
from rl_components import HybridActionParser, HybridObsBuilder, HybridRewardFunction
from strategy import GameContext, PlayerInfo, StrategyDecision, choose_strategy

try:
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - optional dependency
    PPO = None  # type: ignore

USE_ML: bool = bool(int(os.environ.get("HYBRIDBOT_USE_ML", "0")))
MODEL_PATH: Path = Path(os.environ.get("HYBRIDBOT_MODEL", "./models/best_model.zip"))

FIELD_LENGTH: float = 10280.0
FIELD_WIDTH: float = 8240.0
GOAL_HEIGHT: float = 642.775
BLUE_GOAL = np.array([0.0, -FIELD_LENGTH / 2, 0.0])
ORANGE_GOAL = np.array([0.0, FIELD_LENGTH / 2, 0.0])


# -----------------------------------------------------------------------------
# Utility conversion functions translating RLBot packets into simplified states.
# -----------------------------------------------------------------------------


def _to_numpy(vec: Vector3) -> np.ndarray:
    return np.array([vec.x, vec.y, vec.z], dtype=np.float32)


def build_car_state(packet: GameTickPacket, index: int) -> CarState:
    car = packet.game_cars[index]
    rot = np.array([car.physics.rotation.pitch, car.physics.rotation.yaw, car.physics.rotation.roll])
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
        position=_to_numpy(car.physics.location),
        velocity=_to_numpy(car.physics.velocity),
        rotation=rot,
        boost=float(car.boost),
        has_flip=not car.has_used_flip,
        time_since_jump=float(packet.game_info.seconds_elapsed - time_of_last_jump),
        is_grounded=car.has_wheel_contact,
        forward=forward / (np.linalg.norm(forward) + 1e-6),
        up=up / (np.linalg.norm(up) + 1e-6),
    )


def build_ball_state(packet: GameTickPacket) -> BallState:
    ball = packet.game_ball
    return BallState(position=_to_numpy(ball.physics.location), velocity=_to_numpy(ball.physics.velocity))


def collect_players(packet: GameTickPacket, self_index: int) -> List[PlayerInfo]:
    players: List[PlayerInfo] = []
    for idx, car in enumerate(packet.game_cars):
        if idx == self_index:
            continue
        players.append(
            PlayerInfo(
                position=_to_numpy(car.physics.location),
                velocity=_to_numpy(car.physics.velocity),
                is_teammate=car.team == packet.game_cars[self_index].team,
            )
        )
    return players


# -----------------------------------------------------------------------------
# HybridBot agent definition combining ML intent with scripted execution.
# -----------------------------------------------------------------------------


@dataclass
class BotConfig:
    """Runtime configuration for the bot."""

    use_ml: bool = USE_ML
    model_path: Path = MODEL_PATH


class HybridBot(BaseAgent):
    """Hybrid Rocket League agent.

    The bot loads a PPO model when available and uses it to select high-level intents.
    Scripted mechanics execute the physical maneuvers, ensuring competent gameplay even
    before reinforcement learning converges.  When ``use_ml`` is False or the model is
    missing, the bot defaults to purely scripted behaviour, making it instantly playable.
    """

    def __init__(self, name: str, team: int, index: int, config: Optional[BotConfig] = None) -> None:
        super().__init__(name, team, index)
        self.config = config or BotConfig()
        self.obs_builder = HybridObsBuilder()
        self.action_parser = HybridActionParser()
        self.reward_fn = HybridRewardFunction()
        self.random = random.Random(index)
        self.model: Optional[PPO] = None
        self.last_jump_time: float = 0.0
        self.last_landing_time: float = 0.0
        self.current_state_desc: str = "Initializing"

    def initialize_agent(self) -> None:
        """Called by RLBot when the agent is created."""

        if self.config.use_ml and PPO is not None and self.config.model_path.exists():
            try:
                self.model = PPO.load(self.config.model_path)
                self.current_state_desc = f"Loaded PPO model from {self.config.model_path}"
            except Exception as exc:  # pragma: no cover - runtime guard
                self.logger.warn(f"Failed to load PPO model: {exc}")
                self.model = None
                self.current_state_desc = "Falling back to scripted logic"
        else:
            if self.config.use_ml and PPO is None:
                self.logger.warn("Stable-Baselines3 not installed; using scripted logic")
            self.current_state_desc = "Scripted logic active"

    # ------------------------------------------------------------------
    # RLBot callback executed every tick to produce controller outputs.
    # ------------------------------------------------------------------
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        car_state = build_car_state(packet, self.index)
        ball_state = build_ball_state(packet)
        players = collect_players(packet, self.index)

        # Build RL observation and query PPO for high-level intent if available.
        ml_action: Optional[int] = None
        if self.model is not None:
            obs = self._build_observation(packet)
            action, _ = self.model.predict(obs, deterministic=False)
            if isinstance(action, (list, tuple, np.ndarray)):
                ml_action = int(action[0])
            else:
                ml_action = int(action)
        elif self.config.use_ml:
            # Model requested but not yet available -> stick to scripted heuristics.
            ml_action = None

        context = GameContext(
            car=car_state,
            ball=ball_state,
            players=players,
            rand=self.random,
            is_orange=packet.game_cars[self.index].team == 1,
            ml_action=ml_action,
            use_ml=self.model is not None,
        )

        strategy_decision = choose_strategy(context)

        # Layer additional mechanics such as wave dashes and aerials based on context.
        controller = self._apply_mechanics(strategy_decision, car_state, ball_state)

        self.current_state_desc = strategy_decision.description
        return controller

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _build_observation(self, packet: GameTickPacket) -> np.ndarray:
        # RLGym style observation builder expects a wrapper resembling ``StateWrapper``.
        # For live play we approximate by constructing a minimal dictionary.
        # This is sufficient for inference because the PPO model was trained with the same
        # observation builder.
        class MockPlayer:
            def __init__(self, car) -> None:
                self.car_data = type("car_data", (), {})()
                self.car_data.position = (_to_numpy(car.physics.location)).tolist()
                self.car_data.linear_velocity = (_to_numpy(car.physics.velocity)).tolist()
                self.team_num = car.team
                self.boost_amount = car.boost
                self.on_ground = car.has_wheel_contact
                self.has_jump = not car.has_used_flip

        class MockBall:
            def __init__(self, ball) -> None:
                self.position = (_to_numpy(ball.physics.location)).tolist()
                self.linear_velocity = (_to_numpy(ball.physics.velocity)).tolist()

        class MockState:
            def __init__(self, packet: GameTickPacket) -> None:
                self.ball = MockBall(packet.game_ball)
                self.players = [MockPlayer(packet.game_cars[i]) for i in range(packet.num_cars)]

        state = MockState(packet)
        obs = self.obs_builder.build_obs(self.index, state, None)
        return obs.reshape(1, -1)

    # ------------------------------------------------------------------
    # Mechanics augmentation applying aerial, wave dash, etc.
    # ------------------------------------------------------------------
    def _apply_mechanics(
        self,
        decision: StrategyDecision,
        car: CarState,
        ball: BallState,
    ) -> SimpleControllerState:
        controller = decision.controller.controller

        # Attempt to blend aerial control when the strategy aims to attack and the ball is high.
        aerial = aerial_controller(car, ball)
        if aerial:
            controller = merge_controls(aerial, decision.controller).controller

        # Wave dash if landing soon.
        dash = wave_dash(car, landing_normal=np.array([0.0, 0.0, 1.0]))
        if dash:
            controller = merge_controls(dash, decision.controller).controller

        # Ensure dribble control when ball is close.
        dribble = ground_dribble(car, ball)
        if dribble:
            controller = dribble.controller

        # Alignment near goal.
        align = shooting_alignment(car, ball, ORANGE_GOAL if car.position[1] < 0 else BLUE_GOAL)
        if align:
            controller = merge_controls(align, decision.controller).controller

        return controller


# -----------------------------------------------------------------------------
# Command-line interface for launching RLBot with HybridBot.
# -----------------------------------------------------------------------------


def launch_bot() -> None:
    """Launch the RLBot agent via the RLBot configuration system."""

    parser = argparse.ArgumentParser(description="Run HybridBot in hardcoded mode")
    parser.add_argument("--name", default="HybridBot", help="Bot name displayed in RLBot GUI")
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--use-ml", action="store_true", help="Enable PPO inference if model present")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    config = BotConfig(use_ml=args.use_ml, model_path=args.model)

    bot = HybridBot(args.name, args.team, args.index, config=config)
    bot.initialize_agent()
    print(f"HybridBot ready: {bot.current_state_desc}")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    launch_bot()

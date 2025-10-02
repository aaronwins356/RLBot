"""SuperBot RLBot agent entry point."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from boost_pad_tracker import BoostPadTracker
from rl_components import PolicyFactory, PolicyManager, default_dqn_factory
from strategy import DiamondStrategy, Intent, build_context, rl_action_to_intent
from vec import Vec3

LOG_DIR = Path("records")
MODEL_PATH = Path("models/dqn_final.pt")


@dataclass
class TransitionRecord:
    state: List[float]
    action: int
    reward: float


class SuperBot(BaseAgent):
    """Diamond-level baseline with reinforcement learning integration."""

    def __init__(self, name: str, team: int, index: int) -> None:
        super().__init__(name, team, index)
        self.strategy = DiamondStrategy()
        self.policy_factory: PolicyFactory = default_dqn_factory()
        self.policy: Optional[PolicyManager] = None
        self.boost_tracker = BoostPadTracker()
        self.previous_ball_position = Vec3(0.0, 0.0, 0.0)
        self.transition_log: List[TransitionRecord] = []
        self.match_start_time: float = time.time()
        self.random = random.Random()
        self.epsilon_fallback = 0.2  # 20% scripted behaviour

    # ------------------------------------------------------------------
    # RLBot entrypoints
    # ------------------------------------------------------------------

    def initialize_agent(self) -> None:  # type: ignore[override]
        field_info = self.get_field_info()
        self.boost_tracker.initialize(field_info)
        self._maybe_load_model()
        LOG_DIR.mkdir(exist_ok=True)

    def _maybe_load_model(self) -> None:
        if MODEL_PATH.exists():
            dummy_state = np.zeros(self.state_dim(), dtype=np.float32)
            self.policy = self.policy_factory.build(len(dummy_state), len(Intent))
            self.policy.load(MODEL_PATH)
            self.logger.info("Loaded policy from %s", MODEL_PATH)
        else:
            self.logger.info("No model found at %s. Running scripted baseline only.", MODEL_PATH)

    def state_dim(self) -> int:
        return len(self._encode_state_from_arrays(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0))

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:  # type: ignore[override]
        if packet.game_info.paused:
            return SimpleControllerState()

        if self.boost_tracker.pads:
            self.boost_tracker.update(packet)

        state = self.encode_state(packet)
        rl_action: Optional[int] = None
        use_policy = self.policy is not None and self.random.random() > self.epsilon_fallback
        if use_policy and self.policy is not None:
            rl_action = self.policy.select_action(state, epsilon=0.0)

        context = build_context(packet, self.index, rl_action)
        intent = rl_action_to_intent(rl_action)
        if intent is None:
            intent = self.strategy.select_intent(context)
        output = self.strategy.execute(context, intent)

        self._log_transition(state, rl_action if rl_action is not None else int(intent), packet)

        if packet.game_info.is_match_ended:
            self._flush_records()

        return output.controls

    # ------------------------------------------------------------------
    # State encoding and rewards
    # ------------------------------------------------------------------

    def encode_state(self, packet: GameTickPacket) -> np.ndarray:
        me = packet.game_cars[self.index]
        ball = packet.game_ball
        my_pos = Vec3.from_iterable((me.physics.location.x, me.physics.location.y, me.physics.location.z))
        my_vel = Vec3.from_iterable((me.physics.velocity.x, me.physics.velocity.y, me.physics.velocity.z))
        ball_pos = Vec3.from_iterable((ball.physics.location.x, ball.physics.location.y, ball.physics.location.z))
        ball_vel = Vec3.from_iterable((ball.physics.velocity.x, ball.physics.velocity.y, ball.physics.velocity.z))
        own_goal = Vec3(0.0, -5120.0 if self.team == 0 else 5120.0, 0.0)
        opp_goal = Vec3(0.0, 5120.0 if self.team == 0 else -5120.0, 0.0)

        teammates = [packet.game_cars[i] for i in range(packet.num_cars) if packet.game_cars[i].team == me.team and i != self.index]
        opponents = [packet.game_cars[i] for i in range(packet.num_cars) if packet.game_cars[i].team != me.team]

        features = self._encode_state_from_arrays(
            my_pos.to_list(),
            my_vel.to_list(),
            ball_pos.to_list(),
            ball_vel.to_list(),
            float(me.boost) / 100.0,
            my_pos.distance(opp_goal) / 10000.0,
            ball_pos.distance(opp_goal) / 10000.0,
        )

        for collection in (teammates, opponents):
            for player in collection[:3]:
                pos = Vec3.from_iterable((player.physics.location.x, player.physics.location.y, player.physics.location.z))
                vel = Vec3.from_iterable((player.physics.velocity.x, player.physics.velocity.y, player.physics.velocity.z))
                features.extend([(pos - my_pos).x / 5000.0, (pos - my_pos).y / 5000.0, (pos - my_pos).z / 2000.0])
                features.extend([(vel - my_vel).x / 3000.0, (vel - my_vel).y / 3000.0, (vel - my_vel).z / 3000.0])
            if len(collection) < 3:
                features.extend([0.0] * (6 * (3 - len(collection))))
        return np.asarray(features, dtype=np.float32)

    def _encode_state_from_arrays(
        self,
        my_pos: Sequence[float],
        my_vel: Sequence[float],
        ball_pos: Sequence[float],
        ball_vel: Sequence[float],
        boost: float,
        my_goal_distance: float,
        ball_goal_distance: float,
    ) -> List[float]:
        return [
            my_pos[0] / 5000.0,
            my_pos[1] / 5000.0,
            my_pos[2] / 2000.0,
            my_vel[0] / 3000.0,
            my_vel[1] / 3000.0,
            my_vel[2] / 3000.0,
            ball_pos[0] / 5000.0,
            ball_pos[1] / 5000.0,
            ball_pos[2] / 2000.0,
            ball_vel[0] / 4000.0,
            ball_vel[1] / 4000.0,
            ball_vel[2] / 4000.0,
            boost,
            my_goal_distance,
            ball_goal_distance,
        ]

    def _log_transition(self, state: np.ndarray, action: int, packet: GameTickPacket) -> None:
        reward = self._compute_reward(packet)
        self.transition_log.append(TransitionRecord(state=state.tolist(), action=action, reward=reward))

    def _compute_reward(self, packet: GameTickPacket) -> float:
        ball = packet.game_ball
        ball_pos = Vec3.from_iterable((ball.physics.location.x, ball.physics.location.y, ball.physics.location.z))
        opp_goal = Vec3(0.0, 5120.0 if self.team == 0 else -5120.0, 0.0)
        previous_distance = self.previous_ball_position.distance(opp_goal)
        current_distance = ball_pos.distance(opp_goal)
        reward = (previous_distance - current_distance) * 0.001
        self.previous_ball_position = ball_pos
        return reward

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------

    def _flush_records(self) -> None:
        if not self.transition_log:
            return
        timestamp = int(self.match_start_time)
        path = LOG_DIR / f"match_{timestamp}_{self.index}.json"
        data: List[Dict[str, float]] = [
            {"state": record.state, "action": record.action, "reward": record.reward}
            for record in self.transition_log
        ]
        path.write_text(json.dumps(data))
        self.logger.info("Saved %s transitions to %s", len(self.transition_log), path)
        self.transition_log.clear()
        self.match_start_time = time.time()

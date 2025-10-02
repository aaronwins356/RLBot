"""RLBot agent that combines rule-based heuristics with reinforcement learning."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from .base_bot import BaseBotStrategy
from .policy_network import PolicyNetwork
from .replay_buffer import ReplayBuffer, Transition
from .reward_functions import RewardContext, compute_reward
from .state_representation import STATE_DIMENSION, build_state
from .trainer import ExperienceLogger, Trainer, TrainerConfig

ACTION_TEMPLATES: List[dict] = [
    {"throttle": 1.0, "steer": 0.0, "boost": False, "jump": False, "handbrake": False},
    {"throttle": 1.0, "steer": 0.5, "boost": False, "jump": False, "handbrake": False},
    {"throttle": 1.0, "steer": -0.5, "boost": False, "jump": False, "handbrake": False},
    {"throttle": 1.0, "steer": 1.0, "boost": False, "jump": False, "handbrake": True},
    {"throttle": 1.0, "steer": -1.0, "boost": False, "jump": False, "handbrake": True},
    {"throttle": 0.5, "steer": 0.0, "boost": False, "jump": False, "handbrake": False},
    {"throttle": 0.2, "steer": 0.0, "boost": False, "jump": False, "handbrake": False},
    {"throttle": 1.0, "steer": 0.0, "boost": True, "jump": False, "handbrake": False},
    {"throttle": 1.0, "steer": 0.5, "boost": True, "jump": False, "handbrake": False},
    {"throttle": 1.0, "steer": -0.5, "boost": True, "jump": False, "handbrake": False},
    {"throttle": 0.8, "steer": 0.0, "boost": False, "jump": True, "handbrake": False},
    {"throttle": 1.0, "steer": 0.0, "boost": True, "jump": True, "handbrake": False},
    {"throttle": 0.0, "steer": 0.0, "boost": False, "jump": True, "handbrake": False},
    {"throttle": 0.8, "steer": 0.0, "boost": False, "jump": False, "handbrake": True},
    {"throttle": 0.6, "steer": 0.7, "boost": False, "jump": False, "handbrake": False},
    {"throttle": 0.6, "steer": -0.7, "boost": False, "jump": False, "handbrake": False},
]

ACTION_SPACE_SIZE = len(ACTION_TEMPLATES)


class SuperBot(BaseAgent):
    """Self-play reinforcement learning agent."""

    def initialize_agent(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_strategy = BaseBotStrategy(self)
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.reward_context = RewardContext()
        self.previous_state: Optional[np.ndarray] = None
        self.previous_action: Optional[int] = None
        self.match_guid: Optional[str] = None
        self.epsilon = 1.0
        self.min_epsilon = 0.05
        self.epsilon_decay = 5e-5
        self.last_training_time = time.time()

        self.weights_path = (Path(__file__).resolve().parent / "checkpoints/policy.pt").resolve()
        self.replay_log_dir = (Path(__file__).resolve().parent / "replays").resolve()

        self.policy = PolicyNetwork(STATE_DIMENSION, ACTION_SPACE_SIZE)
        self.target = PolicyNetwork(STATE_DIMENSION, ACTION_SPACE_SIZE)
        config = TrainerConfig(device=str(self.device), checkpoint_path=self.weights_path)
        self.experience_logger = ExperienceLogger(self.replay_log_dir, self.name)
        self.trainer = Trainer(self.policy, self.target, self.replay_buffer, config, self.experience_logger)
        self.trainer.load_checkpoint()

        self.set_game_state(None)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        if self.match_guid is None:
            guid = packet.game_info.match_guid
            if not guid:
                guid = f"match-{int(time.time())}"
            self.match_guid = guid
            self.reward_context.last_blue_score = packet.teams[0].score
            self.reward_context.last_orange_score = packet.teams[1].score
            self.reward_context.last_boost = packet.game_cars[self.index].boost
            self.reward_context.last_time = packet.game_info.seconds_elapsed

        state_vector = build_state(self, packet).values
        controller_state: SimpleControllerState

        use_policy = len(self.replay_buffer) >= self.trainer.config.min_buffer_size / 2
        action_index: int

        if use_policy:
            state_tensor = torch.tensor(state_vector, dtype=torch.float32, device=self.device)
            action_index = self.policy.predict_action(state_tensor, self.epsilon)
            controller_state = self._action_to_controller(action_index)
            self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
        else:
            controller_state = self.base_strategy.get_controls(packet)
            action_index = self._approximate_action(controller_state)

        if self.previous_state is not None and self.previous_action is not None:
            reward, done = compute_reward(self, packet, self.reward_context)
            transition = Transition(
                state=self.previous_state,
                action=self.previous_action,
                reward=reward,
                next_state=state_vector,
                done=done,
            )
            self.replay_buffer.add(
                transition.state,
                transition.action,
                transition.reward,
                transition.next_state,
                transition.done,
            )
            self.trainer.log_transition(self.match_guid, transition)
            loss = self.trainer.maybe_train()
            if loss is not None and time.time() - self.last_training_time > 1.0:
                self.logger.info(f"Training loss: {loss:.4f}")
                self.last_training_time = time.time()
            if done:
                self._reset_episode(packet)
        else:
            self.reward_context.last_blue_score = packet.teams[0].score
            self.reward_context.last_orange_score = packet.teams[1].score
            self.reward_context.last_boost = packet.game_cars[self.index].boost
            self.reward_context.last_time = packet.game_info.seconds_elapsed

        self.previous_state = state_vector
        self.previous_action = action_index

        return controller_state

    def _action_to_controller(self, action_index: int) -> SimpleControllerState:
        template = ACTION_TEMPLATES[action_index]
        controller = SimpleControllerState()
        controller.throttle = float(np.clip(template["throttle"], 0.0, 1.0))
        controller.steer = float(np.clip(template["steer"], -1.0, 1.0))
        controller.boost = bool(template["boost"])
        controller.jump = bool(template["jump"])
        controller.handbrake = bool(template["handbrake"])
        controller.pitch = 0.0
        controller.yaw = controller.steer
        controller.roll = 0.0
        return controller

    def _approximate_action(self, controller: SimpleControllerState) -> int:
        best_index = 0
        best_distance = math.inf
        for idx, template in enumerate(ACTION_TEMPLATES):
            distance = 0.0
            distance += (template["throttle"] - float(controller.throttle)) ** 2
            distance += (template["steer"] - float(controller.steer)) ** 2
            distance += (float(template["boost"]) - float(controller.boost)) ** 2
            distance += (float(template["jump"]) - float(controller.jump)) ** 2
            distance += (float(template["handbrake"]) - float(controller.handbrake)) ** 2
            if distance < best_distance:
                best_distance = distance
                best_index = idx
        return best_index

    def _reset_episode(self, packet: GameTickPacket) -> None:
        self.previous_state = None
        self.previous_action = None
        self.reward_context = RewardContext(
            last_blue_score=packet.teams[0].score,
            last_orange_score=packet.teams[1].score,
            last_boost=packet.game_cars[self.index].boost,
            last_time=packet.game_info.seconds_elapsed,
        )
        self.match_guid = packet.game_info.match_guid or self.match_guid
        self.trainer.save_checkpoint()


__all__ = ["SuperBot", "ACTION_SPACE_SIZE"]

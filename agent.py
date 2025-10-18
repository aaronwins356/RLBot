from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from discrete_policy import DiscreteFF
from your_act import YourActionParser
from your_obs import OBS_SIZE


# Hidden-layer sizes for the discrete feed-forward policy network.  The network
# mirrors the architecture used during PPO training and can be tweaked to trade
# latency for policy quality.
POLICY_LAYER_SIZES = [256, 256, 128]


class Agent:
    def __init__(self, policy_path: Optional[Path] = None) -> None:
        self.action_parser = YourActionParser()
        self.num_actions = len(self.action_parser.lookup_table)
        cur_dir = Path(__file__).resolve().parent

        device = torch.device("cpu")
        self.policy = DiscreteFF(OBS_SIZE, self.num_actions, POLICY_LAYER_SIZES, device)
        checkpoint = policy_path or cur_dir / "PPO_POLICY.pt"
        self._weights_loaded = self._load_weights(checkpoint, device)
        self._fallback = HeuristicPilot(self.action_parser)
        torch.set_num_threads(1)

    def act(self, obs: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        if self._weights_loaded:
            self.action_parser.maybe_supervise(context)

            with torch.no_grad():
                action_idx, _ = self.policy.get_action(obs, deterministic=True)

            action = np.asarray(self.action_parser.parse_actions([int(action_idx)], context))
        else:
            action = self._fallback.act(context)

        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]

        if action.ndim != 1:
            raise ValueError(f"Invalid action returned from parser: shape={action.shape}")

        return action

    def _load_weights(self, path: Path, device: torch.device) -> bool:
        if not path.exists():
            print(
                "[agent] No PPO checkpoint found at"
                f" {path.name}. Falling back to heuristic controls until training is complete."
            )
            return False

        try:
            state_dict = torch.load(path, map_location=device)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[agent] Failed to load PPO weights from {path}: {exc}. Using heuristic fallback.")
            return False

        self.policy.load_state_dict(state_dict)
        return True


class HeuristicPilot:
    """Light-weight heuristics providing safe controls when no policy is available."""

    def __init__(self, action_parser: YourActionParser) -> None:
        self._parser = action_parser
        self._name_to_index = action_parser.name_to_index
        self._neutral_idx = self._name_to_index.get("neutral", action_parser.cancel_index)

    def act(self, context: Optional[Dict[str, Any]]) -> np.ndarray:
        action_index = self._choose_action_index(context)
        return np.asarray(self._parser.parse_actions([action_index], context))

    # ------------------------------------------------------------------
    # Heuristic selection

    def _choose_action_index(self, context: Optional[Dict[str, Any]]) -> int:
        if not context:
            return self._neutral_idx

        state = context.get("state")
        player = context.get("player")
        if state is None or player is None:
            return self._neutral_idx

        ball = state.ball
        car = player.car_data
        rel = ball.position - car.position
        flat_dist = float(np.linalg.norm(rel[:2]))
        forward = car.forward()
        forward_flat = forward[:2]
        forward_flat /= np.linalg.norm(forward_flat) + 1e-6
        target_flat = rel[:2]
        target_flat /= np.linalg.norm(target_flat) + 1e-6

        facing = float(np.clip(np.dot(forward_flat, target_flat), -1.0, 1.0))
        cross_z = forward_flat[0] * target_flat[1] - forward_flat[1] * target_flat[0]

        if self._is_kickoff(state):
            return self._name_to_index.get("speed_flip_kickoff", self._neutral_idx)

        if not player.on_ground and np.linalg.norm(car.linear_velocity) > 400:
            return self._name_to_index.get("aerial_recovery", self._neutral_idx)

        if player.on_ground:
            if ball.position[2] > 1500 and player.boost_amount > 0.4:
                return self._name_to_index.get("fast_aerial", self._neutral_idx)

            if facing < -0.3 and flat_dist > 1200:
                return self._name_to_index.get("half_flip", self._neutral_idx)

            if flat_dist < 650 and ball.position[2] < 300:
                return self._name_to_index.get("dribble_carry", self._neutral_idx)

            if flat_dist < 900 and ball.position[2] < 400 and facing > 0.7:
                return self._name_to_index.get("power_shot", self._neutral_idx)

        if facing > 0.9:
            if flat_dist > 2000 and player.boost_amount > 0.3:
                return self._name_to_index.get("boost_forward", self._neutral_idx)
            return self._name_to_index.get("drive_forward", self._neutral_idx)

        if flat_dist < 800 and abs(cross_z) > 0.25:
            if cross_z > 0:
                return self._name_to_index.get("powerslide_left", self._neutral_idx)
            return self._name_to_index.get("powerslide_right", self._neutral_idx)

        if cross_z > 0.05:
            return self._name_to_index.get("sharp_left", self._neutral_idx)
        if cross_z < -0.05:
            return self._name_to_index.get("sharp_right", self._neutral_idx)

        if facing < -0.2:
            return self._name_to_index.get("drive_reverse", self._neutral_idx)

        return self._name_to_index.get("stabilise", self._neutral_idx)

    @staticmethod
    def _is_kickoff(state: Any) -> bool:
        ball = getattr(state, "ball", None)
        if ball is None:
            return False
        return float(np.linalg.norm(ball.position[:2])) < 60 and float(np.linalg.norm(ball.linear_velocity)) < 10

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
    def __init__(self) -> None:
        self.action_parser = YourActionParser()
        self.num_actions = len(self.action_parser.lookup_table)
        cur_dir = Path(__file__).resolve().parent

        device = torch.device("cpu")
        self.policy = DiscreteFF(OBS_SIZE, self.num_actions, POLICY_LAYER_SIZES, device)
        self._load_weights(cur_dir / "PPO_POLICY.pt", device)
        torch.set_num_threads(1)

    def act(self, obs: np.ndarray, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        self.action_parser.maybe_supervise(context)

        with torch.no_grad():
            action_idx, _ = self.policy.get_action(obs, deterministic=True)

        action = np.asarray(self.action_parser.parse_actions([int(action_idx)], context))
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]

        if action.ndim != 1:
            raise ValueError(f"Invalid action returned from parser: shape={action.shape}")

        return action

    def _load_weights(self, path: Path, device: torch.device) -> None:
        if not path.exists():
            raise FileNotFoundError(
                "Missing PPO policy checkpoint. Place your trained weights at "
                f"{path.name} in the bot directory before launching the agent."
            )

        try:
            state_dict = torch.load(path, map_location=device)
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError(f"Failed to load PPO weights from {path}: {exc}") from exc

        self.policy.load_state_dict(state_dict)

"""Entry point for running self-play training matches with SuperBot.

Scaling guide:
    * PPO / A3C: Replace the DQN ``OfflineTrainer`` with an actor-critic
      implementation (e.g. torch.distributions for stochastic policies) and
      collect rollouts by storing log-probabilities alongside state/action data.
    * GPU acceleration: instantiate :class:`TrainerConfig` with
      ``device="cuda"`` when CUDA is available to leverage PyTorch's GPU ops.
    * Curriculum learning: modify :data:`MAX_MATCHES` to iterate through
      increasingly complex ``rlbot.cfg`` variations (e.g. different maps or
      mutators) by copying the configuration file before launching each match.
    * Self-play league: maintain a directory of historic checkpoints and load a
      random opponent weight file in ``super_bot.cfg`` before each episode to
      prevent overfitting to the latest policy.
"""
from __future__ import annotations

import time
from pathlib import Path

from rlbot.setup_manager import SetupManager

from bots.evaluation import Evaluator
from bots.learning_bot import ACTION_SPACE_SIZE
from bots.state_representation import STATE_DIMENSION
from bots.trainer import OfflineTrainer, TrainerConfig

MAX_MATCHES = 50
EVALUATION_INTERVAL = 5


def _wait_for_match_end(manager: SetupManager) -> None:
    while True:
        packet = manager.game_interface.update_live_data_packet()
        if packet and packet.game_info.is_match_ended:
            break
        time.sleep(1.0)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "rlbot.cfg"
    checkpoint_path = project_root / "bots/checkpoints/policy.pt"
    replay_dir = project_root / "bots/replays"

    trainer_config = TrainerConfig(checkpoint_path=checkpoint_path)
    offline_trainer = OfflineTrainer(trainer_config, STATE_DIMENSION, ACTION_SPACE_SIZE)
    evaluator = Evaluator(config_path, project_root / "bots/checkpoints/evaluation_log.jsonl")

    manager = SetupManager()
    manager.load_config(config_path)
    manager.connect_and_start()
    manager.launch_rocket_league_if_needed()

    try:
        for match_index in range(MAX_MATCHES):
            print(f"Starting match {match_index + 1}/{MAX_MATCHES}")
            manager.start_match()
            _wait_for_match_end(manager)

            offline_trainer.load_replay_logs(replay_dir)
            offline_trainer.train()

            if (match_index + 1) % EVALUATION_INTERVAL == 0:
                print("Running evaluation series...")
                evaluator.run_best_of()
    finally:
        manager.shut_down()


if __name__ == "__main__":
    main()

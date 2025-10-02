"""Launches a training-focused RLBot match for SuperBot."""
from __future__ import annotations

import argparse
from pathlib import Path

from rlbot import runner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a training match using rlbot.cfg")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("rlbot.cfg"))
    parser.add_argument("--episodes", type=int, default=10, help="Number of matches to run back-to-back")
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find configuration file at {cfg_path}")

    for episode in range(args.episodes):
        print(f"Training match {episode + 1}/{args.episodes}")
        runner.main(["--config", str(cfg_path)])


if __name__ == "__main__":
    main()

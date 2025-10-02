"""Helper script for launching a SuperBot match via the RLBot runner."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rlbot import runner


def _ensure_config_argument() -> None:
    """Inject the rlbot.cfg path if the user did not supply --config."""

    if any(arg.startswith("--config") for arg in sys.argv[1:]):
        # A config path is already provided by the caller; nothing to change.
        return

    cfg_path = Path(__file__).resolve().with_name("rlbot.cfg")
    # Set the working directory to the project root so relative paths resolve.
    os.chdir(cfg_path.parent)
    # Prepend the --config flag so the RLBot runner receives the expected path.
    sys.argv[1:1] = ["--config", str(cfg_path)]


if __name__ == "__main__":
    print("Starting SuperBot Match...")
    _ensure_config_argument()
    runner.main()

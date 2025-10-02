"""Helper script for launching a SuperBot match via the RLBot runner."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rlbot import runner


def _ensure_config_argument() -> Path:
    """Return the configuration path supplied on the command line or a default."""

    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if not arg.startswith("--config"):
            continue

        if "=" in arg:
            cfg_value = arg.split("=", 1)[1]
        else:
            try:
                cfg_value = args[index + 1]
            except IndexError as exc:  # pragma: no cover - defensive guard
                raise ValueError("--config flag provided without a path") from exc

        cfg_path = Path(cfg_value).expanduser().resolve()
        os.chdir(cfg_path.parent)
        return cfg_path

    cfg_path = Path(__file__).resolve().with_name("rlbot.cfg")
    if not cfg_path.exists():
        raise FileNotFoundError(
            "Unable to locate rlbot.cfg next to run_superbot_match.py."
        )

    # Set the working directory to the project root so relative paths resolve.
    os.chdir(cfg_path.parent)
    return cfg_path


def _launch_runner(cfg_path: Path) -> None:
    """Invoke ``runner.main`` with a guaranteed configuration argument.

    ``runner.main`` has historically accepted either a pre-populated ``sys.argv`` or an
    ``args`` sequence (depending on RLBot version).  We optimistically call the modern API
    first and gracefully fall back to mutating ``sys.argv`` if the signature does not accept
    parameters.  This guards against the ``TypeError`` raised when ``config_location`` is left
    as ``None`` on newer releases.
    """

    try:
        runner.main(["--config", str(cfg_path)])
    except TypeError:
        # Older RLBot versions expect the CLI arguments to be present in ``sys.argv``.
        if not any(arg.startswith("--config") for arg in sys.argv[1:]):
            sys.argv[1:1] = ["--config", str(cfg_path)]
        runner.main()


if __name__ == "__main__":
    print("Starting SuperBot Match...")
    cfg_path = _ensure_config_argument()
    _launch_runner(cfg_path)

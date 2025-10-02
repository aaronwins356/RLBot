# SuperBotProject

SuperBot is a production-ready RLBot agent that combines a Diamond-level scripted
baseline with reinforcement learning and imitation learning pipelines.  The
project is designed so that the bot is competent from the first kickoff while
retaining the ability to self-improve through DQN training and extend to other
algorithms such as PPO or A3C.

## Features

- **Diamond Baseline Strategy** – A robust state machine implements realistic
  rotations, boost management, and situational awareness for 1v1 through 3v3.
- **Advanced Mechanics** – Modular routines cover flip resets, ceiling shots,
  wave dashes, dribbles, fakes, demos, shadow defense, and psycho reads.
- **Reinforcement Learning Ready** – Default DQN policy manager with replay
  buffer logging, imitation pretraining support, and training scripts powered by
  RLGym.
- **Imitation Learning** – Matches optionally log JSON transitions that can be
  converted into datasets for supervised pretraining.
- **Evaluation Harness** – Batch-run matches via RLBot and export aggregate
  statistics for regression tracking.

## Repository Layout

```
SuperBotProject/
├── appearance.cfg          # Car cosmetics used by RLBot
├── bot.cfg                 # Alternate configuration for experimental matches
├── boost_pad_tracker.py    # Field boost tracking helper
├── drive.py                # Low-level control helpers
├── evaluation.py           # Match evaluation automation
├── mechanics.py            # Advanced maneuver implementations
├── orientation.py          # Orientation math helpers
├── rl_components.py        # DQN, replay buffer, and imitation learning utilities
├── rlbot.cfg               # Default match configuration (human vs SuperBot)
├── run_superbot_match.py   # Launch RLBot match from the CLI
├── run_training_match.py   # Loop RLBot matches for dataset generation
├── sequence.py             # Sequencing helper for multi-tick plans
├── spikes.py               # Specialized training scenarios
├── strategy.py             # High-level decision making / state machine
├── super_bot.cfg           # Bot registration for RLBot GUI
├── super_bot.py            # RLBot agent entry point
├── training.py             # DQN + imitation training loop
└── vec.py                  # Vector utilities
```

## Getting Started

1. Install the [RLBot](https://github.com/RLBot/RLBot) Python package and run
   the GUI once to generate required assets.
2. Install extra dependencies for training: `pip install rlgym torch
   stable-baselines3`.
3. Launch a match:

   ```bash
   python run_superbot_match.py
   ```

   By default the configuration spawns a human on blue versus SuperBot on
   orange.  Edit `rlbot.cfg` to run self-play or multi-bot scrimmages.

4. Run repeated training matches to gather imitation data:

   ```bash
   python run_training_match.py --episodes 20
   ```

   Transition logs are written to `./records` and can be consumed by the
   training pipeline.

## Training

- **Reinforcement Learning** – Execute `python training.py` to start DQN
  self-play using RLGym.  Checkpoints are written to `./models`.  Replace the
  default `PolicyFactory` if you want to swap in PPO or A3C implementations.
- **Imitation Learning** – Store JSON replay exports in `./records`, then run
  `python training.py --imitation ./records` to pretrain the policy before RL
  fine-tuning.

## Evaluation

Use the evaluation harness to produce consistent metrics:

```bash
python -c "from evaluation import EvaluationHarness; EvaluationHarness(Path('rlbot.cfg'), matches=3).run()"
```

The harness writes summary statistics to `match_summary.json` (if produced by
RLBot) and can persist aggregated results to disk.

## Extending the Project

- Add new mechanics to `mechanics.py` and expose them via `strategy.py` or the
  reinforcement learning action space.
- Implement alternative policy builders in `rl_components.py` for PPO or A3C.
- Create focused practice scenarios by registering them in `spikes.py` and
  driving them from a custom training script.

Contributions and pull requests are welcome.  Please follow the inline style
comments and docstrings when extending the project to keep the codebase
approachable for new bot developers.

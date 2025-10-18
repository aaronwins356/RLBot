# Hybrid Mechanics PPO Bot

This repository packages a Rocket League bot that blends scripted champion-level
mechanics with a discrete PPO policy.  It ships with a deterministic
observation builder, a hybrid action parser that exposes both low-level control
and timed macros, and an end-to-end training pipeline built on top of
`rlgym-compat`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `agent.py` | Loads the trained policy and exposes a deterministic `act` API. |
| `your_obs.py` | Deterministic observation builder shared between training and inference. |
| `your_act.py` | Hybrid action parser with mechanical macros and rule-based overrides. |
| `mechanics/` | Scripted routines, macro definitions, and the safety supervisor. |
| `training/` | PPO training loop, reward shaping, and evaluation harness. |
| `PPO_POLICY.pt` | (Not tracked) Place your trained checkpoint here before launching the bot. |

## Getting started

1. **Install dependencies** – the runtime requirements are listed in
   `requirements.txt`.  For training you additionally need
   `rlgym`, `rlgym-compat`, and a physics-enabled simulator environment
   (typically Python 3.9 on Windows).  Optional tooling such as
   `stable-baselines3` and `tensorboard` integrate cleanly with the training
   scripts.
2. **Configure the bot** – adjust branding and metadata in `bot.cfg`, ensure the
   trained weights are exported to `PPO_POLICY.pt` (the file is ignored by git), and tweak `POLICY_LAYER_SIZES`
   or `tick_skip` if you target a different latency envelope.
3. **Launch through RLBotGUI** – point the GUI at `bot.cfg`, confirm that the
   dependencies install successfully, and verify the bot stabilises at the
   desired FPS tier.

## Observation design

`YourOBS` produces a 140-length vector that captures:

* Scoreline context and ball physics.
* Full car state for the controlled player, including orientation vectors and
  relative position to the ball.
* Teammate and opponent slices (up to 3v3) padded deterministically.
* Boost pad availability and the previously applied control vector.

All physics quantities are normalised into approximately `[-1, 1]` to stabilise
PPO updates and prevent distribution drift between training and inference.

## Action space and mechanics

`YourActionParser` exposes 16 low-level control templates alongside a curated
set of mechanical macros (fast aerials, half-flips, dribble carries, panic
clears, etc.).  Macros are encoded as timed control sequences in
`mechanics/routines.py`, while `mechanics/supervisor.py` houses a safety gate
that can override the policy in emergencies (kickoffs, recoveries, own-goal
threats).

During inference the parser can be cancelled by selecting the dedicated
`cancel_macro` action, allowing PPO to blend scripted manoeuvres with fine-grain
steering and boost usage.

## Training pipeline

`training/train.py` provides a from-scratch PPO implementation that mirrors the
inference architecture (`DiscreteFF` for the actor plus a value head).  The
script relies on `rlgym-compat` to recreate the RLBot observation/action space
and uses the reward shaping defined in `training/rewards.py` to blend dense
fundamentals with mechanic bonuses.

Run training with:

```bash
python -m training.train --steps 2000000 --rollout 8192 --team-size 2 --tick-skip 8 --checkpoint PPO_POLICY.pt
```

Evaluation against scripted opponents is available via:

```bash
python -m training.evaluate PPO_POLICY.pt --episodes 50
```

Both scripts expect the training dependencies to be installed in the active
environment.

## Packaging checklist

* Update `bot.cfg` with final branding, contact information, and relevant tags.
* Export a trained checkpoint to `PPO_POLICY.pt` (or update `agent.py` to point at your preferred path).
* Document any optional extras (rendering, telemetry) in this README.
* Verify that `requirements.txt` installs cleanly through RLBotGUI.
* Bundle showcase replays or videos when submitting to the RLBot community.


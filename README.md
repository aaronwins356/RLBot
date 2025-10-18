# Hybrid Mechanics PPO Bot

Welcome! This guide explains how to install, configure, and run the Hybrid Mechanics PPO Bot for Rocket League without needing any coding experience. Follow the steps in order and you will have the bot driving in matches through the RLBot framework.

---

## 1. What this project delivers

* A ready-to-use Rocket League bot that mixes reliable scripted moves with a machine-learned decision maker.
* Easy-to-edit settings so you can personalise the bot's name, appearance, and behaviour.
* Optional tools for advanced training and evaluation if you later decide to improve the bot yourself.

The project folder already contains everything required for match play. A single additional file—your trained policy called `PPO_POLICY.pt`—is expected when you are ready to deploy custom behaviour. If you do not provide that file, the bot falls back to a built-in safety pilot so it can still play matches.

---

## 2. Before you start

1. **Install RLBotGUI** (the easiest way to manage bots) from [rlbot.org](https://www.rlbot.org/).
2. **Install Rocket League on Windows** and make sure you can launch the game normally.
3. **Prepare Python 3.9** (32-bit or 64-bit). RLBotGUI will help you install it if it is missing.

That is all the setup you need before working with this repository.

---

## 3. Folder tour (for orientation only)

| Item | Plain-language description |
| --- | --- |
| `bot.cfg` | The configuration RLBotGUI reads. It stores the bot name, author, and which Python file to run. |
| `agent.py` | Loads the trained policy and sends button presses to the game. |
| `your_obs.py` | Describes what information about the match the bot will “see”. |
| `your_act.py` | Converts the bot's decisions into controller actions and special move macros. |
| `mechanics/` | Contains the scripted manoeuvres such as fast aerials, recoveries, and safety rules. |
| `training/` | Extra tools for power users who want to retrain or evaluate the bot. |
| `requirements.txt` | The list of Python packages RLBotGUI will install automatically. |

> **Tip:** You do **not** need to edit any of these files to run the bot. Only adjust `bot.cfg` and the optional `PPO_POLICY.pt` checkpoint.

---

## 4. Quick start: run the bot in RLBotGUI

1. **Open RLBotGUI** and choose **Add > Existing Bot**.
2. Browse to the folder containing this README and select `bot.cfg`.
3. Allow RLBotGUI to install the listed Python packages when prompted.
4. (Optional) Place your trained policy file in the same folder and rename it to `PPO_POLICY.pt`.
5. Press **Launch** in RLBotGUI and start a match. The bot will automatically load the policy (or the built-in fallback) and begin playing.

You can repeat these steps to field multiple bots or to join them with human teammates.

---

## 5. Customise the bot without coding

* **Name, appearance, and loadout** – open `bot.cfg` in any text editor and follow the comments. Update the `name`, `agent_class`, and look settings as desired.
* **Tick rate and performance** – inside `bot.cfg`, the `python_file` points to `bot.py`, which already handles performance tuning. Most users can leave these values at their defaults.
* **Policy upgrades** – replace `PPO_POLICY.pt` with a new checkpoint file whenever you have improved training results. The bot will load the newest file on the next launch.

Always save the edited file and re-launch through RLBotGUI to apply changes.

---

## 6. Advanced: training your own policy (optional)

You only need this section if you want to retrain the machine learning policy yourself. It assumes basic familiarity with Python tools.

1. Create a Python environment with the packages listed in `requirements.txt`, plus `rlgym`, `rlgym-compat`, `stable-baselines3`, and `tensorboard`.
2. Start training with:
   ```bash
   python -m training.train --steps 2000000 --rollout 8192 --team-size 2 --tick-skip 8 --checkpoint PPO_POLICY.pt
   ```
3. Review performance or compare against scripted opponents with:
   ```bash
   python -m training.evaluate PPO_POLICY.pt --episodes 50
   ```
4. Copy the resulting `PPO_POLICY.pt` into the bot folder before launching through RLBotGUI.

If you prefer to avoid command-line work, you can stay with the supplied policy file or the fallback pilot.

---

## 7. Final checklist before sharing the bot

* [ ] Update `bot.cfg` with final branding, contact details, and community tags.
* [ ] Confirm `requirements.txt` installs successfully via RLBotGUI.
* [ ] Include the latest `PPO_POLICY.pt` (or instructions for obtaining it) if you distribute the bot.
* [ ] Consider bundling highlight replays or videos to showcase behaviour.
* [ ] Document any optional telemetry or visualisers you enable.

With these steps complete, the Hybrid Mechanics PPO Bot is ready for local matches or community tournaments—no coding required. Enjoy the games!

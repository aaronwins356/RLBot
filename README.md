# Phoenix Rocket League Bot

Phoenix is a fresh take on your Rocket League automation setup. The bot relies
on a tuned set of heuristics and scripted mechanics (fast aerials, half flips,
panic clears) so the car always has something purposeful to do on the pitch.
The behaviour is intentionally lightweight and ships without any external
training artefacts so you can understand every decision the bot makes.

---

## 1. What this project delivers

* Plug-and-play Rocket League bot driven entirely by transparent heuristics.
* Scripted routines for kick-offs, recoveries, clears, and striking plays.
* Clean Python modules that you can tweak without hunting down training code or
  external assets.

---

## 2. Before you start

1. Install **RLBotGUI** from [rlbot.org](https://www.rlbot.org/).
2. Ensure Rocket League is installed on Windows and launches normally.
3. Allow RLBotGUI to install Python 3.11 when prompted.

No heavyweight ML dependencies are required; everything runs on the core RLBot
stack plus NumPy.

---

## 3. Folder tour

| Item | Description |
| --- | --- |
| `bot.cfg` | RLBot configuration pointing to the Phoenix bot entry point. |
| `bot.py` | RLBot agent wiring which feeds game state into the heuristic agent. |
| `agent.py` | Core decision logic blending heuristics with scripted mechanics. |
| `mechanics/` | Macro actions (fast aerial, half flip, etc.) and their supervisor. |
| `util/` | Small data structures mirroring the info the bot consumes. |
| `appearance.cfg` | Optional cosmetics for the bot. |
The previous heavyweight PPO scaffolding (`discrete_policy.py`, `training/`, and
observation builders) remains retired; the new logic is intentionally lean and
easy to reason about.

---

## 4. Quick start via RLBotGUI

1. Open **RLBotGUI** and choose **Add > Existing Bot**.
2. Select the `bot.cfg` file located next to this README.
3. Allow RLBotGUI to install the listed dependencies (RLBot + NumPy).
4. Click **Launch** and start a match. Phoenix will take the field immediately.

You can clone the folder to field multiple bots or to play against Phoenix with
friends.

---

## 5. Customising behaviour

* **Aggression tweaks** – edit `agent.py` and adjust the constants in
  `_should_fast_aerial`, `_should_power_shot`, or `_choose_target` to suit your
  play style.
* **Macro tuning** – macro definitions live in `mechanics/routines.py`. Update
  the ControlStep timings or throttle values for different results.
* **Control presets** – `your_act.py` keeps reusable controller arrays. Add new
  presets or modify existing ones to influence the bot’s driving posture.

Because everything is still pure Python + NumPy, you can hot reload changes
through RLBotGUI without training a neural network.

---

## 6. Troubleshooting

* **Bot does nothing** – ensure Rocket League is capped at 120/240/360 FPS as
  recommended by RLBot. Check the RLBot console for Python errors.
* **Over-aggressive aerials** – lower the boost threshold in
  `_should_fast_aerial` inside `agent.py`.
* **Rotation feels off** – tweak the shadowing distance in `_choose_target` to
  make Phoenix fall back sooner or stay upfield longer.

---

Phoenix provides a reliable baseline opponent you can iterate on. Have fun in
the arena!

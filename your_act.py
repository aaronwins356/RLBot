"""Hybrid action parser exposing both low level controls and scripted macros."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import gym.spaces
import numpy as np

from mechanics import MacroAction, MechanicSupervisor, routines


ACTION_SIZE = 8


def _controls(
    throttle: float = 0.0,
    steer: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    jump: float = 0.0,
    boost: float = 0.0,
    handbrake: float = 0.0,
) -> np.ndarray:
    arr = np.asarray([throttle, steer, pitch, yaw, roll, jump, boost, handbrake], dtype=np.float32)
    return np.clip(arr, -1.0, 1.0)


@dataclass(frozen=True)
class ActionEntry:
    name: str
    payload: np.ndarray | MacroAction

    def is_macro(self) -> bool:
        return isinstance(self.payload, MacroAction)


class YourActionParser:
    """Decode PPO logits into controller arrays and macros."""

    def __init__(self) -> None:
        self._lookup_table: List[ActionEntry] = self._build_lookup_table()
        self.lookup_table = self._lookup_table  # Public attribute used by the agent
        self.name_to_index = {entry.name: idx for idx, entry in enumerate(self._lookup_table)}
        self.action_space = gym.spaces.Discrete(len(self._lookup_table))

        self._active_macro = None
        self._macro_ticks = 0
        self._macro_entry: Optional[ActionEntry] = None
        self._supervisor = MechanicSupervisor()

        # Index of the "Cancel Macro" action – PPO can choose this to regain
        # manual control, while the supervisor uses it when a scripted routine
        # hands control back to the policy.
        self._cancel_index = next(
            i for i, entry in enumerate(self._lookup_table) if entry.name == "cancel_macro"
        )

    @property
    def cancel_index(self) -> int:
        return self._cancel_index

    def _build_lookup_table(self) -> List[ActionEntry]:
        simple_actions = [
            ActionEntry("neutral", _controls()),
            ActionEntry("drive_forward", _controls(throttle=1.0)),
            ActionEntry("drive_reverse", _controls(throttle=-1.0)),
            ActionEntry("stabilise", _controls(throttle=0.3, pitch=0.0)),
            ActionEntry("boost_forward", _controls(throttle=1.0, boost=1.0)),
            ActionEntry("sharp_left", _controls(throttle=0.8, steer=-1.0)),
            ActionEntry("sharp_right", _controls(throttle=0.8, steer=1.0)),
            ActionEntry("powerslide_left", _controls(throttle=1.0, steer=-1.0, handbrake=1.0)),
            ActionEntry("powerslide_right", _controls(throttle=1.0, steer=1.0, handbrake=1.0)),
            ActionEntry("jump", _controls(jump=1.0)),
            ActionEntry("double_jump", _controls(jump=1.0, pitch=-0.3)),
            ActionEntry("yaw_left_air", _controls(pitch=-0.2, yaw=-1.0)),
            ActionEntry("yaw_right_air", _controls(pitch=-0.2, yaw=1.0)),
            ActionEntry("air_roll_left", _controls(roll=-1.0)),
            ActionEntry("air_roll_right", _controls(roll=1.0)),
            ActionEntry("cancel_macro", _controls()),
        ]

        macro_actions = [
            ActionEntry("fast_aerial", routines.fast_aerial_macro()),
            ActionEntry("half_flip", routines.half_flip_macro()),
            ActionEntry("power_shot", routines.power_shot_macro()),
            ActionEntry("speed_flip_kickoff", routines.speed_flip_kickoff_macro()),
            ActionEntry("aerial_recovery", routines.aerial_recovery_macro()),
            ActionEntry("dribble_carry", routines.ground_dribble_macro()),
            ActionEntry("panic_clear", routines.panic_clear_macro()),
        ]

        return simple_actions + macro_actions

    def parse_actions(
        self, actions: Iterable[int], context: Optional[Dict[str, object]] = None
    ) -> np.ndarray:
        if context is not None:
            self.maybe_supervise(context)

        if self._active_macro is not None:
            return self._advance_macro()

        action_idx = int(next(iter(actions)))

        # Allow PPO to cancel a scripted routine explicitly.
        if action_idx == self._cancel_index:
            self._clear_macro()
            return self._lookup_table[self._cancel_index].payload.copy()

        entry = self._lookup_table[action_idx]
        if entry.is_macro():
            self._start_macro(entry)
            return self._advance_macro()

        return entry.payload.copy()

    # ------------------------------------------------------------------
    # Macro handling

    def _advance_macro(self) -> np.ndarray:
        if self._active_macro is None:
            return self._lookup_table[self._cancel_index].payload.copy()

        controls = self._active_macro.step()
        self._macro_ticks += 1

        if self._active_macro.finished:
            self._clear_macro()

        return controls

    def _clear_macro(self) -> None:
        self._active_macro = None
        self._macro_ticks = 0
        self._macro_entry = None

    def _start_macro(self, entry: ActionEntry) -> None:
        if not entry.is_macro():
            raise ValueError("Attempted to start a macro using a simple action entry")

        self._macro_entry = entry
        macro_action: MacroAction = entry.payload
        self._active_macro = macro_action.instantiate()
        self._macro_ticks = 0

    # ------------------------------------------------------------------
    # Supervisor integration

    def maybe_supervise(self, context: Optional[Dict[str, object]]) -> None:
        """Allow the rule-based supervisor to trigger overrides.

        The supervisor runs before the PPO output is interpreted.  When a
        scripted mechanic is requested, the macro state is updated so that
        :meth:`parse_actions` will emit the macro's control sequence on the next
        tick.
        """

        override = self._supervisor.maybe_override(context, active_macro=self._macro_entry)
        if override is not None:
            self._start_macro(ActionEntry(override.name, override))

"""Evaluation utilities for periodically benchmarking the learning policy."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from rlbot.setup_manager import SetupManager


@dataclass
class MatchStats:
    goals_blue: int = 0
    goals_orange: int = 0
    touches_blue: int = 0
    touches_orange: int = 0

    def winner(self) -> str:
        if self.goals_blue > self.goals_orange:
            return "blue"
        if self.goals_orange > self.goals_blue:
            return "orange"
        return "draw"


@dataclass
class EvaluationResult:
    matches: List[MatchStats] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, float]:
        total_blue_wins = sum(1 for m in self.matches if m.winner() == "blue")
        total_orange_wins = sum(1 for m in self.matches if m.winner() == "orange")
        draws = len(self.matches) - total_blue_wins - total_orange_wins
        total_blue_goals = sum(m.goals_blue for m in self.matches)
        total_orange_goals = sum(m.goals_orange for m in self.matches)
        total_blue_touches = sum(m.touches_blue for m in self.matches)
        total_orange_touches = sum(m.touches_orange for m in self.matches)
        games = len(self.matches) or 1
        return {
            "blue_win_rate": total_blue_wins / games,
            "orange_win_rate": total_orange_wins / games,
            "draw_rate": draws / games,
            "avg_blue_goals": total_blue_goals / games,
            "avg_orange_goals": total_orange_goals / games,
            "avg_blue_touches": total_blue_touches / games,
            "avg_orange_touches": total_orange_touches / games,
        }

    def to_json(self) -> str:
        return json.dumps(
            {
                "matches": [
                    {
                        "goals_blue": m.goals_blue,
                        "goals_orange": m.goals_orange,
                        "touches_blue": m.touches_blue,
                        "touches_orange": m.touches_orange,
                    }
                    for m in self.matches
                ],
                "summary": self.summary,
            }
        )


class Evaluator:
    """Runs short best-of-N series using frozen policies."""

    def __init__(self, config_path: Path, log_path: Path) -> None:
        self.config_path = config_path
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def run_best_of(self, games: int = 3) -> EvaluationResult:
        manager = SetupManager()
        manager.load_config(self.config_path)
        manager.connect_and_start()
        manager.launch_rocket_league_if_needed()

        result = EvaluationResult()
        try:
            for _ in range(games):
                manager.start_match()
                stats = self._play_single_game(manager)
                result.matches.append(stats)
        finally:
            manager.shut_down()

        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(result.to_json() + "\n")
        return result

    def _play_single_game(self, manager: SetupManager) -> MatchStats:
        stats = MatchStats()
        last_touch_time = -1.0
        while True:
            packet = manager.game_interface.update_live_data_packet()
            if packet is None:
                time.sleep(0.1)
                continue
            stats.goals_blue = packet.teams[0].score
            stats.goals_orange = packet.teams[1].score
            touch = packet.game_ball.latest_touch
            if touch and touch.time_seconds > last_touch_time:
                last_touch_time = touch.time_seconds
                if touch.team == 0:
                    stats.touches_blue += 1
                elif touch.team == 1:
                    stats.touches_orange += 1
            if packet.game_info.is_match_ended:
                break
            time.sleep(0.1)
        return stats


__all__ = ["Evaluator", "EvaluationResult", "MatchStats"]

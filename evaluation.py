"""Evaluation utilities for tracking SuperBot performance."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from rlbot import runner


@dataclass
class MatchResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    shots: int = 0
    saves: int = 0
    games: int = 0
    boost_efficiency_sum: float = 0.0
    aerial_attempts: int = 0
    aerial_successes: int = 0
    flip_reset_attempts: int = 0
    flip_reset_successes: int = 0

    def record_game(
        self,
        score_difference: int,
        goals_for: int,
        goals_against: int,
        shots: int,
        saves: int,
        metrics: Dict[str, float],
    ) -> None:
        if score_difference > 0:
            self.wins += 1
        elif score_difference < 0:
            self.losses += 1
        else:
            self.draws += 1
        self.goals_scored += goals_for
        self.goals_conceded += goals_against
        self.shots += shots
        self.saves += saves
        self.games += 1

        boost_used = float(metrics.get("boost_used", 0.0))
        boost_collected = float(metrics.get("boost_collected", 1.0))
        efficiency = boost_used / max(boost_collected, 1.0)
        self.boost_efficiency_sum += efficiency

        aerial_attempts = int(metrics.get("aerial_attempts", 0))
        aerial_successes = int(metrics.get("aerial_successes", 0))
        flip_attempts = int(metrics.get("flip_reset_attempts", 0))
        flip_successes = int(metrics.get("flip_reset_successes", 0))

        self.aerial_attempts += aerial_attempts
        self.aerial_successes += aerial_successes
        self.flip_reset_attempts += flip_attempts
        self.flip_reset_successes += flip_successes

    def aerial_success_rate(self) -> float:
        if self.aerial_attempts == 0:
            return 0.0
        return self.aerial_successes / self.aerial_attempts

    def flip_reset_success_rate(self) -> float:
        if self.flip_reset_attempts == 0:
            return 0.0
        return self.flip_reset_successes / self.flip_reset_attempts

    def boost_efficiency(self) -> float:
        if self.games == 0:
            return 0.0
        return self.boost_efficiency_sum / self.games

    def as_dict(self) -> Dict[str, float]:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "goals_scored": self.goals_scored,
            "goals_conceded": self.goals_conceded,
            "shots": self.shots,
            "saves": self.saves,
            "boost_efficiency": round(self.boost_efficiency(), 3),
            "aerial_success_rate": round(self.aerial_success_rate(), 3),
            "flip_reset_success_rate": round(self.flip_reset_success_rate(), 3),
        }


@dataclass
class EvaluationHarness:
    """Batch evaluation driver using rlbot-runner."""

    config_path: Path
    matches: int = 5
    wait_time: float = 3.0
    csv_path: Optional[Path] = None
    results: MatchResult = field(default_factory=MatchResult)

    def run(self) -> MatchResult:
        for game_index in range(1, self.matches + 1):
            print(f"Starting evaluation match {game_index}/{self.matches} using {self.config_path}")
            runner.main(["--config", str(self.config_path)])
            summary_path = self.config_path.with_name("match_summary.json")
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                metrics = summary.get("metrics", {})
                self.results.record_game(
                    score_difference=int(summary.get("score_difference", 0)),
                    goals_for=int(summary.get("goals_for", 0)),
                    goals_against=int(summary.get("goals_against", 0)),
                    shots=int(summary.get("shots", 0)),
                    saves=int(summary.get("saves", 0)),
                    metrics=metrics,
                )
                self._append_csv(game_index, summary)
            time.sleep(self.wait_time)
        return self.results

    def _append_csv(self, game_index: int, summary: Dict[str, float]) -> None:
        if not self.csv_path:
            return

        metrics = summary.get("metrics", {})
        row = {
            "game": game_index,
            "score_difference": summary.get("score_difference", 0),
            "goals_for": summary.get("goals_for", 0),
            "goals_against": summary.get("goals_against", 0),
            "boost_efficiency": self.results.boost_efficiency(),
            "aerial_success_rate": self.results.aerial_success_rate(),
            "flip_reset_success_rate": self.results.flip_reset_success_rate(),
            "aerial_attempts": metrics.get("aerial_attempts", 0),
            "aerial_successes": metrics.get("aerial_successes", 0),
            "flip_reset_attempts": metrics.get("flip_reset_attempts", 0),
            "flip_reset_successes": metrics.get("flip_reset_successes", 0),
            "boost_used": metrics.get("boost_used", 0.0),
            "boost_collected": metrics.get("boost_collected", 0.0),
        }

        fieldnames = list(row.keys())
        exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def save(self, destination: Path) -> None:
        destination.write_text(json.dumps(self.results.as_dict(), indent=2))

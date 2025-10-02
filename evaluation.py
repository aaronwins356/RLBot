"""Evaluation utilities for tracking SuperBot performance."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

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

    def record_game(self, score_difference: int, goals_for: int, goals_against: int, shots: int, saves: int) -> None:
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

    def as_dict(self) -> Dict[str, int]:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "goals_scored": self.goals_scored,
            "goals_conceded": self.goals_conceded,
            "shots": self.shots,
            "saves": self.saves,
        }


@dataclass
class EvaluationHarness:
    """Batch evaluation driver using rlbot-runner."""

    config_path: Path
    matches: int = 5
    wait_time: float = 3.0
    results: MatchResult = field(default_factory=MatchResult)

    def run(self) -> MatchResult:
        for _ in range(self.matches):
            print(f"Starting evaluation match using {self.config_path}")
            runner.main(["--config", str(self.config_path)])
            summary_path = self.config_path.with_name("match_summary.json")
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                self.results.record_game(
                    score_difference=int(summary.get("score_difference", 0)),
                    goals_for=int(summary.get("goals_for", 0)),
                    goals_against=int(summary.get("goals_against", 0)),
                    shots=int(summary.get("shots", 0)),
                    saves=int(summary.get("saves", 0)),
                )
            time.sleep(self.wait_time)
        return self.results

    def save(self, destination: Path) -> None:
        destination.write_text(json.dumps(self.results.as_dict(), indent=2))

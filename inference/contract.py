"""Data contract between the ML stage and the LLM stage.

StepPrediction / MLOutput are the ONLY types the LLM stage consumes from inference.
Change preprocessing or training internals freely; do not change this module's public
surface without updating llm/context.py in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import List


@dataclass
class LevelPrediction:
    """Top-K prediction for a single hierarchy level at one step."""
    labels: List[str]      # length K, ranked by probability
    probs:  List[float]    # same length, matching positions


@dataclass
class StepPrediction:
    """ML prediction for one future step."""
    step: int                                 # 0-indexed position within n_future
    phase:          LevelPrediction
    phase_step:     LevelPrediction
    major_ops_code: LevelPrediction
    operation:      LevelPrediction
    duration_hours: float                     # un-scaled, back to raw hours

    def top1(self) -> dict:
        return {
            "phase":          self.phase.labels[0],
            "phase_step":     self.phase_step.labels[0],
            "major_ops_code": self.major_ops_code.labels[0],
            "operation":      self.operation.labels[0],
            "duration_hours": self.duration_hours,
        }


@dataclass
class MLOutput:
    well_name: str
    report_date: date
    n_future:  int
    top_k:     int
    steps:     List[StepPrediction] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "well_name": self.well_name,
            "report_date": str(self.report_date),
            "n_future": self.n_future,
            "top_k": self.top_k,
            "steps": [asdict(s) for s in self.steps],
        }

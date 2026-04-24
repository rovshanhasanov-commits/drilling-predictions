"""Data contract between the ML stage and the LLM stage.

StepPrediction / MLOutput are the ONLY types the LLM stage consumes from inference.
Change preprocessing or training internals freely; do not change this module's public
surface without updating llm/context.py in lockstep.

Each step exposes `topk_tuples` — a list of HierarchyTuple (phase, phase_step,
major_ops_code, operation) sorted by joint log-prob, guaranteed legal under the
constraint decoder. This replaces the old per-head top-K lists, which could
emit illegal cross-head combinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import List


@dataclass
class HierarchyTuple:
    """One legal (phase, phase_step, major_ops_code, operation) candidate."""
    phase:          str
    phase_step:     str
    major_ops_code: str
    operation:      str
    log_prob:       float            # joint log-prob under the model (sum over 4 heads)


@dataclass
class StepPrediction:
    """ML prediction for one future step.

    `topk_tuples[0]` is the argmax (highest joint log-prob legal tuple).
    """
    step:           int                          # 0-indexed position within n_future
    topk_tuples:    List[HierarchyTuple]         # length K, sorted by log_prob desc
    duration_hours: float                        # un-scaled, back to raw hours

    def top1(self) -> dict:
        t = self.topk_tuples[0]
        return {
            "phase":          t.phase,
            "phase_step":     t.phase_step,
            "major_ops_code": t.major_ops_code,
            "operation":      t.operation,
            "duration_hours": self.duration_hours,
            "log_prob":       t.log_prob,
        }


@dataclass
class MLOutput:
    well_name:   str
    report_date: date
    n_future:    int
    top_k:       int
    steps:       List[StepPrediction] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "well_name":   self.well_name,
            "report_date": str(self.report_date),
            "n_future":    self.n_future,
            "top_k":       self.top_k,
            "steps":       [asdict(s) for s in self.steps],
        }

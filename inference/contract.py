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
class DurationBin:
    """One ranked duration-bin candidate from the bin classification head.

    `label` is the bin's string class (e.g. `"1-2 hrs"` or sentinel `"EOO"`).
    `prob` is the head's softmax probability.
    `center_hours` is the empirical median of training rows in that bin
        (sentinels: `EOO=0.0`, `Unplanned=NaN`, `UNK=NaN`). Use it as the
        point estimate when picking a duration value within the bin.
    """
    label:        str
    prob:         float
    center_hours: float


@dataclass
class HierarchyTuple:
    """One legal (phase, phase_step, major_ops_code, operation) candidate.

    `log_prob` is the raw joint log-probability (sum of the 4 head log-probs;
    5 when the bin head is part of the joint via `include_duration_bins_in_hierarchy`).
    `exp(log_prob)` is the joint probability under independence, <= P_legal <= 1
    across the full legal set — useful as a confidence health signal.

    `prob` is that same probability **renormalized over the legal set L**: each
    tuple's share of the legal-set mass. Summing `prob` across all legal tuples
    for a given step equals 1.0 exactly; summing just the surfaced top-K gives
    <= 1 and the gap tells you how concentrated the distribution is. This is
    the number to show humans / LLMs when you want an intuitive "how confident
    is the model about this option among all legal next ops".

    Under unconstrained decoding (constraint decoder off), `prob` falls back to
    `exp(log_prob)` — the raw joint under independence, no renormalization.

    `duration_bin` is populated only under 5D constraints
    (`inference.include_duration_bins_in_hierarchy=true`), where the bin head
    joins the joint argmax as the 5th column. Left as `None` under 4D mode so
    the bin-head's top-K still flows through `StepPrediction.duration_bin_topk`
    independently.
    """
    phase:          str
    phase_step:     str
    major_ops_code: str
    operation:      str
    log_prob:       float            # raw joint log-prob (sum over 4 or 5 heads)
    prob:           float            # renormalized over L (legal set); in [0, 1]
    duration_bin:   str | None = None   # populated only under 5D constraints


@dataclass
class StepPrediction:
    """ML prediction for one future step.

    `topk_tuples[0]` is the argmax (highest joint log-prob legal tuple).
    `duration_bin_topk` is the top-K bin candidates when the bin classification
    head is active; empty list when the bin head is off.
    """
    step:              int                          # 0-indexed position within n_future
    topk_tuples:       List[HierarchyTuple]         # length K, sorted by log_prob desc
    duration_hours:    float                        # un-scaled, back to raw hours
    duration_bin_topk: List[DurationBin] = field(default_factory=list)

    def top1(self) -> dict:
        t = self.topk_tuples[0]
        return {
            "phase":          t.phase,
            "phase_step":     t.phase_step,
            "major_ops_code": t.major_ops_code,
            "operation":      t.operation,
            "duration_hours": self.duration_hours,
            "log_prob":       t.log_prob,
            "prob":           t.prob,
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

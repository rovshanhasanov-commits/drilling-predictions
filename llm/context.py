"""Prompt context builders. Each returns a markdown string section for the user message."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from inference.contract import MLOutput, StepPrediction


def context_selected_day(well: str, date, get_day_activities_fn, get_ops_summary_fn) -> str:
    activities = get_day_activities_fn(well, date)
    summary = get_ops_summary_fn(well, date)
    last = activities.sort_values("End_Hour").iloc[-1] if not activities.empty else None

    lines = [f"## Context 1: Target Well Activities — {well} on {date}"]
    lines.append(f"Ops Summary: {summary}")
    if last is not None:
        lines.append(
            f"End-of-day state: Phase={last['Phase']}, Phase_Step={last['Phase_Step']}, "
            f"DepthEnd_ft={last['DepthEnd_ft']}"
        )
    lines.append("")
    lines.append("| Phase | Phase_Step | Major_Ops_Code | Operation | Duration_hrs | DepthStart | DepthEnd |")
    lines.append("|-------|-----------|----------------|-----------|-------------|------------|----------|")
    for _, r in activities.iterrows():
        lines.append(
            f"| {r['Phase']} | {r['Phase_Step']} | {r['Major_Ops_Code']} | {r['Operation']} "
            f"| {r['Duration hours']} | {r['DepthStart_ft']} | {r['DepthEnd_ft']} |"
        )
    return "\n".join(lines)


def context_similar_wells(similar: dict) -> str:
    if not similar:
        return "## Context 2: Similar Wells\nNo similar wells found."
    lines = ["## Context 2: Next-Day Activities from Similar Wells"]
    for i, ((well, nd), activities) in enumerate(similar.items(), 1):
        summary = str(activities["Ops_Summary"].iloc[0]) if not activities.empty else "N/A"
        lines.append(f"\n### Similar Well {i}: {well} — {nd}")
        lines.append(f"Ops Summary: {summary}")
        lines.append("")
        lines.append("| Phase | Phase_Step | Major_Ops_Code | Operation | Duration_hrs | DepthStart | DepthEnd |")
        lines.append("|-------|-----------|----------------|-----------|-------------|------------|----------|")
        for _, r in activities.iterrows():
            lines.append(
                f"| {r['Phase']} | {r['Phase_Step']} | {r['Major_Ops_Code']} | {r['Operation']} "
                f"| {r['Duration hours']} | {r['DepthStart_ft']} | {r['DepthEnd_ft']} |"
            )
    return "\n".join(lines)


def context_constraints(constraints_md: str) -> str:
    return f"## Context 3: Domain Constraints\n\n{constraints_md}"


def context_ml_predictions(ml: MLOutput, fields: Iterable[str]) -> str:
    """Render the ML output as a markdown table, honoring the config's field allowlist.

    `fields` controls which level columns appear. Example: ['phase', 'major_ops_code'] hides
    phase_step and operation. duration_hours is included if present in `fields`.
    """
    fields = list(fields)
    level_fields = [f for f in fields if f in ("phase", "phase_step", "major_ops_code", "operation")]
    show_dur = "duration_hours" in fields
    k = ml.top_k

    lines = [f"## Context 4: ML-Predicted Sequence (top-{k} per step)"]
    lines.append(f"Model horizon: {ml.n_future} steps")
    header_cells = ["Step"] + [f.capitalize() for f in level_fields]
    if show_dur:
        header_cells.append("Dur_hrs")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")

    for s in ml.steps:
        row = [str(s.step + 1)]
        for f in level_fields:
            lvl: "LevelPrediction" = getattr(s, f)
            top_strs = [f"{lab} ({p:.2f})" for lab, p in zip(lvl.labels, lvl.probs)]
            row.append("<br>".join(top_strs))
        if show_dur:
            row.append(f"{s.duration_hours:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def assemble_user_message(
    well: str,
    date,
    get_day_activities_fn,
    get_ops_summary_fn,
    similar: dict,
    constraints_md: str,
    ml: MLOutput,
    ml_fields: Iterable[str],
) -> str:
    parts = [
        context_selected_day(well, date, get_day_activities_fn, get_ops_summary_fn),
        context_similar_wells(similar),
        context_constraints(constraints_md),
        context_ml_predictions(ml, ml_fields),
    ]
    return "\n\n".join(parts)

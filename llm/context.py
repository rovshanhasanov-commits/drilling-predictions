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

    Under the constraint decoder, each step surfaces K **consistent** hierarchy
    tuples (phase / phase_step / major_ops_code / operation move together).
    The `fields` allowlist still hides columns — e.g. `['phase','major_ops_code']`
    suppresses phase_step and operation inside each tuple — but the K tuples
    themselves always travel as one unit.

    `logP` is the joint log-probability (sum over all four head log-probs;
    five when 5D mode is on and the bin head joins the joint).
    `duration_hours` is a per-step scalar; shown only on rank-1 to avoid clutter.

    When the bin classification head is active, an extra "Dur_bin_topK" column
    on rank-1 lists the top-K bin candidates with probabilities (e.g.
    "1-2 hrs (0.47)|2-5 hrs (0.31)"). When 5D mode is also on, each tuple
    also carries its own bin label as a "Dur_bin" column.
    """
    fields = list(fields)
    level_fields = [f for f in fields if f in ("phase", "phase_step", "major_ops_code", "operation")]
    show_dur = "duration_hours" in fields
    k = ml.top_k

    has_bin_topk = any(s.duration_bin_topk for s in ml.steps)
    has_5d_bin = any(
        any(t.duration_bin is not None for t in s.topk_tuples) for s in ml.steps
    )

    lines = [f"## Context 4: ML-Predicted Sequence (top-{k} legal tuples per step)"]
    lines.append(f"Model horizon: {ml.n_future} steps")
    intro = (
        "Each step lists the top-K (phase / phase_step / major_ops_code / operation) "
        "combinations the model judged most likely. Every tuple is guaranteed "
        "consistent with the drilling hierarchy (the ML stage already validated "
        "against Context 3). `prob` is the model's confidence in that tuple as "
        "a share of the legal-set probability mass (all legal tuples at a step "
        "sum to 1; top-K sums to <= 1 and the gap tells you how diffuse the tail "
        "is). `logP` is the raw joint log-probability (useful for diagnostics; "
        "closer to zero = higher confidence across all 4 heads)."
    )
    if has_bin_topk:
        intro += (
            " `Dur_bin_topK` (rank-1 row only) shows the bin classification head's "
            "top-K duration-range candidates with probabilities — treat the top-1 "
            "bin as your range and pick a specific value inside it based on Context 2."
        )
    if has_5d_bin:
        intro += (
            " The per-tuple `Dur_bin` column is the joint argmax's chosen bin "
            "(5D constraint mode)."
        )
    lines.append(intro)

    header_cells = ["Step", "Rank"] + [f.capitalize() for f in level_fields]
    if has_5d_bin:
        header_cells.append("Dur_bin")
    header_cells += ["prob", "logP"]
    if show_dur:
        header_cells.append("Dur_hrs")
    if has_bin_topk:
        header_cells.append("Dur_bin_topK")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")

    for s in ml.steps:
        bin_topk_str = ""
        if has_bin_topk and s.duration_bin_topk:
            bin_topk_str = "|".join(
                f"{db.label} ({db.prob:.2f})" for db in s.duration_bin_topk
            )
        for i, tup in enumerate(s.topk_tuples):
            row = [str(s.step + 1), str(i + 1)]
            for f in level_fields:
                row.append(str(getattr(tup, f)))
            if has_5d_bin:
                row.append(str(tup.duration_bin) if tup.duration_bin else "")
            row.append(f"{tup.prob:.3f}")
            row.append(f"{tup.log_prob:.2f}")
            if show_dur:
                row.append(f"{s.duration_hours:.2f}" if i == 0 else "")
            if has_bin_topk:
                row.append(bin_topk_str if i == 0 else "")
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

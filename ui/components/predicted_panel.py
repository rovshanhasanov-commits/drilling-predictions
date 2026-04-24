"""Right column — predicted next-24 hours from the two-stage (ML -> LLM) pipeline.

Also exposes an expander with the ML stage's top-K per step — useful for the "off-by-1"
investigation the user wants to do.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .actual_panel import render_activities_table
from .data_access import get_next_day_activities, get_ops_summary


def _render_ops_table(ops: list[dict], duration_tol: float):
    if not ops:
        st.warning("No operations in prediction response.")
        return

    pred_df = pd.DataFrame(ops)
    rename = {
        "phase": "Phase", "phase_step": "Phase_Step",
        "major_ops_code": "Major_Ops_Code", "operation": "Operation",
        "duration_hours": "Dur (hrs)",
    }
    pred_df = pred_df.rename(columns={k: v for k, v in rename.items() if k in pred_df.columns})

    col_config = {
        "Phase": st.column_config.TextColumn("Phase", width="small"),
        "Phase_Step": st.column_config.TextColumn("Phase_Step", width="small"),
        "Major_Ops_Code": st.column_config.TextColumn("Major_Ops_Code", width="small"),
        "Operation": st.column_config.TextColumn("Operation", width="small"),
        "Dur (hrs)": st.column_config.NumberColumn("Dur (hrs)", format="%.2f", width="small"),
    }
    st.dataframe(
        pred_df,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        height=min(35 * len(pred_df) + 38, 600),
    )

    if "Dur (hrs)" in pred_df.columns:
        total = float(pred_df["Dur (hrs)"].sum())
        status = st.success if abs(total - 24.0) <= duration_tol else st.warning
        status(f"**{len(ops)}** operations | Total duration: **{total:.2f}** hours")


def _render_ml_debug(ml_output):
    """Flatten the MLOutput into a top-K legal-tuple debug table.

    Each step expands into K rows, one per ranked hierarchy tuple. `prob` is
    the tuple's share of the legal-set probability mass (all legal tuples at
    a step sum to 1). `logP` is the raw joint log-prob — useful for spotting
    low-confidence steps even when top-K probs look similar. Duration is a
    per-step scalar, shown only on rank-1 so the table stays readable.
    """
    rows = []
    for s in ml_output.steps:
        for rank, tup in enumerate(s.topk_tuples, start=1):
            rows.append({
                "Step": s.step + 1,
                "Rank": rank,
                "Phase": tup.phase,
                "Phase_Step": tup.phase_step,
                "Major_Ops_Code": tup.major_ops_code,
                "Operation": tup.operation,
                "prob": round(tup.prob, 3),
                "logP": round(tup.log_prob, 3),
                "Dur (hrs)": round(s.duration_hours, 2) if rank == 1 else None,
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_actual_next_day(master_csv: str, well: str, date) -> None:
    """Render ground-truth operations for the day after `date`, for comparison."""
    activities, next_date = get_next_day_activities(master_csv, well, date)
    st.markdown("---")
    st.subheader(f"Actual Next Day — {next_date}")
    if activities.empty:
        st.info(
            f"No actual operations found for {next_date}. "
            "Either the well ended on the selected date or data is unavailable."
        )
        return
    summary = get_ops_summary(master_csv, well, next_date)
    if summary:
        st.info(f"**Ops Summary:** {summary}")
    render_activities_table(activities)


def render(
    cfg: dict,
    *,
    prediction_clicked: bool,
    result: dict | None,
    master_csv: str | None = None,
    well: str | None = None,
    date=None,
    compare_next_day: bool = False,
):
    st.subheader("Predicted Next Day")
    duration_tol = cfg["llm"]["duration_tolerance_hours"]

    if result is None:
        if not prediction_clicked:
            st.caption("Select a well and date, then click 'Predict Next Day'.")
        return

    llm_response = result["llm_response"]
    ml_output = result["ml_output"]
    user_message = result["user_message"]

    st.info(f"**Predicted Ops Summary:** {llm_response.get('ops_summary', 'N/A')}")
    if llm_response.get("reasoning"):
        st.caption(f"Reasoning: {llm_response['reasoning']}")

    _render_ops_table(llm_response.get("operations", []), duration_tol)

    with st.expander("ML stage — top-K per step (debug)"):
        _render_ml_debug(ml_output)

    with st.expander("Full context sent to Claude"):
        st.code(user_message, language="markdown")

    if compare_next_day and master_csv and well and date:
        _render_actual_next_day(master_csv, well, date)

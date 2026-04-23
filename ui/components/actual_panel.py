"""Left column — actual operations for the selected day."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .data_access import get_day_activities, get_ops_summary


def render_activities_table(activities: pd.DataFrame) -> None:
    """Render a day's activities as a formatted table plus a total-duration caption.

    Exposed so predicted_panel can reuse it for the 'actual next day' compare view.
    """
    if activities.empty:
        return

    act = activities.copy()
    act["Start"] = pd.to_datetime(act["Start_Time"]).dt.strftime("%H:%M")
    act["End"] = pd.to_datetime(act["End_Time"]).dt.strftime("%H:%M")
    act["Duration hours"] = act["Duration hours"].round(2)

    display = act[["Phase", "Phase_Step", "Major_Ops_Code", "Operation", "Start", "End", "Duration hours"]]
    display = display.rename(columns={"Duration hours": "Dur (hrs)"})

    col_config = {
        "Phase": st.column_config.TextColumn("Phase", width="small"),
        "Phase_Step": st.column_config.TextColumn("Phase_Step", width="small"),
        "Major_Ops_Code": st.column_config.TextColumn("Major_Ops_Code", width="small"),
        "Operation": st.column_config.TextColumn("Operation", width="small"),
        "Start": st.column_config.TextColumn("Start", width="small"),
        "End": st.column_config.TextColumn("End", width="small"),
        "Dur (hrs)": st.column_config.NumberColumn("Dur (hrs)", format="%.2f", width="small"),
    }

    st.dataframe(
        display,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        height=min(35 * len(display) + 38, 600),
    )

    total = activities["Duration hours"].sum()
    st.caption(f"**{len(activities)}** operations | Total duration: **{total:.2f}** hours")


def render(master_csv: str, well: str, date):
    st.subheader("Actual Operations")
    if not (well and date):
        return

    summary = get_ops_summary(master_csv, well, date)
    st.info(f"**Ops Summary:** {summary}")

    activities = get_day_activities(master_csv, well, date)
    if activities.empty:
        st.warning("No activities found for this well/date.")
        return

    render_activities_table(activities)

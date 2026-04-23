"""Streamlit-cached data access layer (ported from LLM-powered predictions/data_loader.py)."""

from __future__ import annotations

from datetime import date as _date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st


def get_data_path(cfg: dict) -> Path:
    return (Path(cfg["_repo_root"]) / cfg["data"]["master_csv"]).resolve()


@st.cache_data(show_spinner=False)
def load_master(master_csv: str) -> pd.DataFrame:
    df = pd.read_csv(master_csv)
    df["Report_Date"] = pd.to_datetime(df["Report_Date"]).dt.date
    return df


def get_wells(master_csv: str) -> list[str]:
    df = load_master(master_csv)
    return sorted(df["Well_Name"].dropna().unique().tolist())


def get_dates_for_well(master_csv: str, well: str):
    df = load_master(master_csv)
    return sorted(df.loc[df["Well_Name"] == well, "Report_Date"].dropna().unique())


def get_day_activities(master_csv: str, well: str, date) -> pd.DataFrame:
    df = load_master(master_csv)
    mask = (df["Well_Name"] == well) & (df["Report_Date"] == date)
    rows = df.loc[mask].copy()
    # 6am-to-6am ordering
    rows["_sort_key"] = rows["Start_Hour"].apply(lambda h: h - 6 if h >= 6 else h + 18)
    return rows.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)


def get_next_day_activities(master_csv: str, well: str, date) -> tuple[pd.DataFrame, _date]:
    """Activities for the calendar day immediately after `date`. Returns (df, next_date).

    Used by the UI's 'Compare to actual next day' mode to render a ground-truth
    panel alongside the prediction.
    """
    if isinstance(date, str):
        base = pd.to_datetime(date).date()
    elif isinstance(date, _date):
        base = date
    else:
        base = pd.to_datetime(date).date()
    next_date = base + timedelta(days=1)
    return get_day_activities(master_csv, well, next_date), next_date


def get_ops_summary(master_csv: str, well: str, date) -> str:
    rows = get_day_activities(master_csv, well, date)
    if rows.empty:
        return ""
    return str(rows["Ops_Summary"].iloc[0])


def get_job_report_start_date(master_csv: str, well: str, date) -> str:
    df = load_master(master_csv)
    mask = (df["Well_Name"] == well) & (df["Report_Date"] == date)
    rows = df.loc[mask, "Job_Report_Start_Date"]
    return str(rows.iloc[0]) if not rows.empty else str(date)

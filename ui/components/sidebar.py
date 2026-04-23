"""Sidebar: well / date / model / editable system prompt."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from .data_access import get_dates_for_well, get_wells

MODEL_OPTIONS = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-6"]


def render(cfg: dict, master_csv: str):
    llm_cfg = cfg["llm"]
    repo_root = Path(cfg["_repo_root"])
    prompt_path = repo_root / llm_cfg["prompts_dir"] / llm_cfg["system_prompt_file"]

    with st.sidebar:
        st.header("Well Selection")
        wells = get_wells(master_csv)
        well = st.selectbox("Well Name", wells)

        dates = get_dates_for_well(master_csv, well)
        date = st.selectbox("Report Date", dates)

        st.divider()
        st.header("Model Settings")

        api_key = st.text_input(
            "Anthropic API Key",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            type="password",
            help="Leave blank to use ANTHROPIC_API_KEY from .env",
        )
        default_idx = MODEL_OPTIONS.index(llm_cfg["model"]) if llm_cfg["model"] in MODEL_OPTIONS else 0
        model = st.selectbox("Claude Model", MODEL_OPTIONS, index=default_idx)

        st.divider()
        default_prompt = prompt_path.read_text(encoding="utf-8")
        with st.expander("System Prompt (editable)", expanded=False):
            system_prompt = st.text_area(
                "Prompt", value=default_prompt, height=400, label_visibility="collapsed",
            )

        compare_next_day = st.checkbox(
            "Compare to actual next day",
            value=False,
            help="After predicting, render actual operations from the day after the selected date "
                 "below the predicted plan for side-by-side comparison.",
        )

        predict_clicked = st.button("Predict Next Day", type="primary", use_container_width=True)

    return {
        "well": well,
        "date": date,
        "api_key": api_key or None,
        "model": model,
        "system_prompt": system_prompt,
        "compare_next_day": compare_next_day,
        "predict_clicked": predict_clicked,
    }

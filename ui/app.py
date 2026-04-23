"""Streamlit entry point. Thin — all logic lives in inference/, llm/."""

from __future__ import annotations

import sys
from pathlib import Path

# Make repo root importable when run via `streamlit run ui/app.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st                                                # noqa: E402

from config import load_config                                        # noqa: E402
from llm.client import APIKeyMissing, LLMError                        # noqa: E402
from llm.pipeline import predict_next_day                             # noqa: E402
from ui.components import actual_panel, predicted_panel, sidebar      # noqa: E402
from ui.components.data_access import (                                # noqa: E402
    get_data_path,
    get_day_activities,
    get_job_report_start_date,
    get_ops_summary,
    load_master,
)
from ui.components.styles import CSS                                  # noqa: E402

st.set_page_config(page_title="Next-Day Drilling Predictor (ML + LLM)", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.title("Next-Day Drilling Operations Predictor")

cfg = load_config()
master_csv = str(get_data_path(cfg))

# Sidebar
selection = sidebar.render(cfg, master_csv)
well = selection["well"]
date = selection["date"]

# Header
job_date = get_job_report_start_date(master_csv, well, date)
st.markdown(f"<h3 style='text-align:center; margin-bottom:0.2rem;'>{well}</h3>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align:center; color:gray; margin-top:0;'>Job Report Start: {job_date}</p>",
    unsafe_allow_html=True,
)

col_actual, col_pred = st.columns(2, gap="large")

# Left — actual
with col_actual:
    actual_panel.render(master_csv, well, date)

# Right — predicted (two-stage pipeline)
with col_pred:
    result_key = f"result__{well}__{date}"
    if selection["predict_clicked"]:
        with st.spinner("Running ML inference + LLM..."):
            try:
                master_df = load_master(master_csv)
                master_df_full = master_df.copy()
                master_df_full["Report_Date"] = master_df_full["Report_Date"].astype(str)
                result = predict_next_day(
                    well_name=well,
                    report_date=date,
                    get_day_activities_fn=lambda w, d: get_day_activities(master_csv, w, d),
                    get_ops_summary_fn=lambda w, d: get_ops_summary(master_csv, w, d),
                    master_df=master_df,
                    cfg=cfg,
                    api_key=selection["api_key"],
                    system_prompt_override=selection["system_prompt"],
                )
                st.session_state[result_key] = result
            except APIKeyMissing as e:
                st.error(str(e))
            except LLMError as e:
                st.error(f"LLM error: {e}")
            except FileNotFoundError as e:
                st.error(
                    f"Missing artifact: {e}. Run the preprocessing + training notebooks first."
                )
            except Exception as e:                                    # noqa: BLE001
                st.exception(e)

    predicted_panel.render(
        cfg,
        prediction_clicked=selection["predict_clicked"],
        result=st.session_state.get(result_key),
        master_csv=master_csv,
        well=well,
        date=date,
        compare_next_day=selection["compare_next_day"],
    )

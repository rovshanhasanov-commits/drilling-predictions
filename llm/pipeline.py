"""Two-stage orchestration: ML inference -> LLM call -> final ops JSON."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from config import load_config, resolve
from inference.contract import MLOutput
from inference.predict import predict as ml_predict

from .client import call_claude
from .context import assemble_user_message
from .similarity import find_similar_wells


def _load_prompt(cfg: dict) -> tuple[str, str]:
    pdir = Path(cfg["_repo_root"]) / cfg["llm"]["prompts_dir"]
    system_prompt = (pdir / cfg["llm"]["system_prompt_file"]).read_text(encoding="utf-8")
    constraints = (pdir / cfg["llm"]["constraints_file"]).read_text(encoding="utf-8")
    return system_prompt, constraints


def predict_next_day(
    well_name: str,
    report_date,
    get_day_activities_fn,
    get_ops_summary_fn,
    master_df: pd.DataFrame,
    cfg: dict | None = None,
    ml_output: MLOutput | None = None,
    api_key: str | None = None,
    system_prompt_override: str | None = None,
) -> dict:
    """Returns {'ml_output': MLOutput, 'llm_response': dict, 'user_message': str}.

    `master_df` is the full Master_Data_With_ID DataFrame, passed in so we don't
    read the 150k-row CSV twice in the same session.
    """
    cfg = cfg or load_config()
    llm_cfg = cfg["llm"]

    # Stage 1 — ML
    if ml_output is None:
        ml_output = ml_predict(well_name, report_date, cfg=cfg)

    # Similar wells for Context 2
    similar = find_similar_wells(
        target_well=well_name,
        target_date=report_date,
        df=master_df,
        get_day_activities_fn=get_day_activities_fn,
        top_k=llm_cfg["similar_wells_top_k"],
    )

    # Load prompt files
    system_prompt, constraints_md = _load_prompt(cfg)
    if system_prompt_override is not None:
        system_prompt = system_prompt_override

    user_message = assemble_user_message(
        well=well_name,
        date=report_date,
        get_day_activities_fn=get_day_activities_fn,
        get_ops_summary_fn=get_ops_summary_fn,
        similar=similar,
        constraints_md=constraints_md,
        ml=ml_output,
        ml_fields=llm_cfg["ml_fields_to_include"],
    )

    # Stage 2 — LLM
    llm_response = call_claude(
        system_prompt=system_prompt,
        user_message=user_message,
        model=llm_cfg["model"],
        max_tokens=llm_cfg["max_tokens"],
        api_key=api_key,
    )

    return {
        "ml_output": ml_output,
        "llm_response": llm_response,
        "user_message": user_message,
    }

"""Given (well, date), build the encoder input window using the already-preprocessed parquets.

We do not re-run preprocessing here. The parquets produced by
`preprocessing/run_preprocessing.py` already contain log1p'd + scaled + encoded rows;
inference just finds the right SEQUENCE_LENGTH slice and hands it to the model.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def _concat_all_splits(strategy_dir: Path) -> pd.DataFrame:
    parts = [
        pd.read_parquet(strategy_dir / "df_train.parquet"),
        pd.read_parquet(strategy_dir / "df_val.parquet"),
        pd.read_parquet(strategy_dir / "df_test.parquet"),
    ]
    df = pd.concat(parts, ignore_index=True)
    return df


def build_encoder_input(
    well_name: str,
    report_date,
    strategy_dir: Path,
    data_config: dict,
    seq_len: int,
) -> tuple[list, pd.DataFrame]:
    """Returns (encoder_input_list, window_df).

    encoder_input_list matches the order expected by the saved encoder_model:
      - for strategy 'embed_state':   [state_enc, numeric]
      - for strategy 'embed_separate':[phase_enc, phase_step_enc, ..., numeric]
      - for strategy 'dummies':       [numeric]
    window_df is the raw preprocessed slice (seq_len rows) for debugging / inference bookkeeping.
    """
    df = _concat_all_splits(Path(strategy_dir))

    # Normalize report_date to match preprocessing output (strings in parquet).
    target_date = pd.to_datetime(report_date).date() if not isinstance(report_date, date) else report_date
    df_date = pd.to_datetime(df["Report_Date"]).dt.date

    mask_well = df["Well_Name"] == well_name
    if not mask_well.any():
        raise ValueError(f"Well {well_name!r} not found in preprocessed data")

    mask = mask_well & (df_date <= target_date)
    well_rows = df.loc[mask].sort_values("row_id")
    if len(well_rows) < seq_len:
        raise ValueError(
            f"Only {len(well_rows)} rows available for ({well_name}, {target_date}); "
            f"need at least {seq_len} for the encoder window."
        )

    window = well_rows.iloc[-seq_len:].reset_index(drop=True)

    cat_input_cols = data_config["cat_input_cols"]
    numeric_cols = data_config["cont_cols"] + data_config["bin_cols"] + data_config.get("dummy_col_names", [])

    enc_inputs = []
    for c in cat_input_cols:
        enc_inputs.append(window[c].to_numpy(dtype=np.int32)[None, :])
    enc_inputs.append(window[numeric_cols].to_numpy(dtype=np.float32)[None, :, :])

    return enc_inputs, window

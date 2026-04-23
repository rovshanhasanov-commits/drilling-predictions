"""Defensive, idempotent master-side data cleanup.

Mirrors the logic in ../../data_prep.ipynb so the pipeline is reproducible without
depending on whoever last ran that notebook. Every step is guarded with
`if col in df.columns` and a no-op on already-clean inputs.

Runs between join.py and features.py on the joined DataFrame.
"""

import numpy as np
import pandas as pd


RIG_CONTRACTOR_MAP = {
    "H & P": "H&P",
    "Nabors": "NABORS",
}

OPERATOR_MAP = {
    "XTO Energy": "XTO ENERGY",
}

PHASE_STEP_MAP = {
    "CASE": "CASING",
    "CMT": "CEMENT",
    "POST DRIL": "POST DRL",
    "CASE & CMT": "CASING & CEMENT",
}


def _parse_date_column(s: pd.Series) -> pd.Series:
    """Parse a date column with mixed formats.

    Strategy (matches data_prep.ipynb): try string parse first; for values that
    fail, try ms-since-epoch, then s-since-epoch. Leaves unparseable values as NaT.
    """
    parsed = pd.to_datetime(s, errors="coerce")
    if parsed.isna().any():
        numeric = pd.to_numeric(s, errors="coerce")
        ms = pd.to_datetime(numeric, unit="ms", errors="coerce")
        sec = pd.to_datetime(numeric, unit="s", errors="coerce")
        parsed = parsed.fillna(ms).fillna(sec)
    return parsed


def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")


def clean(
    df: pd.DataFrame,
    unplanned_ops: list[str] | None = None,
    unplanned_token: str = "Unplanned",
) -> pd.DataFrame:
    """Apply all notebook-style cleanup steps to the joined DataFrame.

    Idempotent: safe to run on inputs that have already been partially cleaned.

    Args:
        unplanned_ops: Operation-string list to merge under `unplanned_token` and
            loss-mask on the operation / major_ops_code / duration heads. Config-driven;
            see preprocessing.unplanned_operations in pipeline.yaml.
        unplanned_token: The replacement string for every entry in `unplanned_ops`.
            Becomes a real class at target-encoder fit time, gets its own embedding
            slot, but never receives positive gradient on the op head (mask=0).
    """
    df = df.copy()

    # 1. Parse Job_Report_Start_Date (mixed string / ms / s epoch formats).
    if "Job_Report_Start_Date" in df.columns:
        df["Job_Report_Start_Date"] = _parse_date_column(df["Job_Report_Start_Date"])

    # 2. Derive Report_Date / Start_Hour / End_Hour only if missing.
    _ensure_datetime(df, "Start_Time")
    _ensure_datetime(df, "End_Time")

    if "Report_Date" not in df.columns and "Job_Report_Start_Date" in df.columns:
        df["Report_Date"] = df["Job_Report_Start_Date"].dt.date

    if "Start_Hour" not in df.columns and "Start_Time" in df.columns:
        df["Start_Hour"] = df["Start_Time"].dt.hour + df["Start_Time"].dt.minute / 60

    if "End_Hour" not in df.columns and "End_Time" in df.columns:
        df["End_Hour"] = df["End_Time"].dt.hour + df["End_Time"].dt.minute / 60

    # 3. Rename Planned_Phase_Duration → Planned_Phase_Duration_Hours if the
    #    pre-rename name snuck in.
    if "Planned_Phase_Duration" in df.columns and "Planned_Phase_Duration_Hours" not in df.columns:
        df = df.rename(columns={"Planned_Phase_Duration": "Planned_Phase_Duration_Hours"})

    # 4. Standardize categorical values (no-op on already-clean strings).
    if "Rig_Contractor" in df.columns:
        df["Rig_Contractor"] = df["Rig_Contractor"].replace(RIG_CONTRACTOR_MAP)
    if "Operator" in df.columns:
        df["Operator"] = df["Operator"].replace(OPERATOR_MAP)
    if "Phase_Step" in df.columns:
        df["Phase_Step"] = df["Phase_Step"].replace(PHASE_STEP_MAP)

    # 5. Handle NaN in Major_Ops_Code / Operation. Ported from the memo at
    #    misc/Improvement opportunities/nan_handling_memo.md to replace the
    #    global-mode fallback at run_preprocessing.py:107, which was relabeling
    #    ~12% of Operation rows to "DRL" regardless of MOC.
    if {"Operation", "Major_Ops_Code", "Well_Name", "Phase_Step"}.issubset(df.columns):
        # 5a. Drop wells where every row has a NaN Operation — no within-well
        #     signal to fill from, so any fill would be 100% synthetic.
        nan_frac = df["Operation"].isna().groupby(df["Well_Name"]).mean()
        wells_to_drop = nan_frac[nan_frac == 1.0].index
        n_rows_before = len(df)
        df = df[~df["Well_Name"].isin(wells_to_drop)].copy()
        n_dropped_rows = n_rows_before - len(df)
        print(f"  Dropped {len(wells_to_drop)} fully-unlabeled wells "
              f"({n_dropped_rows:,} rows, "
              f"{n_dropped_rows / max(n_rows_before, 1):.1%})")

        # 5b. Normalize strings + semantic merges. astype(str) turns real NaN
        #     into the string "nan", which would survive fillna — map it back.
        df["Major_Ops_Code"] = (df["Major_Ops_Code"].astype(str).str.strip()
                                .replace({"DRLG": "DRILL",
                                          "CMT": "CEMENT",
                                          "BOP": "BOPE",
                                          "nan": None}))
        df["Operation"] = (df["Operation"].astype(str).str.strip()
                           .replace({"nan": None}))

        # 5c. Fill remaining NaN Major_Ops_Code via Phase_Step-conditional mode.
        moc_nan_before = int(df["Major_Ops_Code"].isna().sum())
        if moc_nan_before > 0:
            moc_mode_by_step = (df.dropna(subset=["Major_Ops_Code"])
                                  .groupby("Phase_Step")["Major_Ops_Code"]
                                  .agg(lambda s: s.mode().iloc[0]))
            df["Major_Ops_Code"] = df["Major_Ops_Code"].fillna(
                df["Phase_Step"].map(moc_mode_by_step)
            )

        # 5d. Fill remaining NaN Operation via (Phase_Step, MOC)-conditional mode.
        op_nan_before = int(df["Operation"].isna().sum())
        if op_nan_before > 0:
            op_mode_by_step_moc = (df.dropna(subset=["Operation"])
                                     .groupby(["Phase_Step", "Major_Ops_Code"])["Operation"]
                                     .agg(lambda s: s.mode().iloc[0]))
            fill_op = pd.Series(
                pd.MultiIndex.from_arrays(
                    [df["Phase_Step"], df["Major_Ops_Code"]]
                ).map(op_mode_by_step_moc),
                index=df.index,
            )
            df["Operation"] = df["Operation"].fillna(fill_op)

        # 5e. Residual-NaN diagnostic.
        moc_residual = int(df["Major_Ops_Code"].isna().sum())
        op_residual = int(df["Operation"].isna().sum())
        print(f"  MOC NaN filled: {moc_nan_before} -> {moc_residual}; "
              f"Operation NaN filled: {op_nan_before} -> {op_residual}")
        if op_residual > 0:
            print("  Residual NaN Operation breakdown by (Phase_Step, Major_Ops_Code):")
            print(df[df["Operation"].isna()]
                  .groupby(["Phase_Step", "Major_Ops_Code"], dropna=False).size()
                  .sort_values(ascending=False)
                  .to_string())

        # 5f. Per-head loss-mask flags. Three independent weights (op / moc / dur)
        #     so the windowing layer can zero the gradient on each head separately:
        #       op_label_real  = 0 if Operation was NaN (residual) or in unplanned list
        #       moc_label_real = 0 if Operation in unplanned list (NPT-style events)
        #       dur_label_real = 0 if Operation was NaN or in unplanned list
        #     Rows with any flag at 0 still contribute encoder context; only the
        #     masked head's gradient is zeroed. Phase_Step is never masked.
        was_nan = df["Operation"].isna()
        is_unpl = df["Operation"].isin(unplanned_ops or [])
        df["op_label_real"]  = (~(was_nan | is_unpl)).astype("float32")
        df["moc_label_real"] = (~is_unpl).astype("float32")
        df["dur_label_real"] = (~(was_nan | is_unpl)).astype("float32")
        print(f"  op_label_real:  {int((df['op_label_real']  == 0).sum())} rows "
              f"masked (NaN: {int(was_nan.sum())}, unplanned: {int(is_unpl.sum())})")
        print(f"  moc_label_real: {int((df['moc_label_real'] == 0).sum())} rows "
              f"masked (unplanned: {int(is_unpl.sum())})")
        print(f"  dur_label_real: {int((df['dur_label_real'] == 0).sum())} rows "
              f"masked (NaN: {int(was_nan.sum())}, unplanned: {int(is_unpl.sum())})")

        # 5g. Merge the 17 unplanned ops under a single `unplanned_token` class.
        #     Must run before target-encoder fit so the token becomes a real class
        #     with a stable embedding slot. Masking (above) keeps the op head from
        #     learning to predict it — same dead-output pattern as UNK.
        if is_unpl.any():
            df.loc[is_unpl, "Operation"] = unplanned_token
            print(f"  Operation: {int(is_unpl.sum())} rows renamed to "
                  f"'{unplanned_token}' (from {len(unplanned_ops or [])} source strings)")

    # 6. Sort chronologically within each well so the neighbor-fill below uses
    #    the correct adjacent rows. Matches the notebook.
    sort_cols = [c for c in ("Well_Name", "Start_Time") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # 7. Well-neighbor null-fill for depth columns.
    depth_before = int(df.get("DepthStart_ft", pd.Series(dtype=float)).isna().sum()) + \
                   int(df.get("DepthEnd_ft",   pd.Series(dtype=float)).isna().sum())

    if {"DepthStart_ft", "DepthEnd_ft", "Well_Name"}.issubset(df.columns):
        df["DepthStart_ft"] = df["DepthStart_ft"].fillna(
            df.groupby("Well_Name")["DepthEnd_ft"].shift(1)
        )
        df["DepthEnd_ft"] = df["DepthEnd_ft"].fillna(
            df.groupby("Well_Name")["DepthStart_ft"].shift(-1)
        )

    depth_after = int(df.get("DepthStart_ft", pd.Series(dtype=float)).isna().sum()) + \
                  int(df.get("DepthEnd_ft",   pd.Series(dtype=float)).isna().sum())

    print(f"Cleaned: {len(df):,} rows; depth nulls filled: {depth_before} -> {depth_after}")
    return df

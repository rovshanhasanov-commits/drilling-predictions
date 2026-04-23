"""Feature engineering + column typing.

Design note: no `*_next` / target-shift columns are produced here. Target shift
happens in training/data.py (windowing). Preprocessing emits current-step state only.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CAT_COLS = ["phase", "phase_step", "major_ops_code", "operation"]
STANDARD_RENAME = {
    "Phase": "phase",
    "Phase_Step": "phase_step",
    "Major_Ops_Code": "major_ops_code",
    "Operation": "operation",
}
LOG_COLS = ["Duration hours", "Planned_Phase_Duration_Hours"]


def engineer(
    df: pd.DataFrame,
    operator_col: str,
    rig_col: str,
    drop_after_derivation: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Common feature engineering shared across all encoding strategies.

    Returns (df, bin_cols) where bin_cols is the list of auto-detected binary columns.

    `drop_after_derivation` is a list of column names that are useful only as
    inputs to derived features above (e.g., Target_Depth → depth_progress) and
    should be removed before the model sees them. Dropped at the end.
    """
    df = df.rename(columns=STANDARD_RENAME).copy()

    # One-hot encode operator + rig before numeric detection.
    for col in (operator_col, rig_col):
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False).astype(int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # Concatenated state string (used by the embed_state strategy).
    df["state"] = df[CAT_COLS].apply(
        lambda r: str(dict(zip(["phase", "step", "major_ops_code", "operation"], r))), axis=1
    )

    # Fill numerical inf/NaN with 0 — required for StandardScaler and for models.
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Log-transform skewed duration columns (Duration hours is needed for the
    # regression target too — training/data.py log1p's it separately when building targets).
    for col in LOG_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # Derived spatial features.
    df["depth_change"] = (df["DepthEnd_ft"] - df["DepthStart_ft"]).fillna(0)
    df["op_sequence_number"] = df.groupby("Well_Name").cumcount()
    df["phase_op_index"] = df.groupby(["Well_Name", "phase"]).cumcount()
    df["depth_progress"] = (df["DepthStart_ft"] / df["Target_Depth"].clip(lower=1)).clip(0, 1)

    # Cumulative progress within the current phase_step block (well-scoped).
    # Baseline = DepthStart_ft at the first row of the block (when the crew
    # entered this phase_step). Numerator uses DepthEnd_ft so each row's
    # progress reflects depth reached at the *end* of that operation, not its
    # start. Denominator = planned extent from baseline to Planned_End_Depth
    # known at block start. 0 at block start, 1 at planned end, >1 = overshoot
    # (capped at 2x).
    prev_step = df.groupby("Well_Name")["phase_step"].shift()
    is_new_block = (df["phase_step"] != prev_step) | prev_step.isna()
    step_block_id = is_new_block.groupby(df["Well_Name"]).cumsum()

    block_key = [df["Well_Name"], step_block_id]
    block_start_depth = df["DepthStart_ft"].groupby(block_key).transform("first")
    block_planned_end = df["Planned_End_Depth"].groupby(block_key).transform("first")

    step_extent = block_planned_end - block_start_depth
    step_extent = step_extent.where(step_extent > 0, other=np.nan)
    df["depth_progress_in_step"] = (
        (df["DepthEnd_ft"] - block_start_depth) / step_extent
    ).clip(lower=0, upper=2).fillna(0)

    # Auto-detect binary columns so StandardScaler can skip them.
    # *_label_real flags are excluded because they're per-row Keras sample_weights
    # on individual heads (op / moc / dur), not model input features.
    exclude = set(CAT_COLS) | {"state", "Well_Name", "row_id", "Report_Date",
                                "Job_Report_Start_Date", "Start_Time", "End_Time",
                                "Ops_Summary",
                                "op_label_real", "moc_label_real", "dur_label_real"}
    bin_cols = []
    for c in df.columns:
        if c in exclude or df[c].dtype == object:
            continue
        vals = set(df[c].dropna().unique())
        if vals.issubset({True, False, 0, 1, 0.0, 1.0}) and len(vals) <= 2:
            df[c] = df[c].astype(int)
            bin_cols.append(c)

    # Drop columns that served only as derivation inputs (e.g., Target_Depth
    # → depth_progress). These never reach the model as predictors.
    if drop_after_derivation:
        to_drop = [c for c in drop_after_derivation if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            bin_cols = [c for c in bin_cols if c not in to_drop]

    return df, bin_cols


def fit_target_encoders(df_train: pd.DataFrame, eoo_token: str, unk_token: str) -> tuple[dict, dict]:
    """Fit one LabelEncoder per hierarchy level. Includes the EOO + UNK sentinels.

    - EOO: emitted by training windowing when a target step falls past a well's last row.
    - UNK: fallback for NaN / unseen-in-train values encountered at val/test time. The
      old behavior here was to relabel those as the global train mode (e.g. `DRL` for
      operation), which manufactured spurious mode-class signal. UNK makes the uncertainty
      explicit in both training data and prediction outputs.

    Fitting both here guarantees stable class ids across train/val/test.
    """
    target_encoders = {}
    n_classes = {}
    for col in CAT_COLS:
        classes = sorted(df_train[col].dropna().unique().tolist())
        for sentinel in (eoo_token, unk_token):
            if sentinel not in classes:
                classes.append(sentinel)
        le = LabelEncoder()
        le.fit(classes)
        target_encoders[col] = le
        n_classes[col] = len(le.classes_)
        print(f"  {col}: {n_classes[col]} classes (incl. EOO + UNK)")
    return target_encoders, n_classes


def detect_continuous_cols(df_train: pd.DataFrame, bin_cols: list[str]) -> list[str]:
    """Numeric columns that are neither binary nor identifiers nor encoded-categoricals."""
    non_predictor = set(bin_cols) | set(CAT_COLS) | {
        "state", "Well_Name", "row_id",
        "Report_Date", "Job_Report_Start_Date", "Start_Time", "End_Time", "Ops_Summary",
        "op_label_real", "moc_label_real", "dur_label_real",
    }
    non_predictor |= {c for c in df_train.columns if c.endswith("_enc")}
    return [
        c for c in df_train.select_dtypes(include="number").columns
        if c not in non_predictor
    ]

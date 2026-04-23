"""Strategy-specific categorical encoding + continuous feature scaling.

Three strategies mirror the previous pipeline:
  - dummies:        one-hot of the 4 hierarchy levels
  - embed_separate: LabelEncoder per level (for 4 separate embedding tables)
  - embed_state:    LabelEncoder on the concatenated state string (one embedding table)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .features import CAT_COLS


def _safe_encode(series: pd.Series, le: LabelEncoder, fallback: str) -> np.ndarray:
    class_to_idx = {c: i for i, c in enumerate(le.classes_)}
    fb_idx = class_to_idx[fallback]
    return series.apply(lambda x: class_to_idx.get(x, fb_idx)).values


def process_dummies(df_train, df_val, df_test):
    td = pd.get_dummies(df_train[CAT_COLS], prefix=CAT_COLS).astype(int)
    vd = pd.get_dummies(df_val[CAT_COLS], prefix=CAT_COLS).reindex(columns=td.columns, fill_value=0).astype(int)
    xd = pd.get_dummies(df_test[CAT_COLS], prefix=CAT_COLS).reindex(columns=td.columns, fill_value=0).astype(int)
    dummy_names = list(td.columns)

    df_train = pd.concat([df_train.drop(columns=CAT_COLS + ["state"]), td], axis=1)
    df_val = pd.concat([df_val.drop(columns=CAT_COLS + ["state"]), vd], axis=1)
    df_test = pd.concat([df_test.drop(columns=CAT_COLS + ["state"]), xd], axis=1)

    print(f"  dummies: {len(dummy_names)} columns")
    return df_train, df_val, df_test, {}, [], dummy_names


def process_embed_separate(df_train, df_val, df_test):
    cat_encoders = {}
    cat_input_cols = []
    for col in CAT_COLS:
        le = LabelEncoder()
        le.fit(df_train[col])
        fb = df_train[col].value_counts().idxmax()
        enc = col + "_enc"
        df_train[enc] = le.transform(df_train[col])
        df_val[enc] = _safe_encode(df_val[col], le, fb)
        df_test[enc] = _safe_encode(df_test[col], le, fb)
        cat_encoders[col] = le
        cat_input_cols.append(enc)
        print(f"  {col}: {len(le.classes_)} classes (fallback={fb!r})")

    for d in (df_train, df_val, df_test):
        d.drop(columns=CAT_COLS + ["state"], inplace=True)
    return df_train, df_val, df_test, cat_encoders, cat_input_cols, []


def process_embed_state(df_train, df_val, df_test):
    le = LabelEncoder()
    le.fit(df_train["state"])
    fb = df_train["state"].value_counts().idxmax()
    df_train["state_enc"] = le.transform(df_train["state"])
    df_val["state_enc"] = _safe_encode(df_val["state"], le, fb)
    df_test["state_enc"] = _safe_encode(df_test["state"], le, fb)
    print(f"  embed_state: {len(le.classes_)} unique states")

    for d in (df_train, df_val, df_test):
        d.drop(columns=CAT_COLS + ["state"], inplace=True)
    return df_train, df_val, df_test, {"state": le}, ["state_enc"], []


STRATEGY_FN = {
    "dummies": process_dummies,
    "embed_separate": process_embed_separate,
    "embed_state": process_embed_state,
}


DUR_COL = "Duration hours"


def scale_features(df_train, df_val, df_test, cont_cols: list[str]) -> tuple[StandardScaler, StandardScaler]:
    """Fit a feature scaler on cont_cols (excluding DUR_COL) and a separate duration scaler.

    DUR_COL carries its own scaler because training windowing reads it as the duration TARGET
    at future step t+k. Having an isolated scaler lets inference inverse-transform the
    predicted duration back to hours without carrying the feature scaler's vector.
    """
    cont_no_dur = [c for c in cont_cols if c != DUR_COL]
    feat_scaler = StandardScaler()
    df_train[cont_no_dur] = feat_scaler.fit_transform(df_train[cont_no_dur])
    df_val[cont_no_dur] = feat_scaler.transform(df_val[cont_no_dur])
    df_test[cont_no_dur] = feat_scaler.transform(df_test[cont_no_dur])

    dur_scaler = StandardScaler()
    df_train[[DUR_COL]] = dur_scaler.fit_transform(df_train[[DUR_COL]])
    df_val[[DUR_COL]] = dur_scaler.transform(df_val[[DUR_COL]])
    df_test[[DUR_COL]] = dur_scaler.transform(df_test[[DUR_COL]])

    return feat_scaler, dur_scaler

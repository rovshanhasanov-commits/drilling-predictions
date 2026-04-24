"""Sequence building. This is where the SINGLE target shift lives.

For each well, slide an encoder window of length SEQUENCE_LENGTH across its rows.
At each window position i, the targets are the next N_FUTURE rows starting at i + SEQUENCE_LENGTH.
When a target row falls past the well's last row, the sentinel 'End of Operations' class
is emitted (with duration = 0 in transformed space).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HIERARCHY = ["phase", "phase_step", "major_ops_code", "operation"]

# Per-head loss-mask sources. preprocessing/clean.py emits these as float32 flags
# (1.0 = trust the label on this row's corresponding head; 0.0 = mask).
# Used to build the Keras `sample_weight` dict windowed over the K-horizon decoder.
SAMPLE_WEIGHT_MAP: list[tuple[str, str]] = [
    ("op_label_real",  "operation"),
    ("moc_label_real", "major_ops_code"),
    ("dur_label_real", "duration"),
    # `duration_bin` shares the same row-level mask as the regression head:
    # Unplanned / NaN-op rows train neither head. Both mappings stay
    # unconditionally — _expand_sample_weight skips heads not in y_fit, so the
    # unused mapping is harmless when only one duration head is active.
    ("dur_label_real", "duration_bin"),
]


def load_strategy_data(strategy_dir: Path) -> dict:
    strategy_dir = Path(strategy_dir)
    df_train = pd.read_parquet(strategy_dir / "df_train.parquet")
    df_val = pd.read_parquet(strategy_dir / "df_val.parquet")
    df_test = pd.read_parquet(strategy_dir / "df_test.parquet")
    with open(strategy_dir / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open(strategy_dir / "config.json", "r") as f:
        config = json.load(f)
    return {
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "encoders": encoders, "config": config,
    }


def compute_numeric_cols(config: dict) -> list[str]:
    """Numeric columns = cont_cols + bin_cols + (dummy_col_names if any)."""
    return config["cont_cols"] + config["bin_cols"] + config.get("dummy_col_names", [])


def build_seq2seq_sequences(
    df: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    target_cols: list[str],
    seq_len: int,
    n_future: int,
    eoo_encoded: dict,
    eoo_duration: float,
    label: str = "",
):
    """Slide windows within each well. Emits encoder inputs + per-target future labels.

    Args:
        cat_cols: strategy-dependent encoder categorical inputs (e.g. ['state_enc'])
        num_cols: all numeric encoder inputs (continuous + binary + dummies)
        target_cols: the four '{level}_target_enc' columns
        eoo_encoded: {target_col: integer_id_of_EOO_in_target_encoder}
        eoo_duration: the transformed duration value for an EOO step (from dur_scaler)

    Returns dict of arrays ready for model.fit.
    """
    cat_store = {c: [] for c in cat_cols}
    num_store = []
    y_store = {c: [] for c in target_cols}
    y_dur = []
    wells_out, idx_out = [], []
    skipped = 0

    # Per-row sample weights flow through the same K-horizon shift as the
    # targets. preprocessing/clean.py emits `op_label_real`, `moc_label_real`,
    # `dur_label_real` — detected lazily so older parquets (pre-masking)
    # continue to work.
    active_weights = [(src, head) for src, head in SAMPLE_WEIGHT_MAP if src in df.columns]
    w_stores: dict[str, list] = {head: [] for _, head in active_weights}

    for well, grp in df.groupby("Well_Name", sort=False):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        if n < seq_len + 1:
            skipped += 1
            continue

        cat_arrays = {c: grp[c].to_numpy(dtype=np.int32) for c in cat_cols}
        num_array = grp[num_cols].to_numpy(dtype=np.float32)
        tgt_arrays = {c: grp[c].to_numpy(dtype=np.int32) for c in target_cols}
        dur_array = grp["Duration hours"].to_numpy(dtype=np.float32)
        w_arrays = {head: grp[src].to_numpy(dtype=np.float32) for src, head in active_weights}

        # Slide windows that require at least one real target (t == i+seq_len must be in range).
        for i in range(n - seq_len):
            for c in cat_cols:
                cat_store[c].append(cat_arrays[c][i:i + seq_len])
            num_store.append(num_array[i:i + seq_len])

            for c in target_cols:
                future = np.full(n_future, eoo_encoded[c], dtype=np.int32)
                dur = np.full(n_future, eoo_duration, dtype=np.float32)
                start = i + seq_len
                end = min(start + n_future, n)
                k = end - start                                  # number of real target rows
                if k > 0:
                    future[:k] = tgt_arrays[c][start:end]
                y_store[c].append(future)

            # Duration built once per window.
            future_dur = np.full(n_future, eoo_duration, dtype=np.float32)
            start = i + seq_len
            end = min(start + n_future, n)
            k = end - start
            if k > 0:
                future_dur[:k] = dur_array[start:end]
            y_dur.append(future_dur)

            # EOO padding counts as a real target, so pad weights with 1.0.
            for head, arr in w_arrays.items():
                w = np.ones(n_future, dtype=np.float32)
                if k > 0:
                    w[:k] = arr[start:end]
                w_stores[head].append(w)

            wells_out.append(well)
            idx_out.append(i + seq_len)

    if skipped:
        print(f"  [{label}] skipped {skipped} wells shorter than seq_len ({seq_len}+1)")

    out = {
        "cat": {c: np.asarray(v) for c, v in cat_store.items()},
        "num": np.asarray(num_store, dtype=np.float32),
        "y":   {c: np.asarray(v) for c, v in y_store.items()},
        "y_dur": np.asarray(y_dur, dtype=np.float32),
        "wells": wells_out,
        "start_idx": idx_out,
    }
    if w_stores:
        out["sample_weight"] = {
            head: np.asarray(v, dtype=np.float32) for head, v in w_stores.items()
        }
    return out


def eoo_encoded_ids(target_encoders: dict, eoo_token: str) -> dict:
    """Return {'phase_target_enc': int, ...} for EOO in each level.

    When the bin encoder is present (`"duration_bin" in target_encoders`), also
    emits `duration_bin_target_enc` -> id of the bin "EOO" sentinel class. The
    bin head's EOO label is the literal string "EOO", not the 4-hierarchy-head
    `eoo_token` ("End of Operations") — see preprocessing/bins.py.
    """
    out = {}
    for level in HIERARCHY:
        classes = list(target_encoders[level].classes_)
        out[f"{level}_target_enc"] = classes.index(eoo_token)
    if "duration_bin" in target_encoders:
        bin_classes = list(target_encoders["duration_bin"].classes_)
        out["duration_bin_target_enc"] = bin_classes.index("EOO")
    return out


def eoo_duration_value(dur_scaler) -> float:
    """Return the transformed duration value for EOO (raw 0 hours).

    The raw column was log1p'd+scaled; a 'zero-hours' target is the scaler's
    transform of log1p(0) = 0.
    """
    zero_log = np.array([[0.0]], dtype=np.float32)
    return float(dur_scaler.transform(zero_log)[0, 0])


def make_decoder_inputs(y: dict, n_classes: dict, targets: list[str], predict_duration: bool, y_dur=None):
    """Teacher-forcing decoder inputs: shift targets right by 1, position 0 is a start token.

    Start token for each categorical head = n_classes[level] (i.e. one beyond the vocabulary).
    Start token for duration = 0.
    """
    dec = {}
    for level in targets:
        enc_col = f"{level}_target_enc"
        arr = y[enc_col]                       # (n, N_future)
        shifted = np.full_like(arr, n_classes[level])
        shifted[:, 1:] = arr[:, :-1]
        dec[level] = shifted
    if predict_duration and y_dur is not None:
        shifted = np.zeros_like(y_dur)
        shifted[:, 1:] = y_dur[:, :-1]
        dec["duration"] = shifted
    return dec


def mix_scheduled_sampling(dec_pure: dict, preds: dict, ss_rate: float,
                           targets: list[str], predict_duration: bool) -> dict:
    """Replace decoder input positions 1..N-1 with model's own prior prediction with prob ss_rate."""
    if ss_rate == 0.0:
        return dec_pure
    out = {}
    some = next(iter(dec_pure.values()))
    n_samples, n_future = some.shape
    for level in targets:
        gt = dec_pure[level].copy()
        pred = np.argmax(preds[level], axis=-1)
        mask = np.random.random((n_samples, n_future - 1)) < ss_rate
        mixed = gt.copy()
        mixed[:, 1:] = np.where(mask, pred[:, :-1], gt[:, 1:])
        out[level] = mixed
    if predict_duration:
        gt = dec_pure["duration"].copy()
        pred = preds["duration"]
        mask = np.random.random((n_samples, n_future - 1)) < ss_rate
        mixed = gt.copy()
        mixed[:, 1:] = np.where(mask, pred[:, :-1], gt[:, 1:])
        out["duration"] = mixed
    return out

"""Load the saved model bundle produced by training/save_artifacts.save_bundle().

Also (optionally) rebuilds the constraint decoder's legal-tuple table from the
cleaned parquet splits at startup. Chosen over a pickled table so the table is
always fresh relative to whatever preprocessing ran last — no bundle-staleness
class of bugs. Cost is a few seconds of parquet reads on first load.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def load_bundle(
    model_dir: str | Path,
    strategy_dir: str | Path | None = None,
    include_bins: bool = False,
):
    """Returns a dict with the model graphs, encoders, configs, and (optionally)
    a legal-tuple table for constrained decoding.

    Keys:
        encoder_model, decoder_step_model, training_model,
        encoders, data_config, model_config,
        legal_tuples   -- np.ndarray (num_legal, 4 or 5) int32, or None if
                          strategy_dir not passed or the parquets are unavailable.

    `include_bins=True` rebuilds the legal-tuple table with the bin head as the
    5th column (requires the bundle to have been trained with the bin head; we
    assert this against the bundle's `target_encoders` to fail loudly).
    """
    import tensorflow as tf  # local import so CLI tools that don't need TF don't pay startup

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_dir}")

    # safe_mode=False allows deserializing the Lambda layers used in the model
    # architecture (training/model.py). Only safe because we produced these files ourselves.
    encoder_model      = tf.keras.models.load_model(model_dir / "encoder_model.keras", compile=False, safe_mode=False)
    decoder_step_model = tf.keras.models.load_model(model_dir / "decoder_step_model.keras", compile=False, safe_mode=False)
    training_model     = tf.keras.models.load_model(model_dir / "training_model.keras", compile=False, safe_mode=False)

    with open(model_dir / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    with open(model_dir / "data_config.json", "r") as f:
        data_config = json.load(f)

    with open(model_dir / "model_config.json", "r") as f:
        model_config = json.load(f)

    # Bundle-self-consistency check: model_config carries bin_edges (injected
    # by training.save_artifacts.save_bundle from encoders.pkl at training
    # time). If the two disagree, someone hand-edited one of the files —
    # silent mismatch would mean the model's softmax classes correspond to a
    # different bin definition than what `bin_centers` / `bin_labels` claim.
    cfg_edges = model_config.get("bin_edges")
    enc_edges = encoders.get("bin_edges")
    if cfg_edges is not None and enc_edges is not None and list(cfg_edges) != list(enc_edges):
        raise ValueError(
            f"Stale bundle: model_config.json bin_edges {cfg_edges} != "
            f"encoders.pkl bin_edges {enc_edges}. Retrain or restore one of "
            f"the files to match."
        )

    if include_bins and "duration_bin" not in encoders.get("target_encoders", {}):
        raise ValueError(
            "include_duration_bins_in_hierarchy=true but bundle was trained "
            "without the bin head (encoders.pkl has no `duration_bin` encoder). "
            "Retrain with duration_bin_next in target_variables, or set the "
            "flag to false."
        )

    legal_tuples = None
    if strategy_dir is not None:
        legal_tuples = _load_legal_tuples(strategy_dir, encoders, include_bins=include_bins)

    return {
        "encoder_model":      encoder_model,
        "decoder_step_model": decoder_step_model,
        "training_model":     training_model,
        "encoders":           encoders,
        "data_config":        data_config,
        "model_config":       model_config,
        "legal_tuples":       legal_tuples,
    }


def _load_legal_tuples(
    strategy_dir: str | Path,
    encoders: dict,
    include_bins: bool = False,
) -> np.ndarray | None:
    """Rebuild L from the cleaned parquet splits. Returns None if parquets are missing."""
    import pandas as pd

    # Local import avoids a TF -> training -> constraints chain when inference is
    # only importing this module to resolve bundle paths.
    from training.constraints import build_legal_tuples, summarize_legal_tuples

    strategy_dir = Path(strategy_dir)
    parquet_files = {
        "df_train": strategy_dir / "df_train.parquet",
        "df_val":   strategy_dir / "df_val.parquet",
        "df_test":  strategy_dir / "df_test.parquet",
    }
    missing = [str(p) for p in parquet_files.values() if not p.exists()]
    if missing:
        print(f"  [legal_tuples] parquet(s) missing, skipping: {missing}")
        return None

    dfs = {k: pd.read_parquet(v) for k, v in parquet_files.items()}
    L = build_legal_tuples(
        dfs["df_train"], dfs["df_val"], dfs["df_test"],
        encoders["target_encoders"],
        eoo_token=encoders.get("eoo_token", "End of Operations"),
        include_bins=include_bins,
    )
    stats = summarize_legal_tuples(L, encoders["target_encoders"])
    suffix = " (5D, bin head joins joint argmax)" if include_bins else ""
    print(f"  [legal_tuples]{suffix} {stats}")
    return L

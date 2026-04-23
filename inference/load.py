"""Load the saved model bundle produced by training/save_artifacts.save_bundle()."""

from __future__ import annotations

import json
import pickle
from pathlib import Path


def load_bundle(model_dir: str | Path):
    """Returns a dict: {encoder_model, decoder_step_model, training_model, encoders, data_config, model_config}."""
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

    return {
        "encoder_model":      encoder_model,
        "decoder_step_model": decoder_step_model,
        "training_model":     training_model,
        "encoders":           encoders,
        "data_config":        data_config,
        "model_config":       model_config,
    }

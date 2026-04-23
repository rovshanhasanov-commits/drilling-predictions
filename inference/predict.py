"""End-to-end ML inference: (well, date) -> MLOutput with top-K predictions per step."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from config import get_model_dir, load_config, resolve

from .contract import LevelPrediction, MLOutput, StepPrediction
from .load import load_bundle
from .preprocess_selection import build_encoder_input

HIERARCHY = ["phase", "phase_step", "major_ops_code", "operation"]


_CACHED_BUNDLE = None


def _get_bundle(model_dir: Path):
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is not None and _CACHED_BUNDLE["_dir"] == str(model_dir):
        return _CACHED_BUNDLE
    bundle = load_bundle(model_dir)
    bundle["_dir"] = str(model_dir)
    _CACHED_BUNDLE = bundle
    return bundle


def _invert_duration(scaled_values: np.ndarray, dur_scaler) -> np.ndarray:
    """Inverse StandardScaler, then expm1, then clip to [0, inf)."""
    unscaled = dur_scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()
    hours = np.expm1(unscaled).clip(min=0.0)
    return hours


def _topk_for_level(probs: np.ndarray, target_encoder, k: int) -> LevelPrediction:
    """probs shape: (n_classes,). Returns top-k labels + probs, descending."""
    order = np.argsort(-probs)[:k]
    labels = [str(target_encoder.classes_[i]) for i in order]
    return LevelPrediction(labels=labels, probs=[float(probs[i]) for i in order])


def predict(
    well_name: str,
    report_date: date | str,
    cfg: dict | None = None,
    model_dir: Path | None = None,
    top_k: int | None = None,
) -> MLOutput:
    cfg = cfg or load_config()
    tcfg = cfg["training"]

    strategy = tcfg["embedding_strategy"]
    strategy_dir = resolve(cfg, cfg["data"]["output_dir"]) / strategy
    model_dir = model_dir or get_model_dir(cfg)
    seq_len = tcfg["sequence_length"]
    n_future = tcfg["n_future"]
    k = top_k if top_k is not None else tcfg.get("top_k_for_eval", 3)

    bundle = _get_bundle(model_dir)
    encoder_model = bundle["encoder_model"]
    decoder_step_model = bundle["decoder_step_model"]
    encoders = bundle["encoders"]
    data_config = bundle["data_config"]
    n_classes = encoders["n_classes"]
    target_encoders = encoders["target_encoders"]
    dur_scaler = encoders["dur_scaler"]

    active_targets = [t.replace("_next", "") for t in tcfg["target_variables"] if t != "duration_next"]
    predict_duration = "duration_next" in tcfg["target_variables"]

    enc_inputs, _ = build_encoder_input(
        well_name, report_date, strategy_dir, data_config, seq_len,
    )

    enc_out, h, c = encoder_model.predict(enc_inputs, verbose=0)

    prev_cat = {t: np.array([[n_classes[t]]], dtype=np.int32) for t in active_targets}
    prev_dur = np.array([[0.0]], dtype=np.float32) if predict_duration else None

    per_step_records: list[dict] = []

    for step in range(n_future):
        dec_step_in = [prev_cat[t] for t in active_targets]
        if predict_duration:
            dec_step_in.append(prev_dur)
        dec_step_in += [enc_out, h, c]

        step_out = decoder_step_model.predict(dec_step_in, verbose=0)

        idx = 0
        level_probs = {}
        for t in active_targets:
            probs = step_out[idx][0, 0, :]        # (n_classes,)
            level_probs[t] = probs
            prev_cat[t] = np.array([[int(probs.argmax())]], dtype=np.int32)
            idx += 1
        dur_scaled = None
        if predict_duration:
            dur_scaled = float(step_out[idx][0, 0, 0])
            prev_dur = np.array([[dur_scaled]], dtype=np.float32)
            idx += 1
        h = step_out[idx]
        c = step_out[idx + 1]

        # Map probs -> top-K labels for each level we actually predicted.
        record = {}
        for t in HIERARCHY:
            if t in level_probs:
                record[t] = _topk_for_level(level_probs[t], target_encoders[t], k)
            else:
                # Level wasn't a training target; emit a single "unknown" prediction
                record[t] = LevelPrediction(labels=["UNKNOWN"], probs=[1.0])
        if predict_duration and dur_scaled is not None:
            record["duration_hours"] = float(_invert_duration(np.array([dur_scaled]), dur_scaler)[0])
        else:
            record["duration_hours"] = 0.0
        per_step_records.append(record)

    report_date_obj = report_date if isinstance(report_date, date) else np.datetime64(report_date).astype("datetime64[D]").item()

    steps = [
        StepPrediction(
            step=i,
            phase=r["phase"],
            phase_step=r["phase_step"],
            major_ops_code=r["major_ops_code"],
            operation=r["operation"],
            duration_hours=r["duration_hours"],
        )
        for i, r in enumerate(per_step_records)
    ]
    return MLOutput(
        well_name=well_name,
        report_date=report_date_obj,
        n_future=n_future,
        top_k=k,
        steps=steps,
    )

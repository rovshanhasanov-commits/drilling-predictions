"""Scheduled-sampling training loop."""

from __future__ import annotations

import numpy as np

from .data import make_decoder_inputs, mix_scheduled_sampling


def prep_encoder_inputs(cat: dict, num: np.ndarray, cat_cols: list[str]) -> list:
    """Encoder input list in the order expected by model construction."""
    return [num] if not cat_cols else [cat[c] for c in cat_cols] + [num]


def prep_model_inputs(enc_X: list, dec: dict, active_targets: list[str], predict_duration: bool) -> list:
    inputs = list(enc_X)
    for t in active_targets:
        inputs.append(dec[t])
    if predict_duration:
        inputs.append(dec["duration"])
    return inputs


def prep_targets(y: dict, y_dur: np.ndarray, active_targets: list[str], predict_duration: bool) -> dict:
    out = {t: y[f"{t}_target_enc"] for t in active_targets}
    if predict_duration:
        out["duration"] = y_dur
    return out


def _expand_sample_weight(sw: dict | None, y_fit: dict) -> dict | None:
    """Pad a partial sample_weight dict to mirror the y_fit structure.

    Keras requires sample_weight to match y's keys exactly. Callers may provide
    weights on any subset of heads (currently operation / major_ops_code / duration
    via op_label_real / moc_label_real / dur_label_real); heads without an explicit
    weight get all-ones arrays of the matching shape.
    """
    if sw is None:
        return None
    ref = next(iter(sw.values()))
    ones = np.ones(ref.shape, dtype=np.float32)
    return {k: sw.get(k, ones) for k in y_fit}


def train(
    training_model,
    enc_X_train, y_train, y_dur_train,
    enc_X_val,   y_val,   y_dur_val,
    active_targets: list[str],
    predict_duration: bool,
    n_classes: dict,
    batch_size: int,
    epochs: int,
    ss_start_rate: float,
    ss_end_rate: float,
    ss_ramp_epochs: int,
    early_stopping_patience: int,
    weights_path: str = "best_ss.weights.h5",
    sample_weight_train: dict | None = None,
    sample_weight_val: dict | None = None,
    lr_patience: int = 0,
    lr_factor: float = 0.5,
    min_lr: float = 1e-7,
):
    """Per-epoch: compute SS rate, optionally mix teacher-forced+predicted decoder inputs, fit 1 epoch,
    manual early stopping with checkpoint save/restore.

    Dynamic LR (ReduceLROnPlateau-style): when `lr_patience > 0`, the optimizer's
    learning rate is multiplied by `lr_factor` after `lr_patience` epochs of no
    val_loss improvement, floored at `min_lr`. `lr_patience=0` disables the
    schedule and keeps LR constant at the optimizer's initial value.
    """

    dec_train_pure = make_decoder_inputs(y_train, n_classes, active_targets, predict_duration, y_dur_train)
    dec_val        = make_decoder_inputs(y_val,   n_classes, active_targets, predict_duration, y_dur_val)

    X_val_full  = prep_model_inputs(enc_X_val,   dec_val,        active_targets, predict_duration)
    y_val_fit   = prep_targets(y_val, y_dur_val, active_targets, predict_duration)
    y_train_fit = prep_targets(y_train, y_dur_train, active_targets, predict_duration)

    # Expand per-head sample_weight dicts to match y's full key structure
    # (Keras requires identical nesting between y and sample_weight).
    sample_weight_train = _expand_sample_weight(sample_weight_train, y_train_fit)
    sample_weight_val   = _expand_sample_weight(sample_weight_val,   y_val_fit)

    # Coerce scheduling kwargs — YAML can deliver ints/strings depending on literal
    # form (e.g. `min_lr: 1e-7` parses as a string under PyYAML SafeLoader; needs
    # `1.0e-7` for float). Fail loudly in the caller rather than deep in the loop.
    lr_patience = int(lr_patience)
    lr_factor   = float(lr_factor)
    min_lr      = float(min_lr)

    history_logs = []
    best_val = np.inf
    wait = 0
    lr_wait = 0                                 # parallel counter for the LR schedule

    for epoch in range(1, epochs + 1):
        progress = min((epoch - 1) / max(ss_ramp_epochs, 1), 1.0) if ss_ramp_epochs > 0 else 1.0
        ss_rate = ss_start_rate + (ss_end_rate - ss_start_rate) * progress
        current_lr = float(training_model.optimizer.learning_rate.numpy())

        if ss_rate == 0.0:
            dec_epoch = dec_train_pure
        else:
            X_tf = prep_model_inputs(enc_X_train, dec_train_pure, active_targets, predict_duration)
            preds = training_model.predict(X_tf, batch_size=batch_size * 2, verbose=0)
            if not isinstance(preds, dict):
                if not isinstance(preds, list):
                    preds = [preds]
                keys = active_targets + (["duration"] if predict_duration else [])
                preds = {k: preds[i] for i, k in enumerate(keys)}
            dec_epoch = mix_scheduled_sampling(dec_train_pure, preds, ss_rate, active_targets, predict_duration)

        X_train_full = prep_model_inputs(enc_X_train, dec_epoch, active_targets, predict_duration)

        validation_data = (
            (X_val_full, y_val_fit, sample_weight_val)
            if sample_weight_val is not None
            else (X_val_full, y_val_fit)
        )
        h = training_model.fit(
            X_train_full, y_train_fit,
            sample_weight=sample_weight_train,
            validation_data=validation_data,
            epochs=1, batch_size=batch_size, verbose=0,
        )
        log = {k: float(v[0]) for k, v in h.history.items()}
        log["epoch"] = epoch
        log["ss_rate"] = float(ss_rate)
        log["lr"] = current_lr
        history_logs.append(log)

        val_loss = log["val_loss"]
        improved = val_loss < best_val

        header = f"Epoch {epoch}/{epochs}  lr={current_lr:.2e}  ss_rate={ss_rate:.3f}"
        if improved:
            header += f"  (new best val_loss: {val_loss:.4f})"
        else:
            header += f"  (best val_loss so far: {best_val:.4f})"
        print(header)
        print(f"  {'metric':<42} {'train':>10} {'val':>10}")
        train_keys = sorted(
            k for k in log
            if not k.startswith("val_") and k not in ("epoch", "ss_rate")
        )
        for k in train_keys:
            t = log.get(k, float("nan"))
            v = log.get(f"val_{k}", float("nan"))
            marker = "  * new best" if improved and k == "loss" else ""
            print(f"  {k:<42} {t:>10.4f} {v:>10.4f}{marker}")

        if improved:
            best_val = val_loss
            wait = 0
            lr_wait = 0
            training_model.save_weights(weights_path)
        else:
            wait += 1
            lr_wait += 1
            # Drop LR when the secondary counter trips, independent of early stopping.
            # Resets lr_wait so the next drop requires another lr_patience epochs of stall.
            if lr_patience > 0 and lr_wait >= lr_patience:
                new_lr = max(current_lr * lr_factor, min_lr)
                if new_lr < current_lr:
                    training_model.optimizer.learning_rate.assign(new_lr)
                    print(f"  LR reduced: {current_lr:.2e} -> {new_lr:.2e}")
                lr_wait = 0
            if wait >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    training_model.load_weights(weights_path)
    print(f"Training complete. Best val_loss: {best_val:.6f}")
    return history_logs

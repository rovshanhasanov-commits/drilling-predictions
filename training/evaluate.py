"""Evaluation utilities. Reports top-1 AND top-3 accuracy per step (per the user's
"off by 1" hypothesis), and autoregressive inference for end-to-end check.
"""

from __future__ import annotations

import numpy as np


def _topk_hit(probs: np.ndarray, true: np.ndarray, k: int) -> np.ndarray:
    """probs: (N, K_classes); true: (N,). Returns bool array of shape (N,)."""
    topk = np.argpartition(-probs, k - 1, axis=-1)[:, :k]
    return (topk == true[:, None]).any(axis=-1)


def per_step_accuracy(
    probs: np.ndarray,
    true: np.ndarray,
    k: int = 3,
    weights: np.ndarray | None = None,
) -> dict:
    """probs: (batch, N_future, n_classes); true: (batch, N_future).

    Returns per-step top-1 and top-k accuracy plus overall.

    If `weights` (same shape as `true`) is provided, positions with weight <= 0
    are excluded from the mean — used to drop masked targets (Unplanned, UNK)
    from accuracy calcs since the model structurally can't predict those classes.
    Returns NaN for a step if every position at that step is masked.
    """
    batch, n_future, _ = probs.shape
    top1 = []
    topk = []
    for s in range(n_future):
        p = probs[:, s, :]
        t = true[:, s]
        pred1 = p.argmax(axis=-1)
        hit1 = (pred1 == t)
        hitk = _topk_hit(p, t, k)
        if weights is not None:
            mask = weights[:, s] > 0
            if not mask.any():
                top1.append(float("nan"))
                topk.append(float("nan"))
                continue
            hit1 = hit1[mask]
            hitk = hitk[mask]
        top1.append(float(hit1.mean()))
        topk.append(float(hitk.mean()))
    return {
        "per_step_top1": top1,
        "per_step_topk": topk,
        "k": k,
        "overall_top1": float(np.nanmean(top1)),
        "overall_topk": float(np.nanmean(topk)),
    }


def autoregressive_predict(
    encoder_model, decoder_step_model,
    enc_X, n_future: int, n_classes: dict,
    active_targets: list[str], predict_duration: bool,
    batch_size: int = 256,
    top_k: int = 3,
):
    """Autoregressive decode. Returns per-target arrays:
        - '{target}_pred':  (batch, N_future)             argmax predictions
        - '{target}_topk':  (batch, N_future, top_k)      top-k class ids per step
        - '{target}_topk_probs': (batch, N_future, top_k) corresponding probs
        - 'duration' (if enabled): (batch, N_future)
    """
    n_samples = enc_X[0].shape[0]
    enc_out, h, c = encoder_model.predict(enc_X, batch_size=batch_size, verbose=0)

    preds = {t: np.zeros((n_samples, n_future), dtype=np.int32) for t in active_targets}
    topk  = {t: np.zeros((n_samples, n_future, top_k), dtype=np.int32) for t in active_targets}
    topkp = {t: np.zeros((n_samples, n_future, top_k), dtype=np.float32) for t in active_targets}
    if predict_duration:
        preds["duration"] = np.zeros((n_samples, n_future), dtype=np.float32)

    prev_cat = {t: np.full((n_samples, 1), n_classes[t], dtype=np.int32) for t in active_targets}
    prev_dur = np.zeros((n_samples, 1), dtype=np.float32) if predict_duration else None

    for step in range(n_future):
        dec_step_in = [prev_cat[t] for t in active_targets]
        if predict_duration:
            dec_step_in.append(prev_dur)
        dec_step_in += [enc_out, h, c]

        step_out = decoder_step_model.predict(dec_step_in, batch_size=batch_size, verbose=0)

        idx = 0
        for t in active_targets:
            probs = step_out[idx][:, 0, :]                       # (batch, n_classes)
            pred_cls = probs.argmax(axis=-1)
            order = np.argsort(-probs, axis=-1)[:, :top_k]
            preds[t][:, step] = pred_cls
            topk[t][:, step, :] = order
            topkp[t][:, step, :] = np.take_along_axis(probs, order, axis=-1)
            prev_cat[t] = pred_cls.reshape(-1, 1)
            idx += 1
        if predict_duration:
            dur_val = step_out[idx][:, 0, 0]
            preds["duration"][:, step] = dur_val
            prev_dur = dur_val.reshape(-1, 1)
            idx += 1

        h = step_out[idx]
        c = step_out[idx + 1]

    return {
        "pred": preds,
        "topk": topk,
        "topk_probs": topkp,
    }

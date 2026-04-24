"""Evaluation utilities. Reports top-1 AND top-3 accuracy per step (per the user's
"off by 1" hypothesis), and autoregressive inference for end-to-end check.

Supports constrained decoding via the optional `legal_tuples` kwarg on
`autoregressive_predict`. When provided, per-step outputs are chosen by joint
argmax over the legal-tuple table (see training/constraints.py) and the winning
tuple's class ids are fed back as the next step's decoder inputs — this keeps
the four heads coordinated throughout the horizon.
"""

from __future__ import annotations

import numpy as np

from .constraints import HIERARCHY, joint_topk_tuples


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
    legal_tuples: np.ndarray | None = None,
    include_bins: bool = False,
):
    """Autoregressive decode. Returns per-target arrays:
        - 'pred[head]':        (batch, N_future)            argmax / joint-argmax predictions
        - 'topk[head]':        (batch, N_future, top_k)     top-k class ids per step
        - 'topk_probs[head]':  (batch, N_future, top_k)     corresponding probs
        - 'duration' (if predict_duration): (batch, N_future)

    When `legal_tuples` is provided (shape (num_legal, D), int32):
        - Per-step argmax becomes joint argmax over the legal-tuple table.
        - `topk` + `topk_probs` are still populated (per-head ranks unchanged)
          so legacy callers work, but the result dict also includes:
            - 'tuple_pred':          (batch, N_future, D)     chosen legal tuple per step
            - 'tuple_topk':          (batch, N_future, K, D)  top-K legal tuples per step
            - 'tuple_topk_logprob':  (batch, N_future, K)     raw joint log-probs
            - 'tuple_topk_prob':     (batch, N_future, K)     probabilities
                                                              RENORMALIZED over all legal
                                                              tuples (sum over full L == 1,
                                                              sum over just top-K <= 1).
        - The winning tuple's class ids feed back as next-step decoder inputs —
          this is what prevents mid-sequence drift (vs. independent per-head argmax).
        - With `include_bins=True`, D=5 and the bin head joins the joint argmax
          as the 5th column. Default `include_bins=False`, D=4 (4-hierarchy
          tuples; the bin head, if active, feeds back via independent argmax).
    """
    use_constraints = legal_tuples is not None
    if use_constraints:
        missing = [h for h in HIERARCHY if h not in active_targets]
        if missing:
            raise ValueError(
                f"legal_tuples requires all 4 hierarchy heads in active_targets; missing {missing}"
            )
    if include_bins:
        if not use_constraints:
            raise ValueError("include_bins=True requires legal_tuples to be provided")
        if "duration_bin" not in active_targets:
            raise ValueError(
                "include_bins=True requires 'duration_bin' in active_targets "
                "(set duration_bin_next in target_variables)"
            )

    head_names_joint = HIERARCHY + (["duration_bin"] if include_bins else [])
    n_joint_heads = len(head_names_joint)

    n_samples = enc_X[0].shape[0]
    enc_out, h, c = encoder_model.predict(enc_X, batch_size=batch_size, verbose=0)

    preds = {t: np.zeros((n_samples, n_future), dtype=np.int32) for t in active_targets}
    topk  = {t: np.zeros((n_samples, n_future, top_k), dtype=np.int32) for t in active_targets}
    topkp = {t: np.zeros((n_samples, n_future, top_k), dtype=np.float32) for t in active_targets}
    if predict_duration:
        preds["duration"] = np.zeros((n_samples, n_future), dtype=np.float32)

    if use_constraints:
        K_tuple = min(int(top_k), int(legal_tuples.shape[0]))
        tuple_pred          = np.zeros((n_samples, n_future, n_joint_heads), dtype=np.int32)
        tuple_topk          = np.zeros((n_samples, n_future, K_tuple, n_joint_heads), dtype=np.int32)
        tuple_topk_logprob  = np.zeros((n_samples, n_future, K_tuple), dtype=np.float32)
        tuple_topk_prob     = np.zeros((n_samples, n_future, K_tuple), dtype=np.float32)

    prev_cat = {t: np.full((n_samples, 1), n_classes[t], dtype=np.int32) for t in active_targets}
    prev_dur = np.zeros((n_samples, 1), dtype=np.float32) if predict_duration else None

    for step in range(n_future):
        dec_step_in = [prev_cat[t] for t in active_targets]
        if predict_duration:
            dec_step_in.append(prev_dur)
        dec_step_in += [enc_out, h, c]

        step_out = decoder_step_model.predict(dec_step_in, batch_size=batch_size, verbose=0)

        # Collect per-head probabilities for this step.
        step_probs: dict = {}
        idx = 0
        for t in active_targets:
            probs = step_out[idx][:, 0, :]                       # (batch, n_classes)
            step_probs[t] = probs
            # Per-head top-K ranks stay populated for back-compat / diagnostics.
            order = np.argsort(-probs, axis=-1)[:, :top_k]
            topk[t][:, step, :] = order
            topkp[t][:, step, :] = np.take_along_axis(probs, order, axis=-1)
            idx += 1

        if use_constraints:
            # Joint top-K first (argmax = top-1). Score sums over `head_names_joint`
            # — 4 hierarchy heads, plus the bin head when include_bins=True.
            tk_tuples, tk_log_probs, tk_probs = joint_topk_tuples(
                step_probs, legal_tuples, K_tuple, head_names=head_names_joint,
            )
            tuple_topk[:, step, :, :]       = tk_tuples
            tuple_topk_logprob[:, step, :]  = tk_log_probs
            tuple_topk_prob[:, step, :]     = tk_probs
            chosen = tk_tuples[:, 0, :]                          # (batch, n_joint_heads)
            tuple_pred[:, step, :] = chosen
            for i, t in enumerate(head_names_joint):
                preds[t][:, step] = chosen[:, i]
                prev_cat[t] = chosen[:, i:i + 1]                 # feedback
            # Heads not in the joint table (e.g. bin head when include_bins=False)
            # fall back to independent argmax. Without this loop, preds["duration_bin"]
            # would silently stay all zeros when the bin head is active in 4D mode.
            for t in active_targets:
                if t not in head_names_joint:
                    pred_cls = step_probs[t].argmax(axis=-1)
                    preds[t][:, step] = pred_cls
                    prev_cat[t] = pred_cls.reshape(-1, 1)
        else:
            for t in active_targets:
                pred_cls = step_probs[t].argmax(axis=-1)
                preds[t][:, step] = pred_cls
                prev_cat[t] = pred_cls.reshape(-1, 1)

        if predict_duration:
            dur_val = step_out[idx][:, 0, 0]
            preds["duration"][:, step] = dur_val
            prev_dur = dur_val.reshape(-1, 1)
            idx += 1

        h = step_out[idx]
        c = step_out[idx + 1]

    out = {
        "pred": preds,
        "topk": topk,
        "topk_probs": topkp,
    }
    if use_constraints:
        out["tuple_pred"]         = tuple_pred
        out["tuple_topk"]         = tuple_topk
        out["tuple_topk_logprob"] = tuple_topk_logprob
        out["tuple_topk_prob"]    = tuple_topk_prob
    return out

"""Legal-tuple constraints for hierarchy-consistent decoding.

The four hierarchy heads (phase, phase_step, major_ops_code, operation) are
trained as independent softmaxes. Per-head argmax can yield tuples that don't
exist in the drilling taxonomy. This module provides:

  - `build_legal_tuples`: enumerate every (p, s, m, o) combination observed in
    the cleaned parquet data (union of all splits), plus the EOO sentinel.
    Excludes `Unplanned` and `UNK` from the op axis — those are mask/sentinel
    classes the op head is trained never to emit.
  - `joint_argmax`: given per-head probability vectors, pick the single legal
    tuple with the highest joint log-prob.
  - `joint_topk_tuples`: same, but return the top-K legal tuples per sample
    ranked by joint log-prob.

No TensorFlow dependency — pure numpy. Decoder inputs feed back as integer
class ids, same as under independent argmax.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HIERARCHY = ["phase", "phase_step", "major_ops_code", "operation"]

# Excluded from L on the op axis: these are sentinel / mask classes that the
# op head is trained never to emit (via `op_label_real=0` in preprocessing/clean.py).
# Including them in L would let constrained decoding pick them, defeating the mask.
DEFAULT_EXCLUDE_OPS: tuple[str, ...] = ("Unplanned", "UNK")


DEFAULT_EXCLUDE_BIN_CLASSES: tuple[str, ...] = ("Unplanned", "UNK")


def build_legal_tuples(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_encoders: dict,
    eoo_token: str = "End of Operations",
    exclude_ops: tuple[str, ...] = DEFAULT_EXCLUDE_OPS,
    include_bins: bool = False,
    exclude_bin_classes: tuple[str, ...] = DEFAULT_EXCLUDE_BIN_CLASSES,
) -> np.ndarray:
    """Enumerate the legal (phase, phase_step, major_ops_code, operation) tuples.

    Source: the `{head}_target_enc` integer-encoded columns from the cleaned
    parquets (union of train + val + test). The string columns are dropped
    during encoding for all three strategies, so we work entirely with
    integer ids mapped through `target_encoders`.

    The EOO tuple `(EOO, EOO, EOO, EOO)` is explicitly added so constrained
    decoding can still emit the well-end sentinel.

    When `include_bins=True`, `duration_bin_target_enc` is stacked as a 5th
    column and bin sentinels (`Unplanned`, `UNK`) are excluded from that axis.
    The EOO row gets the bin "EOO" id appended. Returned shape becomes
    `(num_legal, 5)`. See improvements/Duration Binning.md §8.

    Returns:
        np.ndarray of shape (num_legal, 4) or (num_legal, 5), int32.
        Columns are in HIERARCHY order, optionally followed by `duration_bin`.
    """
    encoded_cols = [f"{h}_target_enc" for h in HIERARCHY]
    if include_bins:
        if "duration_bin" not in target_encoders:
            raise KeyError(
                "build_legal_tuples: include_bins=True requires "
                "target_encoders['duration_bin'] (re-run preprocessing with "
                "duration_bins.enabled=true)."
            )
        encoded_cols = encoded_cols + ["duration_bin_target_enc"]

    frames = []
    for df in (df_train, df_val, df_test):
        missing = [c for c in encoded_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"build_legal_tuples: parquet is missing {missing}. "
                f"Re-run preprocessing to regenerate target_enc columns."
            )
        frames.append(df[encoded_cols])
    df = pd.concat(frames, ignore_index=True)

    # Drop sentinel/mask rows so they don't become legal tuples. We filter by
    # excluded op-axis class ids rather than string labels so we never touch
    # the string columns (which the dummies strategy drops).
    op_le = target_encoders["operation"]
    class_to_idx = {c: i for i, c in enumerate(op_le.classes_)}
    excluded_op_ids = {class_to_idx[o] for o in exclude_ops if o in class_to_idx}
    if excluded_op_ids:
        df = df[~df["operation_target_enc"].isin(excluded_op_ids)]

    # Also drop rows whose phase/phase_step/moc axes landed on UNK (the
    # preprocessing fallback for NaN / unseen-in-train). These aren't real
    # hierarchy members; only the true domain values should enter L.
    unk_token = "UNK"
    for h in HIERARCHY[:3]:
        le = target_encoders[h]
        c2i = {c: i for i, c in enumerate(le.classes_)}
        if unk_token in c2i:
            df = df[df[f"{h}_target_enc"] != c2i[unk_token]]

    # Exclude bin sentinels on the 5th axis (mirror op-axis filter).
    if include_bins:
        bin_le = target_encoders["duration_bin"]
        bin_c2i = {c: i for i, c in enumerate(bin_le.classes_)}
        excluded_bin_ids = {bin_c2i[c] for c in exclude_bin_classes if c in bin_c2i}
        if excluded_bin_ids:
            df = df[~df["duration_bin_target_enc"].isin(excluded_bin_ids)]

    L = df.drop_duplicates().to_numpy(dtype=np.int32)

    eoo_ids_list = [int(_encode_one(target_encoders[h], eoo_token)) for h in HIERARCHY]
    if include_bins:
        eoo_ids_list.append(int(_encode_one(target_encoders["duration_bin"], "EOO")))
    eoo_ids = np.asarray(eoo_ids_list, dtype=np.int32)
    if not _contains_row(L, eoo_ids):
        L = np.vstack([L, eoo_ids[None, :]])

    return L


def _encode_one(le, label: str) -> int:
    classes = list(le.classes_)
    if label not in classes:
        raise ValueError(f"{label!r} not in encoder classes: {classes[:5]}...")
    return classes.index(label)


def _contains_row(arr: np.ndarray, row: np.ndarray) -> bool:
    return bool(np.any(np.all(arr == row, axis=1)))


def _stack_logprobs(probs: dict, head_names: list[str] = HIERARCHY) -> tuple[np.ndarray, ...]:
    """Return log-probs per head, safe for post-softmax or pre-softmax inputs.

    The current model's output layer applies softmax (see training/model.py
    `activation="softmax"`), so inputs here are probabilities in [0, 1]. We
    floor at `eps` before `log` to avoid -inf on zero entries.

    `head_names` defaults to the 4-level hierarchy; pass `HIERARCHY + ["duration_bin"]`
    when scoring 5D legal tuples.
    """
    eps = 1e-12
    return tuple(np.log(probs[h] + eps) for h in head_names)


def _score_tuples(probs: dict, L: np.ndarray, head_names: list[str] = HIERARCHY) -> np.ndarray:
    """Joint log-prob of each legal tuple, per sample.

    probs[h] has shape (B, n_classes[h]); L has shape (num_legal, len(head_names)).
    Returns array of shape (B, num_legal).
    """
    if L.shape[1] != len(head_names):
        raise ValueError(
            f"_score_tuples: L has {L.shape[1]} columns but {len(head_names)} "
            f"head_names provided ({head_names})."
        )
    log_probs = _stack_logprobs(probs, head_names)
    score = log_probs[0][:, L[:, 0]]
    for i in range(1, len(head_names)):
        score = score + log_probs[i][:, L[:, i]]
    return score


def _renormalize_over_L(scores: np.ndarray) -> np.ndarray:
    """Softmax over the legal-tuple axis — numerically stable, no scipy.

    Input  scores: (B, num_legal) raw joint log-probs.
    Output probs:  (B, num_legal), sums to 1 along axis -1.

    Semantic: `probs[b, i]` is tuple i's share of the probability mass the
    four independent head-softmaxes placed on legal tuples. Raw log-probs sum
    to `P_legal <= 1` across L because the 4 softmaxes also put mass on
    illegal combinations; renormalizing strips that out.
    """
    m = scores.max(axis=-1, keepdims=True)
    exp_shifted = np.exp(scores - m)
    return exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)


def joint_argmax(probs: dict, L: np.ndarray, head_names: list[str] = HIERARCHY) -> np.ndarray:
    """Return the single legal tuple per sample with the highest joint log-prob.

    Args:
        probs: {head_name: (B, n_classes_for_head)} — post-softmax probabilities.
        L:     (num_legal, len(head_names)) int32 array of class ids per legal tuple.
        head_names: column order of L. Defaults to the 4-level hierarchy; pass
            `HIERARCHY + ["duration_bin"]` for 5D constrained decoding.

    Returns:
        (B, len(head_names)) int32 array.
    """
    if L.size == 0:
        raise ValueError("joint_argmax: legal-tuple table L is empty")
    scores = _score_tuples(probs, L, head_names)
    best = scores.argmax(axis=-1)
    return L[best].astype(np.int32)


def joint_topk_tuples(
    probs: dict,
    L: np.ndarray,
    k: int,
    head_names: list[str] = HIERARCHY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the top-K legal tuples per sample, sorted by joint log-prob desc.

    Args:
        probs: {head_name: (B, n_classes_for_head)}.
        L:     (num_legal, 4) int32.
        k:     number of tuples to return per sample. Clipped to len(L).

    Returns:
        topk_tuples: (B, k, 4) int32 — chosen legal tuples
        topk_log_probs: (B, k) float32 — raw joint log-prob `log P(tuple)`
                                          (sum of the 4 head log-probs; `exp()` gives
                                          the unnormalized joint probability, which
                                          is `<= P_legal <= 1` across the full L)
        topk_probs: (B, k) float32 — **renormalized over all of L**, i.e. each value
                                      is that tuple's share of the legal-set mass.
                                      Summing all of L's `probs` gives exactly 1;
                                      summing just the top-K gives `<= 1`, and the
                                      gap tells you how sharp the top-K is.
    """
    if L.size == 0:
        raise ValueError("joint_topk_tuples: legal-tuple table L is empty")
    k = min(int(k), L.shape[0])
    scores = _score_tuples(probs, L, head_names)               # (B, num_legal)
    probs_renorm = _renormalize_over_L(scores)                 # (B, num_legal), sums to 1

    if k == L.shape[0]:
        order_full = np.argsort(-scores, axis=-1)
        topk_idx = order_full[:, :k]
    else:
        partial = np.argpartition(-scores, k - 1, axis=-1)[:, :k]
        partial_scores = np.take_along_axis(scores, partial, axis=-1)
        order = np.argsort(-partial_scores, axis=-1)
        topk_idx = np.take_along_axis(partial, order, axis=-1)

    topk_log_probs = np.take_along_axis(scores,       topk_idx, axis=-1).astype(np.float32)
    topk_probs     = np.take_along_axis(probs_renorm, topk_idx, axis=-1).astype(np.float32)
    topk_tuples    = L[topk_idx].astype(np.int32)              # (B, k, len(head_names))
    return topk_tuples, topk_log_probs, topk_probs


def summarize_legal_tuples(L: np.ndarray, target_encoders: dict) -> dict:
    """Diagnostic summary for logging at construction time.

    Returns counts of unique phase/step/moc/op ids represented in L, plus the
    total. Callers can log this to spot mistakes (e.g., exclude list too broad).
    """
    out = {"n_tuples": int(L.shape[0])}
    for i, h in enumerate(HIERARCHY):
        uniq = np.unique(L[:, i])
        out[f"n_unique_{h}"] = int(uniq.size)
        out[f"vocab_{h}"] = int(len(target_encoders[h].classes_))
    return out

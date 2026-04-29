"""Shift + annotate logic for evaluation predictions.

See improvements/Learning rate changes and Eval fixes.md §2b and §2d.

The model is never trained to predict the `Unplanned` or `UNK` tokens (loss
is masked on those positions during training). At eval time we compare AR
predictions against truth — but raw position-by-position comparison forces
the model to be "wrong" at every Unplanned truth position even though the
model couldn't have known. Worse, AR self-feeds its own predictions, so
position i+1's pred was conditioned on position i's pred (planned), not on
the actual Unplanned truth — meaning the AR forecast is implicitly aligned
to the *planned* op stream, not the raw stream.

Solution: shift AR predictions left to skip past Unplanned/UNK truths. The
pred slot at the Unplanned position becomes NULL (the original pred was
moved forward to the next planned position). Trailing positions where we
ran out of preds to shift in get NULL too (`shift_overflow`).

Annotation columns are surfaced in predictions.csv:
    planned_step    1-indexed planned-position counter (NaN on excluded rows)
    exclude         bool — whether to skip this row in summary metrics
    exclude_reason  one of {Unplanned, UNK, shift_overflow,
                            premature_eoo, after_eoo} or empty

Precedence (highest first):
    Unplanned > UNK > shift_overflow > premature_eoo > after_eoo

Post-EOO logic (§2d): once the model emits an EOO prediction, every
subsequent self-fed pred is also EOO (cascade). We don't mask out the EOO
step itself — premature EOO IS graded (correctly counted as wrong if true
isn't EOO there). Strictly later steps get exclude=after_eoo so the cascade
doesn't pile up artificial penalties. If true and pred both have EOO at the
same step (correct well-end detection), that row is NOT excluded — the model
gets credit.
"""

from __future__ import annotations

import numpy as np


def compute_alignment(
    true_op_labels: np.ndarray,            # (n, K) decoded operation labels
    raw_pred_op_labels: np.ndarray,        # (n, K) decoded AR pred operation labels (pre-shift)
    unplanned_token: str = "Unplanned",
    unk_token: str = "UNK",
    eoo_token: str = "End of Operations",
) -> dict:
    """Build shift indices and annotation arrays for a batch of sequences.

    Returns a dict with:
      shift_indices:   (n, K) int.  Source horizon position to gather pred from.
                       Sentinels: -1 = NULL because true is Unplanned/UNK,
                                  -2 = NULL because shifted past available preds.
      planned_step:    (n, K) float. 1-indexed planned-position; NaN for excluded.
      exclude:         (n, K) bool.
      exclude_reason:  (n, K) object array of strings (empty string for non-excluded).
      eval_weights:    (n, K) float. 1.0 where exclude=False, 0.0 elsewhere — pass
                       directly to weighted-mean accuracy helpers.
      first_true_eoo:  (n,) int. First i where true == EOO; -1 if none.
      first_pred_eoo:  (n,) int. First i where shifted pred == EOO; -1 if none.
    """
    n, K = true_op_labels.shape
    is_unplanned = (true_op_labels == unplanned_token)
    is_unk       = (true_op_labels == unk_token)
    is_skipped   = is_unplanned | is_unk

    # cum_skipped_strict_before[s, i] = number of Unplanned/UNK in true[s, 0..i-1].
    cum_inclusive = np.cumsum(is_skipped, axis=1)
    cum_strict    = cum_inclusive - is_skipped.astype(np.int64)

    shift_indices = np.full((n, K), -1, dtype=np.int64)
    for i in range(K):
        # Vectorized per-column assignment: rows where this position is planned
        # take source = i - cum_strict[:, i]; rows where source >= K become -2.
        planned_col = ~is_skipped[:, i]
        if not planned_col.any():
            continue
        sources = i - cum_strict[:, i]
        overflow = sources >= K
        valid = planned_col & ~overflow
        shift_indices[valid, i] = sources[valid]
        shift_indices[planned_col & overflow, i] = -2

    # Compute shifted pred operation labels — needed to find first_pred_eoo.
    shifted_pred_op = shift_along_axis1(raw_pred_op_labels, shift_indices, fill="")

    first_true_eoo = _first_index_of(true_op_labels, eoo_token)
    first_pred_eoo = _first_index_of(shifted_pred_op, eoo_token)

    # Build exclude / exclude_reason with precedence.
    exclude        = np.zeros((n, K), dtype=bool)
    exclude_reason = np.full((n, K), "", dtype=object)

    # 1. Unplanned / UNK in truth.
    exclude_reason[is_unplanned] = "Unplanned"
    exclude_reason[is_unk & (exclude_reason == "")] = "UNK"
    exclude |= is_skipped

    # 2. Shift overflow — only matters where not already excluded.
    overflow_mask = (shift_indices == -2) & (~exclude)
    exclude_reason[overflow_mask] = "shift_overflow"
    exclude |= overflow_mask

    # 3. Premature EOO — pred says EOO, true doesn't, at the FIRST predicted-EOO step.
    #    Only this step itself is labeled premature_eoo; later steps fall into after_eoo.
    premature = np.zeros((n, K), dtype=bool)
    rows_with_pred_eoo = np.where(first_pred_eoo >= 0)[0]
    for s in rows_with_pred_eoo:
        i = first_pred_eoo[s]
        if true_op_labels[s, i] != eoo_token and not exclude[s, i]:
            premature[s, i] = True
    exclude_reason[premature] = "premature_eoo"
    exclude |= premature

    # 4. After-EOO — strictly after first_true_eoo, OR strictly after first_pred_eoo
    #    where true didn't catch up to EOO at this step.
    after = np.zeros((n, K), dtype=bool)
    idx = np.arange(K)[None, :]                                    # (1, K)
    true_eoo_col  = first_true_eoo[:, None]                        # (n, 1)
    pred_eoo_col  = first_pred_eoo[:, None]
    after_true = (true_eoo_col >= 0) & (idx > true_eoo_col)
    after_pred = (
        (pred_eoo_col >= 0) & (idx > pred_eoo_col) & (true_op_labels != eoo_token)
    )
    after = (after_true | after_pred) & (~exclude)
    exclude_reason[after] = "after_eoo"
    exclude |= after

    # planned_step (1-indexed). Defined relative to Unplanned/UNK skips only —
    # not shifted/decremented for EOO-related exclusions (see plan §2b).
    planned_step = np.full((n, K), np.nan, dtype=float)
    planned_mask = ~is_skipped
    i_grid = np.broadcast_to(np.arange(K, dtype=np.int64)[None, :], (n, K))
    planned_step[planned_mask] = (
        i_grid[planned_mask] - cum_strict[planned_mask] + 1
    ).astype(float)

    eval_weights = np.where(exclude, 0.0, 1.0).astype(np.float32)

    return {
        "shift_indices":   shift_indices,
        "planned_step":    planned_step,
        "exclude":         exclude,
        "exclude_reason":  exclude_reason,
        "eval_weights":    eval_weights,
        "first_true_eoo":  first_true_eoo,
        "first_pred_eoo":  first_pred_eoo,
    }


def _first_index_of(arr_2d: np.ndarray, token) -> np.ndarray:
    """First column index of `token` per row; -1 if absent. Shape (n,)."""
    matches = (arr_2d == token)
    has_match = matches.any(axis=1)
    first = matches.argmax(axis=1)
    out = np.where(has_match, first, -1).astype(np.int64)
    return out


def shift_along_axis1(arr: np.ndarray, shift_indices: np.ndarray, fill) -> np.ndarray:
    """Shift `arr` along axis 1 using `shift_indices`.

    `arr` has shape (n, K, ...) — any number of trailing dims. `shift_indices`
    has shape (n, K) and contains source column indices into `arr` (or any
    negative value to indicate NULL — those positions are filled with `fill`).
    Output has the same shape as `arr`.
    """
    n, K = arr.shape[:2]
    safe = np.where(shift_indices >= 0, shift_indices, 0).astype(np.int64)
    rows = np.arange(n)[:, None]
    out = arr[rows, safe]                                          # gather along axis 1
    invalid = shift_indices < 0
    if not invalid.any():
        return out
    # Broadcast the (n, K) mask across any trailing dims of `arr`.
    mask = invalid
    for _ in range(arr.ndim - 2):
        mask = mask[..., None]
    mask_b = np.broadcast_to(mask, out.shape)
    if arr.dtype == object:
        # np.where coerces dtypes — keep object dtype by assigning in place.
        out = out.copy()
        out[mask_b] = fill
        return out
    return np.where(mask_b, fill, out)

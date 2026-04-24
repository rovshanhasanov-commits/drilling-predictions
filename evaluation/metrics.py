"""Pure-function metric primitives used by run_evaluation.py.

Nothing here depends on TF/Keras or on disk I/O. Inputs are numpy arrays (decoded
ids) and plain dicts; outputs are plain dicts and pandas DataFrames.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

HIERARCHY = ["phase", "phase_step", "major_ops_code", "operation"]


# ---------------------------------------------------------------------------
# Hierarchy validity
# ---------------------------------------------------------------------------

def build_hierarchy_sets(master_csv: Path) -> dict:
    """Return parent->set(child) maps built from Master_Data_With_ID.csv.

    Keys: 'phase_to_step', 'step_to_moc', 'moc_to_op'.
    Values: dict[str, set[str]].
    """
    df = pd.read_csv(master_csv, usecols=["Phase", "Phase_Step", "Major_Ops_Code", "Operation"])
    df = df.dropna(subset=["Phase", "Phase_Step", "Major_Ops_Code", "Operation"])

    phase_to_step: dict[str, set[str]] = defaultdict(set)
    step_to_moc: dict[str, set[str]] = defaultdict(set)
    moc_to_op: dict[str, set[str]] = defaultdict(set)

    for p, s, m, o in zip(df["Phase"], df["Phase_Step"], df["Major_Ops_Code"], df["Operation"]):
        phase_to_step[p].add(s)
        step_to_moc[s].add(m)
        moc_to_op[m].add(o)

    return {"phase_to_step": dict(phase_to_step), "step_to_moc": dict(step_to_moc), "moc_to_op": dict(moc_to_op)}


def hierarchy_valid_mask(
    phase_labels: np.ndarray,
    phase_step_labels: np.ndarray,
    major_ops_code_labels: np.ndarray,
    operation_labels: np.ndarray,
    sets: dict,
    eoo_token: str,
) -> np.ndarray:
    """Return bool array (same shape as inputs) where True = all parent->child links valid.

    EOO tuples are treated as valid (well-end sentinel, no hierarchy to check).
    """
    shape = phase_labels.shape
    flat_p = phase_labels.flatten()
    flat_s = phase_step_labels.flatten()
    flat_m = major_ops_code_labels.flatten()
    flat_o = operation_labels.flatten()

    valid = np.ones(flat_p.shape, dtype=bool)
    for i in range(len(flat_p)):
        p, s, m, o = flat_p[i], flat_s[i], flat_m[i], flat_o[i]
        if p == eoo_token or s == eoo_token or m == eoo_token or o == eoo_token:
            continue
        if s not in sets["phase_to_step"].get(p, set()):
            valid[i] = False
            continue
        if m not in sets["step_to_moc"].get(s, set()):
            valid[i] = False
            continue
        if o not in sets["moc_to_op"].get(m, set()):
            valid[i] = False
    return valid.reshape(shape)


def conditional_accuracy(parent_pred: np.ndarray, parent_true: np.ndarray,
                         child_pred: np.ndarray, child_true: np.ndarray,
                         weights: np.ndarray | None = None) -> float:
    """P(child correct | parent correct). Returns nan if no parent is correct.

    If `weights` is provided (same shape as the inputs), positions with weight <= 0
    are dropped before computing the mask — used to exclude masked-target rows
    from the conditional accuracy calculation.
    """
    mask = (parent_pred == parent_true)
    if weights is not None:
        mask &= (weights > 0)
    if not mask.any():
        return float("nan")
    return float((child_pred[mask] == child_true[mask]).mean())


# ---------------------------------------------------------------------------
# Duration regression
# ---------------------------------------------------------------------------

def duration_metrics(
    pred_hours: np.ndarray,
    true_hours: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """MAE, MedAE, p95 abs err — overall and for actual>8h subset.

    If `weights` is provided, positions with weight <= 0 are excluded so that
    masked-duration rows (UNK residuals, Unplanned ops) don't pollute the metrics.
    """
    pred = pred_hours.flatten()
    true = true_hours.flatten()
    if weights is not None:
        keep = weights.flatten() > 0
        pred = pred[keep]
        true = true[keep]
    err = np.abs(pred - true)

    out = {
        "mae_hours": float(err.mean()),
        "medae_hours": float(np.median(err)),
        "p95_abs_err_hours": float(np.percentile(err, 95)),
    }

    long_mask = true > 8.0
    if long_mask.any():
        long_err = np.abs(pred[long_mask] - true[long_mask])
        out["mae_long_ops_hours"] = float(long_err.mean())
        # R² on long ops
        ss_res = float(np.sum((true[long_mask] - pred[long_mask]) ** 2))
        ss_tot = float(np.sum((true[long_mask] - true[long_mask].mean()) ** 2))
        out["r2_long_ops"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        out["mae_long_ops_hours"] = float("nan")
        out["r2_long_ops"] = float("nan")

    return out


# ---------------------------------------------------------------------------
# Duration bin (classification head)
# ---------------------------------------------------------------------------

def bin_center_mae(
    pred_bin_ids: np.ndarray,
    true_hours: np.ndarray,
    bin_centers: dict,
    bin_classes: list[str],
    weights: np.ndarray | None = None,
    long_threshold: float | None = 8.0,
) -> dict:
    """MAE in hours between top-1 bin's center and true raw `Duration hours`.

    Cross-compatible companion to `duration_metrics()` so the same eval report
    can carry numbers from either the bin head or the regression head.

    Skips:
      - rows where `weights` (if given) has weight <= 0 (Unplanned / UNK / NaN-dur masked)
      - rows where the predicted bin's center is NaN (sentinel predicted —
        EOO has center 0.0 and IS counted; Unplanned/UNK have NaN centers
        and are NOT counted, since "hours" doesn't apply)
    """
    pred = pred_bin_ids.flatten()
    true = true_hours.flatten()

    centers_arr = np.asarray(
        [bin_centers.get(c, float("nan")) for c in bin_classes], dtype=np.float64
    )
    pred_centers = centers_arr[pred]

    keep = np.isfinite(pred_centers)
    if weights is not None:
        keep &= (weights.flatten() > 0)
    if not keep.any():
        return {"bin_center_mae_hours": float("nan"),
                "bin_center_mae_long_hours": float("nan")}

    err = np.abs(pred_centers[keep] - true[keep])
    out = {"bin_center_mae_hours": float(err.mean())}

    if long_threshold is not None:
        long_keep = keep & (true > float(long_threshold))
        if long_keep.any():
            out["bin_center_mae_long_hours"] = float(
                np.abs(pred_centers[long_keep] - true[long_keep]).mean()
            )
        else:
            out["bin_center_mae_long_hours"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Confusion pairs
# ---------------------------------------------------------------------------

def top_confused_pairs(
    true_ids: np.ndarray,
    pred_ids: np.ndarray,
    class_labels: list[str],
    top_n: int = 20,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return a tidy DataFrame of the top-N most confused (true, pred) pairs.

    Columns: true_label, pred_label, count, pct_of_true.

    If `weights` is provided (same shape as `true_ids`/`pred_ids`), positions
    with weight <= 0 are dropped before building the confusion matrix. Used to
    exclude Unplanned / UNK targets (the model structurally can't predict them,
    so they'd dominate the confusion table as trivial misses).
    """
    y_true = true_ids.flatten()
    y_pred = pred_ids.flatten()
    if weights is not None:
        keep = weights.flatten() > 0
        y_true = y_true[keep]
        y_pred = y_pred[keep]
    labels = list(range(len(class_labels)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Zero the diagonal so we pick OFF-diagonal pairs.
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat_idx = np.argsort(off.flatten())[::-1][:top_n]

    per_true_totals = cm.sum(axis=1)
    rows = []
    for k in flat_idx:
        i, j = divmod(int(k), cm.shape[1])
        count = int(off[i, j])
        if count == 0:
            break
        denom = int(per_true_totals[i])
        pct = (count / denom) if denom > 0 else 0.0
        rows.append({
            "true_label": class_labels[i],
            "pred_label": class_labels[j],
            "count": count,
            "pct_of_true": round(pct, 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-well aggregation
# ---------------------------------------------------------------------------

def per_well_accuracy(
    wells: list[str],
    pred_ids: dict,
    true_ids: dict,
    pred_duration_hours: np.ndarray | None,
    true_duration_hours: np.ndarray | None,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Group sequences by well, compute mean accuracy per head + duration MAE.

    pred_ids[head], true_ids[head]: (n_sequences, n_future) int arrays.

    `weights` (optional): {head_name: (n_sequences, n_future) float array}. When
    a head has a weight array, rows with weight <= 0 are excluded from that
    head's per-well mean (a well with all masked rows gets NaN for that head).
    Pass `weights["duration"]` to similarly gate the duration MAE.
    """
    weights = weights or {}
    wells_arr = np.asarray(wells)
    rows = []
    for w in pd.unique(wells_arr):
        mask = wells_arr == w
        row = {"well_name": str(w), "n_sequences": int(mask.sum())}
        for h in HIERARCHY:
            if h in pred_ids and h in true_ids:
                hits = (pred_ids[h][mask] == true_ids[h][mask])
                if h in weights:
                    wmask = weights[h][mask] > 0
                    hits = hits[wmask]
                row[f"{h}_top1"] = float(hits.mean()) if hits.size else float("nan")
        if pred_duration_hours is not None and true_duration_hours is not None:
            err = np.abs(pred_duration_hours[mask] - true_duration_hours[mask])
            if "duration" in weights:
                wmask = weights["duration"][mask] > 0
                err = err[wmask]
            row["duration_mae"] = float(err.mean()) if err.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("well_name").reset_index(drop=True)


def well_accuracy_std(per_well_df: pd.DataFrame) -> dict:
    """Cross-well std of per-head accuracy. High std = domain shift across wells."""
    out = {}
    for h in HIERARCHY:
        col = f"{h}_top1"
        if col in per_well_df.columns:
            # nanstd so wells where this head was fully masked don't zero the result
            out[h] = float(per_well_df[col].dropna().std()) if len(per_well_df) > 1 else 0.0
    return out


# ---------------------------------------------------------------------------
# Step-1 vs step-K drop
# ---------------------------------------------------------------------------

def step1_vs_last_drop(per_step_top1: dict) -> dict:
    """Given {head: [per-step top1]}, return {head: acc_step1 - acc_lastStep}."""
    return {h: float(v[0] - v[-1]) for h, v in per_step_top1.items() if len(v) >= 2}

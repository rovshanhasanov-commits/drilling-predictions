"""Disk writers for evaluation outputs. All paths are relative to out_dir."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import HIERARCHY


def write_summary(out_dir: Path, summary: dict) -> Path:
    path = out_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    return path


def write_run_config(out_dir: Path, cli_args: dict, model_config: dict) -> Path:
    payload = {
        "evaluated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cli_args": cli_args,
        "model_config": model_config,
    }
    path = out_dir / "run_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


def write_per_step_accuracy(
    out_dir: Path,
    tf_scores: dict,
    ar_scores: dict,
) -> Path:
    """Tidy table: one row per (head, mode, step)."""
    rows = []
    for head in HIERARCHY:
        for mode, scores in (("tf", tf_scores), ("ar", ar_scores)):
            if head not in scores:
                continue
            s = scores[head]
            for i, (t1, tk) in enumerate(zip(s["per_step_top1"], s["per_step_topk"])):
                rows.append({
                    "head": head, "mode": mode, "step": i + 1,
                    "top1": round(t1, 6), f"top{s['k']}": round(tk, 6),
                })
    df = pd.DataFrame(rows)
    path = out_dir / "per_step_accuracy.csv"
    df.to_csv(path, index=False)
    return path


def write_confusion_csvs(out_dir: Path, confusion_dfs: dict) -> list[Path]:
    paths = []
    for head, df in confusion_dfs.items():
        path = out_dir / f"confusion_{head}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def write_per_well(out_dir: Path, per_well_df: pd.DataFrame) -> Path:
    path = out_dir / "per_well_accuracy.csv"
    per_well_df.to_csv(path, index=False)
    return path


def write_predictions_csv(
    out_dir: Path,
    wells: list[str],
    start_idx: list[int],
    true_labels: dict,        # {head: (n, K) str array}
    pred_top1_labels: dict,
    pred_topk_labels: dict,   # {head: (n, K, top_k) str array}
    true_duration: np.ndarray,
    pred_duration: np.ndarray,
    hierarchy_valid: np.ndarray,   # (n, K) bool
    tuple_topk_labels: np.ndarray | None = None,   # (n, K, top_k, 4) str; HIERARCHY order on last axis
    tuple_topk_logprob: np.ndarray | None = None,  # (n, K, top_k) float, raw joint log-prob
    tuple_topk_prob: np.ndarray | None = None,     # (n, K, top_k) float, renormalized over L
) -> Path:
    """One row per (sequence_idx, step). Large but grep-friendly.

    When `tuple_topk_labels` is provided (constrained AR path), additional
    columns `pred_tuple_{i}_{phase,phase_step,major_ops_code,operation}` +
    `pred_tuple_{i}_logprob` (raw joint log-prob) + `pred_tuple_{i}_prob`
    (renormalized over L, so one row's top-K probs sum to <= 1 and all-of-L
    probs sum to exactly 1) are emitted for i in 0..K-1. The per-head
    pred_{head}_top1 / pred_{head}_top3 columns stay populated (they come
    from the coordinated tuple's first / top-K rows under constraints) so
    downstream consumers don't break.
    """
    n, n_future = hierarchy_valid.shape
    have_tuples = tuple_topk_labels is not None and tuple_topk_logprob is not None
    K_tuple = tuple_topk_labels.shape[2] if have_tuples else 0

    records = []
    for seq in range(n):
        for step in range(n_future):
            row = {
                "sequence_idx": seq,
                "step": step + 1,
                "well_name": wells[seq],
                "start_row_id": start_idx[seq],
            }
            for head in HIERARCHY:
                if head in true_labels:
                    row[f"true_{head}"] = true_labels[head][seq, step]
                    row[f"pred_{head}_top1"] = pred_top1_labels[head][seq, step]
                    row[f"pred_{head}_top3"] = "|".join(pred_topk_labels[head][seq, step].tolist())
                    row[f"{head}_in_top3"] = bool(
                        true_labels[head][seq, step] in pred_topk_labels[head][seq, step]
                    )
            row["true_duration_hours"] = round(float(true_duration[seq, step]), 4)
            row["pred_duration_hours"] = round(float(pred_duration[seq, step]), 4)
            row["duration_abs_err"] = round(
                float(abs(pred_duration[seq, step] - true_duration[seq, step])), 4
            )
            row["hierarchy_valid"] = bool(hierarchy_valid[seq, step])

            if have_tuples:
                for i in range(K_tuple):
                    for j, h in enumerate(HIERARCHY):
                        row[f"pred_tuple_{i}_{h}"] = tuple_topk_labels[seq, step, i, j]
                    row[f"pred_tuple_{i}_logprob"] = round(
                        float(tuple_topk_logprob[seq, step, i]), 4
                    )
                    if tuple_topk_prob is not None:
                        row[f"pred_tuple_{i}_prob"] = round(
                            float(tuple_topk_prob[seq, step, i]), 4
                        )
                # Did the ground-truth tuple appear in the top-K legal tuples?
                true_tup = tuple(true_labels[h][seq, step] for h in HIERARCHY if h in true_labels)
                if len(true_tup) == 4:
                    tuples_at_step = tuple_topk_labels[seq, step]   # (top_k, 4)
                    row["tuple_in_topk"] = bool(
                        any(tuple(tuples_at_step[i]) == true_tup for i in range(K_tuple))
                    )

            records.append(row)

    df = pd.DataFrame.from_records(records)
    path = out_dir / "predictions.csv"
    df.to_csv(path, index=False)
    return path


def _json_default(obj):
    """Make numpy scalars and dates JSON-serializable."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Not JSON serializable: {type(obj)}")

"""Duration-bin classification head — pure functions.

The current single-float duration regression head suffers on the long tail
(r2_long_ops < 0). This module provides the building blocks for an alternative
classification head that predicts duration as one of a small set of bins, with
sentinel classes for EOO / Unplanned / UNK to align with the masking
semantics already used by the four hierarchy heads.

Coexists with the regression head — yaml `target_variables` toggles which one(s)
train. See improvements/Duration Binning.md for the full design.

No I/O, no TensorFlow. Designed to be unit-testable.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SENTINELS: tuple[str, ...] = ("EOO", "Unplanned", "UNK")


def assign_duration_bin(
    hours: pd.Series,
    edges: list[float],
    labels: list[str],
) -> pd.Series:
    """Bucket raw `Duration hours` into one of `labels`.

    Right-inclusive (DAX-style): `0.25` lands in `"≤0.25 hr"`, `0.251` lands in
    `"0.25-0.5 hr"`. NaN inputs map to the `"UNK"` sentinel so the result is
    always a valid string class.

    Args:
        hours:  raw durations in hours (NaNs allowed).
        edges:  ascending interior thresholds, e.g. [0.25, 0.5, 1.0, 2.0, 5.0, 10.0].
        labels: class names, length = len(edges) + 1. The last label covers
                the unbounded right tail (`(edges[-1], inf)`).

    Returns:
        pd.Series[str] with the same index, values in `labels` + `{"UNK"}`.
    """
    if len(labels) != len(edges) + 1:
        raise ValueError(
            f"assign_duration_bin: need {len(edges) + 1} labels for {len(edges)} edges, "
            f"got {len(labels)}."
        )
    bins = [-np.inf, *edges, np.inf]
    cat = pd.cut(hours, bins=bins, labels=labels, include_lowest=True, right=True)
    out = cat.astype(object)
    out = out.where(~hours.isna(), other="UNK")
    return out.astype(str)


def assign_duration_bin_scalar(h: float, edges: list[float], labels: list[str]) -> str:
    """Scalar shim around `assign_duration_bin` — convenient for unit tests."""
    return assign_duration_bin(pd.Series([h]), edges, labels).iloc[0]


def fit_bin_encoder(labels: list[str], sentinels: tuple[str, ...] = SENTINELS) -> LabelEncoder:
    """Construct a LabelEncoder with classes in the EXACT order `labels + sentinels`.

    `LabelEncoder().fit(...)` sorts alphabetically, which would scramble the
    Unicode `≤` and digit-prefixed labels (`"10+ hrs"` would land before
    `"5-10 hrs"`). We bypass `.fit()` entirely and assign `.classes_` directly
    so id-to-label is deterministic and matches the doc / yaml.
    """
    classes = list(labels) + list(sentinels)
    le = LabelEncoder()
    le.classes_ = np.array(classes, dtype=object)
    return le


def compute_bin_centers(
    df_train: pd.DataFrame,
    edges: list[float],
    labels: list[str],
    hours_col: str = "Duration hours",
) -> dict[str, float]:
    """Empirical median of training `Duration hours` per bin, plus sentinel centers.

    Used as the "point estimate" when surfacing top-K bins to the LLM and when
    computing bin-center MAE in eval. Sentinel values:

        EOO       -> 0.0   (matches the well-end semantic)
        Unplanned -> NaN   (skipped by metrics / contract)
        UNK       -> NaN   (skipped by metrics / contract)

    If a non-sentinel bin has zero training rows (rare with default edges +
    137K rows; possible under custom narrow bands), fall back to the arithmetic
    midpoint of that bin's edges and emit a warning. Open-ended first / last
    bins get edges[0]/2 and edges[-1]*1.5 respectively.
    """
    bin_col = assign_duration_bin(df_train[hours_col], edges, labels)
    full_edges = [-np.inf, *edges, np.inf]
    centers: dict[str, float] = {}

    for i, label in enumerate(labels):
        mask = bin_col == label
        if mask.any():
            centers[label] = float(df_train.loc[mask, hours_col].median())
            continue
        lo, hi = full_edges[i], full_edges[i + 1]
        if not np.isfinite(lo):
            fallback = float(hi) / 2.0
        elif not np.isfinite(hi):
            fallback = float(lo) * 1.5
        else:
            fallback = (float(lo) + float(hi)) / 2.0
        warnings.warn(
            f"compute_bin_centers: bin {label!r} has no training rows; "
            f"falling back to midpoint {fallback}.",
            stacklevel=2,
        )
        centers[label] = fallback

    centers["EOO"]       = 0.0
    centers["Unplanned"] = float("nan")
    centers["UNK"]       = float("nan")
    return centers

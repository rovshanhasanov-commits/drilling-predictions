"""Unit tests for preprocessing/bins.py — pure functions, no TF dependency.

Run from the repo root:
    python -m pytest tests/test_duration_bins.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from preprocessing.bins import (
    SENTINELS,
    assign_duration_bin,
    assign_duration_bin_scalar,
    compute_bin_centers,
    fit_bin_encoder,
)

DEFAULT_EDGES = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
DEFAULT_LABELS = [
    "≤0.25 hr", "0.25-0.5 hr", "0.5-1 hr",
    "1-2 hrs", "2-5 hrs", "5-10 hrs", "10+ hrs",
]


# ---------------------------------------------------------------------------
# assign_duration_bin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hours,expected", [
    (0.0,    "≤0.25 hr"),       # left edge of first bin (include_lowest)
    (0.1,    "≤0.25 hr"),
    (0.25,   "≤0.25 hr"),       # right-inclusive
    (0.251,  "0.25-0.5 hr"),
    (0.5,    "0.25-0.5 hr"),    # right-inclusive
    (0.5001, "0.5-1 hr"),
    (1.0,    "0.5-1 hr"),
    (1.5,    "1-2 hrs"),
    (3.0,    "2-5 hrs"),
    (5.0,    "2-5 hrs"),
    (7.0,    "5-10 hrs"),
    (10.0,   "5-10 hrs"),       # right-inclusive
    (10.001, "10+ hrs"),
    (999.0,  "10+ hrs"),
])
def test_assign_duration_bin_right_inclusive(hours, expected):
    assert assign_duration_bin_scalar(hours, DEFAULT_EDGES, DEFAULT_LABELS) == expected


def test_assign_duration_bin_nan_to_unk():
    assert assign_duration_bin_scalar(float("nan"), DEFAULT_EDGES, DEFAULT_LABELS) == "UNK"


def test_assign_duration_bin_vectorized_preserves_index():
    s = pd.Series([0.1, 1.5, 12.0, float("nan")], index=[10, 20, 30, 40])
    out = assign_duration_bin(s, DEFAULT_EDGES, DEFAULT_LABELS)
    assert list(out.index) == [10, 20, 30, 40]
    assert out.tolist() == ["≤0.25 hr", "1-2 hrs", "10+ hrs", "UNK"]


def test_assign_duration_bin_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="labels"):
        assign_duration_bin(pd.Series([0.1]), edges=[0.25, 0.5], labels=["a", "b"])  # need 3 labels


# ---------------------------------------------------------------------------
# fit_bin_encoder
# ---------------------------------------------------------------------------

def test_fit_bin_encoder_classes_in_exact_order():
    """LabelEncoder.fit() would sort alphabetically and break id->label
    determinism. fit_bin_encoder must preserve `labels + sentinels` order."""
    le = fit_bin_encoder(DEFAULT_LABELS)
    expected = DEFAULT_LABELS + list(SENTINELS)
    assert list(le.classes_) == expected
    assert le.classes_.dtype == object


def test_fit_bin_encoder_n_classes_equals_labels_plus_three():
    le = fit_bin_encoder(DEFAULT_LABELS)
    assert len(le.classes_) == len(DEFAULT_LABELS) + 3


def test_fit_bin_encoder_transform_round_trip():
    le = fit_bin_encoder(DEFAULT_LABELS)
    sample = ["≤0.25 hr", "1-2 hrs", "EOO", "Unplanned", "UNK"]
    ids = le.transform(sample)
    back = le.inverse_transform(ids).tolist()
    assert back == sample
    # Sentinel ids are the last three (in declared order).
    assert ids[-3] == len(DEFAULT_LABELS) + 0   # EOO
    assert ids[-2] == len(DEFAULT_LABELS) + 1   # Unplanned
    assert ids[-1] == len(DEFAULT_LABELS) + 2   # UNK


# ---------------------------------------------------------------------------
# compute_bin_centers
# ---------------------------------------------------------------------------

def test_compute_bin_centers_medians_match_by_hand():
    df = pd.DataFrame({"Duration hours": [
        # ≤0.25 hr      -> [0.10, 0.20]              median 0.15
        0.10, 0.20,
        # 0.25-0.5 hr   -> [0.30, 0.40]              median 0.35
        0.30, 0.40,
        # 0.5-1 hr      -> [0.60, 0.80]              median 0.70
        0.60, 0.80,
        # 1-2 hrs       -> [1.50]                    median 1.50
        1.50,
        # 2-5 hrs       -> [3.00, 4.00]              median 3.50
        3.00, 4.00,
        # 5-10 hrs      -> [7.00]                    median 7.00
        7.00,
        # 10+ hrs       -> [12.00, 18.00]            median 15.00
        12.00, 18.00,
    ]})
    centers = compute_bin_centers(df, DEFAULT_EDGES, DEFAULT_LABELS)
    assert centers["≤0.25 hr"]    == pytest.approx(0.15)
    assert centers["0.25-0.5 hr"] == pytest.approx(0.35)
    assert centers["0.5-1 hr"]    == pytest.approx(0.70)
    assert centers["1-2 hrs"]     == pytest.approx(1.50)
    assert centers["2-5 hrs"]     == pytest.approx(3.50)
    assert centers["5-10 hrs"]    == pytest.approx(7.00)
    assert centers["10+ hrs"]     == pytest.approx(15.00)


def test_compute_bin_centers_sentinel_values():
    df = pd.DataFrame({"Duration hours": [0.1, 1.5, 12.0]})
    centers = compute_bin_centers(df, DEFAULT_EDGES, DEFAULT_LABELS)
    assert centers["EOO"] == 0.0
    assert math.isnan(centers["Unplanned"])
    assert math.isnan(centers["UNK"])


def test_compute_bin_centers_empty_bin_falls_back_to_midpoint():
    # Only 0.1 and 12.0 — 5 of the 7 bins are empty; expect midpoint fallbacks
    # for the empty middle bins, edges[0]/2 for the empty first bin (not empty
    # here), edges[-1]*1.5 for the empty last bin (not empty here).
    df = pd.DataFrame({"Duration hours": [0.1, 12.0]})
    with pytest.warns(UserWarning, match="no training rows"):
        centers = compute_bin_centers(df, DEFAULT_EDGES, DEFAULT_LABELS)
    # First and last bins aren't empty (0.1 and 12.0 land there) — medians.
    assert centers["≤0.25 hr"] == pytest.approx(0.10)
    assert centers["10+ hrs"]  == pytest.approx(12.00)
    # Middle bins are empty — arithmetic midpoint of finite edges.
    assert centers["0.25-0.5 hr"] == pytest.approx((0.25 + 0.5) / 2)
    assert centers["0.5-1 hr"]    == pytest.approx((0.5 + 1.0) / 2)
    assert centers["1-2 hrs"]     == pytest.approx((1.0 + 2.0) / 2)
    assert centers["2-5 hrs"]     == pytest.approx((2.0 + 5.0) / 2)
    assert centers["5-10 hrs"]    == pytest.approx((5.0 + 10.0) / 2)


def test_compute_bin_centers_empty_open_ended_bins():
    # Only middle values — both open-ended bins (first and last) are empty.
    df = pd.DataFrame({"Duration hours": [1.5, 3.0]})
    with pytest.warns(UserWarning):
        centers = compute_bin_centers(df, DEFAULT_EDGES, DEFAULT_LABELS)
    # First bin empty -> edges[0] / 2 = 0.125
    assert centers["≤0.25 hr"] == pytest.approx(0.25 / 2)
    # Last bin empty -> edges[-1] * 1.5 = 15.0
    assert centers["10+ hrs"]  == pytest.approx(10.0 * 1.5)

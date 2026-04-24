"""Unit tests for training/constraints.py — pure-numpy helpers, no TF dependency.

Run from the repo root:
    python -m pytest tests/test_constraints.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from training.constraints import (
    HIERARCHY,
    build_legal_tuples,
    joint_argmax,
    joint_topk_tuples,
    summarize_legal_tuples,
)


def _make_encoders(extras: dict | None = None) -> dict:
    """Construct LabelEncoders with a realistic class layout: domain classes +
    EOO + UNK sentinels, plus an `Unplanned` class on the op head.
    """
    classes = {
        "phase":          ["DRL", "CASING", "End of Operations", "UNK"],
        "phase_step":     ["LAT", "CIRC", "RIH", "End of Operations", "UNK"],
        "major_ops_code": ["DRILL", "TRIP", "CEMENT", "End of Operations", "UNK"],
        "operation":      ["DRL", "CIRC", "RIH", "PU", "End of Operations", "UNK", "Unplanned"],
    }
    if extras:
        for k, v in extras.items():
            classes[k] = v
    out = {}
    for h in HIERARCHY:
        le = LabelEncoder()
        le.fit(classes[h])
        out[h] = le
    return out


def _encode_row(encoders: dict, row: dict) -> dict:
    return {
        f"{h}_target_enc": int(list(encoders[h].classes_).index(row[h]))
        for h in HIERARCHY
    }


def _df_from_rows(encoders: dict, rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([_encode_row(encoders, r) for r in rows])


# ---------------------------------------------------------------------------
# build_legal_tuples
# ---------------------------------------------------------------------------

def test_build_legal_tuples_basic_enumeration():
    """Distinct tuples in the parquets should all appear in L, and the EOO
    tuple should be added even if the parquets don't contain it."""
    encoders = _make_encoders()
    rows = [
        {"phase": "DRL",    "phase_step": "LAT",  "major_ops_code": "DRILL",  "operation": "DRL"},
        {"phase": "DRL",    "phase_step": "LAT",  "major_ops_code": "DRILL",  "operation": "DRL"},   # dup
        {"phase": "DRL",    "phase_step": "CIRC", "major_ops_code": "DRILL",  "operation": "CIRC"},
        {"phase": "CASING", "phase_step": "RIH",  "major_ops_code": "TRIP",   "operation": "RIH"},
    ]
    df_train = _df_from_rows(encoders, rows[:2])
    df_val   = _df_from_rows(encoders, rows[2:3])
    df_test  = _df_from_rows(encoders, rows[3:])

    L = build_legal_tuples(df_train, df_val, df_test, encoders)

    # 3 distinct data-driven tuples + EOO tuple.
    assert L.shape == (4, 4)
    assert L.dtype == np.int32

    # EOO tuple must be present.
    eoo_ids = np.asarray([int(list(encoders[h].classes_).index("End of Operations")) for h in HIERARCHY])
    assert np.any(np.all(L == eoo_ids, axis=1))


def test_build_legal_tuples_excludes_unplanned_and_unk_from_op_axis():
    """Unplanned / UNK on the op axis are sentinels — they must never enter L."""
    encoders = _make_encoders()
    rows = [
        {"phase": "DRL", "phase_step": "LAT", "major_ops_code": "DRILL", "operation": "DRL"},
        {"phase": "DRL", "phase_step": "LAT", "major_ops_code": "DRILL", "operation": "Unplanned"},
        {"phase": "DRL", "phase_step": "LAT", "major_ops_code": "DRILL", "operation": "UNK"},
    ]
    df_train = _df_from_rows(encoders, rows)
    df_val   = df_train.iloc[0:0]   # empty but same schema
    df_test  = df_train.iloc[0:0]

    L = build_legal_tuples(df_train, df_val, df_test, encoders)

    unplanned_id = int(list(encoders["operation"].classes_).index("Unplanned"))
    unk_id       = int(list(encoders["operation"].classes_).index("UNK"))
    assert unplanned_id not in L[:, 3].tolist()
    assert unk_id       not in L[:, 3].tolist()


def test_build_legal_tuples_drops_unk_on_upper_axes():
    """UNK fallback on phase/phase_step/moc isn't a real hierarchy member."""
    encoders = _make_encoders()
    rows = [
        {"phase": "DRL", "phase_step": "UNK", "major_ops_code": "DRILL", "operation": "DRL"},  # UNK step
        {"phase": "DRL", "phase_step": "LAT", "major_ops_code": "UNK",   "operation": "DRL"},  # UNK moc
        {"phase": "DRL", "phase_step": "LAT", "major_ops_code": "DRILL", "operation": "DRL"},  # clean
    ]
    df_train = _df_from_rows(encoders, rows)
    df_val   = df_train.iloc[0:0]
    df_test  = df_train.iloc[0:0]

    L = build_legal_tuples(df_train, df_val, df_test, encoders)

    # Only the clean row + EOO should survive.
    assert L.shape[0] == 2


def test_build_legal_tuples_raises_on_missing_columns():
    encoders = _make_encoders()
    bad = pd.DataFrame({"phase_target_enc": [0]})   # missing other 3 enc cols
    with pytest.raises(KeyError, match="missing"):
        build_legal_tuples(bad, bad, bad, encoders)


# ---------------------------------------------------------------------------
# joint_argmax
# ---------------------------------------------------------------------------

def test_joint_argmax_picks_highest_joint_logprob():
    """Three legal tuples, two samples. Verify joint argmax chooses the tuple
    whose sum of per-head log-probs is highest, not the per-head argmaxes."""
    # Classes: phase has 2, step has 2, moc has 2, op has 2.
    L = np.array([
        [0, 0, 0, 0],     # tuple A
        [1, 1, 1, 1],     # tuple B
        [0, 1, 0, 1],     # tuple C (cross-axis)
    ], dtype=np.int32)

    # Sample 0: independent argmax would be (1, 0, 1, 0) but that combo isn't
    # in L. Joint argmax should pick A (0,0,0,0) if per-head 0's combined
    # log-prob exceeds B's.
    # Sample 1: Tune so B wins.
    probs = {
        "phase":          np.array([[0.45, 0.55], [0.10, 0.90]]),
        "phase_step":     np.array([[0.60, 0.40], [0.20, 0.80]]),
        "major_ops_code": np.array([[0.55, 0.45], [0.15, 0.85]]),
        "operation":      np.array([[0.60, 0.40], [0.10, 0.90]]),
    }

    chosen = joint_argmax(probs, L)
    assert chosen.shape == (2, 4)
    # Sample 0 — by hand: log(0.45*0.6*0.55*0.6) vs log(0.55*0.4*0.45*0.4)
    # vs log(0.45*0.4*0.55*0.4). A dominates.
    assert tuple(chosen[0].tolist()) == (0, 0, 0, 0)
    # Sample 1 — B's per-head 1 dominates.
    assert tuple(chosen[1].tolist()) == (1, 1, 1, 1)


def test_joint_argmax_raises_on_empty_L():
    with pytest.raises(ValueError, match="empty"):
        joint_argmax({h: np.ones((1, 2)) for h in HIERARCHY}, np.zeros((0, 4), dtype=np.int32))


# ---------------------------------------------------------------------------
# joint_topk_tuples
# ---------------------------------------------------------------------------

def test_joint_topk_tuples_returns_sorted_desc_logprobs():
    """Top-K result must be ordered by descending joint log-prob, and match
    what joint_argmax would return as position 0."""
    L = np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ], dtype=np.int32)

    probs = {
        "phase":          np.array([[0.55, 0.45]]),
        "phase_step":     np.array([[0.70, 0.30]]),
        "major_ops_code": np.array([[0.60, 0.40]]),
        "operation":      np.array([[0.65, 0.35]]),
    }

    tuples, log_probs, renorm_probs = joint_topk_tuples(probs, L, k=3)
    assert tuples.shape == (1, 3, 4)
    assert log_probs.shape == (1, 3)
    assert renorm_probs.shape == (1, 3)

    # Log-probs (and therefore renorm probs) must be strictly non-increasing.
    assert log_probs[0, 0] >= log_probs[0, 1] >= log_probs[0, 2]
    assert renorm_probs[0, 0] >= renorm_probs[0, 1] >= renorm_probs[0, 2]

    # Top-1 must equal joint_argmax's pick.
    expected_top1 = joint_argmax(probs, L)
    assert tuple(tuples[0, 0].tolist()) == tuple(expected_top1[0].tolist())


def test_joint_topk_tuples_clips_k_to_L_size():
    L = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32)
    probs = {h: np.array([[0.5, 0.5]]) for h in HIERARCHY}

    tuples, log_probs, renorm_probs = joint_topk_tuples(probs, L, k=10)   # k > len(L)
    assert tuples.shape == (1, 2, 4)
    assert log_probs.shape == (1, 2)
    assert renorm_probs.shape == (1, 2)


def test_joint_topk_tuples_renorm_sums_to_one_over_full_L():
    """When k == |L|, the returned renorm_probs must sum to 1.0 per row (up to fp)."""
    L = np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 1],
    ], dtype=np.int32)

    probs = {
        "phase":          np.array([[0.55, 0.45], [0.10, 0.90]]),
        "phase_step":     np.array([[0.70, 0.30], [0.25, 0.75]]),
        "major_ops_code": np.array([[0.60, 0.40], [0.20, 0.80]]),
        "operation":      np.array([[0.65, 0.35], [0.15, 0.85]]),
    }

    _, _, renorm_probs = joint_topk_tuples(probs, L, k=L.shape[0])
    row_sums = renorm_probs.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"rows sum to {row_sums}, expected 1.0"


def test_joint_topk_tuples_renorm_top_k_sum_lt_one_when_truncated():
    """With k < |L|, summed renorm probs should be <= 1 (missing tail mass)."""
    L = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int32)
    # Near-uniform probs → mass spreads across all 4 tuples, so top-2 sum < 1.
    probs = {h: np.array([[0.5, 0.5]]) for h in HIERARCHY}

    _, _, renorm_probs = joint_topk_tuples(probs, L, k=2)
    row_sum = renorm_probs.sum(axis=-1)[0]
    assert 0.0 < row_sum < 1.0, f"top-2 sum should be <1 for uniform priors, got {row_sum}"


def test_joint_topk_tuples_renorm_matches_manual_softmax():
    """Renormalized probs must equal softmax(scores) over L."""
    L = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]], dtype=np.int32)
    probs = {
        "phase":          np.array([[0.6, 0.4]]),
        "phase_step":     np.array([[0.7, 0.3]]),
        "major_ops_code": np.array([[0.5, 0.5]]),
        "operation":      np.array([[0.8, 0.2]]),
    }

    _, log_probs_top, renorm_top = joint_topk_tuples(probs, L, k=L.shape[0])

    # Manual softmax over the raw log-probs.
    lp = log_probs_top[0]                       # (3,)
    m = lp.max()
    exp_shifted = np.exp(lp - m)
    manual = exp_shifted / exp_shifted.sum()

    assert np.allclose(renorm_top[0], manual, atol=1e-6)


# ---------------------------------------------------------------------------
# summarize_legal_tuples
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5D legal tuples (include_bins=True)
# ---------------------------------------------------------------------------

def _make_encoders_with_bin():
    """4-hierarchy encoders + a duration_bin encoder with the forced class
    order (bins + EOO + Unplanned + UNK) used by preprocessing/bins.py."""
    encoders = _make_encoders()
    from preprocessing.bins import fit_bin_encoder
    encoders["duration_bin"] = fit_bin_encoder(
        ["≤0.25 hr", "0.25-0.5 hr", "0.5-1 hr", "1-2 hrs", "2-5 hrs", "5-10 hrs", "10+ hrs"]
    )
    return encoders


def _df_from_rows_with_bin(encoders, rows: list[dict]) -> pd.DataFrame:
    bin_le = encoders["duration_bin"]
    bin_c2i = {c: i for i, c in enumerate(bin_le.classes_)}
    out = []
    for r in rows:
        rec = _encode_row(encoders, r)
        rec["duration_bin_target_enc"] = bin_c2i[r["duration_bin"]]
        out.append(rec)
    return pd.DataFrame(out)


def test_build_legal_tuples_5d_returns_n_by_5():
    encoders = _make_encoders_with_bin()
    rows = [
        {"phase": "DRL", "phase_step": "LAT", "major_ops_code": "DRILL",
         "operation": "DRL", "duration_bin": "1-2 hrs"},
        {"phase": "DRL", "phase_step": "LAT", "major_ops_code": "DRILL",
         "operation": "DRL", "duration_bin": "2-5 hrs"},  # same 4-tuple, different bin
        {"phase": "CASING", "phase_step": "RIH", "major_ops_code": "TRIP",
         "operation": "RIH", "duration_bin": "0.5-1 hr"},
    ]
    df = _df_from_rows_with_bin(encoders, rows)
    L = build_legal_tuples(df, df.iloc[0:0], df.iloc[0:0], encoders, include_bins=True)
    assert L.shape == (4, 5)   # 3 distinct 5-tuples + EOO
    assert L.dtype == np.int32


def test_build_legal_tuples_5d_excludes_bin_sentinels_from_5th_axis():
    encoders = _make_encoders_with_bin()
    rows = [
        {"phase": "DRL", "phase_step": "LAT", "major_ops_code": "DRILL",
         "operation": "DRL", "duration_bin": "1-2 hrs"},     # OK
        {"phase": "DRL", "phase_step": "LAT", "major_ops_code": "DRILL",
         "operation": "DRL", "duration_bin": "Unplanned"},   # excluded
        {"phase": "DRL", "phase_step": "LAT", "major_ops_code": "DRILL",
         "operation": "DRL", "duration_bin": "UNK"},          # excluded
    ]
    df = _df_from_rows_with_bin(encoders, rows)
    L = build_legal_tuples(df, df.iloc[0:0], df.iloc[0:0], encoders, include_bins=True)

    bin_c2i = {c: i for i, c in enumerate(encoders["duration_bin"].classes_)}
    assert bin_c2i["Unplanned"] not in L[:, 4].tolist()
    assert bin_c2i["UNK"]       not in L[:, 4].tolist()


def test_build_legal_tuples_5d_eoo_row_has_bin_eoo_id():
    encoders = _make_encoders_with_bin()
    rows = [
        {"phase": "DRL", "phase_step": "LAT", "major_ops_code": "DRILL",
         "operation": "DRL", "duration_bin": "1-2 hrs"},
    ]
    df = _df_from_rows_with_bin(encoders, rows)
    L = build_legal_tuples(df, df.iloc[0:0], df.iloc[0:0], encoders, include_bins=True)

    bin_eoo_id = list(encoders["duration_bin"].classes_).index("EOO")
    eoo_4d = [int(list(encoders[h].classes_).index("End of Operations")) for h in HIERARCHY]
    expected_eoo_5d = np.asarray(eoo_4d + [bin_eoo_id], dtype=np.int32)
    assert _row_in(L, expected_eoo_5d), \
        f"5D EOO row {expected_eoo_5d} not present in L = {L}"


def _row_in(arr: np.ndarray, row: np.ndarray) -> bool:
    return bool(np.any(np.all(arr == row, axis=1)))


def test_build_legal_tuples_5d_raises_without_bin_encoder():
    encoders = _make_encoders()  # no duration_bin
    rows = [{"phase": "DRL", "phase_step": "LAT", "major_ops_code": "DRILL", "operation": "DRL"}]
    df = _df_from_rows(encoders, rows)
    df["duration_bin_target_enc"] = 0   # column present but encoder missing
    with pytest.raises(KeyError, match="duration_bin"):
        build_legal_tuples(df, df.iloc[0:0], df.iloc[0:0], encoders, include_bins=True)


def test_joint_topk_tuples_5d_accepts_and_ranks():
    """5-axis log-prob sum: bin head contributes a 5th term to the score."""
    L = np.array([
        [0, 0, 0, 0, 0],   # tuple A: all-class-0
        [1, 1, 1, 1, 1],   # tuple B: all-class-1
        [0, 0, 0, 0, 1],   # tuple C: bin differs from A only on the 5th axis
    ], dtype=np.int32)

    probs = {
        "phase":          np.array([[0.6, 0.4]]),
        "phase_step":     np.array([[0.7, 0.3]]),
        "major_ops_code": np.array([[0.55, 0.45]]),
        "operation":      np.array([[0.65, 0.35]]),
        "duration_bin":   np.array([[0.55, 0.45]]),
    }
    head_names = HIERARCHY + ["duration_bin"]

    tuples, log_probs, renorm = joint_topk_tuples(probs, L, k=3, head_names=head_names)
    assert tuples.shape == (1, 3, 5)
    assert log_probs[0, 0] >= log_probs[0, 1] >= log_probs[0, 2]
    # A wins (all per-head 0's strongly dominate).
    assert tuple(tuples[0, 0].tolist()) == (0, 0, 0, 0, 0)


def test_joint_argmax_5d_dimension_check():
    """Pass a 5D L without head_names override → _score_tuples should raise."""
    L = np.zeros((1, 5), dtype=np.int32)
    probs = {h: np.ones((1, 2)) for h in HIERARCHY}
    with pytest.raises(ValueError, match="L has 5 columns"):
        joint_argmax(probs, L)   # default head_names=HIERARCHY (len 4)


# ---------------------------------------------------------------------------
# summarize_legal_tuples
# ---------------------------------------------------------------------------

def test_summarize_legal_tuples_reports_counts():
    encoders = _make_encoders()
    rows = [
        {"phase": "DRL",    "phase_step": "LAT",  "major_ops_code": "DRILL", "operation": "DRL"},
        {"phase": "CASING", "phase_step": "RIH",  "major_ops_code": "TRIP",  "operation": "RIH"},
    ]
    df = _df_from_rows(encoders, rows)
    L = build_legal_tuples(df, df.iloc[0:0], df.iloc[0:0], encoders)

    stats = summarize_legal_tuples(L, encoders)
    # 2 data rows + EOO.
    assert stats["n_tuples"] == 3
    # 3 distinct phase ids (DRL, CASING, EOO).
    assert stats["n_unique_phase"] == 3
    # vocab_{head} mirrors encoder size.
    for h in HIERARCHY:
        assert stats[f"vocab_{h}"] == len(encoders[h].classes_)

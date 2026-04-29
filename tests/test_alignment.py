"""Unit tests for evaluation/alignment.py — see improvements/Learning rate
changes and Eval fixes.md §2b/§2d for the spec these encode."""

from __future__ import annotations

import numpy as np

from evaluation.alignment import compute_alignment, shift_along_axis1


# Matches the worked example in the plan: K=8 horizon, Unplanned at positions
# 4 and 6 (1-indexed -> 0-indexed 3 and 5).
def _worked_example_truth():
    return np.array([["T1", "T2", "T3", "Unplanned", "T5", "Unplanned", "T7", "T8"]], dtype=object)


def _worked_example_pred():
    return np.array([["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]], dtype=object)


def test_shift_indices_match_worked_example():
    true_op = _worked_example_truth()
    pred_op = _worked_example_pred()

    align = compute_alignment(true_op, pred_op)

    # 1-indexed positions in the plan map to 0-indexed columns:
    #   pos 1..3: source 0,1,2  (no shift)
    #   pos 4 (UNP): NULL (-1)
    #   pos 5: source 3 (P4 shifts down)
    #   pos 6 (UNP): NULL (-1)
    #   pos 7: source 4 (P5)
    #   pos 8: source 5 (P6)
    expected = np.array([[0, 1, 2, -1, 3, -1, 4, 5]])
    assert np.array_equal(align["shift_indices"], expected)


def test_shifted_preds_drop_trailing():
    true_op = _worked_example_truth()
    pred_op = _worked_example_pred()
    align = compute_alignment(true_op, pred_op)

    shifted = shift_along_axis1(pred_op, align["shift_indices"], fill="")
    expected = np.array([["P1", "P2", "P3", "", "P4", "", "P5", "P6"]], dtype=object)
    assert (shifted == expected).all()


def test_excluded_unplanned_rows_have_correct_reason():
    true_op = _worked_example_truth()
    pred_op = _worked_example_pred()
    align = compute_alignment(true_op, pred_op)

    expected_exclude = np.array([[False, False, False, True, False, True, False, False]])
    assert np.array_equal(align["exclude"], expected_exclude)
    assert align["exclude_reason"][0, 3] == "Unplanned"
    assert align["exclude_reason"][0, 5] == "Unplanned"
    assert align["exclude_reason"][0, 0] == ""


def test_planned_step_counter():
    true_op = _worked_example_truth()
    pred_op = _worked_example_pred()
    align = compute_alignment(true_op, pred_op)

    ps = align["planned_step"][0]
    assert ps[0] == 1.0 and ps[1] == 2.0 and ps[2] == 3.0
    assert np.isnan(ps[3]) and np.isnan(ps[5])
    assert ps[4] == 4.0
    assert ps[6] == 5.0 and ps[7] == 6.0


def test_unk_treated_like_unplanned():
    true_op = np.array([["T1", "UNK", "T3"]], dtype=object)
    pred_op = np.array([["P1", "P2", "P3"]], dtype=object)
    align = compute_alignment(true_op, pred_op)

    assert align["exclude_reason"][0, 1] == "UNK"
    assert align["shift_indices"][0, 1] == -1
    # Pos 2 (T3) gets P2 shifted in.
    assert align["shift_indices"][0, 2] == 1


def test_shift_overflow_when_too_many_unplanned():
    true_op = np.array([["UNK", "UNK", "T3", "T4"]], dtype=object)
    pred_op = np.array([["P1", "P2", "P3", "P4"]], dtype=object)
    align = compute_alignment(true_op, pred_op)

    # Two upstream UNKs mean position 2 (T3) sources from index 0 (P1),
    # position 3 (T4) sources from index 1 (P2). No shift_overflow yet.
    assert align["shift_indices"][0, 2] == 0
    assert align["shift_indices"][0, 3] == 1
    assert align["exclude_reason"][0, 0] == "UNK"
    assert align["exclude_reason"][0, 1] == "UNK"


def test_premature_eoo_marked_at_first_pred_eoo_step():
    eoo = "End of Operations"
    true_op = np.array([["T1", "T2", "T3", "T4", "T5"]], dtype=object)
    pred_op = np.array([["P1", "P2", eoo, eoo, eoo]], dtype=object)

    align = compute_alignment(true_op, pred_op, eoo_token=eoo)

    # First predicted EOO is at index 2; true is T3 there -> premature_eoo.
    assert align["exclude_reason"][0, 2] == "premature_eoo"
    # Subsequent positions cascade as after_eoo (still pred=EOO, true != EOO).
    assert align["exclude_reason"][0, 3] == "after_eoo"
    assert align["exclude_reason"][0, 4] == "after_eoo"


def test_correct_eoo_at_well_end_not_excluded():
    eoo = "End of Operations"
    true_op = np.array([["T1", "T2", eoo]], dtype=object)
    pred_op = np.array([["P1", "P2", eoo]], dtype=object)

    align = compute_alignment(true_op, pred_op, eoo_token=eoo)

    # Both true and pred have EOO at index 2 -> not excluded (correctly identified well end).
    assert not align["exclude"][0, 2]
    assert align["exclude_reason"][0, 2] == ""


def test_after_eoo_in_truth_excludes_trailing():
    eoo = "End of Operations"
    true_op = np.array([["T1", eoo, "T3", "T4"]], dtype=object)   # synthetic but valid
    pred_op = np.array([["P1", "P2", "P3", "P4"]], dtype=object)

    align = compute_alignment(true_op, pred_op, eoo_token=eoo)

    # i > first_true_eoo (=1) is after_eoo regardless of pred.
    assert align["exclude_reason"][0, 2] == "after_eoo"
    assert align["exclude_reason"][0, 3] == "after_eoo"


def test_eval_weights_match_exclude():
    true_op = _worked_example_truth()
    pred_op = _worked_example_pred()
    align = compute_alignment(true_op, pred_op)

    expected = np.array([[1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    assert np.array_equal(align["eval_weights"], expected)


def test_shift_along_axis1_object_2d():
    arr = np.array([["a", "b", "c", "d"]], dtype=object)
    shift = np.array([[0, -1, 1, 2]])
    out = shift_along_axis1(arr, shift, fill="")
    assert (out == np.array([["a", "", "b", "c"]], dtype=object)).all()


def test_shift_along_axis1_int_3d():
    arr = np.arange(12).reshape(1, 4, 3)   # (1, 4, 3)
    shift = np.array([[0, 2, -1, 3]])
    out = shift_along_axis1(arr, shift, fill=-1)
    expected = np.array([[
        [0, 1, 2],          # source 0
        [6, 7, 8],          # source 2
        [-1, -1, -1],       # NULL
        [9, 10, 11],        # source 3
    ]])
    assert (out == expected).all()


def test_shift_along_axis1_float_4d():
    arr = np.arange(24, dtype=float).reshape(1, 4, 2, 3)
    shift = np.array([[1, 0, -1, 2]])
    out = shift_along_axis1(arr, shift, fill=float("nan"))
    assert out.shape == arr.shape
    assert np.array_equal(out[0, 0], arr[0, 1])
    assert np.array_equal(out[0, 1], arr[0, 0])
    assert np.isnan(out[0, 2]).all()
    assert np.array_equal(out[0, 3], arr[0, 2])

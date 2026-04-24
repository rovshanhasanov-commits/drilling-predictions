"""CLI: offline ML-only evaluation of a saved seq2seq bundle on a data split.

Usage:
    python -m evaluation.run_evaluation                           # test split, current cfg's model
    python -m evaluation.run_evaluation --model-dir models/seq2seq_N30_K8_T5_embed_state
    python -m evaluation.run_evaluation --split val
    python -m evaluation.run_evaluation --limit 500               # first 500 sequences
    python -m evaluation.run_evaluation --no-csv                  # skip predictions.csv
    python -m evaluation.run_evaluation --wells A,B               # restrict to these wells
    python -m evaluation.run_evaluation --no-constraints          # disable constrained AR decoding
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import get_model_dir, load_config, model_folder_name, resolve        # noqa: E402
from inference.load import load_bundle                                           # noqa: E402
from training.constraints import build_legal_tuples, summarize_legal_tuples      # noqa: E402
from training.data import (                                                      # noqa: E402
    HIERARCHY, build_seq2seq_sequences, compute_numeric_cols,
    eoo_duration_value, eoo_encoded_ids, load_strategy_data, make_decoder_inputs,
)
from training.evaluate import autoregressive_predict, per_step_accuracy          # noqa: E402
from training.train import prep_encoder_inputs, prep_model_inputs                # noqa: E402

from . import artifacts, metrics                                                 # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a saved seq2seq bundle on test/val/train split.")
    ap.add_argument("--model-dir", type=str, default=None,
                    help="Override model bundle directory. Default: get_model_dir(cfg).")
    ap.add_argument("--split", choices=("train", "val", "test"), default="test",
                    help="Data split to evaluate on.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N sequences (smoke test).")
    ap.add_argument("--wells", type=str, default=None,
                    help="Comma-separated well-name substring filter.")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="Batch size for decoder step model.")
    ap.add_argument("--no-csv", action="store_true",
                    help="Skip the large per-row predictions.csv.")
    ap.add_argument("--no-constraints", action="store_true",
                    help="Disable constrained (joint-argmax) decoding on the AR path. "
                         "TF path is never constrained — it's the unconstrained upper bound.")
    return ap.parse_args()


def _decode_ids(id_arr: np.ndarray, label_encoder) -> np.ndarray:
    """Map int ids to label strings via target_encoder.classes_."""
    classes = np.asarray(label_encoder.classes_)
    return classes[id_arr]


def _invert_duration(scaled: np.ndarray, dur_scaler) -> np.ndarray:
    """Reverse the log1p + StandardScaler transform applied during preprocessing."""
    shape = scaled.shape
    unscaled = dur_scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()
    hours = np.expm1(unscaled).clip(min=0.0)
    return hours.reshape(shape)


def main():
    args = parse_args()
    cfg = load_config()
    tcfg = cfg["training"]
    icfg = cfg.get("inference", {}) or {}

    # -- resolve model dir and load bundle --
    model_dir = Path(args.model_dir).resolve() if args.model_dir else get_model_dir(cfg)
    print(f"Model bundle: {model_dir}")
    bundle = load_bundle(model_dir)
    encoder_model = bundle["encoder_model"]
    decoder_step_model = bundle["decoder_step_model"]
    training_model = bundle["training_model"]
    encoders = bundle["encoders"]
    data_config = bundle["data_config"]
    model_config = bundle["model_config"]
    target_encoders = encoders["target_encoders"]
    dur_scaler = encoders["dur_scaler"]
    n_classes = encoders["n_classes"]
    eoo_token = encoders["eoo_token"]

    strategy = tcfg["embedding_strategy"]
    strategy_dir = resolve(cfg, cfg["data"]["output_dir"]) / strategy
    seq_len = tcfg["sequence_length"]
    n_future = tcfg["n_future"]
    top_k = tcfg.get("top_k_for_eval", 3)
    # active_targets strips "_next" and excludes the regression head (which is
    # plumbed via predict_duration). The bin classification head ("duration_bin")
    # flows through as another categorical target when present in target_variables.
    active_targets = [t.replace("_next", "") for t in tcfg["target_variables"] if t != "duration_next"]
    predict_duration = "duration_next" in tcfg["target_variables"]
    has_bin_head = "duration_bin" in active_targets

    include_bins_in_hierarchy = bool(icfg.get("include_duration_bins_in_hierarchy", False))
    if include_bins_in_hierarchy and not has_bin_head:
        raise ValueError(
            "inference.include_duration_bins_in_hierarchy=true requires "
            "duration_bin_next in training.target_variables. Either enable the bin "
            "head or set the flag to false."
        )

    # -- load all splits once: the chosen split drives eval, the union drives L --
    split_bundle = load_strategy_data(strategy_dir)
    split_map = {"train": "df_train", "val": "df_val", "test": "df_test"}
    df_split = split_bundle[split_map[args.split]].copy()

    # Legal-tuple table from train ∪ val ∪ test (treated as SME domain knowledge,
    # not a leakage surface — see improvements/Constraint Decoder.md).
    enforce_hierarchy = (not args.no_constraints) and icfg.get("enforce_hierarchy", True)
    use_5d_constraints = enforce_hierarchy and include_bins_in_hierarchy
    legal_tuples = None
    if enforce_hierarchy:
        legal_tuples = build_legal_tuples(
            split_bundle["df_train"], split_bundle["df_val"], split_bundle["df_test"],
            target_encoders, eoo_token=eoo_token,
            include_bins=use_5d_constraints,
        )
        stats = summarize_legal_tuples(legal_tuples, target_encoders)
        mode_str = "5D (bin head joins joint argmax)" if use_5d_constraints else "4D"
        print(f"Legal tuples ({mode_str}): {stats}")
    else:
        print("Constraints disabled (--no-constraints or inference.enforce_hierarchy=false)")

    if args.wells:
        filters = [s.strip() for s in args.wells.split(",") if s.strip()]
        mask = df_split["Well_Name"].apply(lambda w: any(f in str(w) for f in filters))
        df_split = df_split[mask].reset_index(drop=True)
        print(f"Well filter -> {len(df_split):,} rows, {df_split['Well_Name'].nunique()} wells")

    # -- build sequences --
    cat_input_cols = data_config["cat_input_cols"]
    numeric_cols = compute_numeric_cols(data_config)
    target_cols = [f"{h}_target_enc" for h in HIERARCHY]
    if has_bin_head:
        target_cols.append("duration_bin_target_enc")

    eoo_enc = eoo_encoded_ids(target_encoders, eoo_token)
    eoo_dur = eoo_duration_value(dur_scaler)

    seq = build_seq2seq_sequences(
        df_split, cat_input_cols, numeric_cols, target_cols,
        seq_len, n_future, eoo_enc, eoo_dur, args.split,
    )

    if args.limit is not None:
        limit = min(args.limit, seq["num"].shape[0])
        print(f"Limiting to first {limit:,} sequences")
        seq = _trim_sequence_bundle(seq, limit, cat_input_cols, target_cols)

    # Per-head masks — {head: (n_sequences, n_future) float array}. Positions with
    # weight <= 0 are Unplanned / UNK targets the model structurally cannot predict;
    # all metric helpers below accept a `weights` kwarg to exclude those positions.
    sw = seq.get("sample_weight", {}) or {}
    if sw:
        for h, w in sw.items():
            n_masked = int((w == 0).sum())
            total = w.size
            print(f"  masked {h:16s}: {n_masked:,} / {total:,} positions "
                  f"({n_masked / max(total, 1):.2%})")

    print(f"Sequences: {seq['num'].shape[0]:,} | wells: {len(set(seq['wells']))}")

    # -- teacher-forced predictions (unconstrained upper bound) --
    enc_X = prep_encoder_inputs(seq["cat"], seq["num"], cat_input_cols)
    dec_tf = make_decoder_inputs(seq["y"], n_classes, active_targets, predict_duration, seq["y_dur"])
    X_tf = prep_model_inputs(enc_X, dec_tf, active_targets, predict_duration)

    print("Running teacher-forced predictions...")
    tf_raw = training_model.predict(X_tf, batch_size=args.batch_size, verbose=0)
    if not isinstance(tf_raw, dict):
        if not isinstance(tf_raw, list):
            tf_raw = [tf_raw]
        keys = active_targets + (["duration"] if predict_duration else [])
        tf_raw = {k: tf_raw[i] for i, k in enumerate(keys)}

    tf_scores = {}
    for h in active_targets:
        true = seq["y"][f"{h}_target_enc"]
        tf_scores[h] = per_step_accuracy(tf_raw[h], true, k=top_k, weights=sw.get(h))
        print(f"  TF {h:20s}  top1={tf_scores[h]['overall_top1']:.4f}  top{top_k}={tf_scores[h]['overall_topk']:.4f}")

    # -- autoregressive predictions (deployment; constrained when legal_tuples is set) --
    print(f"Running autoregressive predictions (constraints={'on' if enforce_hierarchy else 'off'}"
          f"{', 5D' if use_5d_constraints else ''})...")
    ar = autoregressive_predict(
        encoder_model, decoder_step_model, enc_X, n_future, n_classes,
        active_targets, predict_duration, batch_size=args.batch_size, top_k=top_k,
        legal_tuples=legal_tuples,
        include_bins=use_5d_constraints,
    )
    ar_pred_ids = ar["pred"]                         # {head: (n, n_future)}
    ar_topk_ids = ar["topk"]                         # {head: (n, n_future, top_k)}

    # Per-head top-1 / top-K accuracy. When constraints are on, ar_pred_ids[h]
    # comes from the winning legal tuple (coordinated across heads). topk stays
    # per-head for diagnostics; tuple-level top-K is reported separately.
    ar_scores = {}
    for h in active_targets:
        true = seq["y"][f"{h}_target_enc"]
        w = sw.get(h)
        top1, topk_hits = [], []
        for s in range(n_future):
            hit1 = (ar_pred_ids[h][:, s] == true[:, s])
            hitk = (ar_topk_ids[h][:, s, :] == true[:, s:s + 1]).any(axis=-1)
            if w is not None:
                keep = w[:, s] > 0
                if not keep.any():
                    top1.append(float("nan"))
                    topk_hits.append(float("nan"))
                    continue
                hit1 = hit1[keep]
                hitk = hitk[keep]
            top1.append(float(hit1.mean()))
            topk_hits.append(float(hitk.mean()))
        ar_scores[h] = {
            "per_step_top1": top1,
            "per_step_topk": topk_hits,
            "k": top_k,
            "overall_top1": float(np.nanmean(top1)),
            "overall_topk": float(np.nanmean(topk_hits)),
        }
        print(f"  AR {h:20s}  top1={ar_scores[h]['overall_top1']:.4f}  top{top_k}={ar_scores[h]['overall_topk']:.4f}")

    # Tuple-level top-K accuracy: did the ground-truth tuple appear in the
    # chosen top-K legal tuples at this step? Only meaningful under constraints.
    tuple_topk_hit_rate = None
    if enforce_hierarchy and "tuple_topk" in ar:
        tuple_topk = ar["tuple_topk"]                  # (n, n_future, K, 4)
        y = seq["y"]
        true_stack = np.stack([y[f"{h}_target_enc"] for h in HIERARCHY], axis=-1)  # (n, n_future, 4)
        # Use the op-head mask as the gate (matches legacy behavior: excludes
        # masked-target rows from per-head accuracy). Tuples aren't meaningful
        # when the op head's ground truth is masked anyway.
        w_op = sw.get("operation")
        per_step_hits = []
        for s in range(n_future):
            hits = (tuple_topk[:, s, :, :] == true_stack[:, s:s + 1, :]).all(axis=-1).any(axis=-1)
            if w_op is not None:
                keep = w_op[:, s] > 0
                if not keep.any():
                    per_step_hits.append(float("nan"))
                    continue
                hits = hits[keep]
            per_step_hits.append(float(hits.mean()))
        tuple_topk_hit_rate = {
            "per_step": per_step_hits,
            "overall":  float(np.nanmean(per_step_hits)),
        }
        print(f"  AR tuple_top{top_k}         overall={tuple_topk_hit_rate['overall']:.4f}")

    # -- decode ids -> label strings for downstream reporting --
    ar_pred_labels = {h: _decode_ids(ar_pred_ids[h], target_encoders[h]) for h in active_targets}
    ar_topk_labels = {h: _decode_ids(ar_topk_ids[h], target_encoders[h]) for h in active_targets}
    true_labels = {
        h: _decode_ids(seq["y"][f"{h}_target_enc"], target_encoders[h]) for h in active_targets
    }

    # Top-K tuple labels for predictions.csv (only populated under constraints).
    tuple_topk_labels = None
    tuple_topk_logprob = None
    tuple_topk_prob = None
    if enforce_hierarchy and "tuple_topk" in ar:
        tk = ar["tuple_topk"]                          # (n, n_future, K, n_joint_heads), ids
        tuple_topk_labels = np.empty(tk.shape, dtype=object)
        for i, h in enumerate(HIERARCHY):
            tuple_topk_labels[..., i] = _decode_ids(tk[..., i], target_encoders[h])
        if use_5d_constraints:
            # Last column is the bin head; decode through the bin encoder.
            tuple_topk_labels[..., 4] = _decode_ids(tk[..., 4], target_encoders["duration_bin"])
        tuple_topk_logprob = ar["tuple_topk_logprob"]  # (n, n_future, K) raw joint log-probs
        tuple_topk_prob    = ar["tuple_topk_prob"]     # (n, n_future, K) renormalized over L

    # -- duration (inverse-transform to hours) --
    # Always compute true_duration in raw hours when the bin head is on, so the
    # bin-center MAE has a real reference even when the regression head is off.
    need_true_hours = predict_duration or has_bin_head
    if need_true_hours:
        true_duration = _invert_duration(seq["y_dur"], dur_scaler)
    else:
        true_duration = np.zeros_like(seq["y_dur"])

    if predict_duration:
        pred_duration = _invert_duration(ar["pred"]["duration"], dur_scaler)
        dur_stats = metrics.duration_metrics(pred_duration, true_duration, weights=sw.get("duration"))
        print(
            f"  duration  MAE={dur_stats['mae_hours']:.2f}h  "
            f"MedAE={dur_stats['medae_hours']:.2f}h  p95={dur_stats['p95_abs_err_hours']:.2f}h"
        )
    else:
        pred_duration = np.zeros_like(seq["y_dur"])
        dur_stats = {}

    # -- bin-center MAE (cross-compatible with regression metric) --
    bin_stats = {}
    if has_bin_head:
        bin_classes = list(target_encoders["duration_bin"].classes_)
        bin_centers_dict = encoders.get("bin_centers") or {}
        bin_stats = metrics.bin_center_mae(
            ar_pred_ids["duration_bin"], true_duration, bin_centers_dict,
            bin_classes, weights=sw.get("duration_bin"),
        )
        print(
            f"  duration_bin  center_MAE={bin_stats['bin_center_mae_hours']:.2f}h  "
            f"long_center_MAE={bin_stats['bin_center_mae_long_hours']:.2f}h"
        )

    # -- hierarchy validity --
    master_csv = resolve(cfg, cfg["data"]["master_csv"])
    sets = metrics.build_hierarchy_sets(master_csv)
    hier_valid = metrics.hierarchy_valid_mask(
        ar_pred_labels["phase"], ar_pred_labels["phase_step"],
        ar_pred_labels["major_ops_code"], ar_pred_labels["operation"],
        sets, eoo_token,
    )
    hierarchy_validity_rate = float(hier_valid.mean())
    print(f"  hierarchy_validity_rate = {hierarchy_validity_rate:.4f}"
          f"{' (1.0 by construction under constraints)' if enforce_hierarchy else ''}")

    cond = {
        "phase_step_given_phase": metrics.conditional_accuracy(
            ar_pred_ids["phase"], seq["y"]["phase_target_enc"],
            ar_pred_ids["phase_step"], seq["y"]["phase_step_target_enc"],
            weights=sw.get("phase_step"),
        ),
        "moc_given_phase_step": metrics.conditional_accuracy(
            ar_pred_ids["phase_step"], seq["y"]["phase_step_target_enc"],
            ar_pred_ids["major_ops_code"], seq["y"]["major_ops_code_target_enc"],
            weights=sw.get("major_ops_code"),
        ),
        "operation_given_moc": metrics.conditional_accuracy(
            ar_pred_ids["major_ops_code"], seq["y"]["major_ops_code_target_enc"],
            ar_pred_ids["operation"], seq["y"]["operation_target_enc"],
            weights=sw.get("operation"),
        ),
    }
    print(f"  conditional_acc = {cond}")

    # -- confusion pairs --
    confusion_dfs = {}
    for h in active_targets:
        class_labels = list(target_encoders[h].classes_)
        confusion_dfs[h] = metrics.top_confused_pairs(
            seq["y"][f"{h}_target_enc"], ar_pred_ids[h], class_labels, top_n=20,
            weights=sw.get(h),
        )

    # -- per-well aggregation --
    pw_weights = {h: sw[h] for h in active_targets if h in sw}
    if predict_duration and "duration" in sw:
        pw_weights["duration"] = sw["duration"]
    per_well_df = metrics.per_well_accuracy(
        seq["wells"], ar_pred_ids, {h: seq["y"][f"{h}_target_enc"] for h in active_targets},
        pred_duration if predict_duration else None,
        true_duration if predict_duration else None,
        weights=pw_weights or None,
    )
    well_std = metrics.well_accuracy_std(per_well_df)

    # -- build summary --
    summary = {
        "model_folder": _effective_model_folder_name(cfg, args),
        "evaluated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "split": args.split,
        "n_sequences": int(seq["num"].shape[0]),
        "n_wells": int(per_well_df["well_name"].nunique()),
        "top_k": top_k,
        "constraints_enabled": bool(enforce_hierarchy),
        "constraints_5d": bool(use_5d_constraints),
        "n_legal_tuples": int(legal_tuples.shape[0]) if legal_tuples is not None else None,
        "tf": {
            **{f"{h}_top1": tf_scores[h]["overall_top1"] for h in active_targets},
            **{f"{h}_top{top_k}": tf_scores[h]["overall_topk"] for h in active_targets},
        },
        "ar": {
            **{f"{h}_top1": ar_scores[h]["overall_top1"] for h in active_targets},
            **{f"{h}_top{top_k}": ar_scores[h]["overall_topk"] for h in active_targets},
            "step1_vs_stepK_drop": metrics.step1_vs_last_drop(
                {h: ar_scores[h]["per_step_top1"] for h in active_targets}
            ),
            "hierarchy_validity_rate": hierarchy_validity_rate,
            "conditional_acc": cond,
            **dur_stats,
            **bin_stats,
            "well_accuracy_std": well_std,
        },
    }
    if has_bin_head:
        summary["ar"]["duration_bin_per_step_top1"] = ar_scores["duration_bin"]["per_step_top1"]
        summary["ar"][f"duration_bin_per_step_top{top_k}"] = ar_scores["duration_bin"]["per_step_topk"]
    if tuple_topk_hit_rate is not None:
        summary["ar"][f"tuple_top{top_k}_overall"] = tuple_topk_hit_rate["overall"]
        summary["ar"][f"tuple_top{top_k}_per_step"] = tuple_topk_hit_rate["per_step"]

    # -- write artifacts --
    out_dir = _build_out_dir(cfg, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting artifacts to: {out_dir}")

    artifacts.write_summary(out_dir, summary)
    artifacts.write_run_config(out_dir, vars(args), model_config)
    artifacts.write_per_step_accuracy(
        out_dir, tf_scores, ar_scores,
        extra_heads=(["duration_bin"] if has_bin_head else None),
    )
    artifacts.write_confusion_csvs(out_dir, confusion_dfs)
    artifacts.write_per_well(out_dir, per_well_df)

    if not args.no_csv:
        print("Writing predictions.csv (this may take a moment)...")
        artifacts.write_predictions_csv(
            out_dir, seq["wells"], seq["start_idx"],
            true_labels, {h: ar_pred_labels[h] for h in active_targets}, ar_topk_labels,
            true_duration, pred_duration, hier_valid,
            tuple_topk_labels=tuple_topk_labels,
            tuple_topk_logprob=tuple_topk_logprob,
            tuple_topk_prob=tuple_topk_prob,
            bin_centers=encoders.get("bin_centers"),
            include_bins_in_tuples=use_5d_constraints,
        )

    print("\nDone.")
    return 0


def _trim_sequence_bundle(seq: dict, limit: int, cat_cols: list[str], target_cols: list[str]) -> dict:
    """Slice all arrays in seq to the first `limit` sequences — keeps shapes consistent."""
    trimmed = {
        "cat": {c: seq["cat"][c][:limit] for c in cat_cols},
        "num": seq["num"][:limit],
        "y": {c: seq["y"][c][:limit] for c in target_cols},
        "y_dur": seq["y_dur"][:limit],
        "wells": seq["wells"][:limit],
        "start_idx": seq["start_idx"][:limit],
    }
    if "sample_weight" in seq:
        trimmed["sample_weight"] = {h: w[:limit] for h, w in seq["sample_weight"].items()}
    return trimmed


def _effective_model_folder_name(cfg: dict, args) -> str:
    """Folder name for the model being evaluated.

    When `--model-dir` is passed (as the training notebook's post-train eval
    subprocess does), derive the name from the bundle path so the eval artifacts
    land NEXT TO the model they actually ran against. The alternative —
    `model_folder_name(cfg)` — would use the yaml-default cfg (the subprocess
    re-reads yaml from disk and loses any in-memory overrides from cell 4),
    leaking eval artifacts into a wrong / non-existent folder.

    When `--model-dir` is NOT passed, fall back to the cfg-derived name (matches
    training-notebook behavior when no overrides are in play).
    """
    if args.model_dir:
        return Path(args.model_dir).resolve().name
    return model_folder_name(cfg)


def _build_out_dir(cfg: dict, args) -> Path:
    results_root = resolve(cfg, cfg["training"]["results_dir"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_root / _effective_model_folder_name(cfg, args) / f"eval_{ts}"


if __name__ == "__main__":
    raise SystemExit(main())

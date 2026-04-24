"""End-to-end ML inference: (well, date) -> MLOutput with top-K legal tuples per step.

Under constrained decoding (default), each step's top-K comes from the joint
argmax over the legal-tuple table L (rebuilt at bundle load from parquets).
The winning tuple's class ids feed back as the next step's decoder inputs —
this keeps the four heads coordinated throughout the horizon.

Set `inference.enforce_hierarchy: false` in pipeline.yaml (or build L=None) to
fall back to unconstrained per-head argmax — legacy behavior for ablations only.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from config import get_model_dir, load_config, resolve

from .contract import DurationBin, HierarchyTuple, MLOutput, StepPrediction
from .load import load_bundle
from .preprocess_selection import build_encoder_input

HIERARCHY = ["phase", "phase_step", "major_ops_code", "operation"]


_CACHED_BUNDLE = None


def _get_bundle(model_dir: Path, strategy_dir: Path, include_bins: bool = False):
    global _CACHED_BUNDLE
    # Cache key includes include_bins so flipping the yaml flag rebuilds the
    # 5D legal-tuple table on the next call without restarting the process.
    cache_key = (str(model_dir), str(strategy_dir), bool(include_bins))
    if _CACHED_BUNDLE is not None and _CACHED_BUNDLE["_key"] == cache_key:
        return _CACHED_BUNDLE
    bundle = load_bundle(model_dir, strategy_dir=strategy_dir, include_bins=include_bins)
    bundle["_key"] = cache_key
    _CACHED_BUNDLE = bundle
    return bundle


def _invert_duration(scaled_values: np.ndarray, dur_scaler) -> np.ndarray:
    """Inverse StandardScaler, then expm1, then clip to [0, inf)."""
    unscaled = dur_scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()
    hours = np.expm1(unscaled).clip(min=0.0)
    return hours


def predict(
    well_name: str,
    report_date: date | str,
    cfg: dict | None = None,
    model_dir: Path | None = None,
    top_k: int | None = None,
) -> MLOutput:
    cfg = cfg or load_config()
    tcfg = cfg["training"]
    icfg = cfg.get("inference", {}) or {}

    strategy = tcfg["embedding_strategy"]
    strategy_dir = resolve(cfg, cfg["data"]["output_dir"]) / strategy
    model_dir = model_dir or get_model_dir(cfg)
    seq_len = tcfg["sequence_length"]
    n_future = tcfg["n_future"]
    # `inference.top_k_tuples` is the tuple-level K surfaced to the LLM; falls
    # back to the training eval K so old configs keep working.
    k = (top_k if top_k is not None
         else icfg.get("top_k_tuples", tcfg.get("top_k_for_eval", 3)))
    enforce_hierarchy = bool(icfg.get("enforce_hierarchy", True))
    include_bins_in_hierarchy = bool(icfg.get("include_duration_bins_in_hierarchy", False))

    active_targets = [t.replace("_next", "") for t in tcfg["target_variables"] if t != "duration_next"]
    predict_duration = "duration_next" in tcfg["target_variables"]
    has_bin_head = "duration_bin" in active_targets

    if include_bins_in_hierarchy and not has_bin_head:
        raise ValueError(
            "inference.include_duration_bins_in_hierarchy=true requires "
            "duration_bin_next in training.target_variables."
        )

    use_5d = enforce_hierarchy and include_bins_in_hierarchy
    bundle = _get_bundle(model_dir, strategy_dir, include_bins=use_5d)
    encoder_model = bundle["encoder_model"]
    decoder_step_model = bundle["decoder_step_model"]
    encoders = bundle["encoders"]
    data_config = bundle["data_config"]
    n_classes = encoders["n_classes"]
    target_encoders = encoders["target_encoders"]
    dur_scaler = encoders["dur_scaler"]
    bin_centers = encoders.get("bin_centers") or {}
    legal_tuples = bundle.get("legal_tuples") if enforce_hierarchy else None

    if enforce_hierarchy and legal_tuples is None:
        # Parquets missing OR loader skipped them — fall back to unconstrained
        # per-head argmax rather than silently drop hierarchy guarantees.
        print("[inference] enforce_hierarchy=True but legal_tuples unavailable; "
              "falling back to unconstrained decoding")

    if legal_tuples is not None:
        missing = [h for h in HIERARCHY if h not in active_targets]
        if missing:
            raise ValueError(
                f"Constrained inference requires all 4 hierarchy heads; missing {missing}"
            )

    enc_inputs, _ = build_encoder_input(
        well_name, report_date, strategy_dir, data_config, seq_len,
    )

    enc_out, h, c = encoder_model.predict(enc_inputs, verbose=0)

    prev_cat = {t: np.array([[n_classes[t]]], dtype=np.int32) for t in active_targets}
    prev_dur = np.array([[0.0]], dtype=np.float32) if predict_duration else None

    steps_out: list[StepPrediction] = []
    bin_classes = (list(target_encoders["duration_bin"].classes_)
                   if has_bin_head else [])
    sentinel_top1_count = 0

    for step in range(n_future):
        dec_step_in = [prev_cat[t] for t in active_targets]
        if predict_duration:
            dec_step_in.append(prev_dur)
        dec_step_in += [enc_out, h, c]

        step_out = decoder_step_model.predict(dec_step_in, verbose=0)

        idx = 0
        level_probs: dict = {}
        for t in active_targets:
            level_probs[t] = step_out[idx][0, 0, :]            # (n_classes,)
            idx += 1
        dur_scaled = None
        if predict_duration:
            dur_scaled = float(step_out[idx][0, 0, 0])
            prev_dur = np.array([[dur_scaled]], dtype=np.float32)
            idx += 1
        h = step_out[idx]
        c = step_out[idx + 1]

        # Bin-head top-K (independent of the joint argmax). When 5D mode is on,
        # the joint argmax also chooses a bin and writes it onto each
        # HierarchyTuple; this independent top-K is still surfaced for the LLM /
        # UI debug view.
        duration_bin_topk: list[DurationBin] = []
        if has_bin_head:
            bin_probs = level_probs["duration_bin"]
            bin_order = np.argsort(-bin_probs)[:k]
            duration_bin_topk = [
                DurationBin(
                    label=str(bin_classes[bid]),
                    prob=float(bin_probs[bid]),
                    center_hours=float(bin_centers.get(str(bin_classes[bid]),
                                                       float("nan"))),
                )
                for bid in bin_order
            ]

        if legal_tuples is not None:
            # Batch-dim = 1; joint_topk_tuples expects (B, n_classes) per head.
            head_names_joint = HIERARCHY + (["duration_bin"] if use_5d else [])
            probs_batched = {t: level_probs[t][None, :] for t in head_names_joint}
            from training.constraints import joint_topk_tuples       # local to dodge TF import cost at module load
            tk_tuples, tk_log_probs, tk_probs = joint_topk_tuples(
                probs_batched, legal_tuples, k, head_names=head_names_joint,
            )
            tk_tuples    = tk_tuples[0]                              # (K, n_joint)
            tk_log_probs = tk_log_probs[0]                           # (K,)
            tk_probs     = tk_probs[0]                               # (K,) renormalized over L
            topk_tuples = []
            for i in range(tk_tuples.shape[0]):
                tup_kwargs = dict(
                    phase          = str(target_encoders["phase"].classes_[tk_tuples[i, 0]]),
                    phase_step     = str(target_encoders["phase_step"].classes_[tk_tuples[i, 1]]),
                    major_ops_code = str(target_encoders["major_ops_code"].classes_[tk_tuples[i, 2]]),
                    operation      = str(target_encoders["operation"].classes_[tk_tuples[i, 3]]),
                    log_prob       = float(tk_log_probs[i]),
                    prob           = float(tk_probs[i]),
                )
                if use_5d:
                    tup_kwargs["duration_bin"] = str(
                        target_encoders["duration_bin"].classes_[tk_tuples[i, 4]]
                    )
                topk_tuples.append(HierarchyTuple(**tup_kwargs))
            # Feedback = winning tuple's ids (not independent argmaxes).
            for i, t in enumerate(head_names_joint):
                prev_cat[t] = np.array([[int(tk_tuples[0, i])]], dtype=np.int32)
            # Bin head feedback under 4D constraints: independent argmax (joint
            # didn't choose a bin). Without this, prev_cat["duration_bin"]
            # would never advance and the bin head would condition on the
            # start-token forever.
            if has_bin_head and not use_5d:
                bin_top1_id = int(np.argmax(level_probs["duration_bin"]))
                prev_cat["duration_bin"] = np.array([[bin_top1_id]], dtype=np.int32)
        else:
            # Unconstrained fallback: independent per-head argmax, top-K per head.
            # The surfaced tuples here are NOT guaranteed legal — but they match
            # the old behavior for ablation parity. The LLM should treat these
            # with more skepticism.
            order = {t: np.argsort(-level_probs[t])[:k] for t in HIERARCHY}
            topk_tuples = []
            # Build K "tuples" by zipping each head's top-K. Joint log-prob is
            # just the sum of the per-head log-probs (not a true joint under L).
            # `prob` here is exp(log_prob) — raw joint under independence — since
            # there is no legal set to renormalize against.
            for i in range(k):
                row = {t: order[t][min(i, len(order[t]) - 1)] for t in HIERARCHY}
                lp = sum(
                    float(np.log(level_probs[t][row[t]] + 1e-12)) for t in HIERARCHY
                )
                topk_tuples.append(HierarchyTuple(
                    phase          = str(target_encoders["phase"].classes_[row["phase"]]),
                    phase_step     = str(target_encoders["phase_step"].classes_[row["phase_step"]]),
                    major_ops_code = str(target_encoders["major_ops_code"].classes_[row["major_ops_code"]]),
                    operation      = str(target_encoders["operation"].classes_[row["operation"]]),
                    log_prob       = lp,
                    prob           = float(np.exp(lp)),
                ))
            for t in HIERARCHY:
                prev_cat[t] = np.array([[int(order[t][0])]], dtype=np.int32)
            if has_bin_head:
                bin_top1_id = int(np.argmax(level_probs["duration_bin"]))
                prev_cat["duration_bin"] = np.array([[bin_top1_id]], dtype=np.int32)

        # duration_hours sourcing:
        # - regression head on  -> inverse-transform of its scalar
        # - bin head on (only)  -> top-1 bin's center; sentinels collapse to 0.0
        # - both off            -> 0.0 (legacy behavior)
        if predict_duration and dur_scaled is not None:
            duration_hours = float(_invert_duration(np.array([dur_scaled]), dur_scaler)[0])
        elif has_bin_head and duration_bin_topk:
            top1_bin = duration_bin_topk[0]
            if top1_bin.center_hours == top1_bin.center_hours:   # not NaN -> real bin
                duration_hours = float(top1_bin.center_hours)
            else:
                # Sentinel predicted (Unplanned / UNK); render as 0.0 so the
                # 24h-sum indicator stays meaningful. EOO has center 0.0 so it
                # naturally falls into the real-bin branch above.
                sentinel_top1_count += 1
                duration_hours = 0.0
        else:
            duration_hours = 0.0

        steps_out.append(StepPrediction(
            step=step,
            topk_tuples=topk_tuples,
            duration_hours=duration_hours,
            duration_bin_topk=duration_bin_topk,
        ))

    if has_bin_head and sentinel_top1_count > 0:
        print(f"[inference] bin top-1 was a sentinel (Unplanned/UNK) on "
              f"{sentinel_top1_count}/{n_future} steps; duration_hours set to 0.0 "
              f"for those steps.")

    report_date_obj = (
        report_date if isinstance(report_date, date)
        else np.datetime64(report_date).astype("datetime64[D]").item()
    )
    return MLOutput(
        well_name=well_name,
        report_date=report_date_obj,
        n_future=n_future,
        top_k=k,
        steps=steps_out,
    )

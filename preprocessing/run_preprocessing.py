"""CLI entry point for the preprocessing stage.

Usage:
    python -m preprocessing.run_preprocessing
    python -m preprocessing.run_preprocessing --config path/to/pipeline.yaml
"""

import argparse
import sys
from pathlib import Path

# Make the repo importable whether run via `python -m` or `python preprocessing/run_preprocessing.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import load_config, resolve                                       # noqa: E402
from preprocessing.clean import clean                                          # noqa: E402
from preprocessing.encode import STRATEGY_FN, scale_features                  # noqa: E402
from preprocessing.features import (                                           # noqa: E402
    detect_continuous_cols,
    engineer,
    fit_target_encoders,
)
from preprocessing.join import load_and_join                                   # noqa: E402
from preprocessing.save import save_strategy                                   # noqa: E402
from preprocessing.split import split_wells                                    # noqa: E402


# Columns that must never be scaled, encoded, or treated as model features but kept
# on the row for traceability / row-level lookup at inference time.
KEEP_METADATA = {
    "Well_Name",
    "row_id",
    "Report_Date",
    "Job_Report_Start_Date",
    "Start_Time",
    "End_Time",
    "Ops_Summary",
}


def _drop_untrainable(df):
    """Drop object-dtype cols that aren't needed at training time.

    Keeps Well_Name + row_id + Report_Date for traceability.
    """
    drop = [c for c in df.columns
            if df[c].dtype == object
            and c not in KEEP_METADATA
            and c not in ("phase", "phase_step", "major_ops_code", "operation", "state")]
    if drop:
        df = df.drop(columns=drop)
    return df


def main(cfg_path: str | None = None):
    cfg = load_config(cfg_path)

    master_csv = resolve(cfg, cfg["data"]["master_csv"])
    comments_csv = resolve(cfg, cfg["data"]["comments_csv"])
    out_root = resolve(cfg, cfg["data"]["output_dir"])

    print("=" * 60)
    print("PREPROCESSING — E2E Drilling Pipeline")
    print("=" * 60)
    print(f"  master:   {master_csv}")
    print(f"  comments: {comments_csv}")
    print(f"  output:   {out_root}")

    # 1. Join raw CSVs
    print("\n--- Join ---")
    df = load_and_join(master_csv, comments_csv, cfg)

    # 2. Clean (idempotent, mirrors data_prep.ipynb)
    print("\n--- Clean ---")
    df = clean(
        df,
        unplanned_ops=cfg.get("preprocessing", {}).get("unplanned_operations", []),
        unplanned_token=cfg.get("preprocessing", {}).get("unplanned_token", "Unplanned"),
    )

    # 3. Feature engineering
    print("\n--- Feature engineering ---")
    df, bin_cols = engineer(
        df,
        operator_col=cfg["features"]["operator_column"],
        rig_col=cfg["features"]["rig_column"],
        drop_after_derivation=cfg.get("preprocessing", {}).get("drop_after_derivation", []),
    )
    df = _drop_untrainable(df)
    print(f"  {df.shape[0]:,} rows x {df.shape[1]} cols after engineering")
    print(f"  binary columns ({len(bin_cols)}): {bin_cols}")

    # 3. Well-level split
    print("\n--- Split ---")
    df_train, df_val, df_test, train_wells, val_wells, test_wells = split_wells(df, cfg["split"])

    # 4. Target encoders (shared across strategies) — fit with EOO + UNK sentinels
    print("\n--- Target encoding ---")
    target_encoders, n_classes = fit_target_encoders(
        df_train, cfg["encoding"]["eoo_token"], cfg["encoding"]["unk_token"]
    )

    # 4b. Materialize {col}_target_enc on every split so training can window targets
    # independent of which encoding strategy is in use. NaN / unseen-in-train values
    # fall back to UNK so uncertainty stays visible; the old global-mode fallback
    # (DRL for operation, NON_DRILL for MOC) manufactured spurious majority-class signal.
    from preprocessing.features import CAT_COLS                              # local import
    unk_token = cfg["encoding"]["unk_token"]
    for split_df in (df_train, df_val, df_test):
        for col in CAT_COLS:
            le = target_encoders[col]
            class_to_idx = {c: i for i, c in enumerate(le.classes_)}
            unk_idx = class_to_idx[unk_token]
            enc = split_df[col].map(class_to_idx).fillna(unk_idx).astype("int32")
            split_df[col + "_target_enc"] = enc.values

    # 5. Continuous column detection (before strategy mutates shapes)
    cont_cols = detect_continuous_cols(df_train, bin_cols)
    print(f"\n  continuous columns ({len(cont_cols)}): {cont_cols}")

    # 6. Per-strategy: copy → encode cats → scale numerics → save
    for strategy in cfg["encoding"]["strategies"]:
        print(f"\n{'=' * 60}\nSTRATEGY: {strategy}\n{'=' * 60}")

        dft = df_train.copy()
        dfv = df_val.copy()
        dfx = df_test.copy()

        dft, dfv, dfx, cat_enc, cat_input_cols, dummy_names = STRATEGY_FN[strategy](dft, dfv, dfx)

        # Recompute cont cols post-strategy (some cols may have been dropped)
        strat_cont_cols = [c for c in cont_cols if c in dft.columns]
        feat_scaler, dur_scaler = scale_features(dft, dfv, dfx, strat_cont_cols)
        print(f"  scaled {len(strat_cont_cols)} continuous cols  +  Duration hours (separate scaler)")

        out_dir = out_root / strategy
        save_strategy(
            out_dir=out_dir,
            strategy=strategy,
            df_train=dft, df_val=dfv, df_test=dfx,
            cat_encoders=cat_enc,
            target_encoders=target_encoders,
            n_classes=n_classes,
            feat_scaler=feat_scaler,
            dur_scaler=dur_scaler,
            cat_input_cols=cat_input_cols,
            cont_cols=strat_cont_cols,
            bin_cols=bin_cols,
            dummy_col_names=dummy_names,
            train_wells=train_wells, val_wells=val_wells, test_wells=test_wells,
            eoo_token=cfg["encoding"]["eoo_token"],
            split_cfg=cfg["split"],
        )

    print(f"\nDone — all strategies saved under {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args.config)

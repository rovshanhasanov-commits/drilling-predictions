"""Persist a strategy's train/val/test parquets + encoders.pkl + config.json."""

import json
import os
import pickle
from pathlib import Path


def save_strategy(
    out_dir: Path,
    strategy: str,
    df_train, df_val, df_test,
    cat_encoders: dict,
    target_encoders: dict,
    n_classes: dict,
    feat_scaler,
    dur_scaler,
    cat_input_cols: list,
    cont_cols: list,
    bin_cols: list,
    dummy_col_names: list,
    train_wells: set,
    val_wells: set,
    test_wells: set,
    eoo_token: str,
    split_cfg: dict,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(out_dir / "df_train.parquet", index=False)
    df_val.to_parquet(out_dir / "df_val.parquet", index=False)
    df_test.to_parquet(out_dir / "df_test.parquet", index=False)

    with open(out_dir / "encoders.pkl", "wb") as f:
        pickle.dump({
            "cat_encoders": cat_encoders,
            "target_encoders": target_encoders,
            "n_classes": n_classes,
            "feat_scaler": feat_scaler,
            "dur_scaler": dur_scaler,
            "eoo_token": eoo_token,
        }, f)

    config = {
        "strategy": strategy,
        "eoo_token": eoo_token,
        "cat_input_cols": cat_input_cols,
        "cont_cols": cont_cols,
        "bin_cols": bin_cols,
        "dummy_col_names": dummy_col_names,
        "n_classes": n_classes,
        "random_state": split_cfg["random_state"],
        "test_well_fraction": split_cfg["test_well_fraction"],
        "val_well_fraction": split_cfg["val_well_fraction"],
        "n_train_wells": len(train_wells),
        "n_val_wells": len(val_wells),
        "n_test_wells": len(test_wells),
        "n_train_rows": len(df_train),
        "n_val_rows": len(df_val),
        "n_test_rows": len(df_test),
        # target vocab is implicit in target_encoders; names reproduced for clarity
        "target_cols": ["phase", "phase_step", "major_ops_code", "operation"],
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    sizes = [
        os.path.getsize(out_dir / fn) / 1e6
        for fn in ("df_train.parquet", "df_val.parquet", "df_test.parquet")
    ]
    print(f"  Saved to {out_dir}/  (parquet {sizes[0]:.1f}/{sizes[1]:.1f}/{sizes[2]:.1f} MB)")

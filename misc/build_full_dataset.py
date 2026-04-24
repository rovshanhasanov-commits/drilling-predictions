"""Build the pre-encode / pre-scale snapshot of the training data.

Runs the shared preprocessing steps — join -> clean -> engineer -> drop-untrainable —
and writes the resulting frame to misc/full_dataset.parquet for exploration.

No new logic: every step is imported from preprocessing/ so the output stays in
sync with whatever run_preprocessing.py would feed to the strategy loop.

Usage:
    python -m misc.build_full_dataset
    python misc/build_full_dataset.py --config path/to/pipeline.yaml
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import load_config, resolve
from preprocessing.clean import clean
from preprocessing.features import engineer
from preprocessing.join import load_and_join
from preprocessing.run_preprocessing import _drop_untrainable


OUT_PATH = Path(__file__).resolve().parent / "full_dataset.csv"


def build(cfg_path: str | None = None):
    cfg = load_config(cfg_path)
    master_csv = resolve(cfg, cfg["data"]["master_csv"])
    comments_csv = resolve(cfg, cfg["data"]["comments_csv"])

    df = load_and_join(master_csv, comments_csv, cfg)
    df = clean(
        df,
        unplanned_ops=cfg.get("preprocessing", {}).get("unplanned_operations", []),
        unplanned_token=cfg.get("preprocessing", {}).get("unplanned_token", "Unplanned"),
    )
    df, bin_cols = engineer(
        df,
        operator_col=cfg["features"]["operator_column"],
        rig_col=cfg["features"]["rig_column"],
        drop_after_derivation=cfg.get("preprocessing", {}).get("drop_after_derivation", []),
    )
    df = _drop_untrainable(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {df.shape[0]:,} rows x {df.shape[1]} cols -> {OUT_PATH}")
    print(f"Binary cols ({len(bin_cols)}): {bin_cols}")
    return df, bin_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    build(args.config)

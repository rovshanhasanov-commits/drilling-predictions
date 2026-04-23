"""Join Master_Data_With_ID + Comment_Features on row_id, keeping only columns
named in pipeline.yaml. Sort order is owned by clean.py; join.py does not sort.
"""

from pathlib import Path

import pandas as pd


def load_and_join(master_csv: Path, comments_csv: Path, cfg: dict) -> pd.DataFrame:
    """Load both CSVs, keep only the yaml-listed columns, and merge on row_id.

    Reads keep-lists from cfg:
      - cfg["preprocessing"]["master_keep"]
      - cfg["features"]["comment_numeric"] + cfg["features"]["comment_obs_flags"]
    """
    master = pd.read_csv(master_csv)
    comments = pd.read_csv(comments_csv)

    master_keep = cfg["preprocessing"]["master_keep"]
    master = master[[c for c in master_keep if c in master.columns]]

    features_cfg = cfg["features"]
    keep_from_comments = ["row_id"] + features_cfg["comment_numeric"] + features_cfg["comment_obs_flags"]
    keep_from_comments = [c for c in keep_from_comments if c in comments.columns]
    comments = comments[keep_from_comments]

    df = master.merge(comments, on="row_id", how="left")
    df = df.reset_index(drop=True)

    print(f"Joined: {len(df):,} rows x {df.shape[1]} cols  ({df['Well_Name'].nunique()} wells)")
    return df

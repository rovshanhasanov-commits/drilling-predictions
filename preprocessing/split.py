"""Well-level train/val/test split with leakage asserts."""

import numpy as np
import pandas as pd


def split_wells(df: pd.DataFrame, split_cfg: dict) -> tuple:
    seed = split_cfg["random_state"]
    test_frac = split_cfg["test_well_fraction"]
    val_frac = split_cfg["val_well_fraction"]

    rng = np.random.default_rng(seed)
    wells = df["Well_Name"].unique().tolist()
    rng.shuffle(wells)

    n_test = max(1, int(len(wells) * test_frac))
    test_wells = set(wells[-n_test:])
    remain = wells[:-n_test]
    n_val = max(1, int(len(remain) * val_frac))
    val_wells = set(remain[-n_val:])
    train_wells = set(remain[:-n_val])

    assert not (train_wells & val_wells), "LEAKAGE: well in train & val!"
    assert not (train_wells & test_wells), "LEAKAGE: well in train & test!"
    assert not (val_wells & test_wells), "LEAKAGE: well in val & test!"

    df_train = df[df["Well_Name"].isin(train_wells)].reset_index(drop=True)
    df_val = df[df["Well_Name"].isin(val_wells)].reset_index(drop=True)
    df_test = df[df["Well_Name"].isin(test_wells)].reset_index(drop=True)

    print(
        f"Wells total/train/val/test: "
        f"{len(wells)}/{len(train_wells)}/{len(val_wells)}/{len(test_wells)}"
    )
    print(f"Rows  train: {len(df_train):,}  val: {len(df_val):,}  test: {len(df_test):,}")

    return df_train, df_val, df_test, train_wells, val_wells, test_wells

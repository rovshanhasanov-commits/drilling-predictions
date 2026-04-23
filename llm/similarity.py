"""Find the top-K most similar historical wells to the target (well, date).

Ported from LLM-powered predictions/similarity.py. TF-IDF embeddings are looked up
from the original `LLM-powered predictions/embeddings/` folder so we don't duplicate
the 4MB matrix.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Reuse the existing TF-IDF artifacts from the previous app — no need to regenerate.
EMB_DIR = Path(__file__).resolve().parent.parent.parent / "LLM-powered predictions" / "embeddings"


def _load_embeddings(emb_dir: Path = EMB_DIR):
    with open(emb_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(emb_dir / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(emb_dir / "well_date_index.pkl", "rb") as f:
        well_date_index = pickle.load(f)
    return vectorizer, tfidf_matrix, well_date_index


def _last_activity_per_day(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values("End_Hour")
        .groupby(["Well_Name", "Report_Date"])
        .last()
        .reset_index()
    )


def find_similar_wells(
    target_well: str,
    target_date,
    df: pd.DataFrame,
    get_day_activities_fn,
    top_k: int = 5,
    shortlist_size: int = 15,
    emb_dir: Path = EMB_DIR,
) -> dict:
    """Returns {(well, next_date): next_day_DataFrame} for up to top_k similar wells."""
    _, tfidf_matrix, well_date_index = _load_embeddings(emb_dir)
    index_lookup = {(w, d): i for i, (w, d) in enumerate(well_date_index)}

    target_activities = get_day_activities_fn(target_well, target_date)
    if target_activities.empty:
        return {}
    last = target_activities.sort_values("End_Hour").iloc[-1]
    t_phase = last["Phase"]
    t_step = last["Phase_Step"]
    t_depth = last["DepthEnd_ft"]
    if pd.isna(t_phase) or pd.isna(t_depth):
        return {}

    last_per_day = _last_activity_per_day(df)
    candidates = last_per_day[
        (last_per_day["Well_Name"] != target_well)
        & (last_per_day["Phase"] == t_phase)
        & (last_per_day["Phase_Step"] == t_step)
    ].copy()
    if candidates.empty:
        candidates = last_per_day[
            (last_per_day["Well_Name"] != target_well)
            & (last_per_day["Phase"] == t_phase)
        ].copy()
    if candidates.empty:
        return {}

    candidates["depth_diff"] = (candidates["DepthEnd_ft"] - t_depth).abs()
    candidates = candidates.nsmallest(shortlist_size, "depth_diff")

    t_idx = index_lookup.get((target_well, str(target_date)))
    if t_idx is None:
        candidates = candidates.head(top_k)
    else:
        cand_indices, cand_rows = [], []
        for _, row in candidates.iterrows():
            idx = index_lookup.get((row["Well_Name"], str(row["Report_Date"])))
            if idx is not None:
                cand_indices.append(idx)
                cand_rows.append(row)
        if cand_indices:
            sims = cosine_similarity(tfidf_matrix[t_idx], tfidf_matrix[cand_indices]).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k]
            candidates = pd.DataFrame([cand_rows[i] for i in top_idx])
        else:
            candidates = candidates.head(top_k)

    result = {}
    well_dates_map = df.groupby("Well_Name")["Report_Date"].apply(
        lambda x: sorted(x.unique())
    ).to_dict()
    for _, row in candidates.iterrows():
        well = row["Well_Name"]
        date = row["Report_Date"]
        wd = well_dates_map.get(well, [])
        try:
            i = list(wd).index(date)
        except ValueError:
            continue
        if i + 1 >= len(wd):
            continue
        nd = wd[i + 1]
        nxt = get_day_activities_fn(well, nd)
        if not nxt.empty:
            result[(well, nd)] = nxt
        if len(result) >= top_k:
            break
    return result

"""Assemble & execute analysis.ipynb.

Run with the interpreter that has TF/pandas/nbformat available:
  c:/Users/rovshan/anaconda3/python.exe misc/"Improvement opportunities"/_build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # .../Drilling - End to End Prediction
NB_PATH = HERE / "analysis.ipynb"

CELLS: list[tuple[str, str]] = []  # (type, source)


def md(s: str) -> None:
    CELLS.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    CELLS.append(("code", s.strip("\n")))


# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
md(r"""
# Drilling Ops Seq2Seq — Improvement Opportunities

Current autoregressive operation top-1 accuracy is **0.389** (0.577 at step 1 → 0.307 at step 8).
This notebook looks for concrete, prioritised ways to lift that number.

**Scope.** Consumes the existing evaluation artifacts (`results/.../eval_*/`), the raw
`misc/full_dataset.csv`, and the three split parquets. No TensorFlow / no re-inference.

**Convention.** Each section opens with the question it answers; results feed
`synopsis.md`.
""")

code(r"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["figure.dpi"] = 110
sns.set_style("whitegrid")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)

HERE = Path.cwd()  # expected to be .../misc/Improvement opportunities
REPO = HERE.parent.parent

# ensure repo importable
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA_CSV   = REPO / "misc" / "full_dataset.csv"
MASTER_CSV = (REPO / ".." / "Data" / "Master_Data_With_ID.csv").resolve()
MODEL_DIR  = REPO / "models" / "seq2seq_N25_K8_T5_dummies"
PARQ_DIR   = (REPO / ".." / "Data" / "Data for model (E2E)" / "dummies").resolve()
EVAL_ROOT  = REPO / "results" / "seq2seq_N25_K8_T5_dummies"
# newest eval_* folder
EVAL_DIR   = sorted(EVAL_ROOT.glob("eval_*"))[-1]
print("EVAL_DIR:", EVAL_DIR.name)

FINDINGS: dict = {}  # populated section-by-section, persisted at the end
HIERARCHY = ["phase", "phase_step", "major_ops_code", "operation"]
""")

# ---------------------------------------------------------------------------
# 1. Schema & null audit
# ---------------------------------------------------------------------------
md(r"""
## 1. Schema & null audit

**Question.** Does the training pipeline silently corrupt labels — e.g., are the 19k NaN
`operation` rows I saw earlier being quietly relabeled as the most-frequent class?
""")

code(r"""
df = pd.read_csv(DATA_CSV, low_memory=False)
print(f"rows: {len(df):,}  wells: {df['Well_Name'].nunique()}  columns: {df.shape[1]}")
null_counts = df.isna().sum().sort_values(ascending=False)
print("\nColumns with nulls:")
print(null_counts[null_counts > 0].to_string())
FINDINGS["n_rows"]   = int(len(df))
FINDINGS["n_wells"]  = int(df["Well_Name"].nunique())
FINDINGS["nan_operation"]      = int(df["operation"].isna().sum())
FINDINGS["nan_major_ops_code"] = int(df["major_ops_code"].isna().sum())
""")

code(r"""
# What does the label-encoder map NaN rows to?
with open(MODEL_DIR / "encoders.pkl", "rb") as f:
    enc = pickle.load(f)
te = enc["target_encoders"]

# The preprocessing pipeline (run_preprocessing.py:110) uses:
#   lambda x: classes.index(x) if x in classes else fb_idx
# where fb_idx = most-frequent TRAIN value for that column.
# Because NaN ∉ classes, every NaN target is silently remapped to the majority class.
print("Target-encoder vocab sizes (post-EOO):", {k: len(v.classes_) for k, v in te.items()})
train_mode_op = df["operation"].value_counts().idxmax()
train_mode_moc = df["major_ops_code"].value_counts().idxmax()
print(f"\nfallback for operation NaN   -> {train_mode_op!r}  (count={df['operation'].value_counts().max():,})")
print(f"fallback for major_ops_code NaN -> {train_mode_moc!r}")
FINDINGS["operation_fallback"] = str(train_mode_op)
FINDINGS["moc_fallback"]       = str(train_mode_moc)

# Quantify: how often does the fallback win?
# Top-N operations in TRAIN parquet (which has the post-fallback labels)
train_parq = pd.read_parquet(PARQ_DIR / "df_train.parquet", columns=["operation_target_enc"])
id_to_class = {i: c for i, c in enumerate(te["operation"].classes_)}
train_op_counts = train_parq["operation_target_enc"].map(id_to_class).value_counts()
# Compare to raw (non-null) counts for the same wells: we can't join on well easily
# here without re-splitting, so instead contrast parquet majority-class count to
# raw CSV majority-class count.
raw_top1 = df["operation"].value_counts().iloc[0]
parq_top1 = train_op_counts.iloc[0]
print(f"\nraw {train_mode_op} count (CSV, all wells): {raw_top1:,}")
print(f"train-parquet '{train_op_counts.index[0]}' count:         {parq_top1:,}")
print(f"parquet surplus (incl. NaN-remaps):      {parq_top1 - int(round(raw_top1 * 662/972)):+,}"
      f"  (scaled for train-well fraction 662/972)")
FINDINGS["raw_top1_operation_csv"]   = int(raw_top1)
FINDINGS["parquet_top1_operation"]   = int(parq_top1)
""")

# ---------------------------------------------------------------------------
# 2. Hierarchy cardinalities & legality
# ---------------------------------------------------------------------------
md(r"""
## 2. Hierarchy cardinalities & legality

**Question.** Of the `8 · 14 · 41 · 78 = 358,176` theoretical (phase, phase_step, MOC,
operation) tuples, how many actually occur? A constrained decoder can exploit the gap.
""")

code(r"""
card = {c: df[c].nunique(dropna=True) for c in HIERARCHY}
print("Unique counts:", card)
theoretical = card["phase"] * card["phase_step"] * card["major_ops_code"] * card["operation"]
actual_tuples = df.dropna(subset=HIERARCHY).drop_duplicates(HIERARCHY).shape[0]
print(f"theoretical tuples: {theoretical:,}")
print(f"actual (in data)  : {actual_tuples:,}")
print(f"coverage ratio    : {actual_tuples/theoretical:.4%}")
FINDINGS["cardinality"] = card
FINDINGS["actual_tuples"] = int(actual_tuples)
FINDINGS["theoretical_tuples"] = int(theoretical)
FINDINGS["tuple_coverage"] = float(actual_tuples / theoretical)
""")

code(r"""
# Parent -> child legality sets (identical to what evaluation/metrics.build_hierarchy_sets produces,
# but reused locally to avoid re-reading Master_Data_With_ID).
clean = df.dropna(subset=HIERARCHY)
phase_to_step  = clean.groupby("phase")["phase_step"].unique().apply(set).to_dict()
step_to_moc    = clean.groupby("phase_step")["major_ops_code"].unique().apply(set).to_dict()
moc_to_op      = clean.groupby("major_ops_code")["operation"].unique().apply(set).to_dict()

rows = []
for p, steps in phase_to_step.items():
    rows.append({"level": "phase -> phase_step", "parent": p, "n_legal_children": len(steps)})
legal_steps_df = pd.DataFrame(rows).sort_values("n_legal_children", ascending=False)
print("\nPer-phase # legal phase_steps (max possible 14):")
print(legal_steps_df.to_string(index=False))

# Compactly: cross-entropy-ish "branching factor" per level
branching = {
    "phase_step | phase":   float(np.mean([len(v) for v in phase_to_step.values()])),
    "moc | phase_step":     float(np.mean([len(v) for v in step_to_moc.values()])),
    "operation | moc":      float(np.mean([len(v) for v in moc_to_op.values()])),
}
print("\nMean legal children per parent:")
for k, v in branching.items():
    print(f"  {k:20s} {v:.2f}")
FINDINGS["branching"] = branching
""")

code(r"""
# How often does the MODEL predict a tuple that's illegal vs. legal? from summary.json
with open(EVAL_DIR / "summary.json") as f:
    eval_summary = json.load(f)

print(f"hierarchy_validity_rate = {eval_summary['ar']['hierarchy_validity_rate']:.4f}  "
      f"(share of AR predictions that are structurally legal)")
print(f"\nCompare with conditional accuracies:")
for k, v in eval_summary["ar"]["conditional_acc"].items():
    print(f"  P({k}) = {v:.4f}")
FINDINGS["hierarchy_validity_rate"] = eval_summary["ar"]["hierarchy_validity_rate"]
FINDINGS["conditional_acc"] = eval_summary["ar"]["conditional_acc"]
""")

# ---------------------------------------------------------------------------
# 3. Class distribution & rare classes
# ---------------------------------------------------------------------------
md(r"""
## 3. Class distribution & rare classes

**Question.** How skewed are the four target distributions, and how much of the loss is
coming from noise in the long tail?
""")

code(r"""
def tail_stats(series):
    vc = series.value_counts()
    tot = vc.sum()
    out = {"n_classes": int(len(vc))}
    for t in [10, 50, 100, 500]:
        under = vc[vc < t]
        out[f"classes_<{t}"] = int(len(under))
        out[f"share_<{t}%"]  = round(100 * float(under.sum()) / tot, 3)
    return vc, out

summary_rows = {}
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, level in zip(axes.flat, HIERARCHY):
    vc, tab = tail_stats(df[level].dropna())
    summary_rows[level] = tab
    top = vc.head(30)
    ax.bar(range(len(top)), top.values, color="steelblue")
    ax.set_yscale("log")
    ax.set_title(f"{level} — top 30 (of {vc.shape[0]})")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=75, fontsize=8)
    ax.set_ylabel("count (log)")
plt.tight_layout()
plt.show()

print("\nRare-class tails:")
print(pd.DataFrame(summary_rows).T.to_string())
FINDINGS["rare_class_tails"] = summary_rows
""")

code(r"""
# Cumulative-coverage plot: how few classes cover 90% of rows?
fig, ax = plt.subplots(figsize=(10, 4))
for level, colour in zip(HIERARCHY, ["#4c72b0", "#dd8452", "#55a467", "#c44e52"]):
    vc = df[level].dropna().value_counts()
    cov = vc.cumsum() / vc.sum()
    ax.plot(range(1, len(vc) + 1), cov.values, label=level, color=colour)
ax.axhline(0.9, color="grey", linestyle=":", linewidth=1)
ax.axhline(0.99, color="grey", linestyle=":", linewidth=1)
ax.set_xlabel("classes (sorted by freq, desc)")
ax.set_ylabel("cumulative coverage")
ax.set_title("Class-coverage curves (dotted lines at 90% / 99%)")
ax.legend()
plt.show()

cov90 = {}
for level in HIERARCHY:
    vc = df[level].dropna().value_counts()
    cum = vc.cumsum() / vc.sum()
    cov90[level] = int((cum < 0.9).sum() + 1)   # +1 = class that crosses 90%
print(f"\n# classes needed to cover 90% of rows: {cov90}")
FINDINGS["classes_for_90pct_cov"] = cov90
""")

# ---------------------------------------------------------------------------
# 4. Vocabulary-reduction candidates
# ---------------------------------------------------------------------------
md(r"""
## 4. Vocabulary-reduction candidates

**Question.** Are some operations near-duplicates that hurt training signal more than
they help?  We look for operations that (a) live inside the same `(phase_step, MOC)`
bucket, (b) have similar successor distributions, and (c) share a name prefix suggesting
they're variants of the same activity.
""")

code(r"""
# (a) Operations co-occurring in the same (phase_step, MOC) bucket
same_bucket = (
    df.dropna(subset=HIERARCHY)
      .groupby(["phase_step", "major_ops_code"])["operation"]
      .agg(lambda s: sorted(set(s)))
)
multi = same_bucket[same_bucket.apply(len) > 1]
print(f"buckets with >1 operation: {len(multi)} / {len(same_bucket)}")
print("\nSample of multi-op buckets:")
print(multi.head(10).to_string())
""")

code(r"""
# (b) Successor-distribution similarity between operations that share a MOC.
# For each op, compute empirical P(next op | current op) within well.
clean = df.dropna(subset=HIERARCHY).sort_values(["Well_Name", "Start_Time"]).reset_index(drop=True)
clean["next_op"] = clean.groupby("Well_Name")["operation"].shift(-1)
pair = clean.dropna(subset=["next_op"])
pair = pair[pair["Well_Name"] == pair["Well_Name"].shift(0)]  # same well (already true)

# Normalise counts to distributions per op
trans = pd.crosstab(pair["operation"], pair["next_op"])
trans_dist = trans.div(trans.sum(axis=1), axis=0).fillna(0)

def jsd(p, q, eps=1e-12):
    p = np.asarray(p) + eps; q = np.asarray(q) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

ops = trans_dist.index.tolist()
sims = []
for i, a in enumerate(ops):
    for b in ops[i+1:]:
        # Only pair ops that SHARE at least one (phase_step, MOC) bucket
        a_bk = set(zip(clean[clean["operation"] == a]["phase_step"],
                       clean[clean["operation"] == a]["major_ops_code"]))
        b_bk = set(zip(clean[clean["operation"] == b]["phase_step"],
                       clean[clean["operation"] == b]["major_ops_code"]))
        if not (a_bk & b_bk):
            continue
        jd = jsd(trans_dist.loc[a].values, trans_dist.loc[b].values)
        sims.append((a, b, float(jd),
                     int(trans.loc[a].sum()), int(trans.loc[b].sum())))

merge_candidates = (
    pd.DataFrame(sims, columns=["op_a", "op_b", "js_div", "count_a", "count_b"])
      .sort_values("js_div")
      .head(20)
      .reset_index(drop=True)
)
print("Top-20 candidate merges by Jensen-Shannon divergence of successor dist (low = similar):")
print(merge_candidates.to_string())
FINDINGS["merge_candidates_top20"] = merge_candidates.to_dict(orient="records")
""")

code(r"""
# (c) Name-prefix grouping — identify "families" by leading token
prefix_groups = defaultdict(list)
for op in df["operation"].dropna().unique():
    root = op.split("_")[0]
    prefix_groups[root].append(op)
families = {k: v for k, v in prefix_groups.items() if len(v) > 1}
print(f"Name families with >1 variant: {len(families)}")
for k, v in sorted(families.items(), key=lambda kv: -len(kv[1]))[:15]:
    print(f"  {k:8s} -> {v}")
FINDINGS["name_families"] = {k: v for k, v in families.items()}
""")

# ---------------------------------------------------------------------------
# 5. Transition entropy by level
# ---------------------------------------------------------------------------
md(r"""
## 5. Transition entropy by level

**Question.** Where is the prediction task actually hard vs. near-deterministic?
We compute H(next | current) empirically per level; low entropy means a rules-based
or lookup baseline already does well.
""")

code(r"""
def shifted_entropy(frame, col):
    frame = frame.sort_values(["Well_Name", "Start_Time"]).copy()
    frame["_next"] = frame.groupby("Well_Name")[col].shift(-1)
    frame = frame.dropna(subset=[col, "_next"])
    tab = pd.crosstab(frame[col], frame["_next"]).astype(float)
    probs = tab.div(tab.sum(axis=1), axis=0).replace(0, np.nan)
    per_row_H = -(probs * np.log2(probs)).sum(axis=1, skipna=True)
    # Marginal H of next-token
    marg = frame["_next"].value_counts(normalize=True)
    Hnext = -(marg * np.log2(marg)).sum()
    # Conditional H: Σ p(curr) * H(next|curr)
    pcurr = frame[col].value_counts(normalize=True)
    Hcond = (pcurr.reindex(per_row_H.index).fillna(0) * per_row_H.fillna(0)).sum()
    return float(Hnext), float(Hcond), float(Hnext - Hcond)

ent_rows = []
for level in HIERARCHY:
    Hnext, Hcond, mi = shifted_entropy(df, level)
    ent_rows.append({
        "level": level,
        "H(next)_bits":       round(Hnext, 3),
        "H(next|curr)_bits":  round(Hcond, 3),
        "I(curr; next)_bits": round(mi,    3),
    })
entropy_df = pd.DataFrame(ent_rows)
print(entropy_df.to_string(index=False))
FINDINGS["entropy_by_level"] = entropy_df.set_index("level").to_dict(orient="index")
""")

# ---------------------------------------------------------------------------
# 6. Per-position narrowing / horizon difficulty
# ---------------------------------------------------------------------------
md(r"""
## 6. Per-position narrowing (how hard is step-k vs step-1?)

**Question.** The model drops from 0.577 at step 1 to 0.307 at step 8. Is that because
step 8's true distribution is irreducibly wider, or because the model isn't learning
long-range structure?
""")

code(r"""
# Data-side: for every row, what's the empirical distribution of operation_{t+k} given
# operation_t? Higher entropy at larger k = irreducibly harder.
per_k = []
op_series = df.dropna(subset=["operation"]).sort_values(["Well_Name", "Start_Time"])
groups = op_series.groupby("Well_Name")["operation"]

for k in range(1, 9):
    shifted = groups.shift(-k)
    frame = pd.DataFrame({"curr": op_series["operation"].values, "fut": shifted.values}).dropna()
    tab = pd.crosstab(frame["curr"], frame["fut"]).astype(float)
    probs = tab.div(tab.sum(axis=1), axis=0).replace(0, np.nan)
    per_row_H = -(probs * np.log2(probs)).sum(axis=1, skipna=True)
    pcurr = frame["curr"].value_counts(normalize=True)
    Hcond = (pcurr.reindex(per_row_H.index).fillna(0) * per_row_H.fillna(0)).sum()
    per_k.append((k, float(Hcond)))

horizon_df = pd.DataFrame(per_k, columns=["k", "H(op_{t+k}|op_t)_bits"])
print(horizon_df.to_string(index=False))

# overlay model's per-step accuracy
per_step = pd.read_csv(EVAL_DIR / "per_step_accuracy.csv")
op_ar = per_step[(per_step["head"] == "operation") & (per_step["mode"] == "ar")]

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(horizon_df["k"], horizon_df.iloc[:, 1], "o-", color="#c44e52", label="H(op_{t+k} | op_t)")
ax1.set_ylabel("Entropy (bits)", color="#c44e52")
ax2 = ax1.twinx()
ax2.plot(op_ar["step"].values, op_ar["top1"].values, "s-", color="#4c72b0", label="AR operation top-1")
ax2.set_ylabel("Model accuracy", color="#4c72b0")
ax1.set_xlabel("k (steps ahead)")
ax1.set_title("Irreducible difficulty vs. model accuracy")
plt.show()

FINDINGS["horizon_entropy_op"] = {int(k): float(h) for k, h in per_k}
FINDINGS["ar_operation_per_step"] = op_ar[["step", "top1", "top3"]].to_dict(orient="records")
""")

# ---------------------------------------------------------------------------
# 7. Repetition patterns
# ---------------------------------------------------------------------------
md(r"""
## 7. Repetition patterns

**Question.** How often does the same operation repeat? If runs of N≥3 are common a
copy-state head (or a simple "repeat previous token" bias) buys accuracy cheaply.
""")

code(r"""
# Run-length encoding per well
clean = df.dropna(subset=["operation"]).sort_values(["Well_Name", "Start_Time"])
runs = []
for well, sub in clean.groupby("Well_Name", sort=False):
    ops = sub["operation"].values
    if len(ops) == 0:
        continue
    cur, n = ops[0], 1
    for nxt in ops[1:]:
        if nxt == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur, n = nxt, 1
    runs.append((cur, n))

rl_df = pd.DataFrame(runs, columns=["op", "run_len"])
print("Run-length stats, overall:")
print(rl_df["run_len"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())

repeat_frac = (rl_df["run_len"] > 1).mean()
print(f"\n% of runs with length>1: {100*repeat_frac:.2f}%")

# For the top-10 ops, median run length
top10 = clean["operation"].value_counts().head(10).index
rl_top = rl_df[rl_df["op"].isin(top10)].groupby("op")["run_len"].agg(["count","mean","median","max"])
rl_top = rl_top.reindex(top10)
print("\nRun lengths for top-10 operations:")
print(rl_top.to_string())

# Share of same-op transitions
trans_rows = []
for well, sub in clean.groupby("Well_Name", sort=False):
    ops = sub["operation"].values
    if len(ops) < 2:
        continue
    trans_rows.append(int((ops[1:] == ops[:-1]).sum()))
    trans_rows.append(-(len(ops) - 1))   # negative = total transitions
total_trans = -sum(x for x in trans_rows if x < 0)
self_trans  =  sum(x for x in trans_rows if x > 0)
same_frac = self_trans / total_trans
print(f"\nSame-op (self) transitions: {self_trans:,} / {total_trans:,}  ({100*same_frac:.2f}%)")

FINDINGS["same_op_transition_frac"] = float(same_frac)
FINDINGS["run_length_mean"]   = float(rl_df["run_len"].mean())
FINDINGS["run_length_median"] = float(rl_df["run_len"].median())
FINDINGS["run_length_p90"]    = float(rl_df["run_len"].quantile(0.9))
""")

# ---------------------------------------------------------------------------
# 8. Leakage & split diagnostics
# ---------------------------------------------------------------------------
md(r"""
## 8. Leakage & per-well variance

**Question.** Split is well-level (see `preprocessing/split.py`). Are wells really
disjoint across train / val / test? And does the huge 0.233 per-well std on operation
accuracy come from a handful of catastrophic wells or a long tail?
""")

code(r"""
train_wells = set(pd.read_parquet(PARQ_DIR / "df_train.parquet", columns=["Well_Name"])["Well_Name"].unique())
val_wells   = set(pd.read_parquet(PARQ_DIR / "df_val.parquet",   columns=["Well_Name"])["Well_Name"].unique())
test_wells  = set(pd.read_parquet(PARQ_DIR / "df_test.parquet",  columns=["Well_Name"])["Well_Name"].unique())

print(f"wells: train={len(train_wells)}  val={len(val_wells)}  test={len(test_wells)}  "
      f"sum={len(train_wells)+len(val_wells)+len(test_wells)}  raw={df['Well_Name'].nunique()}")
print("overlap train∩val  :", len(train_wells & val_wells))
print("overlap train∩test :", len(train_wells & test_wells))
print("overlap val∩test   :", len(val_wells & test_wells))

# Time overlap: are test wells operating in the same calendar window as train wells?
df["Report_Date"] = pd.to_datetime(df["Report_Date"], errors="coerce")
by_well = df.groupby("Well_Name")["Report_Date"].agg(["min", "max"])
by_well["split"] = by_well.index.map(
    lambda w: "train" if w in train_wells else "val" if w in val_wells else "test" if w in test_wells else "none"
)
fig, ax = plt.subplots(figsize=(10, 3))
for sp, c in zip(["train", "val", "test"], ["#4c72b0", "#dd8452", "#c44e52"]):
    sub = by_well[by_well["split"] == sp]
    ax.scatter(sub["min"], [sp]*len(sub), s=3, c=c, label=f"{sp} ({len(sub)})", alpha=0.5)
    ax.scatter(sub["max"], [sp]*len(sub), s=3, c=c, alpha=0.5)
ax.set_title("Well date ranges per split — temporal overlap check")
ax.set_xlabel("Report date")
plt.tight_layout(); plt.show()

FINDINGS["split_well_counts"] = {"train": len(train_wells), "val": len(val_wells), "test": len(test_wells)}
FINDINGS["split_overlap"] = {
    "train_val":  len(train_wells & val_wells),
    "train_test": len(train_wells & test_wells),
    "val_test":   len(val_wells & test_wells),
}
""")

code(r"""
# Distribution of per-well operation accuracy on test
per_well = pd.read_csv(EVAL_DIR / "per_well_accuracy.csv")
print(per_well.describe().to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for ax, col in zip(axes, ["operation_top1", "major_ops_code_top1"]):
    ax.hist(per_well[col], bins=30, color="#4c72b0", edgecolor="white")
    ax.axvline(per_well[col].mean(), color="k", linestyle=":")
    ax.set_title(f"{col} (mean={per_well[col].mean():.3f})")
plt.tight_layout(); plt.show()

# Near-zero wells
worst = per_well.sort_values("operation_top1").head(15)[
    ["well_name", "n_sequences", "phase_top1", "phase_step_top1",
     "major_ops_code_top1", "operation_top1"]
]
print("\nWorst-15 wells (operation top1):")
print(worst.to_string(index=False))

# Enrich with well meta (operator, rig)
meta_cols = [c for c in df.columns if c.startswith(("Operator_", "Rig_Contractor_"))]
well_meta = df.groupby("Well_Name")[meta_cols].first()
worst_meta = worst.merge(well_meta, left_on="well_name", right_index=True)
print("\nOperator/Rig flags for worst wells:")
print(worst_meta[["well_name", "operation_top1"] + meta_cols].to_string(index=False))

FINDINGS["per_well_op_mean"] = float(per_well["operation_top1"].mean())
FINDINGS["per_well_op_std"]  = float(per_well["operation_top1"].std())
FINDINGS["worst_15_wells"]   = worst[["well_name","operation_top1","n_sequences"]].to_dict(orient="records")
""")

# ---------------------------------------------------------------------------
# 9. Consume eval artifacts — cross-tabs
# ---------------------------------------------------------------------------
md(r"""
## 9. Consuming eval artifacts — confusion & per-step stratification

**Question.** Which specific misclassifications drive the operation-head failure, and
what does step-by-step top-1 vs top-3 tell us?
""")

code(r"""
# Confusion pairs — operation
cm_op = pd.read_csv(EVAL_DIR / "confusion_operation.csv")
print("Top confused operation pairs (from artifact):")
print(cm_op.head(15).to_string(index=False))

cm_moc = pd.read_csv(EVAL_DIR / "confusion_major_ops_code.csv")
print("\nTop confused MOC pairs:")
print(cm_moc.head(15).to_string(index=False))

# How much of the error budget is concentrated in the top-5 pairs?
def error_concentration(csv, level):
    total_mistakes = csv["count"].sum()
    top5 = csv.head(5)["count"].sum()
    return float(top5 / total_mistakes)

FINDINGS["err_top5_concentration"] = {
    "operation":      error_concentration(cm_op, "operation"),
    "major_ops_code": error_concentration(cm_moc, "major_ops_code"),
}
print("\nShare of errors from top-5 confusion pairs:")
for k, v in FINDINGS["err_top5_concentration"].items():
    print(f"  {k:20s} {v:.2%}")
""")

code(r"""
# Per-step top-1 + top-3 all heads
per_step = pd.read_csv(EVAL_DIR / "per_step_accuracy.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, head in zip(axes.flat, HIERARCHY):
    sub = per_step[per_step["head"] == head]
    for mode, style in [("tf", "--"), ("ar", "-")]:
        d = sub[sub["mode"] == mode]
        ax.plot(d["step"], d["top1"],   style, label=f"{mode} top1", color="#c44e52" if mode=="ar" else "#4c72b0")
        ax.plot(d["step"], d["top3"],   style, label=f"{mode} top3", color="#c44e52" if mode=="ar" else "#4c72b0", alpha=0.4)
    ax.set_title(head)
    ax.set_xlabel("step"); ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# TF-AR gap summary
gap = {}
for head in HIERARCHY:
    tf = per_step[(per_step["head"]==head) & (per_step["mode"]=="tf")]["top1"].mean()
    ar = per_step[(per_step["head"]==head) & (per_step["mode"]=="ar")]["top1"].mean()
    gap[head] = {"tf_top1": round(float(tf),4), "ar_top1": round(float(ar),4),
                 "tf_minus_ar": round(float(tf-ar),4)}
print("TF vs AR gap:")
print(pd.DataFrame(gap).T.to_string())
FINDINGS["tf_ar_gap"] = gap
""")

# ---------------------------------------------------------------------------
# Findings persistence
# ---------------------------------------------------------------------------
md(r"""
## Persist findings

Everything above feeds `findings.json`, which `synopsis.md` cites.
""")

code(r"""
def _clean(o):
    if isinstance(o, dict):  return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list):  return [_clean(x) for x in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o

out = HERE / "findings.json"
with open(out, "w") as f:
    json.dump(_clean(FINDINGS), f, indent=2, default=str)
print(f"wrote {out}")
""")


# ---------------------------------------------------------------------------
# Assemble and execute
# ---------------------------------------------------------------------------
def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    for kind, src in CELLS:
        if kind == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src))

    print(f"Writing {NB_PATH} ({len(nb.cells)} cells)")
    nbf.write(nb, NB_PATH)

    print("Executing notebook...")
    client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}})
    client.execute()
    nbf.write(nb, NB_PATH)

    findings = json.loads((HERE / "findings.json").read_text())
    print(f"\nfindings.json keys: {list(findings.keys())}")


if __name__ == "__main__":
    main()

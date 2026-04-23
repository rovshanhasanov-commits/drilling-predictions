# Memo — NaN handling in MOC and Operation columns

Written 2026-04-18, updated 2026-04-21. Self-contained; readable by both
the project owner and a future Claude session.

## What we're working on

Improving the seq2seq drilling-operation predictor. Current autoregressive
operation top-1 accuracy is **0.389** (test split, 24,733 sequences, 193
wells). Full EDA and recommendation list live in
[`synopsis.md`](./synopsis.md); evidence in [`analysis.ipynb`](./analysis.ipynb);
machine-readable numbers in [`findings.json`](./findings.json); latest
evaluation artifacts in
[`../../results/seq2seq_N25_K8_T5_dummies/eval_20260418_152124/`](../../results/seq2seq_N25_K8_T5_dummies/eval_20260418_152124/).

## The bug that motivated this memo

`full_dataset.csv` has **19,090 NaN `operation`** rows (12.4 %) and **15 NaN
`major_ops_code`** rows. The current training pipeline —
[`preprocessing/run_preprocessing.py:107`](../../preprocessing/run_preprocessing.py#L107)
— fills them via a global-mode fallback:

```python
fb_idx = {c: i for i, c in enumerate(le.classes_)}[
    df_train[col].value_counts().idxmax()        # <<-- global mode
]
split_df[col + "_target_enc"] = split_df[col].apply(
    lambda x: list(le.classes_).index(x) if x in classes else fb_idx
)
```

Every NaN `operation` becomes `DRL` (the overall mode) regardless of MOC, and
every NaN `major_ops_code` becomes `NON_DRILL`. Because NaN is not in
`le.classes_`, the lambda always falls through. This:

- Relabels ~12 % of training targets with a single class biased toward the
  majority — classic mode-collapse recipe.
- Manufactures **illegal** (MOC, Op) pairs (e.g. MOC=`SAFETY`, Op=`DRL`),
  which matches the observed **19 % hierarchy-invalid** AR predictions and
  the observed confusion pattern (RIG_SVC→DRL 39 %, RIG→DRILL 30 %, etc.).
- Parquet evidence: `df_train.parquet`'s top operation id has **25,033** rows,
  vs. the ~10,832 expected for `DRL` scaled to 662 train wells — surplus ≈
  NaN count.

## Structure of the missingness (critical — it's not random)

Per-well breakdown of NaN `operation` fraction (10 % bins, with 0 % and 100 %
called out separately):

| nan_frac(Operation) | # wells | rows    | # NaN ops | % of all NaN |
|---------------------|--------:|--------:|----------:|-------------:|
| 0 %                 |     789 | 129,982 |         0 |        0.0 % |
| 0–10 %              |      17 |   2,936 |        81 |        0.4 % |
| 10–20 %             |       4 |     548 |        89 |        0.5 % |
| 20–30 %             |       9 |   1,079 |       274 |        1.4 % |
| 30–40 %             |       9 |     787 |       276 |        1.4 % |
| 40–50 %             |       2 |     282 |       129 |        0.7 % |
| 50–60 %             |       3 |     388 |       214 |        1.1 % |
| 60–70 %             |       1 |     113 |        68 |        0.4 % |
| 70–80 %             |       2 |     253 |       187 |        1.0 % |
| 80–90 %             |       1 |     105 |        90 |        0.5 % |
| 90–100 %            |       5 |     734 |       729 |        3.8 % |
| **100 %**           | **130** | **16,953** | **16,953** | **88.8 %** |

So **89 % of all NaN operations are concentrated in 130 wells where *every*
row has a NaN operation** — these are systematic reporting failures
(Big Eddy Unit DI, James Ranch Unit Apache, Poker Lake Unit families). Not
random missingness. The remaining 41 partial-NaN wells account for ~2,137
NaN rows across ~7,225 total rows.

## The existing notebook does the right thing — but doesn't reach the model

[`../../../Reformat data and include features.ipynb`](../../../Reformat%20data%20and%20include%20features.ipynb)
(Cells 6–7) already applies a much better cleaning:

- **MOC**: `fillna("UNKNOWN")` then replace `DRLG → DRILL`, `CMT → CEMENT`,
  `BOP → BOPE` (three semantic merges that collapse ~4k duplicate-class rows).
- **Operation**: strip spaces (fixes `"LWD_ GAMMA"` etc.), then fill via
  **MOC-conditional mode** (13,464 of the 19,090 NaNs caught this way), then
  fall through to `Operation = Major_Ops_Code` string for the remaining
  5,626 (wells where the MOC has *zero* non-null Operation).

That notebook writes to `Data/Prepped data with extracted features.csv` — a
completely separate file. The model pipeline reads `full_dataset.csv`
instead, so none of this cleaning is in use today.

## Temporal safety — why we can't drop individual rows

Sequences are per-well, windowed (N=25 context, K=8 horizon). Dropping row N
mid-well breaks the encoder window, corrupts `depth_change`,
`op_sequence_number`, `phase_op_index`, and shifts every `*_next` target that
comes from `groupby("Well_Name").shift(-1)`. Dropping an *entire* well is
safe (sequences never cross wells, splits are already well-level).

## Plan (decided 2026-04-21)

Two distinct populations of NaN, treated differently. Applied to the
pre-split dataframe in `preprocessing/clean.py`, *before* `split.py` and
`fit_target_encoders`.

**Rationale for the split treatment.** The 100 %-NaN wells have zero
within-well signal — any fill is 100 % synthetic and becomes a deterministic
function of (Phase_Step, MOC). Training on those labels teaches the model an
artificial MOC→Op mapping that's tighter than the real data supports; it
looks like a gentler version of the original F1 bug. Partial-NaN wells are
different: their non-NaN rows are real ground truth, and conditional-mode
fill borrows from the clean majority (789 fully-labeled wells) without
fabricating correlations that aren't there.

### Step 0 — Drop fully-unlabeled wells

Drop the 130 wells where every row has a NaN operation. Cost: ~17k rows
(~11 % of data), eliminates 89 % of all NaN operations, zero loss of real
operation signal.

```python
nan_frac = df["Operation"].isna().groupby(df["Well_Name"]).mean()
wells_to_drop = nan_frac[nan_frac == 1.0].index
n_rows_before = len(df)
df = df[~df["Well_Name"].isin(wells_to_drop)].copy()
print(f"Dropped {len(wells_to_drop)} wells, "
      f"{n_rows_before - len(df):,} rows "
      f"({(n_rows_before - len(df)) / n_rows_before:.1%})")
```

### Step 1 — Normalize and strip

Do this before any conditional-mode computation so the grouping keys are
clean.

```python
df["Major_Ops_Code"] = (df["Major_Ops_Code"].astype(str).str.strip()
                        .replace({"DRLG": "DRILL",
                                  "CMT": "CEMENT",
                                  "BOP": "BOPE",
                                  "nan": None}))
df["Operation"] = (df["Operation"].astype(str).str.strip()
                   .replace({"nan": None}))
```

Note the `"nan"` → `None` replacement: `astype(str)` turns real `NaN` into
the string `"nan"`, which would then silently survive `fillna`.

### Step 2 — Fill MOC NaNs via Phase_Step-conditional mode

Only 15 rows — global mode would suffice, but Phase_Step-conditional is
cheap and tighter:

```python
moc_mode_by_step = (df.dropna(subset=["Major_Ops_Code"])
                      .groupby("Phase_Step")["Major_Ops_Code"]
                      .agg(lambda s: s.mode().iloc[0]))
df["Major_Ops_Code"] = df["Major_Ops_Code"].fillna(
    df["Phase_Step"].map(moc_mode_by_step)
)
```

### Step 3 — Fill Operation NaNs via (Phase_Step, MOC)-conditional mode

One level tighter than the old notebook, which conditioned only on MOC:

```python
op_mode_by_step_moc = (df.dropna(subset=["Operation"])
                         .groupby(["Phase_Step", "Major_Ops_Code"])["Operation"]
                         .agg(lambda s: s.mode().iloc[0]))
df["Operation"] = df["Operation"].fillna(
    df.set_index(["Phase_Step", "Major_Ops_Code"])
      .index.map(op_mode_by_step_moc)
      .to_series()
      .values
)
```

### Step 4 — Residual-NaN diagnostic

After Step 3 there may still be rows where the `(Phase_Step, MOC)` bucket
has no observed Operation in the training majority. Report and decide:

```python
residual = df["Operation"].isna().sum()
print(f"Residual NaN operations: {residual}")
if residual > 0:
    print("Residual NaN breakdown by (Phase_Step, MOC):")
    print(df[df["Operation"].isna()]
          .groupby(["Phase_Step", "Major_Ops_Code"]).size()
          .sort_values(ascending=False))
```

Decision rule for residuals:

- **0 residuals**: done.
- **Handful (<100) of residuals**: loss-mask them via an `op_label_real`
  flag fed to Keras `sample_weight` (keeps them as input context for the
  encoder but kills the gradient on the fabricated target).
- **Hundreds+**: stop and inspect the unmatched `(Phase_Step, MOC)`
  buckets before proceeding — this would indicate a data issue, not just
  sparse labels.

## Open items deferred to the next working session

- **Port the cleanup into the model pipeline.** Target location:
  [`preprocessing/clean.py`](../../preprocessing/clean.py) (already handles
  depth/date cleaning; MOC + Operation cleaning belongs here too). Do this
  BEFORE `split.py` runs so dropped wells never enter a split and the
  conditional modes are computed on the full pre-split dataframe.
- **Confirm the 3 MOC semantic merges with SME** (`DRLG→DRILL`,
  `CMT→CEMENT`, `BOP→BOPE`). `ops_codes.docx` Section 3 explicitly lists
  DRLG, CMT, and BOP as legacy/summary-level codes that map to the same
  activities as their detailed counterparts — point the SME at that
  section for fast sign-off.
- **Implement loss-masking plumbing** if Step 4 returns non-zero residuals.
  Adds an `op_label_real` flag in `preprocessing/features.py` and a
  `sample_weight` argument in `training/train.py`. Skip this work entirely
  if residuals are zero.
- **Rerun evaluation after the fix** and compare against the current
  artifacts in `eval_20260418_152124/`. Expect the `DRL` / `DRILL` mode
  collapse in the confusion tables to weaken materially without any
  architecture change. Also **recompute the JSD-nearest pairs from F4** —
  if near-duplicate candidates partially reflect NaN-fill artifacts, the
  merge candidate list will shift. Do not lock in vocab merges on
  pre-fix JSD numbers.

## Resolved / no longer open

- **Whether to drop wells vs. fill vs. loss-mask** (was open). Decision:
  drop 100 %-NaN wells, conditional-mode fill the rest, loss-mask any
  residuals. Rationale in the "Plan" preamble above.
- **Whether the 10 %-threshold rule of thumb from earlier analysis still
  applies** (was open). No — the per-well bin breakdown shows the
  missingness is essentially bimodal (0 % or 100 %), with a thin tail in
  between. A threshold rule isn't the right tool; the 100 %-vs-partial
  split is.

## What should be left alone (not this memo's scope)

- The shift-alignment diagnostic in `evaluation/` was removed earlier
  per user request; don't re-add.
- The larger architectural recommendations in `synopsis.md` (constrained
  hierarchical decoder, cascaded decoder, longer training, label smoothing,
  duration-head overhaul) are separate workstreams — *do not* bundle them
  into the NaN-handling change. Fix NaN first; measure; then revisit.

## Where to resume

1. Open [`preprocessing/clean.py`](../../preprocessing/clean.py) and add
   Steps 0–3 above, in order, before any sort / fill / derived-feature
   logic that depends on `Major_Ops_Code` or `Operation`.
2. Run `python -m preprocessing.run_preprocessing` to regenerate parquets
   under `../Data/Data for model (E2E)/`.
3. Print the Step 4 diagnostic (residual NaN operations + affected
   buckets).
4. Based on the residual count, either proceed directly to retrain or
   implement loss-masking first, then retrain, then re-run
   `python -m evaluation.run_evaluation --split test`.
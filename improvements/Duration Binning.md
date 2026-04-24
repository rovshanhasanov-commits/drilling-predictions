# Duration Binning — classification head for duration instead of (or alongside) regression

> **Reading this doc**: It is both an implementation instruction and a record of the
> discussion that produced the design. Sections 1–4 are the rationale and decisions,
> sections 5–12 are implementation detail, sections 13–14 are verification and open items,
> and section 15 is the discussion transcript for future-me / future-Claude to pick up
> mid-stream. Pick up implementation from **section 5 onwards**.

---

## 1. Problem

The current duration head is a single-float regression trained with Huber loss on
`StandardScaler(log1p(raw_hours))`. Eval numbers on the test split from
`seq2seq_N25_K8_T5_dummies` (2026-04-18):

| metric | value |
|---|---:|
| `mae_hours` | 1.56 |
| `medae_hours` | 0.56 |
| `p95_abs_err_hours` | 7.58 |
| `mae_long_ops_hours` (ops > 8h) | **11.19** |
| `r2_long_ops` | **−7.21** |

R² < 0 means the regressor is **worse than predicting the mean** on the long-duration
tail. `p95 = 7.58h` tells you this isn't a handful of outliers — the top 5% of
errors are ≥7.58 hours, which is a full tour of drilling operations.

Root causes:

1. **Bimodal / heavy-tailed target distribution.** Drilling durations cluster at 15-min
   increments for prep ops and hours for drilling runs; `log1p` compresses but doesn't
   linearize the structure.
2. **MSE/Huber weights all errors equally in transformed space.** The long tail
   dominates the gradient and pushes the model to a "middle" that hurts everyone.
3. **No way to express uncertainty.** A regressor is forced to pick a point, even when
   the distribution is genuinely bimodal ("this op is either 15 min or 4 hours").

## 2. Goal

Add a **classification head** that predicts duration as one of a small set of bins,
with the following design properties:

- **Coexists** with the existing regression head — both are toggleable from yaml, so
  we can ablate and compare without deleting code.
- **Uniform with the other 4 softmax heads** — same teacher forcing, same scheduled
  sampling, same sample-weight masking. No special-case plumbing.
- **Surfaces top-K bins** (with probabilities) to the LLM stage, so the LLM can pick a
  point value within the most likely bin and reason about uncertainty explicitly.
- **Reports a bin-center MAE in hours** as a backward-compat metric so results can be
  diffed against the regression head.
- **Optionally couples into the constraint decoder** (legal tuples → 5D) behind a yaml
  flag — disabled by default so bin accuracy can be measured in isolation first.

## 3. Why both heads coexist

Binning is expected to dominate on the long tail but lose precision on 15-min ops.
Keeping the regression head available:

- Lets us **ablate**: run training with `duration_next` only, `duration_bin_next` only,
  and both. Diff the eval artifacts.
- Costs almost nothing: both heads read from the same decoder hidden state; adding the
  bin head is one extra softmax layer + one extra CE loss term.
- Preserves the existing inference contract's `duration_hours: float` field (populated
  from regression head when on, from bin-center when off).

Once binning is clearly superior (or clearly worse), we can remove the regression path
in a follow-up. Don't delete it in this change.

## 4. Decisions locked in

| Decision | Chosen |
|---|---|
| New target variable name | `duration_bin_next` (parallel to `operation_next`). Strip `_next` → `duration_bin`. |
| Default bin edges | `[0.25, 0.5, 1.0, 2.0, 5.0, 10.0]` → 7 natural bins |
| Default bin labels | `["≤0.25 hr", "0.25-0.5 hr", "0.5-1 hr", "1-2 hrs", "2-5 hrs", "5-10 hrs", "10+ hrs"]` |
| Bin edges configurable via yaml? | **Yes** — `preprocessing.duration_bins.{edges, labels}`. Alternate fallback variant documented in §5.3. |
| Sentinels | `EOO`, `Unplanned`, `UNK` each get their own bin class → **10 classes total** under the default bins. Aligns with how the other 4 heads already handle sentinels. |
| Bin-center computation | **Empirical median of training data per bin**, computed at preprocessing time and saved in `encoders.pkl`. Not arithmetic midpoints (non-uniform distribution within a bin). |
| Apply `include_duration_bins_in_hierarchy` by default? | **No** — default `false`. Measure bin accuracy in isolation first; decide on coupling based on results. |
| Loss weight default for new head | `duration_bin_next: 1.0` — tunable later. Regression head keeps its existing `duration_next: 1.5` when enabled. |
| Class weighting for underrepresented bins? | **Not in first pass.** Smallest class (`10+ hrs` ≈ 4.58%) is viable under plain CE. Revisit only if top-1 on long bins underperforms. |
| Scheduled sampling | Bin head participates in SS like any other categorical head — teacher-forcing input is an embedded bin id. |
| Raw regression head behavior | **Unchanged** — keep it, keep its Huber loss, keep its inverse-transform at inference. Just toggle via `target_variables` yaml list. |
| LLM surface | Top-K **bins with probabilities** per step. Under 5D tuples, top-K tuples include the bin as a 5th column. |
| Report bin-center MAE? | **Yes** — alongside top-1/top-3 accuracy; gives apples-to-apples vs. the regression baseline. |
| Loss vs metric for bin | Loss = sparse CE. Bin-center MAE is a **reporting metric only**, not part of the loss. |

## 5. Bin definitions and sentinels

### 5.1 Default bins (7 data-driven + 3 sentinel)

| Bin | Range | Label | Approx share* |
|---|---|---|---:|
| 0 | `[0, 0.25]` | `≤0.25 hr` | 30.1% |
| 1 | `(0.25, 0.5]` | `0.25-0.5 hr` | 20.1% |
| 2 | `(0.5, 1.0]` | `0.5-1 hr` | 8.1% |
| 3 | `(1.0, 2.0]` | `1-2 hrs` | 17.5% |
| 4 | `(2.0, 5.0]` | `2-5 hrs` | 14.6% |
| 5 | `(5.0, 10.0]` | `5-10 hrs` | 5.1% |
| 6 | `(10.0, ∞)` | `10+ hrs` | 4.6% |

\* From Power BI distribution screenshot (2026-04-23, 137,131 rows).

Plus three sentinel classes (zero training share, reserved for special rows):

| Bin | Meaning | Applied when |
|---|---|---|
| `EOO` | End of operations | Target window extends past well's last real row (matches the existing EOO sentinel on phase/phase_step/MOC/op) |
| `Unplanned` | NPT / sidetrack operations | Row's Operation is in `preprocessing.unplanned_operations` |
| `UNK` | Unknown / fallback | Row's Operation is UNK OR the bin assignment falls through (e.g. NaN duration) |

So the bin encoder fits on **10 classes** under the default configuration. Use the same
`LabelEncoder` + sentinel-append pattern as `fit_target_encoders` in
[preprocessing/features.py:112](../preprocessing/features.py#L112).

### 5.2 Why sentinels are their own bins

You'd expect `EOO` to have duration 0, which would land in `≤0.25 hr` — so why not let
it? Two reasons:

1. **Hard coupling with the hierarchy's EOO sentinel.** When the 4-level hierarchy
   tuple predicts `(EOO, EOO, EOO, EOO)`, the bin head should predict `EOO` too.
   Having it as its own class gives the model a clean signal to learn. Forcing it
   into `≤0.25 hr` is a soft coupling that competes with legitimate 15-min prep ops.
2. **Unmasked vs. masked.** EOO rows are NOT masked (they're real emitted labels for
   beyond-well-end predictions). `Unplanned` and `UNK` rows ARE masked via
   `dur_label_real=0`. Making all three sentinels gives the bin head a consistent
   vocabulary regardless of masking status — you don't have to remember "EOO lives in
   bin 0 but UNK lives where?"

Cost of adding the three extra classes: 3 × `dec_target_edims["duration_bin_next"]`
extra embedding parameters + 3 output logits. Negligible.

### 5.3 Alternate binning (documented fallback)

If the long bins (`5-10`, `10+`) show weak top-1 accuracy, consolidate the tail:

```yaml
duration_bins:
  edges:  [0.25, 0.5, 1.0, 2.0, 4.0, 6.0]
  labels: ["≤0.25 hr", "0.25-0.5 hr", "0.5-1 hr", "1-2 hrs", "2-4 hrs", "4-6 hrs", "6+ hrs"]
```

This merges everything above 6h into one class (approx 9.6% of rows) — gives the
classifier more signal per long-bin class but loses LLM-side resolution. Switch by
editing `pipeline.yaml` only; no code change. Treat as an experiment to try **after**
the first training run with default bins, not something to do preemptively.

### 5.4 Bin-center computation

At preprocessing time, for each non-sentinel bin, compute the **median raw duration of
training rows** falling in that bin:

```python
centers = {}
for bin_id, (lo, hi) in enumerate(zip([-inf] + edges, edges + [inf])):
    mask = (df_train["Duration hours"] > lo) & (df_train["Duration hours"] <= hi)
    centers[bin_id] = float(df_train.loc[mask, "Duration hours"].median())
```

For sentinel bins: `centers["EOO"] = 0.0`, `centers["Unplanned"] = nan`,
`centers["UNK"] = nan`. (NaN means "don't contribute to bin-center MAE" — skipped at
metric time.)

Persist in `encoders.pkl` under `bin_centers: dict[str, float]`. Used by
`evaluation/run_evaluation.py` and `inference/predict.py` alike — identical numbers
train-to-eval-to-inference.

**Note**: bins are computed from the **raw** `Duration hours` column, *before* the
`log1p` step in [preprocessing/features.py:54-56](../preprocessing/features.py#L54-L56).
Easiest path is to add `duration_bin` assignment BEFORE `engineer()` runs log1p, in
`clean.py` or a new `preprocessing/bins.py` module (see §7.1).

## 6. Configuration (yaml additions)

```yaml
preprocessing:
  # ... existing ...
  duration_bins:
    enabled: true                                      # turns the bin COLUMN on at preprocessing time
    edges:  [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]           # 6 edges -> 7 bins
    labels: ["≤0.25 hr", "0.25-0.5 hr", "0.5-1 hr",
             "1-2 hrs", "2-5 hrs", "5-10 hrs", "10+ hrs"]
    # Sentinels EOO / Unplanned / UNK auto-appended; total classes = len(labels) + 3.
    # Bin centers = empirical median per bin, computed at fit time, saved in encoders.pkl.

training:
  target_variables:
    - phase_next
    - phase_step_next
    - major_ops_code_next
    - operation_next
    - duration_next                                    # EXISTING raw regression head (toggleable)
    - duration_bin_next                                # NEW classification head (toggleable)
  dec_target_edims:
    phase_next:          4
    phase_step_next:     8
    major_ops_code_next: 16
    operation_next:      32
    duration_bin_next:   8                             # NEW; 10 classes fits in 8 dims
  loss_weights:
    phase_next:          0.2
    phase_step_next:     0.3
    major_ops_code_next: 1.0
    operation_next:      2.0
    duration_next:       1.5                           # regression head (if enabled)
    duration_bin_next:   1.0                           # NEW; tunable

inference:
  enforce_hierarchy: true
  top_k_tuples: 3
  include_duration_bins_in_hierarchy: false            # NEW; if true, legal tuples go 5D
```

**Toggle semantics**:
- Turn the bin head **off** → remove `duration_bin_next` from `target_variables`. `preprocessing.duration_bins.enabled` can stay true (produces the column but nothing trains on it) — convenient for exploration.
- Turn the regression head **off** → remove `duration_next` from `target_variables`. Duration hours in the inference contract falls back to bin-center of top-1 bin.
- Turn **both** off → no duration prediction. LLM gets no duration field.
- Turn **5D constraints** on → requires both `duration_bin_next` in `target_variables` AND `include_duration_bins_in_hierarchy: true`. Fails loudly if only one is set.

## 7. Files that will change

| File | Purpose of change |
|---|---|
| `config/pipeline.yaml` | Add `preprocessing.duration_bins`, `duration_bin_next` entries in `target_variables` / `dec_target_edims` / `loss_weights`, `inference.include_duration_bins_in_hierarchy`. |
| `preprocessing/bins.py` *(new)* | `assign_duration_bin(raw_hours, edges) -> str` (+sentinel handling), `fit_bin_encoder(df_train, labels, sentinels)`, `compute_bin_centers(df_train, edges)`. Pure functions, no I/O. |
| `preprocessing/clean.py` | After the unplanned-merge step, assign `duration_bin` column from raw `Duration hours`. Sentinel logic: if Operation in unplanned list → `"Unplanned"`; if NaN residual → `"UNK"`; else bin by edges. Mask flag `dur_label_real` continues to govern whether the bin loss applies to a row (mask = 0 → contributes no bin CE). |
| `preprocessing/features.py` | In `fit_target_encoders`, optionally fit the bin encoder too (or keep it separate for clarity — recommend separate `fit_bin_encoder` in `preprocessing/bins.py`). Add `duration_bin` to the set of encoded-string columns whose `_target_enc` column must land in each split. |
| `preprocessing/run_preprocessing.py` | Wire in bin column creation before split; fit bin encoder; compute bin centers from df_train; materialize `duration_bin_target_enc` on every split DataFrame (same pattern as the existing 4 hierarchy levels); plumb `bin_centers` through `save_strategy`. |
| `preprocessing/save.py` | Persist `bin_labels`, `bin_edges`, `bin_centers`, and the `duration_bin` LabelEncoder in `encoders.pkl`. Extend `config.json` with `duration_bin_n_classes` if useful. |
| `training/data.py` | `HIERARCHY` is unchanged (stays 4 levels). Extend the `SAMPLE_WEIGHT_MAP` so `dur_label_real` also maps to `duration_bin` head. Extend `build_seq2seq_sequences` to emit `duration_bin_target_enc` windowed with EOO padding (use the bin encoder's `"EOO"` class id). Add a helper `duration_bin_eoo_id(bin_encoder)`. Keep `y_dur` for the regression head. |
| `training/model.py` | If `"duration_bin_next"` in `target_variables`: treat it as another categorical head in the per-head loop — embedding in decoder inputs, Dense softmax head, CE loss, participates in teacher forcing and scheduled sampling identically to the others. No Lambda / Reshape tricks. The regression head (under `predict_duration`) continues to work exactly as today. |
| `training/train.py` | `prep_targets` and `prep_model_inputs` already handle any set of categorical targets cleanly — bin head flows through naturally. `_expand_sample_weight` needs to fill ones for the bin head if no explicit weight arrives. |
| `training/evaluate.py` | `autoregressive_predict` treats the bin as another categorical head — same top-K extraction, same feedback. Joint-argmax path (constraints) unaffected unless `include_duration_bins_in_hierarchy=true` (see §8). |
| `training/constraints.py` | Extend to support 5D tuples behind a flag. `build_legal_tuples(..., include_bins=False)` defaults unchanged; with `include_bins=True`, stack `duration_bin_target_enc` as the 5th column and exclude bin sentinels from L. `joint_argmax` / `joint_topk_tuples` generalize naturally — the score is a sum of 5 log-probs instead of 4. |
| `evaluation/run_evaluation.py` | Add `duration_bin` to `active_targets` handling. Per-step top-1/top-3 like other heads. Compute bin-center MAE in hours using `bin_centers` from the bundle. Add `summary.json` keys: `ar.duration_bin_top1`, `ar.duration_bin_top3`, `ar.duration_bin_center_mae_hours`, and `ar.duration_bin_per_step_*` arrays. |
| `evaluation/metrics.py` | New helper `bin_center_mae(pred_bin_ids, true_hours, bin_centers, sample_weight)` returning a scalar. Skip rows where predicted bin's center is NaN (sentinels) or where sample_weight == 0. |
| `evaluation/artifacts.py` | `write_predictions_csv` gains columns: `true_duration_bin`, `pred_duration_bin_top1`, `pred_duration_bin_top3`, `duration_bin_top1_center_hours`, `duration_bin_abs_err_hours`. Keep the existing `true_duration_hours` / `pred_duration_hours` columns if the regression head is on. |
| `evaluation/eval_report.ipynb` | Add a "Duration bin" section with: (a) per-step top-1/top-3 curves; (b) confusion matrix on bins; (c) bin-center MAE vs. regression MAE comparison chart; (d) for each bin, the within-bin distribution of TRUE hours so the reader can see bin granularity. See §9 for details. |
| `inference/contract.py` | Add `duration_bin_topk: List[DurationBin]` to `StepPrediction` — each `DurationBin` has `{label: str, prob: float, center_hours: float}`. Keep `duration_hours: float` on the dataclass (populated from the regression head if enabled, else from top-1 bin's center). |
| `inference/predict.py` | If bin head is in `active_targets`, run it through the per-head top-K extraction (constraint-aware or not, depending on `include_duration_bins_in_hierarchy`). Populate `duration_bin_topk` on each StepPrediction. Feed bin id back as decoder input for next step. |
| `inference/load.py` | Hand `bin_centers` and `bin_labels` through from `encoders.pkl` into the bundle dict. |
| `llm/context.py` | `context_ml_predictions` renders a "Duration bin top-K" column per step when enabled. Each row shows `"1-2 hrs (0.47)"` style candidates. Point value for the LLM's final plan still must sum to 24h — the prompt should ask the LLM to pick a point within the top-1 bin's range (or weighted average across top-K if it's genuinely torn). |
| `llm/prompts/system_prompt.md` | Add a paragraph to Context 4 explaining the duration bin: "Each step also has a top-K duration-bin prediction with probabilities. Treat the top-1 bin as your range; pick a specific duration inside it based on Context 2's similar wells. Bins are labeled inclusive on the right." |
| `ui/components/predicted_panel.py` | `_render_ml_debug` gets a new "Dur bin top-K" column per step when bin head is enabled. |
| `misc/build_full_dataset.py` | Adds the `duration_bin` column to the snapshot so analysts can slice by bin in Power BI. Expose raw-hours + bin-center-hours + bin label side by side. |
| `misc/exploration.md` | Add a short section documenting the bin taxonomy and sentinel handling for analysts. |
| `tests/test_duration_bins.py` *(new)* | Unit tests — see §11. |

## 8. Legal-tuple 5D variant (optional, flag-gated)

When `inference.include_duration_bins_in_hierarchy: true`:

### What changes

- `build_legal_tuples` stacks `duration_bin_target_enc` as the 5th column. Exclude bin
  sentinels on the 5th axis (`EOO`, `Unplanned`, `UNK`), same pattern as the existing
  op-axis exclusions. Final L shape: `(num_legal, 5)` int32.
- `joint_argmax` / `joint_topk_tuples` sum 5 log-probs instead of 4. Straightforward
  generalization.
- `autoregressive_predict` with constraints feeds the winning tuple's bin id back as
  the bin decoder input for step t+1 (coordinated with the other 4 head feedbacks).
- `HierarchyTuple` in `inference/contract.py` gains a `duration_bin: str` field when
  the flag is on. Top-K tuples rendered in the LLM prompt include the bin column.

### What stays the same

- When the flag is off, bin head is predicted independently (top-K via per-head
  argsort, not joint argmax).
- The 4D constraints and tuple shape are unchanged under the flag-off default.
- `build_legal_tuples` signature accepts `include_bins: bool = False`; `False` path
  continues to work identically for callers that haven't opted in.

### Expected L size

Under 4D, |L| ≈ 1.3K (observed). Under 5D, rough upper bound is 1.3K × 7 = 9.1K, but
in practice same (phase, step, MOC, op) typically appears with 2–4 distinct bins in
training data, so the realistic |L_5D| ≈ 3–5K. Still negligible vs. the unconstrained
product `10 × 16 × 28 × 65 × 10 ≈ 2.9M`.

### Why default off

- Bin accuracy in isolation tells us whether the classifier is useful at all. If bin
  top-1 is e.g. 0.35, coupling it into L will drag the other heads' accuracy down.
- Coupling is additive work; measure first, commit second.
- The flag costs nothing — `include_duration_bins_in_hierarchy: true` is a single-line
  change to flip once we're satisfied with isolated bin numbers.

## 9. Evaluation changes

The eval stage must report **both** the classification metrics (top-1, top-3) and a
cross-compatible regression metric (bin-center MAE). Add these to `summary.json::ar`:

```jsonc
{
  "ar": {
    // ... existing ...
    "duration_bin_top1":          0.51,
    "duration_bin_top3":          0.84,
    "duration_bin_per_step_top1": [0.62, 0.55, 0.50, ...],   // length n_future
    "duration_bin_per_step_top3": [0.89, 0.86, 0.84, ...],
    "duration_bin_center_mae_hours":     1.72,                // derived from top-1 bin center
    "duration_bin_center_mae_long_hours": 2.34                // same but for true hours > 8
  }
}
```

Keep the existing `mae_hours`, `medae_hours`, `p95_abs_err_hours`,
`mae_long_ops_hours`, `r2_long_ops` when the regression head is also enabled — so the
same eval report carries both sets of numbers.

### Per-step accuracy.csv

Add rows for `head="duration_bin", mode="tf"` and `mode="ar"` — same schema as the
existing 4 hierarchy heads.

### Confusion pairs

Emit `confusion_duration_bin.csv` with the same tidy format as the other heads.
Mask Unplanned / UNK positions via `dur_label_real` so the confusion table doesn't
fill with trivial misses.

### `predictions.csv`

Add columns:

| column | content |
|---|---|
| `true_duration_bin` | label string |
| `pred_duration_bin_top1` | predicted label |
| `pred_duration_bin_top3` | `\|`-joined top-3 labels |
| `duration_bin_in_top3` | bool |
| `duration_bin_top1_center_hours` | bin center for the top-1 bin |
| `duration_bin_abs_err_hours` | `|top1_center − true_duration_hours|` |

Keep the existing `true_duration_hours` / `pred_duration_hours` / `duration_abs_err`
columns populated by the regression head when enabled.

### `evaluation/eval_report.ipynb`

Add a new section 11 (before the current "What to look at first"):

- **11a — per-step top-1 / top-3 accuracy for duration_bin** — same shape as existing per-step charts.
- **11b — confusion matrix on bins** — reuse `plot_confusion` helper from section 8; order class labels by raw bin order (not alphabetical).
- **11c — bin-center MAE vs. regression MAE** — a 2-bar chart; if both heads ran, both bars present. If only one ran, show the one number. Annotate the structural-floor note ("bin-center MAE has a floor from bin width — this metric is a sanity check, not a head-to-head on precision").
- **11d — within-bin distribution of TRUE durations** — 7 histograms, one per bin, showing true-hours distribution for rows that landed in that bin. Lets the reader see the bin granularity empirically and spot if a bin is unexpectedly wide.
- **11e — per-well bin accuracy scatter** — optional; same logic as `per_well_accuracy` but for the bin head only, to catch wells where binning breaks down.

## 10. Exploration changes

`misc/build_full_dataset.py` should export the bin label alongside raw hours so Power
BI users can filter by bin without recomputing:

```python
# in build(), after clean() and before engineer():
df["duration_bin"] = df["Duration hours"].apply(
    lambda h: assign_duration_bin(h, cfg["preprocessing"]["duration_bins"]["edges"],
                                    cfg["preprocessing"]["duration_bins"]["labels"])
)
```

Update `misc/exploration.md` with a short "Duration bins" subsection describing:

- The 7 default bins + 3 sentinels
- Bin label convention (right-inclusive)
- Where bin centers live (in `encoders.pkl`, not in the exploration CSV)
- How to back-transform if needed (`np.expm1` on the parquet-scaled column, bins on
  the raw column)

## 11. Verification plan

### 11.1 Unit tests — `tests/test_duration_bins.py`

1. **`assign_duration_bin`**:
   - 0.1 → `"≤0.25 hr"`
   - 0.25 → `"≤0.25 hr"` (right-inclusive)
   - 0.251 → `"0.25-0.5 hr"`
   - 10.0 → `"5-10 hrs"`
   - 10.001 → `"10+ hrs"`
   - NaN → `"UNK"`
2. **`fit_bin_encoder`**: vocab = labels + `["EOO", "Unplanned", "UNK"]` in that order. `n_classes == len(labels) + 3`.
3. **`compute_bin_centers`**: on a fixture of known durations, median per bin matches by hand.
4. **5D legal tuples** (extension of existing `test_constraints.py`): `build_legal_tuples(..., include_bins=True)` produces `(N, 5)` array; bin sentinels excluded; joint_topk_tuples accepts and ranks 5D tuples.
5. **Sentinel mask sanity**: rows with `dur_label_real == 0` contribute no CE on the bin head — verify by building a tiny dataset and checking that the sample_weight dict passed to `fit()` zeros those positions on the `duration_bin` key.

### 11.2 Integration checks

1. `python -m preprocessing.run_preprocessing` — new parquets contain `duration_bin_target_enc` column on every split; `encoders.pkl` has `bin_labels`, `bin_edges`, `bin_centers`, and a `duration_bin` LabelEncoder.
2. Training notebook runs to completion with `duration_bin_next` in `target_variables`. Loss curves show a separate `duration_bin_loss` trending down.
3. `python -m evaluation.run_evaluation` — `summary.json` has the new `duration_bin_*` keys; `predictions.csv` has the new columns; `eval_report.ipynb` renders all four new sub-sections.
4. Streamlit UI renders the new "Dur bin top-K" column when the head is enabled, gracefully hides it otherwise.
5. With `include_duration_bins_in_hierarchy: true` — `summary.json` shows |L| increased; `tuple_topK` tuples have 5 columns; LLM prompt includes a duration-bin column in the tuple table.

### 11.3 Result comparisons to expect

- Bin top-1 per-step curve should decay with step (like the other heads), probably
  hitting 0.55–0.75 at step 1 and 0.30–0.50 at step 8.
- Bin-center MAE should land in the 1.5–3.0 hr range — better than 1.56 overall
  means binning plus point-picking beat the regressor; worse means binning sacrificed
  short-op precision. Long-ops bin-center MAE (on rows with true hours > 8) is the
  real headline — should be materially below the 11.19h from the current regression.
- If the LLM's final plan shows "pick a value inside the predicted bin" behavior in a
  few smoke-test wells, the contract is working end-to-end.

## 12. Edge cases

1. **Raw duration = 0.** Lands in `≤0.25 hr` bin naturally. EOO padding emits
   `"EOO"` regardless, not `≤0.25 hr`.
2. **Raw duration > 10.** Lands in `10+ hrs`. Bin-center for this bin might be ~12h
   (training-data median); the MAE there will be wider by construction.
3. **Unplanned op with long duration.** `dur_label_real=0` masks it from loss; the
   parquet row has `duration_bin = "Unplanned"` regardless of raw hours. Doesn't
   contaminate legal-tuple construction either (bin sentinels are explicitly excluded
   from L).
4. **NaN residual duration** (after conditional-mode fill). `assign_duration_bin`
   returns `"UNK"`; `dur_label_real=0` masks it from loss.
5. **Bin boundaries and floating point.** `pd.cut` with `include_lowest=True,
   right=True` matches the DAX semantics (`<= 0.25` goes to bin 0). Enforce this in
   `assign_duration_bin` tests.
6. **Alternate bin edges.** If the user swaps edges in yaml, the parquet gets
   regenerated, encoders get refitted, bin centers get recomputed. All downstream
   (LLM, UI, eval) reads labels from the bundle, not from hard-coded strings —
   verify no hard-coded labels leak anywhere. Grep for `"≤0.25"`, `"10+"`, etc.
7. **Model bundle staleness.** If a bundle was trained with a different bin edges
   set than the current yaml, inference should refuse to load with a clear error
   rather than silently mismatch. Store `bin_edges` in `model_config.json` and
   assert agreement with `encoders.pkl` at load time.
8. **Top-K bins from independent softmax when `include_duration_bins_in_hierarchy
   = false`.** These may include sentinels (`EOO`, `Unplanned`, `UNK`) at low
   probability. That's fine for the LLM debug table but the LLM prompt text should
   tell the model to ignore sentinels when picking a value — they're not durations.

## 13. Not in scope

- **Quantile bins / learned bin edges.** Could be nice but adds complexity; default
  to hand-picked edges informed by the data distribution.
- **Mixture density duration head.** A Gaussian mixture over hours would preserve
  real-valued precision and express uncertainty, but it's a bigger architectural
  change. Try binning first; revisit if binning is promising but the bin-center MAE
  ceiling is too high.
- **Replacing `include_duration_bins_in_hierarchy` with a per-tuple learned
  distribution.** I.e. for each legal 4-tuple, learn a distribution over bins.
  Reasonable, but requires more infrastructure than a flat 5D L.
- **Removing the raw regression head.** Keep it for now. Separate PR once bin head
  is validated as better.

## 14. Open items

- **Loss weight of 1.0 for `duration_bin_next`** is a first-pass guess. Tune based
  on first training run. If MOC/op accuracy degrades when bin is added, the bin
  head may need a lower weight; if bin top-1 stagnates, raise it.
- **Whether 5D constraints help or hurt.** Pure empirical question; answer after
  running the eval with the flag on vs. off.
- **Whether to surface bin-center MAE at step level** (per-step chart in the eval
  notebook) or only overall. Probably yes — matches how per-step accuracy is
  surfaced for the categorical heads. Easy add once the overall number exists.

## 15. Context / discussion history (for future-me and future-Claude)

This design emerged during a 2026-04-23 session after the constraint decoder
(see `improvements/Constraint Decoder.md`) shipped. Key points from the conversation
that shaped decisions — preserved here so a future session picks up without
re-relitigating:

- **Regression performance was the trigger.** `r2_long_ops = -7.21` on the T5 eval
  was the concrete red flag. Regression head was actively anti-predictive on the
  long tail.
- **User proposed the DAX binning**: 7 bins with edges `[0.25, 0.5, 1, 2, 5, 10]`.
  First-pass labels had a minor bug (the `≤0.25` branch was labeled `"0.25-0.5 hr"`),
  corrected in §5.1.
- **User flagged EOO as the critical sentinel**: "we dont wanna predict anything
  other than 0" for EOO steps. Led to §5.2 — sentinel-as-own-class for EOO
  (Unplanned/UNK similarly for consistency, though masking already handles them).
- **`include_duration_bins_in_hierarchy` was introduced as a config flag** rather
  than a hard decision because user wanted to measure bin accuracy in isolation
  first — "i would actually wanna see how the bins accuracy work before i add this
  to the legal tuple constraint."
- **Top-K bins (not top-1)** surfaced to the LLM — "top-k is good." Mirrors how
  the 4 hierarchy heads already surface top-K.
- **Binning edges configurable via yaml** — user asked for this explicitly so the
  alternate `[0.25, 0.5, 1, 2, 4, 6]` variant in §5.3 can be tried without code
  edits.
- **Bin-center MAE is a reporting metric, not a loss.** Clarified during the
  session: cross-entropy is the loss; bin-center MAE is a convenience number for
  diffing against the regression baseline. An alternative "expected-value MAE" as
  an auxiliary loss was considered and rejected — it biases the head toward
  hedging between neighboring bins even when the truth is genuinely bimodal.
- **Raw regression head is NOT being removed.** User was explicit: "lets not
  eliminate raw duration hours regression just yet. we can always turn it off
  anyways." Both heads coexist, controlled by `target_variables` in yaml.
- **First implementation target**: default bins, default sentinels, 4D constraints,
  top-K bins to LLM. Measure. Then decide on 5D constraints and/or alternate bins.

### What this doc is NOT

- Not a replacement for the actual implementation PRs. It's the design contract.
- Not a commitment to every detail — if something in the code clashes with the
  doc while implementing, update the doc.
- Not a promise that binning will beat regression. It's an A/B experiment with
  a reasonable prior that it will help on the long tail.

# Drilling Seq2Seq — improvement opportunities (synopsis)

All figures below cite `analysis.ipynb` sections and the evaluation artifact
`results/seq2seq_N25_K8_T5_dummies/eval_20260418_152124/`.

## TL;DR

1. **12.4 % of all `operation` labels are NaN in `full_dataset.csv` and silently relabelled
   to the majority class `DRL`** by the target-encoding step. Train parquet's `DRL`
   count is 25,033 vs. ~10,832 expected — every extra row is a ghost. This single
   bug plausibly explains the mode-collapse toward `DRL` seen in confusion pairs. *(§1)*
2. **19 % of AR predictions are structurally illegal** (P → S → MOC → Op violates
   observed parent→child sets). Only **1,233 of 358,176** tuples ever occur —
   0.34 % coverage. A constrained decoder is "free" accuracy. *(§2)*
3. **The MOC head is the bottleneck**: `P(MOC correct | phase_step correct) = 0.455`.
   Operation-given-MOC is already 0.765, so anything that lifts MOC lifts operation.
   *(§2, §9)*
4. **Near-duplicate operations exist** — top JSD pairs include `CONC_BOPE↔TEST_BOPE`
   (0.059), `FIT↔LOT` (0.061), `LNR_W/OWASH↔LNR_WWASH` (0.067),
   `CSG_W/O ROTATION↔CSG_W/ROTATION` (0.081). Merging the top ~10 cuts the
   78-class operation vocab by ~13 % without semantic loss. *(§4)*
5. **Step-k irreducibility is ~1.1 bits of entropy growth from k=1 to k=8** while the
   model loses ~0.27 in top-1 accuracy — most of the 8-step degradation is *achievable*
   structure the model isn't using. *(§6)*
6. **Exposure bias is large**: TF-minus-AR for operation is 0.208, for MOC 0.206.
   Training stopped at epoch 20 of 500, so scheduled-sampling only reached ~0.38 of
   its target 0.75 — the very mechanism meant to close this gap was under-utilised.
   *(§9)*
7. **Per-well operation accuracy std = 0.233** (mean 0.422). Worst wells (e.g.
   TEINERT-SALLY 26F 206H at 0.11) are *not* leakage — splits are well-disjoint — but
   a long tail of low-accuracy wells is masked by the mean. *(§8)*
8. **Self-repetition is rare (10.7 % of transitions, P90 run-length = 1)**, so a
   copy-state head or "repeat previous token" bias buys little. Do not prioritise. *(§7)*

## Findings

### F1 — NaN-operation label corruption *(observation)*
- `misc/full_dataset.csv` has 19,090 rows with `operation = NaN` and 15 with
  `major_ops_code = NaN` *(§1)*.
- [preprocessing/run_preprocessing.py:110](../../preprocessing/run_preprocessing.py#L110)
  encodes targets with
  ```python
  lambda x: list(le.classes_).index(x) if x in classes else fb_idx
  ```
  where `fb_idx` is the mode of the TRAIN column. NaN is never in `classes_`, so
  every NaN is silently relabelled to the mode (**DRL** for `operation`,
  **NON_DRILL** for `major_ops_code`).
- Evidence: train-parquet's top operation has 25,033 rows; the raw-CSV DRL count
  scaled to 662 train wells is ~10,832. The surplus ≈ the number of NaN rows in
  training wells. *(§1 final cell)*
- **Why it matters**: 12 % of training targets are systematic noise biased toward
  the majority class — this is the textbook recipe for mode collapse, and it
  matches the observed confusion pattern (RIG_SVC → DRL 39 %, FLOW_CHK → DRL 25 %,
  RIG → DRILL 30 %).

### F2 — Tuple legality is drastically sparser than the cross-product *(observation)*
- Cardinalities: 8 × 14 × 41 × 78 = 358,176 theoretical tuples, **1,233 actual
  (0.34 %)** *(§2)*.
- Mean legal children per parent: phase_step | phase = 4.9; MOC | phase_step = 12.9;
  operation | MOC = 4.6 *(§2)*.
- `evaluation/metrics.hierarchy_valid_mask` already reports **0.810 AR validity**
  on the test set.
- **Why it matters**: ~19 % of predictions lose accuracy simply because they're
  structurally impossible. A decoder that masks illegal children recovers a large
  share at no training cost.

### F3 — MOC is the accuracy bottleneck *(observation)*
- Conditional accuracy (AR, from `summary.json`):
  `phase_step | phase` = 0.834, **`MOC | phase_step` = 0.455**,
  `operation | MOC` = 0.765 *(§2, §9)*.
- Top MOC confusions: RIG→DRILL (30 %), TEST→DRILL (26 %), TRIP↔NON_DRILL,
  SAFETY→CASING/NON_DRILL *(§9)*.
- **Why it matters**: Operation learning is already decent given a correct MOC.
  Any improvement concentrated on MOC (constrained decoding, MOC-specific head
  capacity, curriculum) compounds.

### F4 — Near-duplicate operations *(hypothesis — validate with SME)*
- Top-20 JSD pairs sharing a `(phase_step, MOC)` bucket surface several
  semantic merges that shouldn't hurt a drilling SME's ability to interpret
  outputs *(§4)*. High-confidence candidates:
  - `CONC_BOPE ≈ TEST_BOPE` (JSD 0.059, counts 915/1063)
  - `FIT ≈ LOT` (0.061, 1187/49) — formation-integrity vs. leak-off test
  - `LNR_W/OWASH ≈ LNR_WWASH` (0.067, 2446/129) — wash-vs-no-wash variant
  - `CSG_W/O ROTATION ≈ CSG_W/ROTATION` (0.081) — rotation modifier
  - `LD_BHA ≈ LD_DIR` (0.098) — lay-down variants
  - `TIH_ELEV ≈ TIH_NONELEV`, `TOOH_ELEV ≈ TOOH_NONELEV` — elevator trip variants
- Name-prefix families with ≥3 variants: `RIG`, `PU`, `DRL`, `HDL`, `LD`, `RU`,
  `CSG`, `RD` *(§4)*.
- **Why it matters**: merging the top 10 candidates cuts the 78-class op vocab
  by ~13 %, roughly doubles samples/class for the merged entries, and lifts the
  floor on rare-class accuracy.

### F5 — Long tail of rare operations *(observation)*
- Operations: 36 of 78 classes have <500 examples (3.6 % of rows); 12 classes
  have <50; 5 have <10 *(§3)*.
- MOC: 15 of 41 classes have <100 examples (~0.1 % of rows) *(§3)*.
- 31 of 78 operations cover 90 % of rows *(§3)*.
- **Why it matters**: with no label smoothing and cross-entropy on a hot target,
  rare classes get almost-zero gradient signal but the model still has to reserve
  output capacity for them. Either merge (F4), bucket into OTHER, or apply focal
  loss.

### F6 — Hierarchical determinism collapses at MOC/operation *(observation)*
- Empirical transition entropy (bits) *(§5)*:

  | level          | H(next) | H(next \| curr) | I(curr; next) |
  |----------------|--------:|----------------:|--------------:|
  | phase          |    1.75 |            0.15 |          1.61 |
  | phase_step     |    2.89 |            0.49 |          2.40 |
  | major_ops_code |    3.71 |            2.76 |          0.95 |
  | operation      |    4.85 |            3.16 |          1.70 |

- **Why it matters**: phase/phase_step are near-deterministic given the previous
  token (explains ~95 %/80 % AR accuracy). MOC and operation carry much more
  residual uncertainty — this is *where* deeper context, better features, or
  a constrained decoder matters most.

### F7 — Step-k difficulty grows less than the model's accuracy drop *(observation)*
- Irreducible H(op_{t+k} | op_t) grows from **3.16 → 4.27 bits** over k=1..8 (1.1
  bits), while AR top-1 drops **0.577 → 0.307** *(§6)*.
- **Why it matters**: the 27-point accuracy loss across the horizon is only
  partly explained by genuine horizon uncertainty. A lot is exposure-bias-driven
  error compounding (see F8).

### F8 — Exposure bias / under-trained scheduled sampling *(observation)*
- TF-minus-AR top-1: phase 0.03, phase_step 0.13, MOC 0.21, operation 0.21
  *(§9 final table)*.
- Training halted at epoch 20 of 500 (early-stopping patience = 10); scheduled
  sampling ramp target 0.75 only reached ~0.38 *(from `training/train_seq2seq.ipynb`)*.
- **Why it matters**: the single biggest controllable lever between TF and AR is
  already in the pipeline but wasn't applied long enough to bite.

### F9 — Per-well accuracy has a long low-end tail, not leakage *(observation)*
- Splits are well-disjoint: train/val, train/test, val/test overlap = 0 *(§8)*.
- Mean per-well operation top-1 = 0.422, std 0.233; worst wells down to 0.11
  (TEINERT-SALLY 26F 206H with 30 sequences) *(§8)*.
- Worst-well list is a mix of short wells (5–30 sequences) and long unusual wells
  — suggests the model generalises badly to well-level "dialects" (operator,
  rig, pad order) not captured by the current feature set.
- **Why it matters**: adding explicit well-metadata features or a small per-well
  adaptation (LoRA-style head / per-operator embedding) could attack this directly.

### F10 — Self-repetition is low; copy-head is not a high-leverage fix *(observation)*
- Only **10.7 %** of within-well transitions are same-op; median run-length = 1;
  P90 = 1 *(§7)*.
- **Why it matters**: the sequence is genuinely one-step-at-a-time. Architectures
  that lean on copy/stay-in-state buy little. Deprioritise.

### F11 — Duration head is non-learning on long operations *(observation, from eval)*
- `mae_hours = 1.56`, **`mae_long_ops_hours = 11.19`, `r2_long_ops = -7.21`**
  (worse than predicting the mean) *(eval summary.json)*.
- **Why it matters**: the Huber loss + log1p scaling collapses predictions to
  the median. Long, important events (sidetracks, NPT) are invisible to the
  model's duration signal. Either drop the head, re-weight it toward long tail,
  or model as a classification over duration buckets.

## Recommendations (sorted by impact ÷ effort)

| # | Recommendation | Expected impact | Implementation effort | Risk |
|--:|----------------|-----------------|-----------------------|------|
| 1 | **Fix NaN-target fallback in [`preprocessing/run_preprocessing.py:110`](../../preprocessing/run_preprocessing.py#L110)** — either `dropna` those 19k rows, add an explicit `UNK` class and mask the loss, or back-fill from SME rules. Pair with a data-quality check in `clean.py`. | HIGH — addresses 12 % mislabelled training signal; expected to directly reduce DRL mode collapse (F1, confusion table) | LOW (data plumbing) | LOW — worst case we drop 12 % of rows and retrain cleanly |
| 2 | **Constrained hierarchical decoder** — at each step, mask logits to children legal under the currently-predicted parents (use sets from `evaluation/metrics.build_hierarchy_sets`). | HIGH — the 19 pp illegal-tuple rate is the ceiling; realistic lift 3-8 pp on operation top-1 | MEDIUM — ~50 lines in `training/model.py`, inference-time only | LOW — legality is data-derived, deterministic |
| 3 | **Train longer / relax early stopping** and ensure scheduled-sampling reaches its 0.75 target before stopping. | MEDIUM-HIGH — directly shrinks F8's 20 pp TF-AR gap | LOW | LOW — already implemented, just run more epochs |
| 4 | **Mask EOO padding in the loss** (`data.py` currently emits EOO at well end; loss treats them as real) | MEDIUM — reduces gradient bias toward EOO; helps rare classes | LOW (Keras `sample_weight`) | LOW |
| 5 | **Merge top-10 near-duplicate operations** (F4) and retrain. Also collapse rare classes (<50 examples) into a single `OTHER` bucket. | MEDIUM — ~13 % vocab reduction, better rare-class resolution | MEDIUM — affects `features.py`, encoders, evaluation labels; requires SME sign-off | LOW-MED (SME may reject some merges) |
| 6 | **Cascaded hierarchical decoder** — predict `phase → phase_step → MOC → operation` with each conditioned on the predicted ancestor (as opposed to today's independent heads). | HIGH — directly attacks the `MOC \| phase_step` = 0.455 bottleneck (F3) | HIGH — architectural change in `training/model.py` | MED — design complexity, interacts with scheduled sampling |
| 7 | **Label smoothing (0.1) + class-frequency-weighted or focal loss on operation/MOC heads**. | MEDIUM | LOW | LOW |
| 8 | **Per-operator/per-rig embedding or well-metadata features** to reduce the 0.233 per-well std (F9). | MEDIUM | MEDIUM | LOW |
| 9 | **Duration head overhaul** — predict bucketed durations (e.g. <1h, 1-4h, 4-8h, 8-24h, >24h) or add explicit long-tail loss re-weighting (F11). | MEDIUM (improves an orthogonal output) | MEDIUM | LOW |
| 10 | Copy-state / repeat-previous bias head. | LOW (F10: only 10.7 % self-transitions) | MEDIUM | LOW — but poor return |

### Explicit architecture changes

Items **2, 6, 9** change the model architecture, not just data or training
hyper-parameters. Items **1, 3, 4, 5, 7, 8, 10** are data or loss/training-level.

## Open questions (need model/SME, not data-only)

- Does fixing F1 + item 4 alone close enough of the gap that item 6 (cascaded
  decoder) is unnecessary? Needs a re-train.
- For the worst wells in F9, is the issue covariate shift (novel operator/rig
  combo) or short-well truncation interacting with the 25-step encoder window?
  Needs per-well attribute probing on the trained model.
- Are the JSD-nearest merges in F4 semantically valid to a drilling engineer?
  The data agrees they are statistically interchangeable, but SME sign-off is
  needed before deploying a merged vocab.
- Would a longer encoder window (N=50 or 100 vs. 25) meaningfully reduce
  horizon-k entropy for operation? We can only measure the irreducible H(k)
  here; the "achievable with longer context" H needs a new training run.
- The top-5 confusion pairs account for ~40 % of operation errors and ~45 % of
  MOC errors (§9). Are these the *same* confusions as the NaN→DRL mis-relabels
  (F1)? If so, fixing F1 might remove a large chunk without any architectural
  changes — worth checking by recomputing confusion after the fix.

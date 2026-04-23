# Constraint Decoder — hierarchy-consistent predictions via legal-tuple joint argmax

## Problem

The current seq2seq model emits four independent softmaxes per decoder step
(phase → phase_step → major_ops_code → operation). Nothing couples them, so the
model can — and does — emit illegal combinations, e.g. `(DRL, RIH, CEMENT, CIRC)`
where `RIH` is not a legal step under phase `DRL`, and `CEMENT` is not a legal MOC
under step `RIH`. The evaluation artifact `summary.json::hierarchy_validity_rate`
quantifies how often the four heads produce a legal tuple; the remaining fraction
is delivered to the LLM stage as-is today.

## Goal

At inference time, guarantee every predicted step forms a legal
`(phase, phase_step, major_ops_code, operation)` tuple, while preserving as much
of the model's per-head predictive signal as possible. No changes to training.

## Core idea — joint argmax over legal tuples

Under independent per-head softmaxes, the log joint probability of any tuple
`(p, s, m, o)` is the sum of four log-probabilities:

```
log P(p, s, m, o)  =  log P(p) + log P(s) + log P(m) + log P(o)
```

Instead of `argmax` per head, we:
1. Precompute the set **L** of all legal tuples (offline, once).
2. At each decoder step, sum the four log-probs at each legal tuple's indices.
3. Pick the legal tuple with the highest joint score.

The chosen tuple's class ids **also feed back as the decoder inputs for step
t+1** — so autoregressive history stays self-consistent throughout the horizon.

This lets "lower levels vote on which higher level was really right": if phase
argmax mildly prefers `DRL` but step logits strongly prefer `RIH` (which only
exists under `CASING`), the joint can still pick the `CASING`-rooted tuple.

## Decisions locked in

| Decision | Chosen |
|---|---|
| Legal-tuple source | Post-cleaned `df_train ∪ df_val ∪ df_test` parquet string columns (NOT raw master CSV; NOT df_train-only). Treated as **domain knowledge**, not a leakage surface. |
| Excluded classes | `UNK` and `Unplanned` — these are sentinels/masks the op head is trained never to emit. `End of Operations` is explicitly added as a legal tuple. |
| Apply to TF eval path? | **No.** TF metrics stay as the unconstrained upper bound. |
| Apply to AR eval path? | **Yes.** `evaluation/run_evaluation.py` AR predictions go through joint argmax. |
| Apply to production inference? | **Yes.** `inference/predict.py` emits hierarchy-valid `MLOutput`. |
| Top-K contract | **Top-K full tuples per step** (replacing top-K per head). Before: `3×3×3×3 = 81` cartesian combinations; after: `K` curated consistent tuples with their joint log-prob. |

## Files that will change

| File | What changes |
|---|---|
| `training/constraints.py` *(new)* | `build_legal_tuples(...)`, `joint_argmax(...)`, `joint_topk_tuples(...)` — pure numpy, no TF dep. |
| `training/evaluate.py` | `autoregressive_predict` accepts optional `legal_tuples: np.ndarray \| None`; when provided, replaces independent argmax + top-K extraction with joint argmax + top-K tuples. Feeds winning tuple's classes as next-step decoder inputs. |
| `evaluation/run_evaluation.py` | Build `L` from parquets; pass into `autoregressive_predict`. Adjust AR scoring to read top-K tuples instead of top-K per head. `hierarchy_validity_rate` will be `1.0` by construction under constraints; report it alongside the pre-constraint value from the TF path for comparison. |
| `evaluation/artifacts.py` | `write_predictions_csv` contract changes: columns become `topk_tuple_{0..K-1}_{phase,phase_step,major_ops_code,operation}` + `topk_tuple_{0..K-1}_logprob`. Keep `target_*` columns unchanged for side-by-side comparison. |
| `inference/contract.py` | `StepPrediction` replaces per-head top-K lists with a single `topk_tuples: list[HierarchyTuple]` where `HierarchyTuple = {phase, phase_step, major_ops_code, operation, log_prob}`. Keep `duration_hours` as a scalar. |
| `inference/predict.py` | After model inference, pass logits through `joint_topk_tuples`. Build the new `StepPrediction` shape. Legal-tuple table loaded from the model bundle at startup. |
| `llm/ml_to_llm.py` *(and any prompt templates)* | Consume top-K tuples instead of top-K per head. Simpler prompt: "here are K consistent next-step options" vs. "here are K per-level options that may not combine legally." |
| `training/save.py` *(or equivalent bundle writer)* | Pickle the legal-tuple table alongside `encoders.pkl` so inference can load it without re-enumerating. Cache key: master-CSV hash or parquet-split hash, so a rerun detects staleness. |
| `config/pipeline.yaml` | Add `inference.top_k_tuples` (default 3) and `inference.enforce_hierarchy` (default true, flag to bypass for debugging). |

## Legal-tuple construction (offline, once)

Enumerate from the cleaned, split-union parquet data. Pseudocode:

```python
# training/constraints.py
import numpy as np
import pandas as pd

HIERARCHY = ["phase", "phase_step", "major_ops_code", "operation"]

def build_legal_tuples(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    target_encoders: dict,
    eoo_token: str = "End of Operations",
    exclude_ops: tuple = ("Unplanned", "UNK"),
) -> np.ndarray:
    """Return (num_legal, 4) int32 array of class-id tuples."""

    df = pd.concat([df_train, df_val, df_test], ignore_index=True)[HIERARCHY]
    df = df.dropna(how="any")                       # drops UNK-sourced rows (NaN in string col)
    df = df[~df["operation"].isin(exclude_ops)]     # drops Unplanned-sourced rows explicitly
    uniq = df.drop_duplicates().reset_index(drop=True)

    L = np.stack([
        target_encoders[col].transform(uniq[col].values)
        for col in HIERARCHY
    ], axis=1).astype(np.int32)                     # (num_legal, 4)

    # EOO tuple — all four levels at their EOO sentinel id
    eoo_ids = np.asarray([
        int(target_encoders[col].transform([eoo_token])[0]) for col in HIERARCHY
    ], dtype=np.int32)
    L = np.vstack([L, eoo_ids[None, :]])

    return L
```

**Expected size**: ~1.3K tuples (per user estimate). Compare to unconstrained
product `10 × 16 × 28 × 65 ≈ 291K`.

**Sanity checks to run at construction time** (log these):
- `|L|` total count
- Number of unique phases/steps/mocs/ops represented in L
- Count of tuples whose op class is `Unplanned` or `UNK` (must be 0)
- Whether EOO tuple is present

## Joint argmax mechanics

Per decoder step, with batch size `B`:

```python
# training/constraints.py
def joint_argmax(
    logits: dict,          # {head_name: (B, n_classes_for_head)} — softmax inputs
    L:      np.ndarray,    # (num_legal, 4) int32 — class ids per (phase, step, moc, op)
) -> np.ndarray:
    """Return (B, 4) int32 — chosen legal tuple per sample."""
    import scipy.special
    # Numerically stable log-softmax per head (avoid log(0)):
    logp_phase = scipy.special.log_softmax(logits["phase"],          axis=-1)
    logp_step  = scipy.special.log_softmax(logits["phase_step"],     axis=-1)
    logp_moc   = scipy.special.log_softmax(logits["major_ops_code"], axis=-1)
    logp_op    = scipy.special.log_softmax(logits["operation"],      axis=-1)

    # Fancy-indexed gather: pick each head's log-prob at that tuple's column
    scores = (
        logp_phase[:, L[:, 0]] +
        logp_step [:, L[:, 1]] +
        logp_moc  [:, L[:, 2]] +
        logp_op   [:, L[:, 3]]
    )                                      # (B, num_legal)
    best = scores.argmax(axis=-1)          # (B,)
    return L[best]                         # (B, 4)
```

**Per-step compute**: one gather of shape `(B, 1.3K)` + one argmax. Negligible
vs. the decoder forward pass.

**Numerical note**: if the model's final layer returns post-softmax probabilities
(not logits), use `np.log(probs + 1e-12)` instead of `log_softmax`. Check what
`step_out` from `decoder_step_model.predict` actually contains — the current
code at [training/evaluate.py:74](training/evaluate.py#L74) treats it as
probabilities (applies `argmax` and `np.argsort` directly), which is consistent
with the model having a softmax activation on the output layer.

## Top-K legal tuples

For `predictions.csv`, `MLOutput`, and LLM consumption:

```python
def joint_topk_tuples(
    logits: dict,
    L: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (B, k, 4) chosen tuples and (B, k) joint log-probs, sorted desc."""
    scores = _compute_scores(logits, L)              # (B, num_legal)
    topk_idx = np.argpartition(-scores, k - 1, axis=-1)[:, :k]
    # Sort within the top-k for stable ordering:
    topk_scores = np.take_along_axis(scores, topk_idx, axis=-1)
    order = np.argsort(-topk_scores, axis=-1)
    topk_idx    = np.take_along_axis(topk_idx,    order, axis=-1)  # (B, k)
    topk_scores = np.take_along_axis(topk_scores, order, axis=-1)  # (B, k)
    topk_tuples = L[topk_idx]                                       # (B, k, 4)
    return topk_tuples, topk_scores
```

Argmax pick = first element of top-K.

## Autoregressive feedback loop — the critical detail

At [training/evaluate.py:80](training/evaluate.py#L80), the current loop feeds
each head's independent argmax back as that head's next-step decoder input:

```python
prev_cat[t] = pred_cls.reshape(-1, 1)         # per head, independently
```

Under constraints, all four `prev_cat[head]` entries at step t+1 come from the
**same winning tuple** at step t. That's what prevents mid-sequence drift:

```python
chosen = joint_argmax(logits_at_step_t, L)    # (B, 4)
for i, h in enumerate(HIERARCHY):
    prev_cat[h] = chosen[:, i:i+1]            # (B, 1)
```

## Contract changes in detail

### `predictions.csv`

| Before | After |
|---|---|
| `pred_phase_top1`, `pred_phase_top2`, `pred_phase_top3`, `prob_phase_top1`, ... (and same for phase_step, moc, op — 8 × 3 × 4 = ~96 columns, many illegal combos) | `pred_tuple_0_phase`, `pred_tuple_0_phase_step`, `pred_tuple_0_major_ops_code`, `pred_tuple_0_operation`, `pred_tuple_0_logprob`, repeated for `_1` and `_2`. ~15 columns, all guaranteed legal. `target_*` columns unchanged. |

### `inference.contract.StepPrediction`

Current (sketch, verify before editing):
```python
class StepPrediction:
    phase: list[tuple[str, float]]           # [(label, prob), ...] top-K
    phase_step: list[tuple[str, float]]
    major_ops_code: list[tuple[str, float]]
    operation: list[tuple[str, float]]
    duration_hours: float
```

Proposed:
```python
class HierarchyTuple:
    phase: str
    phase_step: str
    major_ops_code: str
    operation: str
    log_prob: float                          # joint log-probability under the model

class StepPrediction:
    topk_tuples: list[HierarchyTuple]        # length K, sorted by log_prob desc
    duration_hours: float                    # unchanged
```

### LLM stage

The current prompt presumably shows the LLM top-K per head. Post-change, the
prompt shows top-K **consistent** tuples. Likely simpler prompt, since the LLM
no longer has to reason about hierarchy consistency — the ML stage guarantees
it. Needs a prompt update in `llm/prompts/system_prompt.md`.

## Edge cases

1. **Empty L collision.** If somehow `L` is empty (e.g., bundle ships without
   the table and fallback path fires), raise explicitly. No silent passthrough.

2. **Sentinel-containing tuples.** UNK and Unplanned won't appear in L by
   construction. The op head's softmax still has slots for them, but those
   logits are effectively unused during joint argmax — they get ignored because
   no tuple in L references those class ids.

3. **A well's true next step is a combination never seen in the union data.**
   Under the post-clean union, this should be nearly impossible — the legal set
   subsumes training, val, and test. But if it happens (future data drift), the
   joint argmax will pick the closest legal tuple by log-prob. Worth logging
   when the chosen tuple's joint log-prob is below some threshold so drift is
   detectable.

4. **EOO handling at well end.** The EOO tuple `(EOO, EOO, EOO, EOO)` is
   explicitly in L. When the model approaches the true end of a well and
   strongly predicts EOO on all four heads, the joint will pick it. Mid-well,
   the EOO tuple's joint log-prob is typically very low (model isn't predicting
   EOO on any head), so it won't spuriously win.

5. **Bundle staleness.** If the master CSV is updated (new operations added by
   SMEs) but the bundled legal-tuple table is stale, the model may not be able
   to predict the new combination even if its unconstrained argmax would. Mitigation:
   log the legal-tuple hash at inference startup and compare to what the parquets
   would produce; warn on mismatch. Or re-enumerate at startup from live parquets.

## Verification plan

1. **Unit tests for `training/constraints.py`**:
   - `build_legal_tuples` on a 5-tuple toy dataset returns the expected array.
   - `joint_argmax` on a batch of 2 with 3 legal tuples returns the correct argmax per sample.
   - `joint_topk_tuples` with `k=2` returns sorted indices and matching log-probs.
   - Sentinels (`UNK`, `Unplanned`) never appear in the output of `build_legal_tuples`.

2. **End-to-end eval comparison**:
   - Run `python -m evaluation.run_evaluation --split test` with constraints disabled (flag).
   - Run again with constraints enabled.
   - Expected deltas:
     - `hierarchy_validity_rate` → 1.0 (from whatever baseline reports).
     - Per-head top-1 accuracy: phase ≈ same or slightly better, phase_step likely better, MOC ≈ same, operation likely better. The "unconstrained top-1 was illegal and the joint rescued a correct alternative" case contributes positive delta.
     - Per-head top-1 accuracy can also get *worse* when the unconstrained argmax was correct-but-illegal-combination; the joint then picks something consistent but wrong. If this case dominates, the unconstrained per-head model is actually more accurate despite illegal tuples — which would be useful information.
   - Compare `summary.json` diffs; document the mechanical vs. real-quality split.

3. **LLM smoke test**:
   - Via UI, pick a well + date, run Predict Next Day with constraints on.
   - Expand the ML debug table — every step's top-K should be K legal tuples.
   - Check the LLM's final ops list for plausible durations summing to ~24h.

4. **Bundle integrity check**:
   - Load a fresh bundle in `inference/predict.py`.
   - Assert legal-tuple table loads, has expected shape, contains EOO tuple, contains no sentinel ops.

## Open items (non-blocking)

- **Which field does `inference/contract.StepPrediction` actually expose today?**
  I sketched the `HierarchyTuple` shape but need to read `inference/contract.py`
  at implementation time to confirm the exact migration path. Worth grep-ing
  for consumers of the current per-head shape before writing the new one.

- **Prompt rewrite scope.** If `llm/prompts/system_prompt.md` heavily references
  per-head top-K, the prompt needs an audit. A smaller prompt (top-K tuples
  only) may perform better or worse — empirical question; easy to A/B.

- **Beam search / lookahead as a future step.** Current design is greedy per step
  (joint argmax at each t, feed back winner). A more sophisticated decoder could
  do beam search across the full K-step horizon — score partial sequences by
  cumulative joint log-prob. Out of scope here; flag as a possible follow-up if
  greedy-per-step shows cascade errors in autoregressive eval.

- **Config flag for bypass.** Add `inference.enforce_hierarchy: true` so debugging
  or ablation comparisons can quickly toggle it off without code edits.

## Not in scope for this work

- Constrained decoder *during training* (option 4 from the design discussion —
  would zero illegal-child logits during teacher forcing). Separate architectural
  change, tracked as its own `synopsis.md` item.
- Soft penalty / auxiliary loss for hierarchy consistency during training
  (option 3). Rarely worthwhile standalone; skip unless constrained inference
  plus constrained training both turn out to underperform this approach.
- Changes to MOC / Phase / Duration heads beyond what's needed to thread the
  top-K-tuple contract through.

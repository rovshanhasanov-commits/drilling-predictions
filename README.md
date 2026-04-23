# Drilling Operations — Next-24-Hour Prediction (E2E)

Two-stage pipeline: an ML seq2seq model predicts the next N steps of the
phase → phase_step → major_ops_code → operation hierarchy (plus duration);
a Claude call then refines those predictions into a full next-day operations
plan with final operation codes and durations summing to 24 hours.

## Layout

```
config/            ← pipeline.yaml (single source of truth) + loader
preprocessing/     ← joins Master_Data_With_ID + Comment_Features, engineers features, encodes, splits, saves
training/          ← sequence builder (single target shift, EOO padding), model, SS loop, eval, notebook
inference/         ← (well, date) → MLOutput with top-K per step
llm/               ← prompts + Claude client + similar-wells + ML→LLM orchestration
ui/                ← thin Streamlit layer (sidebar + actual/predicted panels)
models/            ← trained model bundle (written by training, read by inference)
```

Stages are intentionally decoupled. The contracts:

- **Preprocessing → Training/Inference**: parquet + encoders.pkl + config.json per strategy under `Data/Data for model (E2E)/{strategy}/`.
- **Training → Inference**: bundle under `models/seq2seq_N{seq_len}_K{n_future}_T{target_count}_lr{lr}_p{patience}_{strategy}/` (`encoder_model.keras`, `decoder_step_model.keras`, `encoders.pkl`, `data_config.json`, `model_config.json`).
- **Inference → LLM**: [`inference.contract.MLOutput`](inference/contract.py) with `steps: list[StepPrediction]` — each step has top-K labels+probs for all four hierarchy levels plus a duration.
- **LLM → UI**: a dict `{ops_summary, reasoning, operations: [...]}` with durations summing to 24h (± tolerance).

## Critical bug fixed — single target shift

The previous pipeline shifted targets twice (once in `prepare_data_for_model.py`
via `*_next` columns, again via windowing in training). The new pipeline shifts
**only once, in `training/data.py::build_seq2seq_sequences`**. Preprocessing
emits current-step columns only (no `*_next` columns).

Well boundaries are handled with an `"End of Operations"` sentinel class present
in all four LabelEncoders. When a target step falls past the last row of a well,
training emits EOO (categorical) + 0 hours duration. No rows are dropped.

A second sentinel, `"UNK"`, is fit into every LabelEncoder as the fallback for
NaN / unseen-in-train values. It replaces an older global-mode fallback that
silently relabeled missing operations to `DRL`, which manufactured spurious
majority-class signal. The operation encoder additionally has a merged
`"Unplanned"` class for the NPT-style operations listed in
`preprocessing.unplanned_operations` (see below).

## Loss-masking on unpredictable rows

The model is trained to predict the **planned workflow**, not reality with
unplanned interruptions. Two classes of rows are loss-masked via per-row
`sample_weight` flags emitted by preprocessing:

- **`op_label_real=0`, `dur_label_real=0`**: rows where Operation was NaN
  (residual after conditional-mode fill) or was in the unplanned SME list.
- **`moc_label_real=0`**: rows where Operation was in the unplanned list.

Masked positions still flow through the encoder as context and through the
decoder as teacher-forcing inputs — only the loss (cross-entropy / MSE) at
those output positions is zeroed. Phase_Step is never masked. The SME list
lives at `preprocessing.unplanned_operations` in `pipeline.yaml` and is
straightforward to edit.

## Usage

Install deps (see `requirements.txt`):

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your ANTHROPIC_API_KEY in .env
```

### 1. Preprocessing

```bash
python -m preprocessing.run_preprocessing
```

Joins raw CSVs, engineers features, produces the three encoding strategies under
`Data/Data for model (E2E)/{dummies,embed_separate,embed_state}/`.

### 2. Training

Open [training/train_seq2seq.ipynb](training/train_seq2seq.ipynb) and run all
cells. Config values are pulled from `config/pipeline.yaml`. Current defaults:
- strategy = `dummies`
- sequence_length = 25, n_future = 8
- scheduled sampling 0.0 → 0.75 over 200 epochs

Evaluation reports **top-1 AND top-3** accuracy per step — if `top-3 ≫ top-1`,
the user's "off by 1" hypothesis is supported.

The notebook's final cell saves the model bundle to
`models/seq2seq_N{seq_len}_K{n_future}_T{target_count}_lr{lr}_p{patience}_{strategy}/`
(see `config.get_model_dir`). The naming embeds the window shape plus learning
rate and early-stopping patience so multiple runs with different hyperparameters
coexist without clobbering.

### 3. Regenerate constraints (once after preprocessing)

```bash
python -m llm.generate_constraints
```

Writes `llm/prompts/constraints.md` from `Master_Data_With_ID.csv`.

### 4. Run the UI

```bash
streamlit run ui/app.py
```

Select a well, a date, click **Predict Next Day**. The right panel shows the
LLM's final plan. Expand **ML stage — top-K per step (debug)** to inspect what
the ML stage passed to the LLM.

## Iterating on one stage only

Each folder is independently iterable so long as the contract (above) holds:

- Change preprocessing features? Edit `config/pipeline.yaml::features` and `preprocessing/features.py`. Re-run preprocessing, re-train.
- Change the unplanned-operations list? Edit `config/pipeline.yaml::preprocessing::unplanned_operations`. Re-run preprocessing (the list becomes a merged `Unplanned` class + loss-mask flags), re-train.
- Change the model? Edit `training/model.py` + `config/pipeline.yaml::training`. Re-run the notebook. The saved bundle is a drop-in for inference.
- Change which ML outputs the LLM sees? Edit `config/pipeline.yaml::llm::ml_fields_to_include`. No retraining.
- Iterate on the prompt? Edit `llm/prompts/system_prompt.md`. The UI's sidebar also has a live editor.

## Verification checklist

1. **Preprocessing**: `python -m preprocessing.run_preprocessing` → three strategy folders with df_train/val/test.parquet + encoders.pkl + config.json. `n_classes` adds two sentinels (EOO + UNK) on every head; the operation head adds one further merged `Unplanned` class. Parquets carry `op_label_real`, `moc_label_real`, `dur_label_real` flags (float32, 1.0=train, 0.0=loss-mask).
2. **Training**: notebook runs to completion; top-3 > top-1 per step; the `models/seq2seq_N{n}_K{k}_T{t}_lr{lr}_p{p}_{strategy}/` bundle directory is populated.
3. **Inference**: `python -c "from inference.predict import predict; from config import load_config; cfg = load_config(); out = predict('<well>', '<date>', cfg=cfg); print(out.to_dict())"` → MLOutput with `n_future` StepPredictions (currently 8), top-3 per level. The first predicted step corresponds to the op **immediately after** the selected window (not one-off).
4. **LLM**: `streamlit run ui/app.py`, pick a well+date, click Predict. Both panels render. Totals sum to ~24h.
5. **Shift regression**: pick a test well+date where the next day's actuals are known; manually diff against the MLOutput's step-1 prediction to confirm no double-shift.
6. **EOO sanity**: on a well near its end, verify `EOO` appears in top-K of late steps in the ML debug table.

## Out of scope

- `comment_parser.py` (LLM-based extraction from drilling comments) is treated as upstream — the new pipeline reads the already-extracted `Data/Comment_Features.csv`.
- TF-IDF embeddings for similar-well lookup are reused from `LLM-powered predictions/embeddings/` (no regeneration here).
- Productionization / auth / multi-user state.

# Training

## 1. Build the dataset

```bash
python -m src.data.pipeline
```

Runs the full Phase 7 pipeline (load -> validate -> clean/dedupe -> label engineering -> preprocess -> stratified split + time-based holdout) and writes to `data/processed/`:

- `train.parquet`, `val.parquet`, `test.parquet` — stratified 70/15/15
- `time_holdout_train.parquet` (filed <= 2023), `time_holdout_test.parquet` (filed after) — a secondary drift check, since categories like `FORWARD COLLISION AVOIDANCE` barely existed in earlier complaint years
- `label_taxonomy.json` — the kept primary-component classes (everything else collapses to `OTHER`)

Takes ~80s on the full 271,517-row export (text preprocessing is the majority of that).

## 2. Train the baseline

```bash
python -m src.training.train
```

Trains `LogisticRegression` and `LinearSVC` (both `class_weight="balanced"`) on TF-IDF features, selects the winner by validation macro-F1, evaluates the winner once on the held-out test set, and:

- Logs params/metrics/timing to MLflow (`experiments/mlflow.db` — view with `mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db`)
- Writes a full report to `artifacts/<timestamp>/` (classification report, confusion matrix, most-confused-pairs, metrics.json)
- Promotes the winner to `models/production/` (`model.joblib`, `vectorizer.joblib`, `metadata.json`)

Takes ~4-8 minutes on the full 188,056-row training split (LinearSVC's probability calibration, 3-fold, is the slower of the two).

## 3. Benchmark a transformer (optional)

```bash
python -m src.training.train_transformer
```

By default, trains DistilBERT on a stratified **4,000-row subsample** as a fast smoke-test. `TransformerClassifier` uses `AutoModelForSequenceClassification`/`AutoTokenizer`, so any HF encoder checkpoint works, and the script is fully config-driven via environment variables — no code edits needed to scale up or swap models:

| Env var | Default | Purpose |
|---|---|---|
| `TRANSFORMER_BASE_MODEL` | `distilbert-base-uncased` | Any HF encoder checkpoint, e.g. `microsoft/deberta-v3-base` |
| `TRANSFORMER_TRAIN_SUBSAMPLE_SIZE` | `4000` | `0` = use the full training split |
| `TRANSFORMER_VAL_SUBSAMPLE_SIZE` | `800` | `0` = use the full validation split |
| `TRANSFORMER_TEST_SUBSAMPLE_SIZE` | `800` | `0` = use the full test split |
| `TRANSFORMER_NUM_EPOCHS` | `2` | Max training epochs (early stopping may end sooner) |
| `TRANSFORMER_EARLY_STOPPING_PATIENCE` | `2` | Stop after N eval epochs with no val macro-F1 improvement |
| `TRANSFORMER_TRAIN_BATCH_SIZE` | `16` | Raise to use more GPU VRAM (e.g. 32-64 on a 16GB+ card; a 6GB card fits ~8-32 depending on model size) |
| `TRANSFORMER_EVAL_BATCH_SIZE` | `32` | Same idea for eval |
| `TRANSFORMER_LEARNING_RATE` | `2e-5` | Lower than the HF default (5e-5) — larger encoders are more sensitive |
| `TRANSFORMER_WARMUP_RATIO` | `0.1` | LR warmup fraction of total steps — **don't set this to 0** for DeBERTa-v2/v3, see caveat below |
| `TRANSFORMER_USE_WEIGHTED_LOSS` | `0` (off) | See caveat below before enabling |

Saves to `artifacts/transformer_runs/<timestamp>/` and promotes to `models/transformer/` (so `SERVING_MODEL_BACKEND=transformer` can serve it immediately, config-only, no code changes) — but only for full-dataset runs by default; a subsample smoke-test won't silently overwrite a good full-data model (`TRANSFORMER_FORCE_PROMOTE=1` opts back in).

### Before you scale up: two real training-stability gotchas

Both were discovered the hard way while fine-tuning DeBERTa-v3-base on this project — full writeup in [model_benchmark.md](model_benchmark.md#getting-deberta-v3-base-to-actually-train-correctly-took-real-debugging):

1. **Don't enable `TRANSFORMER_USE_WEIGHTED_LOSS` without validating on a subsample first.** Raw "balanced" class weights can be wide enough (160x ratio, in this project's case) to stop a larger encoder from learning at all, even though the identical setup with unweighted loss trains fine.
2. **Don't drop `warmup_ratio` to 0, especially for DeBERTa-v2/v3.** Without it, a full-scale run collapsed to predicting only the majority class after epoch 1 — and a quick 250-step smoke test didn't catch it, because the collapse only shows up over a longer horizon. Validate any new base model or hyperparameter change on a few-thousand-row subsample with `eval_strategy=steps` and several eval checkpoints (not just one final number) before committing hours to a full run.

### Scaling to GPU / full dataset / a different base model

```bash
# Windows (PowerShell)
$env:TRANSFORMER_BASE_MODEL="microsoft/deberta-v3-base"; $env:TRANSFORMER_TRAIN_SUBSAMPLE_SIZE=0; $env:TRANSFORMER_VAL_SUBSAMPLE_SIZE=0; $env:TRANSFORMER_TEST_SUBSAMPLE_SIZE=0
python -m src.training.train_transformer

# bash
TRANSFORMER_BASE_MODEL="microsoft/deberta-v3-base" TRANSFORMER_TRAIN_SUBSAMPLE_SIZE=0 TRANSFORMER_VAL_SUBSAMPLE_SIZE=0 TRANSFORMER_TEST_SUBSAMPLE_SIZE=0 \
  python -m src.training.train_transformer
```

Compare `artifacts/transformer_runs/<run_id>/metrics.json`'s `test_metrics.macro_f1` against the baseline's 0.625. If it wins, set `SERVING_MODEL_BACKEND=transformer` in `.env` — the artifact is already saved in the exact shape `src/inference/predictor.py` expects.

**GPU memory note**: a 6GB card (e.g. RTX 4050 Laptop) comfortably fits DistilBERT at `TRANSFORMER_TRAIN_BATCH_SIZE=16-32`, and DeBERTa-v3-base (2.8x the params) at `TRANSFORMER_TRAIN_BATCH_SIZE=8` (16 got tight enough — ~87% VRAM — that throughput actually dropped). If you hit a CUDA out-of-memory error, lower the batch size before anything else. Per-epoch checkpointing (`save_total_limit=2`, `load_best_model_at_end=True`) means an interrupted multi-hour run only costs you the current epoch, not the whole thing.

## Retraining in production

`POST /api/v1/retrain` (see [API.md](API.md)) runs step 2 above as a background job and hot-swaps the promoted model without restarting the API process. It does **not** rebuild the dataset (step 1) — if the underlying complaints data changed, run `python -m src.data.pipeline` first.

## Reproducibility

- All splits use `random_state=42` (`src/config/settings.py::SplitConfig`).
- Both baseline models use `random_state=42` and deterministic `class_weight="balanced"`.
- The transformer run sets `seed=42` in `TrainingArguments` and disables experiment tracking side-effects by default (`report_to="none"`) — the original notebook's `trainer.train()` hit an interactive `wandb` login prompt mid-run; this is opt-in only now.

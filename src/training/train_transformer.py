"""Transformer fine-tuning benchmark (Phase 10).

Device-agnostic (`TransformerClassifier` auto-detects CUDA) and config-driven
via environment variables, so scaling from a quick CPU smoke-test to a full
GPU run — or swapping in a stronger pretrained model — is a settings change,
not a code edit:

| Env var | Default | Purpose |
|---|---|---|
| `TRANSFORMER_BASE_MODEL` | `distilbert-base-uncased` | Any HF encoder checkpoint, e.g. `microsoft/deberta-v3-base` |
| `TRANSFORMER_TRAIN_SUBSAMPLE_SIZE` | `4000` | Set to `0`/empty to use the full training split |
| `TRANSFORMER_VAL_SUBSAMPLE_SIZE` | `800` | Same, for validation |
| `TRANSFORMER_TEST_SUBSAMPLE_SIZE` | `800` | Same, for test |
| `TRANSFORMER_NUM_EPOCHS` | `2` | Max training epochs (early stopping may end sooner) |
| `TRANSFORMER_EARLY_STOPPING_PATIENCE` | `2` | Stop after N eval epochs with no val macro-F1 improvement |
| `TRANSFORMER_TRAIN_BATCH_SIZE` | `16` | Per-device train batch size (lower for bigger models / less VRAM) |
| `TRANSFORMER_EVAL_BATCH_SIZE` | `32` | Per-device eval batch size |
| `TRANSFORMER_USE_WEIGHTED_LOSS` | `0` (off) | See "Class-weighted loss" below before enabling |
| `TRANSFORMER_WARMUP_RATIO` | `0.1` | LR warmup fraction of total steps — DeBERTa-v2/v3 is documented to collapse without this |
| `TRANSFORMER_LEARNING_RATE` | `2e-5` | Lower than the HF default (5e-5); large pretrained encoders are more sensitive |

Two things beyond the original DistilBERT smoke-test:

1. **Class-weighted loss (opt-in, off by default — read this before enabling).**
   The baseline models use `class_weight="balanced"`, and matching that for the
   transformer sounds like a free win given how imbalanced the label
   distribution is (majority-class baseline accuracy is only 0.16 across 28
   classes). In practice, raw inverse-frequency ("balanced") weights span a
   ~160x ratio here (rarest to most common class), and empirically that wide a
   spread destabilizes DeBERTa-v3-base training with the `Trainer`'s default
   optimizer/clipping settings — loss gets stuck near `ln(28)≈3.33` (i.e. the
   model doesn't learn at all), confirmed via a small-scale A/B (unweighted:
   loss 3.09→2.18, accuracy 18%→35% over 2 epochs; real "balanced" weights:
   loss flat at ~3.3, accuracy stuck ~10-14%, on the *same* data/model/seed).
   `sqrt`-compressing the weights (`WeightedLossTrainer(..., compress="sqrt")`)
   narrows the ratio to ~13x and recovers *some* of the gap but not all of it,
   and combining it with a raised `max_grad_norm` made things worse, not
   better, in testing — this isn't a settings knob with an obvious fix, it's
   a genuine optimization-stability tradeoff. Unweighted cross-entropy is the
   safe default; only enable weighting if you've verified convergence on a
   small subsample first (`TRANSFORMER_TRAIN_SUBSAMPLE_SIZE=2000` or so) —
   don't discover this at hour 4 of a full run like this project did.
2. **Per-epoch checkpointing + early stopping**, keeping only the best
   checkpoint by val macro-F1 (`save_total_limit=2`, `load_best_model_at_end=True`).
   A multi-hour fine-tune on a personal machine can be interrupted (this
   project has already lost one in-progress run to an unplanned laptop
   restart) — checkpointing bounds the loss to one epoch instead of the whole run.

Run as a script: `python -m src.training.train_transformer`
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments

from src.config.settings import label_config, paths
from src.evaluation.metrics import compute_metrics, get_classification_report, lenient_accuracy
from src.evaluation.multi_label_metrics import multi_label_metrics_at_k
from src.models.transformer import DEFAULT_MAX_LENGTH, TransformerClassifier
from src.preprocessing.text_cleaning import preprocess_for_transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

LABEL_COL = label_config.PRIMARY_LABEL_COLUMN
RAW_TEXT_COL = label_config.TEXT_COLUMN


def _env_int_or_none(name: str, default: int) -> int | None:
    raw = os.environ.get(name, str(default)).strip()
    if not raw or raw == "0":
        return None
    return int(raw)


BASE_MODEL_NAME = os.environ.get("TRANSFORMER_BASE_MODEL", "distilbert-base-uncased")
TRAIN_SUBSAMPLE_SIZE = _env_int_or_none("TRANSFORMER_TRAIN_SUBSAMPLE_SIZE", 4000)
VAL_SUBSAMPLE_SIZE = _env_int_or_none("TRANSFORMER_VAL_SUBSAMPLE_SIZE", 800)
TEST_SUBSAMPLE_SIZE = _env_int_or_none("TRANSFORMER_TEST_SUBSAMPLE_SIZE", 800)
NUM_EPOCHS = int(os.environ.get("TRANSFORMER_NUM_EPOCHS", 2))
EARLY_STOPPING_PATIENCE = int(os.environ.get("TRANSFORMER_EARLY_STOPPING_PATIENCE", 2))
TRAIN_BATCH_SIZE = int(os.environ.get("TRANSFORMER_TRAIN_BATCH_SIZE", 16))
EVAL_BATCH_SIZE = int(os.environ.get("TRANSFORMER_EVAL_BATCH_SIZE", 32))
USE_WEIGHTED_LOSS = os.environ.get("TRANSFORMER_USE_WEIGHTED_LOSS", "0") == "1"
# DeBERTa-v2/v3 is documented to be prone to training collapse without LR
# warmup (see module docstring) — 0.1 (10% of steps) is the standard
# recommendation in DeBERTa's own fine-tuning examples. Harmless for
# DistilBERT too, so it's not gated behind the base model choice.
WARMUP_RATIO = float(os.environ.get("TRANSFORMER_WARMUP_RATIO", 0.1))
LEARNING_RATE = float(os.environ.get("TRANSFORMER_LEARNING_RATE", 2e-5))


class WeightedLossTrainer(Trainer):
    """Trainer with per-class weighted cross-entropy, mirroring the
    baseline's `class_weight="balanced"` — see the module docstring for why
    this is opt-in, not the default, and why weights are `sqrt`-compressed
    rather than used raw."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _stratified_subsample(df: pd.DataFrame, n: int | None, label_col: str, seed: int = 42) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    subsample, _ = train_test_split(
        df, train_size=n, random_state=seed, stratify=df[label_col]
    )
    return subsample.reset_index(drop=True)


def _tokenize_dataset(df: pd.DataFrame, tokenizer, label2id: dict[str, int]) -> Dataset:
    texts = df[RAW_TEXT_COL].astype(str).apply(preprocess_for_transformer).tolist()
    labels = [label2id[label] for label in df[LABEL_COL]]
    ds = Dataset.from_dict({"text": texts, "label": labels})

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=DEFAULT_MAX_LENGTH)

    return ds.map(_tok, batched=True)


def train_transformer_benchmark() -> dict:
    train_full = pd.read_parquet(paths.train_parquet)
    val_full = pd.read_parquet(paths.val_parquet)
    test_full = pd.read_parquet(paths.test_parquet)

    # Label space must match the baseline model's so the benchmark is
    # apples-to-apples and the two backends are interchangeable at serving time.
    labels_sorted = sorted(train_full[LABEL_COL].unique())
    logger.info("Label space: %d classes", len(labels_sorted))

    train_df = _stratified_subsample(train_full, TRAIN_SUBSAMPLE_SIZE, LABEL_COL)
    val_df = _stratified_subsample(val_full, VAL_SUBSAMPLE_SIZE, LABEL_COL)
    test_df = _stratified_subsample(test_full, TEST_SUBSAMPLE_SIZE, LABEL_COL)
    is_full_run = TRAIN_SUBSAMPLE_SIZE is None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Device=%s | %s: train=%d val=%d test=%d, epochs=%d, train_batch_size=%d",
        device,
        "full dataset" if is_full_run else "stratified subsample",
        len(train_df), len(val_df), len(test_df), NUM_EPOCHS, TRAIN_BATCH_SIZE,
    )

    clf = TransformerClassifier.new_for_training(BASE_MODEL_NAME, labels_sorted)
    label2id = {label: i for i, label in enumerate(clf.labels)}
    id2label = {i: label for label, i in label2id.items()}

    train_ds = _tokenize_dataset(train_df, clf.tokenizer, label2id)
    val_ds = _tokenize_dataset(val_df, clf.tokenizer, label2id)

    # compute_class_weight requires every class in `classes` to actually appear
    # in `y`, which doesn't hold for small subsamples (smoke tests) even though
    # it always holds for the full training set. Compute weights only for
    # classes present, default absent ones to 1.0 (neutral — they contribute
    # no loss anyway since they never appear in a training batch).
    class_weights = None
    if USE_WEIGHTED_LOSS:
        train_label_ids = np.array([label2id[label] for label in train_df[LABEL_COL]])
        present_classes = np.unique(train_label_ids)
        present_weights = compute_class_weight("balanced", classes=present_classes, y=train_label_ids)
        present_weights = np.sqrt(present_weights)  # see module docstring — raw
                                                      # "balanced" weights destabilize training
        class_weights = torch.ones(len(labels_sorted), dtype=torch.float)
        class_weights[present_classes] = torch.tensor(present_weights, dtype=torch.float)
        logger.info(
            "Weighted loss enabled (sqrt-compressed): weight min/max/mean = %.3f/%.3f/%.3f",
            class_weights.min().item(), class_weights.max().item(), class_weights.mean().item(),
        )
    else:
        logger.info("Weighted loss disabled — using plain unweighted cross-entropy (see module docstring for why)")

    def _compute_metrics_hf(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return compute_metrics(labels, preds)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = paths.artifacts_dir / "transformer_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # keep only the 2 most recent epoch checkpoints —
                              # bounds disk usage and interruption loss to ~1 epoch
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",  # the original notebook hit an interactive wandb
                            # login prompt mid-run; tracking is disabled by
                            # default here and should be opted into explicitly.
        seed=42,
        disable_tqdm=False,
    )

    trainer_cls = WeightedLossTrainer if USE_WEIGHTED_LOSS else Trainer
    trainer_kwargs = {"class_weights": class_weights} if USE_WEIGHTED_LOSS else {}
    trainer = trainer_cls(
        model=clf.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics_hf,
        data_collator=DataCollatorWithPadding(tokenizer=clf.tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        **trainer_kwargs,
    )

    t0 = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t0
    epochs_trained = trainer.state.epoch
    logger.info(
        "Fine-tuning took %.1fs for %.1f/%d epochs (early stopping patience=%d) on %d examples",
        train_time, epochs_trained, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, len(train_df),
    )

    clf.model.eval()

    # Latency benchmark: single-sample predict, matches how the baseline
    # benchmark measures inference_latency_ms_per_sample in src/training/train.py.
    sample_texts = test_df[RAW_TEXT_COL].astype(str).apply(preprocess_for_transformer).tolist()
    bench_texts = sample_texts[:100] if len(sample_texts) >= 100 else sample_texts
    t0 = time.perf_counter()
    clf.predict(bench_texts, batch_size=1)
    latency_ms = (time.perf_counter() - t0) / len(bench_texts) * 1000

    # Final test evaluation.
    y_test_pred = clf.predict(sample_texts, batch_size=32)
    y_test_true = test_df[LABEL_COL].tolist()
    test_metrics = compute_metrics(y_test_true, y_test_pred)
    # See docs/model_benchmark.md's "honest evaluation" section — quantifies
    # how much "error" is really just predicting a non-primary listed component.
    test_metrics["lenient_accuracy"] = lenient_accuracy(test_df["components"], y_test_pred)
    # v2 scoping step (see docs/model_benchmark.md) — how well would this
    # single-label model do if its top-k output were scored as a multi-label
    # prediction set, with no retraining?
    y_test_proba = clf.predict_proba(sample_texts, batch_size=32)
    multi_label_metrics = [
        multi_label_metrics_at_k(test_df["components"], y_test_proba, clf.classes_, k=k)
        for k in (1, 3)
    ]
    report_text = get_classification_report(y_test_true, y_test_pred, labels=labels_sorted)
    logger.info("Transformer test metrics: %s", test_metrics)

    n_params = sum(p.numel() for p in clf.model.parameters())
    model_memory_mb = n_params * 4 / 1e6  # float32 params

    clf.save(run_dir / "model")
    (run_dir / "classification_report.txt").write_text(report_text)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_type": f"{BASE_MODEL_NAME}-finetuned",
                "base_model": BASE_MODEL_NAME,
                "device": clf.device,
                "n_train_subsample": len(train_df),
                "n_val_subsample": len(val_df),
                "n_test_subsample": len(test_df),
                "max_epochs": NUM_EPOCHS,
                "epochs_trained": epochs_trained,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "train_time_seconds": train_time,
                "inference_latency_ms_per_sample": latency_ms,
                "n_parameters": n_params,
                "model_memory_mb": model_memory_mb,
                "test_metrics": test_metrics,
                "multi_label_metrics": multi_label_metrics,
                "is_full_dataset_run": is_full_run,
                "note": (
                    f"Trained on the full training set on {device.upper()}."
                    if is_full_run
                    else (
                        "Trained on a stratified subsample as a pipeline smoke-test, "
                        "not the full training set. See docs/Training.md for full-scale "
                        "run instructions (set TRANSFORMER_TRAIN_SUBSAMPLE_SIZE=0)."
                    )
                ),
            },
            indent=2,
        )
    )

    # Promote to models/transformer/ so SERVING_MODEL_BACKEND=transformer can
    # load it for serving — but only for full-dataset runs by default. A
    # subsample smoke-test has silently clobbered a good full-data model here
    # twice already; TRANSFORMER_FORCE_PROMOTE=1 opts back in if you really
    # want to serve a subsample-trained model (e.g. a quick demo).
    should_promote = is_full_run or os.environ.get("TRANSFORMER_FORCE_PROMOTE") == "1"
    if should_promote:
        transformer_serving_dir = paths.models_dir / "transformer"
        clf.save(transformer_serving_dir)
        (transformer_serving_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "model_type": f"{BASE_MODEL_NAME}-finetuned",
                    "trained_at": run_id,
                    "classes": labels_sorted,
                    "test_metrics": test_metrics,
                    "multi_label_metrics": multi_label_metrics,
                    "is_full_dataset_run": is_full_run,
                    "n_train_examples": len(train_df),
                    "device": device,
                    "inference_latency_ms_per_sample": latency_ms,
                },
                indent=2,
            )
        )
        logger.info("Saved transformer benchmark to %s and promoted to %s", run_dir, transformer_serving_dir)
    else:
        logger.info(
            "Saved transformer benchmark to %s (NOT promoted to models/transformer/ — "
            "this was a subsample run; set TRANSFORMER_FORCE_PROMOTE=1 to override)",
            run_dir,
        )

    return {
        "run_dir": str(run_dir),
        "test_metrics": test_metrics,
        "train_time_seconds": train_time,
        "inference_latency_ms_per_sample": latency_ms,
        "n_parameters": n_params,
    }


if __name__ == "__main__":
    train_transformer_benchmark()

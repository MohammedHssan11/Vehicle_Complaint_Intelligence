"""Baseline model training orchestrator (Phase 10/11).

Trains every model in MODEL_REGISTRY on the TF-IDF features, selects the
winner by macro-F1 on the validation split, does a final held-out test
evaluation + error analysis for the winner only (never touch test before
model selection is final), and promotes the winning artifacts to
`models/production/`.

Run as a script: `python -m src.training.train`
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import mlflow
import pandas as pd

from src.config.settings import PROJECT_ROOT, feature_config, label_config, model_config, paths, settings
from src.evaluation.error_analysis import most_confused_pairs
from src.evaluation.metrics import (
    compute_metrics,
    get_classification_report,
    get_confusion_matrix_df,
    majority_class_baseline_accuracy,
)
from src.features.tfidf import TfidfFeaturizer
from src.models.baseline import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TEXT_COL = label_config.CLEAN_TEXT_COLUMN
LABEL_COL = label_config.PRIMARY_LABEL_COLUMN


def _load_processed_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in (paths.train_parquet, paths.val_parquet, paths.test_parquet):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run `python -m src.data.pipeline` first to build processed splits."
            )
    return (
        pd.read_parquet(paths.train_parquet),
        pd.read_parquet(paths.val_parquet),
        pd.read_parquet(paths.test_parquet),
    )


def _benchmark_latency_ms(model, X_sample) -> float:
    n = X_sample.shape[0]
    t0 = time.perf_counter()
    model.predict(X_sample)
    elapsed = time.perf_counter() - t0
    return (elapsed / n) * 1000


def train_and_select() -> dict:
    (PROJECT_ROOT / "experiments").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("vehicle-complaint-baseline")

    train, val, test = _load_processed_splits()
    logger.info("Loaded splits: train=%d val=%d test=%d", len(train), len(val), len(test))

    baseline_acc = majority_class_baseline_accuracy(train[LABEL_COL])
    logger.info("Majority-class baseline accuracy: %.4f", baseline_acc)

    featurizer = TfidfFeaturizer()
    t0 = time.perf_counter()
    X_train = featurizer.fit_transform(train[TEXT_COL])
    logger.info("TF-IDF fit_transform took %.1fs, vocab size=%d", time.perf_counter() - t0, X_train.shape[1])
    X_val = featurizer.transform(val[TEXT_COL])
    y_train, y_val = train[LABEL_COL], val[LABEL_COL]

    results = {}
    for name in model_config.BASELINE_MODELS:
        logger.info("Training %s ...", name)
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "model_type": name,
                    "tfidf_max_features": feature_config.TFIDF_MAX_FEATURES,
                    "tfidf_ngram_range": str(feature_config.TFIDF_NGRAM_RANGE),
                    "tfidf_min_df": feature_config.TFIDF_MIN_DF,
                    "n_train": len(train),
                }
            )

            model_cls = MODEL_REGISTRY[name]
            model = model_cls()

            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            y_val_pred = model.predict(X_val)
            val_metrics = compute_metrics(y_val, y_val_pred)
            latency_ms = _benchmark_latency_ms(model, X_val[:200])

            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metric("train_time_seconds", train_time)
            mlflow.log_metric("inference_latency_ms_per_sample", latency_ms)

            logger.info(
                "%s -> val macro_f1=%.4f weighted_f1=%.4f acc=%.4f train_time=%.1fs latency=%.3fms/sample",
                name,
                val_metrics["macro_f1"],
                val_metrics["weighted_f1"],
                val_metrics["accuracy"],
                train_time,
                latency_ms,
            )

            results[name] = {
                "model": model,
                "val_metrics": val_metrics,
                "train_time_seconds": train_time,
                "inference_latency_ms_per_sample": latency_ms,
            }

    best_name = max(results, key=lambda n: results[n]["val_metrics"]["macro_f1"])
    best = results[best_name]
    logger.info("Selected best model by val macro_f1: %s", best_name)

    # Final test evaluation — winner only, test set touched exactly once.
    X_test = featurizer.transform(test[TEXT_COL])
    y_test = test[LABEL_COL]
    y_test_pred = best["model"].predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)
    labels = sorted(y_train.unique())
    report_text = get_classification_report(y_test, y_test_pred, labels=labels)
    confusion_df = get_confusion_matrix_df(y_test, y_test_pred, labels=labels)
    confused_pairs = most_confused_pairs(test[TEXT_COL], y_test, y_test_pred, top_n=20)

    logger.info("Test metrics for %s: %s", best_name, test_metrics)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = paths.artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "classification_report.txt").write_text(report_text)
    confusion_df.to_csv(run_dir / "confusion_matrix.csv")
    confused_pairs.to_csv(run_dir / "most_confused_pairs.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_model": best_name,
                "val_metrics": best["val_metrics"],
                "test_metrics": test_metrics,
                "majority_class_baseline_accuracy": baseline_acc,
                "train_time_seconds": best["train_time_seconds"],
                "inference_latency_ms_per_sample": best["inference_latency_ms_per_sample"],
                "all_models_val_metrics": {n: r["val_metrics"] for n, r in results.items()},
            },
            indent=2,
        )
    )

    # Promote winner to models/production/ for serving.
    paths.production_model_dir.mkdir(parents=True, exist_ok=True)
    best["model"].save(paths.production_model_dir / "model.joblib")
    featurizer.save(paths.production_model_dir / "vectorizer.joblib")
    (paths.production_model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": best_name,
                "trained_at": run_id,
                "classes": sorted(best["model"].classes_.tolist()),
                "test_metrics": test_metrics,
                "val_metrics": best["val_metrics"],
            },
            indent=2,
        )
    )
    logger.info("Promoted %s to %s", best_name, paths.production_model_dir)

    return {
        "best_model_name": best_name,
        "test_metrics": test_metrics,
        "all_results": results,
        "run_dir": str(run_dir),
    }


if __name__ == "__main__":
    train_and_select()

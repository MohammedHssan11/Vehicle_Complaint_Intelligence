"""End-to-end dataset build: raw CSV -> validated, cleaned, labeled,
preprocessed, split parquet files ready for training (Phase 7).

Run as a script: `python -m src.data.pipeline`
"""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from src.config.settings import label_config, paths, split_config
from src.data.cleaning import clean_complaints
from src.data.labels import engineer_labels
from src.data.loader import load_complaints
from src.data.splits import assert_no_text_leakage, stratified_split, time_based_split
from src.preprocessing.text_cleaning import preprocess_for_classical

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_dataset(save: bool = True) -> dict[str, pd.DataFrame]:
    t0 = time.time()

    logger.info("Loading raw complaints.csv ...")
    df = load_complaints()
    logger.info("Loaded %d rows", len(df))

    df = clean_complaints(df)
    logger.info("After cleaning/dedup: %d rows", len(df))

    df, kept_labels = engineer_labels(df)
    logger.info("Label taxonomy: %d kept classes + OTHER", len(kept_labels))

    logger.info("Preprocessing text (classical pipeline) ...")
    t_text = time.time()
    df[label_config.CLEAN_TEXT_COLUMN] = df[label_config.TEXT_COLUMN].astype(str).apply(
        preprocess_for_classical
    )
    logger.info("Text preprocessing took %.1fs", time.time() - t_text)

    train, val, test = stratified_split(df)
    assert_no_text_leakage(train, test)
    assert_no_text_leakage(train, val)
    assert_no_text_leakage(val, test)
    logger.info("Stratified split: train=%d val=%d test=%d", len(train), len(val), len(test))

    time_train, time_test = time_based_split(df)
    logger.info(
        "Time-based holdout: train(<=%d)=%d test(>%d)=%d",
        split_config.TIME_HOLDOUT_TRAIN_MAX_YEAR,
        len(time_train),
        split_config.TIME_HOLDOUT_TRAIN_MAX_YEAR,
        len(time_test),
    )

    result = {
        "train": train,
        "val": val,
        "test": test,
        "time_holdout_train": time_train,
        "time_holdout_test": time_test,
    }

    if save:
        paths.processed_dir.mkdir(parents=True, exist_ok=True)
        train.to_parquet(paths.train_parquet, index=False)
        val.to_parquet(paths.val_parquet, index=False)
        test.to_parquet(paths.test_parquet, index=False)
        time_train.to_parquet(paths.time_holdout_train_parquet, index=False)
        time_test.to_parquet(paths.time_holdout_test_parquet, index=False)

        taxonomy_path = paths.processed_dir / "label_taxonomy.json"
        taxonomy_path.write_text(json.dumps(sorted(kept_labels), indent=2))
        logger.info("Saved splits + label taxonomy to %s", paths.processed_dir)

    logger.info("Dataset build finished in %.1fs", time.time() - t0)
    return result


if __name__ == "__main__":
    build_dataset()

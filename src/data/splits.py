"""Train/val/test splitting (Phase 7/11).

Two splits are produced:
1. A stratified 70/15/15 split on the primary_component label — used for
   model training/selection/testing.
2. A time-based holdout (train: filed <= TIME_HOLDOUT_TRAIN_MAX_YEAR, test:
   filed after) — used only as a secondary drift check, since categories
   like FORWARD COLLISION AVOIDANCE and LANE DEPARTURE are recent-tech
   categories that barely existed in earlier complaint years. A pure random
   split would hide this non-stationarity; this holdout surfaces it without
   changing what the model is actually trained/selected on.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import label_config, split_config


def stratified_split(
    df: pd.DataFrame,
    label_col: str = label_config.PRIMARY_LABEL_COLUMN,
    test_size: float = split_config.TEST_SIZE,
    val_size: float = split_config.VAL_SIZE,
    random_state: int = split_config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    # val_size is expressed as a fraction of the *original* df; rescale it
    # to be a fraction of train_val (what remains after removing test).
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val[label_col],
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def time_based_split(
    df: pd.DataFrame,
    date_col: str = "dateComplaintFiled",
    train_max_year: int = split_config.TIME_HOLDOUT_TRAIN_MAX_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = pd.to_datetime(df[date_col]).dt.year
    train = df[years <= train_max_year].reset_index(drop=True)
    test = df[years > train_max_year].reset_index(drop=True)
    return train, test


def assert_no_text_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    text_col: str = label_config.TEXT_COLUMN,
) -> None:
    """Guard against the exact failure mode the notebook's dead upsampling
    code would have caused: identical complaint text present in both splits."""
    overlap = set(train[text_col]) & set(test[text_col])
    if overlap:
        raise ValueError(
            f"{len(overlap)} complaint summaries appear in both splits — "
            "train/test contamination. Check deduplication ran before splitting."
        )

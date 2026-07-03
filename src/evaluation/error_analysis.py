"""Misclassification analysis (Phase 12): most-confused label pairs and
concrete example texts, so errors are debuggable rather than just a number."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_frame(texts, y_true, y_pred) -> pd.DataFrame:
    # texts/y_true are typically pandas Series (from a dataframe column),
    # y_pred is typically a raw numpy array (from model.predict) — normalize
    # both through np.asarray so this works regardless of which was passed.
    return pd.DataFrame(
        {"text": np.asarray(texts), "true": np.asarray(y_true), "pred": np.asarray(y_pred)}
    )


def most_confused_pairs(texts, y_true, y_pred, top_n: int = 20) -> pd.DataFrame:
    df = _to_frame(texts, y_true, y_pred)
    errors = df[df["true"] != df["pred"]]
    pair_counts = (
        errors.groupby(["true", "pred"]).size().reset_index(name="count").sort_values(
            "count", ascending=False
        )
    )
    return pair_counts.head(top_n).reset_index(drop=True)


def sample_errors_for_pair(
    texts, y_true, y_pred, true_label: str, pred_label: str, n: int = 5
) -> pd.DataFrame:
    df = _to_frame(texts, y_true, y_pred)
    mask = (df["true"] == true_label) & (df["pred"] == pred_label)
    return df[mask].head(n).reset_index(drop=True)


def false_negative_report(texts, y_true, y_pred, label: str, n: int = 10) -> pd.DataFrame:
    """Rows truly belonging to `label` that the model missed."""
    df = _to_frame(texts, y_true, y_pred)
    mask = (df["true"] == label) & (df["pred"] != label)
    return df[mask].head(n).reset_index(drop=True)


def false_positive_report(texts, y_true, y_pred, label: str, n: int = 10) -> pd.DataFrame:
    """Rows the model predicted as `label` that actually belong elsewhere."""
    df = _to_frame(texts, y_true, y_pred)
    mask = (df["pred"] == label) & (df["true"] != label)
    return df[mask].head(n).reset_index(drop=True)

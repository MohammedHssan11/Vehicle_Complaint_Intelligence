"""Evaluation metrics (Phase 12).

Macro-F1 is the primary metric, not accuracy: the label distribution is
heavily imbalanced (top class ~23% of train, several tail classes <1%), so a
model that only ever predicts the top 2-3 classes would still score high
accuracy while being useless for rare-but-important categories like AIR BAGS
or SEAT BELTS. Macro-F1 weights every class equally regardless of support.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def get_classification_report(y_true, y_pred, labels=None) -> str:
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)


def get_classification_report_dict(y_true, y_pred, labels=None) -> dict:
    return classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)


def get_confusion_matrix_df(y_true, y_pred, labels) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def majority_class_baseline_accuracy(y_true) -> float:
    """Accuracy of always predicting the single most frequent class —
    the floor any real model must clear."""
    counts = pd.Series(y_true).value_counts()
    return counts.iloc[0] / counts.sum()


def lenient_accuracy(components_raw: pd.Series, y_pred) -> float:
    """Fraction of predictions matching ANY listed component, not just the
    primary (first-listed) one used as the training/scoring label.

    `components` is a genuinely multi-label field (e.g. "POWER TRAIN,ENGINE")
    collapsed to primary-only for this task. A prediction matching a listed
    but non-primary component isn't really wrong — it's scored as an error
    only because of that collapse. See docs/model_benchmark.md's "honest
    evaluation" section for the full analysis (this metric alone accounted
    for ~9 points of "error" on the DeBERTa-v3-base run)."""
    components_raw = pd.Series(components_raw).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    matches = [
        pred in {p.strip() for p in components.split(",")}
        for components, pred in zip(components_raw, y_pred)
    ]
    return sum(matches) / len(matches)

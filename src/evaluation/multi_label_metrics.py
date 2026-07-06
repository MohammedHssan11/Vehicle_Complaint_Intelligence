"""Multi-label evaluation — v2 scoping step (Phase: post-M4 accuracy work).

`components` is genuinely multi-label (e.g. "POWER TRAIN,ENGINE"), collapsed
to primary-only for the single-label task the current models are trained
on (see docs/model_benchmark.md's "honest evaluation" section — that
collapse alone accounts for ~9 points of apparent "error").

These metrics evaluate the EXISTING single-label models' top-k probability
outputs against the FULL listed component set, without retraining anything.
This is a deliberately low-risk first step toward a real multi-label
classifier (sigmoid head, subset/Hamming-loss training objective) — it
answers "how much of the true label set would top-k predictions already
cover?" using models that were never trained with multi-label in mind, as
evidence for whether a full multi-label retrain is worth the much larger
engineering lift (new training objective, new API/UI shape, explainability
that doesn't extend the same way — see docs/model_benchmark.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def parse_component_set(components_raw: str) -> set[str]:
    return {p.strip() for p in components_raw.split(",")}


def multi_label_metrics_at_k(
    components_raw: pd.Series, proba: np.ndarray, classes: np.ndarray, k: int
) -> dict:
    """Treats a single-label model's top-k predictions as a multi-label
    prediction set and scores them against the full listed component set.

    - recall_at_k: mean fraction of each row's true labels captured within
      the top-k predictions (partial credit on multi-label rows).
    - full_coverage_at_k: fraction of rows where ALL true labels appear in
      the top-k predictions — a stricter, subset-accuracy-style metric.
    - precision_at_k: mean fraction of the k predicted slots that were
      correct. Expected to be low by construction (most rows only have 1-2
      true labels but k slots are spent regardless) — not meant to look
      good, just to be honest about the precision/recall tradeoff of using
      a fixed k instead of a calibrated per-label threshold.
    """
    if proba.shape[0] != len(components_raw):
        raise ValueError(
            f"proba has {proba.shape[0]} rows but components_raw has {len(components_raw)}"
        )
    components_raw = pd.Series(components_raw).reset_index(drop=True)
    classes = np.asarray(classes)
    top_k_idx = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    top_k_labels = classes[top_k_idx]  # shape (n_rows, k)

    recalls, full_coverage, precisions = [], [], []
    for i, components in enumerate(components_raw):
        true_set = parse_component_set(components)
        pred_set = set(top_k_labels[i])
        overlap = len(true_set & pred_set)
        recalls.append(overlap / len(true_set))
        full_coverage.append(1.0 if true_set.issubset(pred_set) else 0.0)
        precisions.append(overlap / k)

    return {
        "k": k,
        "recall_at_k": float(np.mean(recalls)),
        "full_coverage_at_k": float(np.mean(full_coverage)),
        "precision_at_k": float(np.mean(precisions)),
    }

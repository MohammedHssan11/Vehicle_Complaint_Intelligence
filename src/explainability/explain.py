"""Phase 13 explainability: confidence, top-3 categories, and top
contributing words for a prediction.

Word-level attribution is computed with `shap.LinearExplainer` against the
model's extracted linear coefficients (see `coefficients.py`) — cheap and
exact for a linear model over a sparse TF-IDF space, unlike SHAP's
model-agnostic explainers (KernelExplainer/Permutation), which would be far
too slow for real-time inference over a 30k-dim vocabulary. If SHAP ever
fails for an unexpected reason (version mismatch, etc.), this degrades to a
manual coefficient x feature-value attribution — mathematically the same
quantity SHAP's LinearExplainer computes internally (relative to a
zero/mean background), just without the library, so explanations never
silently disappear.
"""
from __future__ import annotations

import logging

import numpy as np
import shap
from scipy.sparse import spmatrix

from src.explainability.coefficients import extract_class_coefficients

logger = logging.getLogger(__name__)

_explainer_cache: dict[int, tuple] = {}


def _get_explainer(model, n_features: int):
    key = id(model)
    if key not in _explainer_cache:
        coef, intercept, classes = extract_class_coefficients(model)
        background = np.zeros((1, n_features))
        explainer = shap.LinearExplainer((coef, intercept), background)
        _explainer_cache[key] = (explainer, coef, intercept, classes)
    return _explainer_cache[key]


def _coefficient_terms(
    coef: np.ndarray, classes: np.ndarray, predicted_label: str, x_vector: spmatrix, feature_names, top_k: int
) -> list[tuple[str, float]]:
    class_idx = list(classes).index(predicted_label)
    row = x_vector.tocsr()[0]
    nonzero_idx = row.indices
    contributions = coef[class_idx, nonzero_idx] * row.data
    terms = list(zip((feature_names[i] for i in nonzero_idx), contributions.tolist()))
    terms.sort(key=lambda t: t[1], reverse=True)
    return terms[:top_k]


def top_contributing_terms(
    model, x_vector: spmatrix, predicted_label: str, feature_names, top_k: int = 10
) -> tuple[list[dict], str]:
    """Returns (terms, method) where method is "shap" or "coefficient" (fallback)."""
    try:
        explainer, coef, intercept, classes = _get_explainer(model, x_vector.shape[1])
        class_idx = list(classes).index(predicted_label)
        shap_values = explainer.shap_values(x_vector)[0, :, class_idx]
        row = x_vector.tocsr()[0]
        nonzero_idx = row.indices
        terms = [(feature_names[i], float(shap_values[i])) for i in nonzero_idx]
        terms.sort(key=lambda t: t[1], reverse=True)
        method = "shap"
    except Exception:
        logger.warning("SHAP explanation failed, falling back to raw coefficients", exc_info=True)
        coef, intercept, classes = extract_class_coefficients(model)
        terms = _coefficient_terms(coef, classes, predicted_label, x_vector, feature_names, top_k)
        method = "coefficient"

    terms = terms[:top_k]
    return [{"term": t, "contribution": round(c, 4)} for t, c in terms], method


def top_k_predictions(proba_row: np.ndarray, classes: np.ndarray, k: int = 3) -> list[dict]:
    top_idx = np.argsort(proba_row)[::-1][:k]
    return [{"label": classes[i], "confidence": round(float(proba_row[i]), 4)} for i in top_idx]

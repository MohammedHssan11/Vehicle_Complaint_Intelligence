"""Extract per-class linear coefficients regardless of whether the model is
a plain linear classifier (LogisticRegression: has `coef_` directly) or a
`CalibratedClassifierCV`-wrapped one (LinearSVC, our M1-selected production
model: coefficients live on each fold's `.estimator.coef_`, averaged here).

Both baseline models are linear over the same TF-IDF feature space, so this
is what makes coefficient-based and SHAP-linear explanations possible at all
without retraining a separate "explainer model".
"""
from __future__ import annotations

import numpy as np


def extract_class_coefficients(model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (coef [n_classes, n_features], intercept [n_classes], classes [n_classes])."""
    inner = getattr(model, "model", model)  # unwrap our BaseClassifier subclasses

    if hasattr(inner, "coef_"):
        return inner.coef_, inner.intercept_, inner.classes_

    if hasattr(inner, "calibrated_classifiers_"):
        fold_coefs = []
        fold_intercepts = []
        for calibrated_clf in inner.calibrated_classifiers_:
            estimator = getattr(calibrated_clf, "estimator", None) or getattr(
                calibrated_clf, "base_estimator"
            )
            fold_coefs.append(estimator.coef_)
            fold_intercepts.append(estimator.intercept_)
        coef = np.mean(fold_coefs, axis=0)
        intercept = np.mean(fold_intercepts, axis=0)
        return coef, intercept, inner.classes_

    raise TypeError(
        f"Don't know how to extract linear coefficients from {type(inner).__name__}"
    )

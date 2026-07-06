"""Classical TF-IDF baseline models.

Class imbalance (top class ~23% of train, tail classes <1%) is handled via
`class_weight="balanced"`, not duplication. The original notebook's approach
(upsample minority classes to match the majority via `resample(replace=True)`,
i.e. literal duplicate rows) was never actually wired into its training cells,
but it is exactly the wrong tool for text: duplicating rows before a split
risks train/test contamination (Phase 4), and even done correctly (post-split)
it just teaches the model to memorize repeated exact strings rather than
generalizing. `class_weight="balanced"` re-weights the loss instead — no
duplication, no leakage risk, and it's the standard sklearn mechanism for
this exact problem.

LinearSVC has no `predict_proba` (only `decision_function`), but Phase 13
(explainability) requires a confidence score and top-3 probabilities for
every model backend. It's wrapped in `CalibratedClassifierCV` so both
baseline models expose a real, comparable probability interface.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.config.settings import model_config
from src.models.base import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, **overrides):
        params = dict(
            max_iter=1000,
            class_weight="balanced",
            random_state=model_config.RANDOM_STATE,
            n_jobs=-1,
        )
        params.update(overrides)
        self.model = LogisticRegression(**params)

    def fit(self, X, y) -> "LogisticRegressionClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_


class LinearSVCClassifier(BaseClassifier):
    def __init__(self, calibration_cv: int = 3, **overrides):
        params = dict(class_weight="balanced", random_state=model_config.RANDOM_STATE)
        params.update(overrides)
        base = LinearSVC(**params)
        self.model = CalibratedClassifierCV(base, cv=calibration_cv)

    def fit(self, X, y) -> "LinearSVCClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_


def _lazy_xgboost_classifier(**kwargs):
    from src.models.xgboost_model import XGBoostClassifier  # heavy import (xgboost), load lazily

    return XGBoostClassifier(**kwargs)


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionClassifier,
    "linear_svc": LinearSVCClassifier,
    "xgboost": _lazy_xgboost_classifier,
}

"""XGBoost baseline (Phase 10) — added alongside LogisticRegression/LinearSVC
to check whether a nonlinear model finds interactions the linear models miss
over the same TF-IDF features.

XGBoost needs integer-encoded labels (unlike sklearn, which accepts strings
directly), so the label encoding is handled internally here and is invisible
to callers — `predict()` still returns the original string labels, matching
`BaseClassifier`'s contract. Class imbalance is handled via per-sample
weights (`compute_sample_weight("balanced", y)`), the XGBoost equivalent of
sklearn's `class_weight="balanced"`.

**CPU-only, deliberately** — this is not an oversight. `device="cuda"` was
tried (this project otherwise uses the GPU for DistilBERT) and produced
silently broken models: on an 8,000-row diagnostic, CPU scored 0.523
macro-F1 while GPU scored 0.042 (worse than random across 28 classes), with
only a benign-looking "mismatched devices, falling back to DMatrix" warning
as a clue. This is a real correctness bug in how XGBoost 3.2.0's GPU `hist`
path handles a `scipy.sparse.csr_matrix` TF-IDF input on this setup, not a
performance tradeoff — so GPU is switched off here rather than "fixed" by
converting inputs, since that fix wasn't verified and shipping a fast-but-wrong
model would be worse than a slow-but-correct one.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.config.settings import model_config
from src.models.base import BaseClassifier


class XGBoostClassifier(BaseClassifier):
    def __init__(self, **overrides):
        params = dict(
            n_estimators=80,
            max_depth=5,
            learning_rate=0.3,
            tree_method="hist",
            device="cpu",
            max_bin=64,  # lower memory footprint than the 256 default; this is a
                         # sparse, high-cardinality (30-50K feature) TF-IDF matrix,
                         # not a small tabular dataset — histogram memory adds up fast,
                         # and building histograms over sparse text features is
                         # inherently slower than XGBoost's typical dense-tabular case.
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=model_config.RANDOM_STATE,
            n_jobs=6,  # capped rather than -1 (all logical cores): thread
                       # oversubscription on a 12-thread laptop competing with an
                       # already-loaded Streamlit/uvicorn process caused this to
                       # thrash under memory pressure rather than actually parallelize.
            verbosity=1,
        )
        params.update(overrides)
        self._params = params
        self._label_encoder = LabelEncoder()
        self.model = None

    def fit(self, X, y) -> "XGBoostClassifier":
        y_encoded = self._label_encoder.fit_transform(y)
        sample_weight = compute_sample_weight("balanced", y_encoded)
        self.model = xgb.XGBClassifier(
            num_class=len(self._label_encoder.classes_), **self._params
        )
        self.model.fit(X, y_encoded, sample_weight=sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        return self._label_encoder.inverse_transform(self.model.predict(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self._label_encoder.classes_

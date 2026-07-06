"""Inference service (Phase 14): the single place that knows how to go from
raw complaint text to a prediction, regardless of which model backend is
serving. This is the actual swappable abstraction the plan called for —
`TransformerClassifier` and the baseline `BaseClassifier` subclasses have
different input contracts (pre-vectorized matrix vs. raw text), and this
class hides that behind one `predict(text) -> PredictionResult` method so
`src/api` and the Streamlit app never need to know which backend is active.

Flow: Request -> validation -> preprocessing -> model -> postprocessing -> response.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from src.config.settings import label_config, paths, settings
from src.explainability.explain import top_contributing_terms, top_k_predictions
from src.models.base import BaseClassifier
from src.features.tfidf import FEATURIZER_REGISTRY, TfidfFeaturizer
from src.preprocessing.text_cleaning import preprocess_for_classical, preprocess_for_transformer

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 5
MAX_TEXT_LENGTH = 10_000


class InvalidInputError(ValueError):
    pass


@dataclass
class PredictionResult:
    predicted_label: str
    confidence: float
    top_k: list[dict]
    explanation: list[dict] | None
    explanation_method: str | None
    model_backend: str
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise InvalidInputError("complaint text must be a non-empty string")
    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        raise InvalidInputError(f"complaint text is too short (min {MIN_TEXT_LENGTH} chars)")
    if len(text) > MAX_TEXT_LENGTH:
        raise InvalidInputError(f"complaint text is too long (max {MAX_TEXT_LENGTH} chars)")
    return text


class BaselinePredictor:
    """TF-IDF + linear model backend — the default production model (M1).
    Supports full word-level explainability."""

    backend_name = "baseline"

    def __init__(self, model_dir=None):
        model_dir = model_dir or paths.production_model_dir
        self.model: BaseClassifier = BaseClassifier.load(model_dir / "model.joblib")

        metadata_path = model_dir / "metadata.json"
        featurizer_type = "word_tfidf"
        if metadata_path.exists():
            featurizer_type = json.loads(metadata_path.read_text()).get("featurizer_type", "word_tfidf")
        featurizer_cls = FEATURIZER_REGISTRY.get(featurizer_type, TfidfFeaturizer)
        self.vectorizer = featurizer_cls.load(model_dir / "vectorizer.joblib")
        self.feature_names = self.vectorizer.get_feature_names()

    def predict(self, text: str, explain: bool = True) -> PredictionResult:
        t0 = time.perf_counter()
        clean_text = preprocess_for_classical(text)
        X = self.vectorizer.transform([clean_text])
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        predicted_idx = proba.argmax()
        predicted_label = classes[predicted_idx]

        explanation, method = (None, None)
        if explain:
            explanation, method = top_contributing_terms(
                self.model, X, predicted_label, self.feature_names, top_k=10
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        return PredictionResult(
            predicted_label=predicted_label,
            confidence=round(float(proba[predicted_idx]), 4),
            top_k=top_k_predictions(proba, classes, k=3),
            explanation=explanation,
            explanation_method=method,
            model_backend=self.backend_name,
            latency_ms=round(latency_ms, 3),
        )


class TransformerPredictor:
    """DistilBERT backend (M2 benchmark). No word-level explanation yet —
    SHAP/coefficient attribution only applies to the linear baseline; a
    transformer explainer (e.g. integrated gradients via captum) is a
    documented future extension, not implemented here."""

    backend_name = "transformer"

    def __init__(self, model_dir=None):
        from src.models.transformer import TransformerClassifier  # heavy import, load lazily

        model_dir = model_dir or (paths.models_dir / "transformer")
        self.clf = TransformerClassifier.load(model_dir)

    def predict(self, text: str, explain: bool = True) -> PredictionResult:
        t0 = time.perf_counter()
        clean_text = preprocess_for_transformer(text)
        proba = self.clf.predict_proba([clean_text])[0]
        classes = self.clf.classes_
        predicted_idx = proba.argmax()
        predicted_label = classes[predicted_idx]

        latency_ms = (time.perf_counter() - t0) * 1000
        return PredictionResult(
            predicted_label=predicted_label,
            confidence=round(float(proba[predicted_idx]), 4),
            top_k=top_k_predictions(proba, classes, k=3),
            explanation=None,
            explanation_method=None,
            model_backend=self.backend_name,
            latency_ms=round(latency_ms, 3),
        )


class PredictorService:
    def __init__(self, backend: str | None = None):
        backend = backend or settings.serving_model_backend
        if backend == "baseline":
            self._impl = BaselinePredictor()
        elif backend == "transformer":
            self._impl = TransformerPredictor()
        else:
            raise ValueError(f"Unknown SERVING_MODEL_BACKEND: {backend!r}")
        logger.info("PredictorService initialized with backend=%s", backend)

    def predict(self, text: str, explain: bool = True) -> PredictionResult:
        text = _validate_text(text)
        return self._impl.predict(text, explain=explain)

    def predict_batch(self, texts: list[str], explain: bool = True) -> list[PredictionResult]:
        return [self.predict(text, explain=explain) for text in texts]


@lru_cache(maxsize=None)
def get_predictor(backend: str | None = None) -> PredictorService:
    """Process-wide singleton so FastAPI/Streamlit don't reload model
    artifacts on every request."""
    return PredictorService(backend=backend)

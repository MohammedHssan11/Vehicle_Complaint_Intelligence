"""Shared application service used by both the FastAPI routes and the
Streamlit app, so prediction + logging + example-lookup logic lives in one
place instead of being duplicated across the two front doors."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.config.settings import label_config, paths
from src.inference.predictor import PredictionResult, get_predictor
from src.monitoring.prediction_log import get_prediction_logger


class ClassificationService:
    def __init__(self, backend: str | None = None):
        self.predictor = get_predictor(backend)
        self.logger = get_prediction_logger()

    def classify(self, text: str, explain: bool = True) -> PredictionResult:
        result = self.predictor.predict(text, explain=explain)
        self.logger.log(
            text=text,
            predicted_label=result.predicted_label,
            confidence=result.confidence,
            model_backend=result.model_backend,
            latency_ms=result.latency_ms,
        )
        return result

    def classify_batch(self, texts: list[str], explain: bool = True) -> list[PredictionResult]:
        return [self.classify(text, explain=explain) for text in texts]

    def get_metrics(self) -> dict:
        return self.logger.get_metrics()

    def get_example_complaints(self, n: int = 10, seed: int = 42) -> pd.DataFrame:
        """Sample real complaints from the held-out test split (never used
        for training) so the "Examples" page can show predicted-vs-actual
        on genuine data rather than made-up text."""
        if not paths.test_parquet.exists():
            return pd.DataFrame(columns=["summary", label_config.PRIMARY_LABEL_COLUMN])
        test_df = pd.read_parquet(paths.test_parquet, columns=["summary", label_config.PRIMARY_LABEL_COLUMN])
        return test_df.sample(n=min(n, len(test_df)), random_state=seed).reset_index(drop=True)


@lru_cache(maxsize=None)
def get_classification_service(backend: str | None = None) -> ClassificationService:
    return ClassificationService(backend=backend)

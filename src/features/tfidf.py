"""TF-IDF featurization for the classical baseline models."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.settings import feature_config


class TfidfFeaturizer:
    def __init__(self, **overrides):
        params = dict(
            max_features=feature_config.TFIDF_MAX_FEATURES,
            ngram_range=feature_config.TFIDF_NGRAM_RANGE,
            min_df=feature_config.TFIDF_MIN_DF,
        )
        params.update(overrides)
        self.vectorizer = TfidfVectorizer(**params)

    def fit_transform(self, texts: pd.Series) -> spmatrix:
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: pd.Series) -> spmatrix:
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path: Path) -> "TfidfFeaturizer":
        obj = cls.__new__(cls)
        obj.vectorizer = joblib.load(path)
        return obj

"""TF-IDF featurization for the classical baseline models.

Two featurizers are available for the model bake-off in src/training/train.py:

- `TfidfFeaturizer`: word n-grams only (the original M1 baseline).
- `CombinedTfidfFeaturizer`: word n-grams + character n-grams. The text
  audit found ~46% of the vocabulary is hapax legomena, much of it
  misspellings ("burnig", "misalligned", "sewrious") — word-level features
  can't generalize across a typo and its correct spelling, but overlapping
  character n-grams ("burn", "urni", "rnig") partially can.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, spmatrix
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


class CombinedTfidfFeaturizer:
    """Word n-grams + character n-grams, hstacked into one sparse matrix.
    Column order (word features, then char features) is fixed and must
    match between fit_transform/transform and get_feature_names for
    coefficient-based explainability to line up correctly."""

    def __init__(self, **overrides):
        word_params = dict(
            max_features=feature_config.TFIDF_MAX_FEATURES,
            ngram_range=feature_config.TFIDF_NGRAM_RANGE,
            min_df=feature_config.TFIDF_MIN_DF,
        )
        char_params = dict(
            max_features=feature_config.CHAR_MAX_FEATURES,
            ngram_range=feature_config.CHAR_NGRAM_RANGE,
            min_df=feature_config.CHAR_MIN_DF,
            analyzer="char_wb",
        )
        word_params.update(overrides.get("word", {}))
        char_params.update(overrides.get("char", {}))

        self.word_vectorizer = TfidfVectorizer(**word_params)
        self.char_vectorizer = TfidfVectorizer(**char_params)

    def fit_transform(self, texts: pd.Series) -> spmatrix:
        word_X = self.word_vectorizer.fit_transform(texts)
        char_X = self.char_vectorizer.fit_transform(texts)
        return hstack([word_X, char_X], format="csr")

    def transform(self, texts: pd.Series) -> spmatrix:
        word_X = self.word_vectorizer.transform(texts)
        char_X = self.char_vectorizer.transform(texts)
        return hstack([word_X, char_X], format="csr")

    def get_feature_names(self):
        word_names = self.word_vectorizer.get_feature_names_out()
        char_names = [f"char:{n}" for n in self.char_vectorizer.get_feature_names_out()]
        return np.concatenate([word_names, char_names])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"word": self.word_vectorizer, "char": self.char_vectorizer}, path)

    @classmethod
    def load(cls, path: Path) -> "CombinedTfidfFeaturizer":
        obj = cls.__new__(cls)
        data = joblib.load(path)
        obj.word_vectorizer = data["word"]
        obj.char_vectorizer = data["char"]
        return obj


FEATURIZER_REGISTRY = {
    "word_tfidf": TfidfFeaturizer,
    "word_char_tfidf": CombinedTfidfFeaturizer,
}

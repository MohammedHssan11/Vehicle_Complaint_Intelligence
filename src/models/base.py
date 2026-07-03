"""Common interface all model backends (classical + transformer) implement,
so `src/inference` and `src/api` can swap backends via config without
touching calling code."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np


class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X, y) -> "BaseClassifier": ...

    @abstractmethod
    def predict(self, X) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray: ...

    @property
    @abstractmethod
    def classes_(self) -> np.ndarray: ...

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "BaseClassifier":
        return joblib.load(path)

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.baseline import LogisticRegressionClassifier, LinearSVCClassifier, MODEL_REGISTRY
from src.models.base import BaseClassifier
from src.models.xgboost_model import XGBoostClassifier


def _toy_data():
    # 3 classes, linearly separable, 3 examples each — LinearSVCClassifier's
    # default 3-fold probability calibration needs >=3 examples per class.
    X = csr_matrix(
        np.array(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 0.5],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 0, 0.5],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 0.5],
            ],
            dtype=float,
        )
    )
    y = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])
    return X, y


@pytest.mark.parametrize("cls", [LogisticRegressionClassifier, LinearSVCClassifier, XGBoostClassifier])
class TestBaselineModelsShared:
    def test_fit_predict_roundtrip(self, cls):
        X, y = _toy_data()
        model = cls()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds) <= {"A", "B", "C"}

    def test_predict_proba_shape_and_range(self, cls):
        X, y = _toy_data()
        model = cls()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_classes_property(self, cls):
        X, y = _toy_data()
        model = cls()
        model.fit(X, y)
        assert set(model.classes_) == {"A", "B", "C"}

    def test_save_and_load_roundtrip(self, cls, tmp_path):
        X, y = _toy_data()
        model = cls()
        model.fit(X, y)
        save_path = tmp_path / "model.joblib"
        model.save(save_path)

        loaded = BaseClassifier.load(save_path)
        np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


class TestModelRegistry:
    def test_registry_contains_expected_models(self):
        assert set(MODEL_REGISTRY.keys()) == {"logistic_regression", "linear_svc", "xgboost"}

    def test_registry_values_are_instantiable(self):
        for cls in MODEL_REGISTRY.values():
            instance = cls()
            assert isinstance(instance, BaseClassifier)

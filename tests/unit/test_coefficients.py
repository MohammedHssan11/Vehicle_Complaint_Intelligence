import numpy as np
import pytest

from src.explainability.coefficients import extract_class_coefficients


class _FakeLinearModel:
    """Mimics a plain sklearn linear model (e.g. LogisticRegression)."""

    def __init__(self, coef, intercept, classes):
        self.coef_ = coef
        self.intercept_ = intercept
        self.classes_ = classes


class _FakeBaseClassifierWrapper:
    """Mimics our BaseClassifier subclasses, which store the real sklearn
    model under `.model`."""

    def __init__(self, model):
        self.model = model


class _FakeFoldEstimator:
    def __init__(self, coef, intercept):
        self.coef_ = coef
        self.intercept_ = intercept


class _FakeCalibratedClassifier:
    """Mimics sklearn's CalibratedClassifierCV after fitting."""

    def __init__(self, fold_coefs, fold_intercepts, classes):
        self.calibrated_classifiers_ = [
            type("Fold", (), {"estimator": _FakeFoldEstimator(c, i)})()
            for c, i in zip(fold_coefs, fold_intercepts)
        ]
        self.classes_ = classes


class TestExtractClassCoefficients:
    def test_plain_linear_model_direct_attributes(self):
        coef = np.array([[1.0, 2.0], [3.0, 4.0]])
        intercept = np.array([0.1, 0.2])
        classes = np.array(["A", "B"])
        model = _FakeBaseClassifierWrapper(_FakeLinearModel(coef, intercept, classes))

        result_coef, result_intercept, result_classes = extract_class_coefficients(model)

        np.testing.assert_array_equal(result_coef, coef)
        np.testing.assert_array_equal(result_intercept, intercept)
        np.testing.assert_array_equal(result_classes, classes)

    def test_calibrated_classifier_averages_fold_coefficients(self):
        fold_coefs = [np.array([[1.0, 1.0]]), np.array([[3.0, 3.0]])]
        fold_intercepts = [np.array([0.0]), np.array([2.0])]
        classes = np.array(["A"])
        calibrated = _FakeCalibratedClassifier(fold_coefs, fold_intercepts, classes)
        model = _FakeBaseClassifierWrapper(calibrated)

        result_coef, result_intercept, result_classes = extract_class_coefficients(model)

        np.testing.assert_array_almost_equal(result_coef, [[2.0, 2.0]])
        np.testing.assert_array_almost_equal(result_intercept, [1.0])
        np.testing.assert_array_equal(result_classes, classes)

    def test_unknown_model_type_raises(self):
        model = _FakeBaseClassifierWrapper(object())
        with pytest.raises(TypeError):
            extract_class_coefficients(model)

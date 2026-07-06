import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_metrics,
    get_confusion_matrix_df,
    lenient_accuracy,
    majority_class_baseline_accuracy,
)


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = ["A", "B", "A", "B"]
        y_pred = ["A", "B", "A", "B"]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = ["A", "A"]
        y_pred = ["B", "B"]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_returns_expected_keys(self):
        metrics = compute_metrics(["A", "B"], ["A", "A"])
        assert set(metrics.keys()) == {
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_precision",
            "macro_recall",
        }


class TestMajorityClassBaseline:
    def test_computes_correct_ratio(self):
        y_true = ["A"] * 7 + ["B"] * 3
        acc = majority_class_baseline_accuracy(y_true)
        assert acc == 0.7

    def test_uniform_distribution(self):
        y_true = ["A", "B", "C", "D"]
        acc = majority_class_baseline_accuracy(y_true)
        assert acc == 0.25


class TestLenientAccuracy:
    def test_credits_secondary_label_match(self):
        components = pd.Series(["POWER TRAIN,ENGINE", "SEAT BELTS"])
        y_pred = ["ENGINE", "SEAT BELTS"]  # first is a secondary-label match, second is exact
        assert lenient_accuracy(components, y_pred) == 1.0

    def test_does_not_credit_unlisted_prediction(self):
        components = pd.Series(["POWER TRAIN,ENGINE"])
        y_pred = ["AIR BAGS"]
        assert lenient_accuracy(components, y_pred) == 0.0

    def test_matches_primary_label_alone(self):
        components = pd.Series(["ENGINE"])
        y_pred = ["ENGINE"]
        assert lenient_accuracy(components, y_pred) == 1.0

    def test_strips_whitespace_around_components(self):
        components = pd.Series(["SERVICE BRAKES, HYDRAULIC"])
        y_pred = ["HYDRAULIC"]
        assert lenient_accuracy(components, y_pred) == 1.0

    def test_mixed_batch_ratio(self):
        components = pd.Series(["ENGINE,POWER TRAIN", "SEAT BELTS", "AIR BAGS"])
        y_pred = ["POWER TRAIN", "SEAT BELTS", "STEERING"]
        assert lenient_accuracy(components, y_pred) == pytest.approx(2 / 3)


class TestConfusionMatrix:
    def test_shape_matches_labels(self):
        labels = ["A", "B", "C"]
        cm = get_confusion_matrix_df(["A", "B"], ["A", "C"], labels=labels)
        assert cm.shape == (3, 3)
        assert list(cm.index) == labels
        assert list(cm.columns) == labels

    def test_diagonal_counts_correct_predictions(self):
        cm = get_confusion_matrix_df(["A", "A", "B"], ["A", "A", "B"], labels=["A", "B"])
        assert cm.loc["A", "A"] == 2
        assert cm.loc["B", "B"] == 1

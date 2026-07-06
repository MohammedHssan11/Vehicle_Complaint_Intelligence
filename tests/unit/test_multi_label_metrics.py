import numpy as np
import pandas as pd
import pytest

from src.evaluation.multi_label_metrics import multi_label_metrics_at_k, parse_component_set


class TestParseComponentSet:
    def test_single_label(self):
        assert parse_component_set("ENGINE") == {"ENGINE"}

    def test_multi_label_strips_whitespace(self):
        assert parse_component_set("ENGINE, POWER TRAIN") == {"ENGINE", "POWER TRAIN"}


class TestMultiLabelMetricsAtK:
    def _make(self, components, proba, classes, k):
        return multi_label_metrics_at_k(pd.Series(components), np.array(proba), np.array(classes), k)

    def test_perfect_single_label_coverage(self):
        # top-1 prediction exactly matches the (single) true label for every row.
        classes = ["A", "B"]
        components = ["A", "B"]
        proba = [[0.9, 0.1], [0.1, 0.9]]
        result = self._make(components, proba, classes, k=1)
        assert result["recall_at_k"] == 1.0
        assert result["full_coverage_at_k"] == 1.0
        assert result["precision_at_k"] == 1.0

    def test_multi_label_row_partial_recall_at_k1(self):
        # true set has 2 labels, only top-1 prediction available -> half credit.
        classes = ["A", "B", "C"]
        components = ["A,B"]
        proba = [[0.7, 0.2, 0.1]]  # top-1 = A
        result = self._make(components, proba, classes, k=1)
        assert result["recall_at_k"] == pytest.approx(0.5)
        assert result["full_coverage_at_k"] == 0.0
        assert result["precision_at_k"] == pytest.approx(1.0)

    def test_multi_label_row_full_coverage_at_k2(self):
        classes = ["A", "B", "C"]
        components = ["A,B"]
        proba = [[0.6, 0.3, 0.1]]  # top-2 = A, B
        result = self._make(components, proba, classes, k=2)
        assert result["recall_at_k"] == 1.0
        assert result["full_coverage_at_k"] == 1.0
        assert result["precision_at_k"] == pytest.approx(1.0)

    def test_no_overlap_scores_zero(self):
        classes = ["A", "B", "C"]
        components = ["C"]
        proba = [[0.6, 0.3, 0.1]]  # top-1 = A, true = C
        result = self._make(components, proba, classes, k=1)
        assert result["recall_at_k"] == 0.0
        assert result["full_coverage_at_k"] == 0.0
        assert result["precision_at_k"] == 0.0

    def test_precision_drops_with_larger_k_same_true_set_size(self):
        classes = ["A", "B", "C", "D"]
        components = ["A"]
        proba = [[0.4, 0.3, 0.2, 0.1]]
        result_k1 = self._make(components, proba, classes, k=1)
        result_k3 = self._make(components, proba, classes, k=3)
        assert result_k1["precision_at_k"] == pytest.approx(1.0)
        assert result_k3["precision_at_k"] == pytest.approx(1 / 3)
        # recall/full_coverage unaffected since the true label is already captured at k=1
        assert result_k1["recall_at_k"] == result_k3["recall_at_k"] == 1.0

    def test_row_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            multi_label_metrics_at_k(pd.Series(["A", "B"]), np.array([[0.5, 0.5]]), np.array(["A", "B"]), k=1)

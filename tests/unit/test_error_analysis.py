import pandas as pd

from src.evaluation.error_analysis import (
    false_negative_report,
    false_positive_report,
    most_confused_pairs,
    sample_errors_for_pair,
)

TEXTS = pd.Series(["t1", "t2", "t3", "t4", "t5"])
Y_TRUE = pd.Series(["A", "A", "B", "B", "C"])
Y_PRED = ["B", "B", "A", "B", "C"]  # A->B twice, B->A once, B->B once, C correct


class TestMostConfusedPairs:
    def test_counts_only_errors(self):
        result = most_confused_pairs(TEXTS, Y_TRUE, Y_PRED, top_n=10)
        assert (result["true"] != result["pred"]).all()

    def test_most_frequent_pair_first(self):
        result = most_confused_pairs(TEXTS, Y_TRUE, Y_PRED, top_n=10)
        top_row = result.iloc[0]
        assert top_row["true"] == "A"
        assert top_row["pred"] == "B"
        assert top_row["count"] == 2

    def test_respects_top_n(self):
        result = most_confused_pairs(TEXTS, Y_TRUE, Y_PRED, top_n=1)
        assert len(result) == 1

    def test_works_with_raw_numpy_array_predictions(self):
        import numpy as np

        result = most_confused_pairs(TEXTS, Y_TRUE, np.array(Y_PRED), top_n=10)
        assert len(result) > 0


class TestSampleErrorsForPair:
    def test_filters_to_exact_pair(self):
        result = sample_errors_for_pair(TEXTS, Y_TRUE, Y_PRED, true_label="A", pred_label="B", n=10)
        assert len(result) == 2
        assert (result["true"] == "A").all()
        assert (result["pred"] == "B").all()


class TestFalseNegativeReport:
    def test_finds_missed_true_label(self):
        result = false_negative_report(TEXTS, Y_TRUE, Y_PRED, label="A", n=10)
        assert len(result) == 2
        assert (result["true"] == "A").all()
        assert (result["pred"] != "A").all()

    def test_correct_label_has_no_false_negatives(self):
        result = false_negative_report(TEXTS, Y_TRUE, Y_PRED, label="C", n=10)
        assert len(result) == 0


class TestFalsePositiveReport:
    def test_finds_incorrectly_predicted_label(self):
        result = false_positive_report(TEXTS, Y_TRUE, Y_PRED, label="B", n=10)
        assert len(result) == 2
        assert (result["pred"] == "B").all()
        assert (result["true"] != "B").all()

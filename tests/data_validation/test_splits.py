import pandas as pd
import pytest

from src.data.splits import assert_no_text_leakage, stratified_split, time_based_split


def _labeled_df(n_per_class: int = 40) -> pd.DataFrame:
    rows = []
    for cls in ["A", "B", "C"]:
        for i in range(n_per_class):
            rows.append({"summary": f"{cls} complaint text number {i}", "primary_component": cls})
    return pd.DataFrame(rows)


class TestStratifiedSplit:
    def test_no_text_overlap_between_splits(self):
        df = _labeled_df()
        train, val, test = stratified_split(df, test_size=0.15, val_size=0.15, random_state=42)
        assert_no_text_leakage(train, test)
        assert_no_text_leakage(train, val)
        assert_no_text_leakage(val, test)

    def test_split_sizes_roughly_correct(self):
        df = _labeled_df(n_per_class=100)  # 300 rows total
        train, val, test = stratified_split(df, test_size=0.15, val_size=0.15, random_state=42)
        assert len(train) + len(val) + len(test) == len(df)
        assert abs(len(test) - 45) <= 2  # 15% of 300
        assert abs(len(val) - 45) <= 2

    def test_stratification_preserves_class_balance(self):
        df = _labeled_df(n_per_class=100)
        train, val, test = stratified_split(df, test_size=0.15, val_size=0.15, random_state=42)
        for split in (train, val, test):
            counts = split["primary_component"].value_counts()
            assert set(counts.index) == {"A", "B", "C"}
            # roughly balanced since source classes are equal-sized
            assert counts.max() - counts.min() <= 3


class TestAssertNoTextLeakage:
    def test_raises_on_overlap(self):
        train = pd.DataFrame({"summary": ["same text", "unique train text"]})
        test = pd.DataFrame({"summary": ["same text", "unique test text"]})
        with pytest.raises(ValueError, match="contamination"):
            assert_no_text_leakage(train, test)

    def test_passes_when_disjoint(self):
        train = pd.DataFrame({"summary": ["train text a", "train text b"]})
        test = pd.DataFrame({"summary": ["test text a", "test text b"]})
        assert_no_text_leakage(train, test)  # should not raise


class TestTimeBasedSplit:
    def test_partitions_by_year_correctly(self):
        df = pd.DataFrame(
            {
                "dateComplaintFiled": pd.to_datetime(
                    ["2020-01-01", "2023-06-15", "2024-01-01", "2025-12-31"]
                ),
                "summary": ["a", "b", "c", "d"],
            }
        )
        train, test = time_based_split(df, train_max_year=2023)
        assert train["summary"].tolist() == ["a", "b"]
        assert test["summary"].tolist() == ["c", "d"]

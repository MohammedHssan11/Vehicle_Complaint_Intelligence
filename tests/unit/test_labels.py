import pandas as pd

from src.data.labels import (
    build_label_taxonomy,
    collapse_to_taxonomy,
    engineer_labels,
    extract_primary_component,
)


class TestExtractPrimaryComponent:
    def test_single_component_unchanged(self):
        result = extract_primary_component(pd.Series(["ENGINE"]))
        assert result.tolist() == ["ENGINE"]

    def test_compound_takes_first_tag(self):
        result = extract_primary_component(pd.Series(["POWER TRAIN,ENGINE"]))
        assert result.tolist() == ["POWER TRAIN"]

    def test_strips_whitespace(self):
        result = extract_primary_component(pd.Series(["POWER TRAIN, ENGINE"]))
        assert result.tolist() == ["POWER TRAIN"]


class TestBuildLabelTaxonomy:
    def test_keeps_classes_above_threshold(self):
        primary = pd.Series(["A"] * 10 + ["B"] * 5 + ["C"] * 1)
        kept = build_label_taxonomy(primary, min_samples=5)
        assert kept == {"A", "B"}

    def test_empty_when_all_below_threshold(self):
        primary = pd.Series(["A"] * 2 + ["B"] * 1)
        kept = build_label_taxonomy(primary, min_samples=5)
        assert kept == set()


class TestCollapseToTaxonomy:
    def test_kept_labels_unchanged(self):
        primary = pd.Series(["A", "B"])
        result = collapse_to_taxonomy(primary, kept_labels={"A", "B"})
        assert result.tolist() == ["A", "B"]

    def test_unkept_labels_become_other(self):
        primary = pd.Series(["A", "RARE_TAG"])
        result = collapse_to_taxonomy(primary, kept_labels={"A"})
        assert result.tolist() == ["A", "OTHER"]


class TestEngineerLabels:
    def test_fits_taxonomy_when_none_given(self):
        df = pd.DataFrame({"components": ["A"] * 5 + ["RARE"] * 1})
        result_df, kept = engineer_labels(df.assign(components=df["components"]), kept_labels=None)
        # with default MIN_SAMPLES_PER_CLASS=200, nothing survives -> all OTHER
        assert set(result_df["primary_component"].unique()) <= {"OTHER"} | kept

    def test_unseen_tag_at_inference_time_collapses_to_other(self):
        df = pd.DataFrame({"components": ["NEVER_SEEN_BEFORE"]})
        result_df, kept = engineer_labels(df, kept_labels={"ENGINE", "STEERING"})
        assert result_df["primary_component"].tolist() == ["OTHER"]

    def test_known_tag_at_inference_time_preserved(self):
        df = pd.DataFrame({"components": ["ENGINE"]})
        result_df, kept = engineer_labels(df, kept_labels={"ENGINE", "STEERING"})
        assert result_df["primary_component"].tolist() == ["ENGINE"]

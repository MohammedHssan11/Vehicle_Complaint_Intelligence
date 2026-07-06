import pandas as pd

from src.data.cleaning import MIN_SUMMARY_CHARS, clean_complaints


def _base_df(**overrides) -> pd.DataFrame:
    data = {
        "odiNumber": [1, 2, 3],
        "summary": ["A valid complaint description here.", "Also valid complaint text.", "Third one."],
        "components": ["ENGINE", "STEERING", "BRAKES"],
        "vin": ["VIN1", "VIN2", "VIN3"],
        "products": ["{}", "{}", "{}"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestCleanComplaints:
    def test_drops_null_summary(self):
        df = _base_df(summary=["Valid complaint text here.", None, "Also valid text."])
        result = clean_complaints(df)
        assert result["summary"].notna().all()
        assert len(result) == 2

    def test_drops_too_short_summary(self):
        short_text = "hi"
        assert len(short_text) < MIN_SUMMARY_CHARS
        df = _base_df(summary=["Valid complaint text here.", short_text, "Also valid text."])
        result = clean_complaints(df)
        assert len(result) == 2

    def test_drops_vin_and_products_columns(self):
        df = _base_df()
        result = clean_complaints(df)
        assert "vin" not in result.columns
        assert "products" not in result.columns

    def test_dedupes_on_summary_text_alone_even_with_different_labels(self):
        """Regression test: dedup must key on summary text alone, not
        (summary, components) — identical text with a different label was
        found in the real data and would otherwise leak across splits."""
        df = _base_df(
            summary=["Identical complaint text here.", "Identical complaint text here.", "Unique third one."],
            components=["ENGINE", "STEERING", "BRAKES"],
        )
        result = clean_complaints(df)
        assert len(result) == 2
        assert result["summary"].duplicated().sum() == 0

    def test_drops_null_components(self):
        df = _base_df(components=["ENGINE", None, "BRAKES"])
        result = clean_complaints(df)
        assert len(result) == 2

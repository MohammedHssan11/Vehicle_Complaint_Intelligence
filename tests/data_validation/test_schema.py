import pandas as pd
import pytest

from src.data.schema import REQUIRED_COLUMNS, DataValidationError, validate_complaints_schema


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame({col: ["x"] for col in REQUIRED_COLUMNS} | {"odiNumber": [1]})


class TestValidateComplaintsSchema:
    def test_passes_on_well_formed_dataframe(self):
        df = _valid_df()
        validate_complaints_schema(df)  # should not raise

    def test_raises_on_missing_column(self):
        df = _valid_df().drop(columns=["summary"])
        with pytest.raises(DataValidationError, match="missing required columns"):
            validate_complaints_schema(df)

    def test_raises_on_empty_dataframe(self):
        df = _valid_df().iloc[0:0]
        with pytest.raises(DataValidationError, match="zero rows"):
            validate_complaints_schema(df)

    def test_raises_on_duplicate_odi_number(self):
        df = pd.concat([_valid_df(), _valid_df()], ignore_index=True)
        with pytest.raises(DataValidationError, match="unique primary key"):
            validate_complaints_schema(df)

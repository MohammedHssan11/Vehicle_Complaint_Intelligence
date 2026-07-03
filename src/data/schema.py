"""Schema contract for the raw NHTSA complaints export.

The original notebook read this file with `engine='python', on_bad_lines='warn'`,
suggesting the author suspected malformed rows. Re-validated during the audit:
the current export parses cleanly with the fast C engine and has no bad lines.
We still validate the schema explicitly at load time so a future re-export
with a changed/missing column fails loudly instead of silently corrupting
downstream training.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "odiNumber",
    "manufacturer",
    "crash",
    "fire",
    "numberOfInjuries",
    "numberOfDeaths",
    "dateOfIncident",
    "dateComplaintFiled",
    "vin",
    "components",
    "summary",
    "products",
    "make",
    "model",
    "modelYear",
}


class DataValidationError(Exception):
    """Raised when a raw dataset does not match the expected schema."""


def validate_complaints_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            f"complaints.csv is missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise DataValidationError("complaints.csv loaded with zero rows")

    if df["odiNumber"].duplicated().any():
        n_dupes = int(df["odiNumber"].duplicated().sum())
        raise DataValidationError(
            f"odiNumber is expected to be a unique primary key, found {n_dupes} duplicates"
        )

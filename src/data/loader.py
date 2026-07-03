"""Raw data loading for the vehicle complaint classifier."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import paths
from src.data.schema import validate_complaints_schema

# Columns actually needed downstream. odiNumber/vin/products are read then
# dropped early (vin is a unique ID with no predictive value; products is a
# redundant JSON encoding of make/model/manufacturer already present as
# plain columns), but odiNumber is kept as the row identifier for traceability
# and dedup diagnostics.
_DTYPES = {
    "odiNumber": "int64",
    "manufacturer": "string",
    "crash": "boolean",
    "fire": "boolean",
    "numberOfInjuries": "Int64",
    "numberOfDeaths": "Int64",
    "vin": "string",
    "components": "string",
    "summary": "string",
    "products": "string",
    "make": "string",
    "model": "string",
}


def load_complaints(path: Path | None = None, validate: bool = True) -> pd.DataFrame:
    """Load complaints.csv with the fast C engine and explicit dtypes.

    The original notebook used `engine='python', on_bad_lines='warn'` out of
    caution. That engine is ~10-20x slower and is unnecessary here: the file
    parses cleanly with the C engine (verified during the audit).
    """
    csv_path = path or paths.complaints_csv
    df = pd.read_csv(
        csv_path,
        engine="c",
        dtype=_DTYPES,
        parse_dates=["dateOfIncident", "dateComplaintFiled"],
        date_format="%m/%d/%Y",
    )
    # modelYear has 1 null in the full export; keep as nullable Int64 rather
    # than forcing float64 (pandas default for int-with-nulls).
    df["modelYear"] = df["modelYear"].astype("Int64")

    if validate:
        validate_complaints_schema(df)

    return df

"""Row/column-level cleaning for the raw complaints export.

Findings this addresses (from the data audit):
- 13 null `summary` rows, 314 rows with summary <10 chars carry no signal.
- `vin` is a unique identifier with no predictive value (already correctly
  excluded in the original notebook) and is 2% null / only partial VINs anyway.
- `products` is a JSON-encoded duplicate of make/model/manufacturer, already
  available as plain columns — parsing it buys nothing.
- 2,272 duplicate (summary, components) pairs exist, and a further check
  found the exact same summary text occasionally attached to *different*
  components values (same boilerplate text submitted/coded differently).
  Deduplication therefore keys on `summary` alone, not the (summary,
  components) pair: two rows with identical text can never safely be split
  across train/test regardless of label, since the model would be tested on
  text it (or its label-mate) has already memorized. Left undeduplicated,
  this causes train/test contamination — fixed here, once, before any split.
"""
from __future__ import annotations

import pandas as pd

MIN_SUMMARY_CHARS = 10
COLUMNS_TO_DROP = ("vin", "products")


def clean_complaints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df["summary"].notna()]
    df = df[df["summary"].str.strip().str.len() >= MIN_SUMMARY_CHARS]
    df = df[df["components"].notna()]

    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

    before = len(df)
    df = df.drop_duplicates(subset=["summary"], keep="first")
    n_deduped = before - len(df)
    if n_deduped:
        import logging

        logging.getLogger(__name__).info(
            "Dropped %d duplicate (summary, components) rows", n_deduped
        )

    return df.reset_index(drop=True)

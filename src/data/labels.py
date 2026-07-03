"""Label engineering — Phase 3 problem-formulation decision.

The raw `components` field is comma-joined and can list multiple fault
systems per complaint (32.7% of rows do). The original notebook kept these
compound strings as their own classes (2,341 raw / 21 arbitrarily grouped),
producing near-duplicate classes with no shared signal separation — its own
SVM baseline scored F1=0.00 on several of them.

Confirmed v1 formulation: single-label multiclass on the *primary*
(first-listed) component. Splitting on comma and taking the first tag gives
45 raw classes; classes below LabelConfig.MIN_SAMPLES_PER_CLASS are folded
into OTHER (see settings.py docstring for why 200 was chosen).
"""
from __future__ import annotations

import pandas as pd

from src.config.settings import label_config


def extract_primary_component(components: pd.Series) -> pd.Series:
    return components.str.split(",").str[0].str.strip()


def build_label_taxonomy(primary: pd.Series, min_samples: int | None = None) -> set[str]:
    """Return the set of primary-component tags kept as their own class.
    Everything else collapses into OTHER."""
    min_samples = min_samples if min_samples is not None else label_config.MIN_SAMPLES_PER_CLASS
    counts = primary.value_counts()
    return set(counts[counts >= min_samples].index)


def collapse_to_taxonomy(primary: pd.Series, kept_labels: set[str]) -> pd.Series:
    return primary.where(primary.isin(kept_labels), other=label_config.OTHER_LABEL)


def engineer_labels(df: pd.DataFrame, kept_labels: set[str] | None = None) -> tuple[pd.DataFrame, set[str]]:
    """Add `primary_component` (collapsed taxonomy) to df.

    The taxonomy (which raw tags are frequent enough to be their own class)
    is a fixed label-space decision, not a fitted statistical estimator — it
    doesn't use any row's true label to predict that same row, so computing
    it on the full cleaned dataset before splitting is intentional (splitting
    needs the final label column to stratify by). Pass an explicit
    `kept_labels` set when applying the same taxonomy at inference time so
    new/unseen tags collapse to OTHER instead of erroring.
    """
    df = df.copy()
    primary = extract_primary_component(df[label_config.TARGET_COLUMN])

    if kept_labels is None:
        kept_labels = build_label_taxonomy(primary)

    df[label_config.PRIMARY_LABEL_COLUMN] = collapse_to_taxonomy(primary, kept_labels)
    return df, kept_labels

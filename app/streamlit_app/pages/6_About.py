from __future__ import annotations

import streamlit as st

from theme import hero_header, inject_global_css

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
inject_global_css()
hero_header("About", "The full story — data, redesign decisions, and architecture.")

st.markdown(
    """
### Data

271,517 real complaint records from the NHTSA public complaints database
(2016-2026 model years), joined against auxiliary NHTSA investigations,
recalls, and crash-test rating datasets (not currently used for modeling —
see `docs/Architecture.md` for the documented leakage risk in naively
joining recall/investigation data without a temporal cutoff).

### Key redesign decisions

- **Label formulation**: the raw `components` field is comma-joined and can
  list multiple fault systems per complaint. Treating each unique combination
  as its own class (the original approach) produced near-duplicate classes
  with unrecoverable signal separation. This system instead classifies on the
  **primary (first-listed) component**, collapsed to ~27 classes with enough
  support for a stable macro-F1, with rare tags folded into `OTHER`.
- **Negation-safe preprocessing**: contractions are expanded *before*
  punctuation stripping, so `"doesn't"` becomes `"does not"` instead of the
  unmatched `"doesnt"` — negation is exactly the signal that distinguishes
  "brake did **not** engage" from "brake engaged."
- **Leakage guards**: deduplication on complaint text (not just
  text+label) before splitting, plus an explicit assertion that no text
  appears in more than one split.
- **Class imbalance** handled via `class_weight="balanced"`, not row
  duplication — duplicating text before a split risks train/test
  contamination for no generalization benefit.

### Architecture

```
data/raw -> src/data (validate, clean, label) -> src/preprocessing
    -> src/features (TF-IDF) -> src/models (baseline + transformer)
    -> src/training (MLflow-tracked) -> src/evaluation
    -> src/inference (config-swappable backend) -> src/explainability (SHAP)
    -> src/api (FastAPI) / app/streamlit_app (this app)
```

### Tech stack

scikit-learn, HuggingFace Transformers, MLflow, SHAP, FastAPI, Streamlit, Docker.

See `docs/` for the full audit report, architecture writeup, and model
benchmark comparison.
"""
)

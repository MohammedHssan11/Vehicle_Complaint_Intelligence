from __future__ import annotations

import streamlit as st

from theme import hero_header, inject_global_css
from utils import get_service

st.set_page_config(page_title="Examples", page_icon="🧪", layout="wide")
inject_global_css()
hero_header(
    "Examples",
    "Real complaints sampled from the held-out test split (never used for training), "
    "with the model's prediction shown next to the true label.",
)

n = st.slider("Number of examples", min_value=5, max_value=50, value=10)
seed = st.number_input("Random seed", value=42, step=1)

if st.button("Sample new examples", type="primary"):
    st.session_state["examples_seed"] = seed
    st.session_state["examples_n"] = n

seed = st.session_state.get("examples_seed", 42)
n = st.session_state.get("examples_n", 10)

service = get_service("baseline")
examples_df = service.get_example_complaints(n=n, seed=seed)

if examples_df.empty:
    st.warning("No processed test split found — run `python -m src.data.pipeline` first.")
else:
    results = service.classify_batch(examples_df["summary"].tolist(), explain=False)
    examples_df = examples_df.copy()
    examples_df["predicted_label"] = [r.predicted_label for r in results]
    examples_df["confidence"] = [r.confidence for r in results]
    examples_df["correct"] = examples_df["predicted_label"] == examples_df["primary_component"]

    accuracy = examples_df["correct"].mean()
    st.metric("Accuracy on this sample", f"{accuracy:.0%}")

    st.dataframe(
        examples_df.rename(columns={"summary": "complaint_text", "primary_component": "true_label"})[
            ["complaint_text", "true_label", "predicted_label", "confidence", "correct"]
        ],
        width='stretch',
        hide_index=True,
    )

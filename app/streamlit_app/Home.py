"""Dashboard — landing page for the Streamlit app.
Run with: `streamlit run app/streamlit_app/Home.py`
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import load_production_metadata, load_transformer_metadata, transformer_display_name

st.set_page_config(page_title="Vehicle Complaint Classifier", page_icon="🚗", layout="wide")

st.title("🚗 Vehicle Complaint Classification")
st.caption("NLP system for classifying NHTSA vehicle complaints into fault categories")

metadata = load_production_metadata()

if metadata is None:
    st.error(
        "No production model found at `models/production/`. Run "
        "`python -m src.data.pipeline` then `python -m src.training.train` first."
    )
    st.stop()

test_metrics = metadata["test_metrics"]
classes = metadata["classes"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Macro-F1", f"{test_metrics['macro_f1']:.3f}")
col2.metric("Test Accuracy", f"{test_metrics['accuracy']:.3f}")
col3.metric("Fault Categories", len(classes))
col4.metric("Model", metadata["model_type"])

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("What this system does")
    st.markdown(
        """
        Predicts the NHTSA fault-category taxonomy (e.g. `ENGINE`, `SERVICE BRAKES`,
        `AIR BAGS`) from a free-text vehicle complaint description, with:

        - **Confidence + top-3 alternatives** for every prediction
        - **Word-level explanations** (SHAP) showing which terms drove the prediction
        - **Batch prediction** for CSV uploads
        - **Config-swappable backend** — TF-IDF+LinearSVC (fast, production) or
          a fine-tuned transformer (see the Model Metrics page for the honest
          benchmark comparison)

        Trained on 271,517 real NHTSA complaint records. See the **About** page
        for the full data audit and architecture writeup.
        """
    )

with right:
    st.subheader("Fault category coverage")
    class_df = pd.DataFrame({"category": classes})
    st.dataframe(class_df, width='stretch', height=300, hide_index=True)

transformer_metadata = load_transformer_metadata()
if transformer_metadata:
    st.divider()
    st.subheader("Available model backends")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("**Baseline (production)**")
        st.write(f"Macro-F1: {test_metrics['macro_f1']:.3f} | Accuracy: {test_metrics['accuracy']:.3f}")
    with bcol2:
        st.markdown(f"**{transformer_display_name(transformer_metadata)} (benchmark)**")
        t_metrics = transformer_metadata["test_metrics"]
        st.write(f"Macro-F1: {t_metrics['macro_f1']:.3f} | Accuracy: {t_metrics['accuracy']:.3f}")
        if transformer_metadata.get("is_full_dataset_run"):
            st.caption("Trained on the full dataset on GPU — see Model Metrics page for details")
        else:
            st.caption("Trained on a data subsample — see Model Metrics page for details")

st.divider()
st.page_link("pages/1_Single_Prediction.py", label="Try a single prediction", icon="🔍")
st.page_link("pages/2_Compare_Backends.py", label="Compare baseline vs. transformer", icon="⚖️")
st.page_link("pages/3_Batch_Prediction.py", label="Run batch predictions on a CSV", icon="📄")

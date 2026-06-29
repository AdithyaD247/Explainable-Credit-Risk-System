import shap
import streamlit as st

from utils.rag_utils import (
    load_retriever
)

from utils.model_utils import (
    load_dataset_model,
    load_manual_model,
    load_manual_features,
    load_data
)

from utils.session_state import initialize_session_state

from pages.dataset_applicant import show_dataset_page

from pages.manual_entry import show_manual_page


# =====================================
# Streamlit Configuration
# =====================================

st.set_page_config(
    page_title="Explainable Credit Risk System",
    page_icon="📊",
    layout="wide"
)

# =====================================
# Load Models and Data
# =====================================

dataset_model = load_dataset_model()

manual_model = load_manual_model()

manual_features = load_manual_features()

df = load_data()

@st.cache_resource
def load_explainers(dataset_model, manual_model):

    return (
        shap.TreeExplainer(dataset_model),
        shap.TreeExplainer(manual_model)
    )

dataset_explainer, manual_explainer = load_explainers(
    dataset_model,
    manual_model
)

# =====================================
# Initialize Application
# =====================================

initialize_session_state()

retriever = load_retriever()

# =====================================
# Main Interface
# =====================================

st.title("📊 Explainable Credit Risk System")

st.divider()

input_mode = st.radio(
    "Input Mode",
    [
        "Dataset Applicant",
        "Manual Entry"
    ],
    horizontal=True
)

st.divider()

st.caption("✅ Models and explainability components loaded successfully.")

st.divider()

if input_mode == "Dataset Applicant":

    show_dataset_page(
        df,
        dataset_model,
        dataset_explainer,
        retriever
    )

else:

    show_manual_page(
        manual_model,
        manual_explainer,
        manual_features,
        retriever
    )
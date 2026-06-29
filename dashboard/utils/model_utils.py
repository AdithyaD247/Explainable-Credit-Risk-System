import joblib
import pandas as pd
import streamlit as st

# =============================
# Dataset Model
# =============================
@st.cache_resource
def load_dataset_model():
    return joblib.load(
        r"C:\Users\Asus\OneDrive\Desktop\Explainable-Credit-Risk-System\models\best_xgb_model.joblib"
    )

# =============================
# Manual Model
# =============================
@st.cache_resource
def load_manual_model():
    return joblib.load(
        r"C:\Users\Asus\OneDrive\Desktop\Explainable-Credit-Risk-System\models\manual_entry_xgb_model.joblib"
    )

# =============================
# Manual Features
# =============================
@st.cache_resource
def load_manual_features():
    return joblib.load(
        r"C:\Users\Asus\OneDrive\Desktop\Explainable-Credit-Risk-System\models\manual_entry_features.joblib"
    )

# =============================
# Dataset
# =============================
@st.cache_data
def load_data():
    return pd.read_csv(
        r"C:\Users\Asus\OneDrive\Desktop\Explainable-Credit-Risk-System\data\processed\preprocessed_v2_encoded.csv"
    )


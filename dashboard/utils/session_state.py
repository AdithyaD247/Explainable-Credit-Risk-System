import streamlit as st


def initialize_session_state():

    defaults = {

        # Dataset Applicant
        "dataset_report": None,
        "dataset_sample": None,
        "dataset_probability": None,
        "dataset_risk_level": None,
        "dataset_top_features": None,
        "last_dataset_applicant": None,

        # Manual Entry
        "manual_prediction": None,
        "manual_sample": None,
        "manual_probability": None,
        "manual_risk_level": None,
        "manual_top_features": None,
        "manual_report": None,

    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value
import streamlit as st

from utils.predict_utils import predict_credit_risk

from utils.ui_utils import (
    display_prediction_summary,
    display_applicant_information,
    display_explainability,
    display_ai_report
)

from utils.rag_utils import generate_credit_risk_report

def show_dataset_page(
    df,
    dataset_model,
    dataset_explainer,
    retriever
):

    # =============================
    # Applicant Selection
    # =============================

    st.subheader("Select Applicant")

    applicant_index = st.number_input(
        "Applicant Index",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    if st.session_state.last_dataset_applicant != applicant_index:

        st.session_state.last_dataset_applicant = applicant_index

        st.session_state.dataset_report = None
        st.session_state.dataset_sample = None
        st.session_state.dataset_probability = None
        st.session_state.dataset_risk_level = None
        st.session_state.dataset_top_features = None

    st.info(
        f"Selected Applicant Index: {applicant_index}"
    )

    st.divider()

    # =============================
    # Prediction
    # =============================

    sample = df.drop(
        "TARGET",
        axis=1
    ).iloc[[int(applicant_index)]]

    (
        probability,
        risk_level,
        top_features
    ) = predict_credit_risk(
        dataset_model,
        dataset_explainer,
        sample
    )

    display_prediction_summary(
        probability,
        risk_level
    )

    display_applicant_information(sample)

    display_explainability(top_features)

    # =============================
    # AI Report
    # =============================

    if st.button("Generate Credit Risk Report"):

        with st.spinner("Generating AI credit risk assessment..."):

            report = generate_credit_risk_report(
                retriever=retriever,
                sample=sample,
                probability=probability,
                risk_level=risk_level,
                top_features=top_features
            )

            st.session_state.dataset_report = report
            st.session_state.dataset_sample = sample
            st.session_state.dataset_probability = probability
            st.session_state.dataset_risk_level = risk_level
            st.session_state.dataset_top_features = top_features

        st.divider()

    if st.session_state.dataset_report:

        display_ai_report(
            sample=st.session_state.dataset_sample,
            probability=st.session_state.dataset_probability,
            risk_level=st.session_state.dataset_risk_level,
            top_features=st.session_state.dataset_top_features,
            report=st.session_state.dataset_report,
            file_name="Credit_Risk_Report.pdf"
        )
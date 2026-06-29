import streamlit as st
import pandas as pd

from utils.pdf_utils import generate_pdf_report

from utils.shap_utils import (
    plot_shap_bar,
    format_shap_table
)

def show_risk_status(risk_level):

    if risk_level == "Low Risk":
        st.success("🟢 Low Risk")

    elif risk_level == "Medium Risk":
        st.warning("🟠 Medium Risk")

    else:
        st.error("🔴 High Risk")

def display_prediction_summary(probability, risk_level):
    """Display prediction probability and risk level."""

    st.subheader("📋 Prediction Summary")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Default Probability",
            f"{probability * 100:.2f}%"
        )

    with metric_col2:

        st.metric(
            "Risk Level",
            risk_level
        )

        show_risk_status(risk_level)

    st.divider()

def display_applicant_information(sample):
    """Display applicant information."""

    st.subheader("👤 Applicant Information")

    feature_df = pd.DataFrame({
        "Applicant Information": [
            "Age",
            "Employment Duration",
            "Annual Income",
            "Credit Amount",
            "Loan Annuity",
            "Credit / Income Ratio",
            "Annuity / Income Ratio"
        ],
        "Value": [
            f"{sample['AGE_YEARS'].iloc[0]:.1f} years",
            f"{sample['EMPLOYMENT_YEARS'].iloc[0]:.1f} years",
            f"₹ {sample['AMT_INCOME_TOTAL'].iloc[0]:,.0f}",
            f"₹ {sample['AMT_CREDIT'].iloc[0]:,.0f}",
            f"₹ {sample['AMT_ANNUITY'].iloc[0]:,.0f}",
            f"{sample['CREDIT_INCOME_RATIO'].iloc[0]:.2f}",
            f"{sample['ANNUITY_INCOME_RATIO'].iloc[0]:.2f}"
        ]
    })

    st.dataframe(
        feature_df,
        hide_index=True,
        use_container_width=True
    )

def display_explainability(top_features):
    """Display explainability information using SHAP values."""

    st.subheader("🔍 Explainability")

    st.markdown("**Top Feature Contributions**")

    display_shap = format_shap_table(top_features)

    st.dataframe(
        display_shap,
        hide_index=True,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Feature Contribution Chart**")

        plot_shap_bar(top_features)

    with col2:

        st.markdown("**SHAP Interpretation**")

        st.info(
            """
• Positive SHAP values increase default risk.

• Negative SHAP values decrease default risk.

• Larger absolute SHAP values have a stronger influence on the prediction.
"""
        )

    st.divider()

def display_ai_report(
    sample,
    probability,
    risk_level,
    top_features,
    report,
    file_name
):
    """Display AI-generated credit risk assessment report."""

    st.divider()

    st.subheader("🤖 AI Credit Risk Assessment")

    st.markdown(report)

    pdf_buffer = generate_pdf_report(
        sample=sample,
        probability=probability,
        risk_level=risk_level,
        top_features=top_features,
        ai_report=report
    )

    st.download_button(
        label="📄 Download Credit Risk Report (PDF)",
        data=pdf_buffer,
        file_name=file_name,
        mime="application/pdf",
        use_container_width=True
    )
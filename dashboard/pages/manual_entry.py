import streamlit as st

from utils.feature_engineering import (
    create_empty_dataframe,
    calculate_ratios,
    encode_categorical_features
)

from utils.predict_utils import predict_credit_risk

from utils.rag_utils import generate_credit_risk_report

from utils.ui_utils import (
    display_prediction_summary,
    display_applicant_information,
    display_explainability,
    display_ai_report
)

def show_manual_page(
    manual_model,
    manual_explainer,
    manual_features,
    retriever
):
    
    with st.form("manual_application"):

        st.subheader("Manual Loan Application")

        st.markdown("### 👤 Personal Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age (Years)",
                min_value=18,
                max_value=100,
                value=35
            )

            gender = st.selectbox(
                "Gender",
                [
                    "Female",
                    "Male"
                ]
            )

            family_status = st.selectbox(
                "Family Status",
                [
                    "Married",
                    "Single / not married",
                    "Separated",
                    "Widow"
                ]
            )

        with col2:

            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=10,
                value=0
            )

            family_members = st.number_input(
                "Family Members",
                min_value=1,
                max_value=15,
                value=2
            )

            education = st.selectbox(
                "Education",
                [
                    "Higher education",
                    "Incomplete higher",
                    "Secondary / secondary special",
                    "Lower secondary"
                ]
            )

        st.divider()

        st.markdown("### 💰 Financial Information")

        col1, col2 = st.columns(2)

        with col1:

            income = st.number_input(
                "Annual Income",
                min_value=10000.0,
                value=150000.0,
                step=10000.0
            )

            credit = st.number_input(
                "Credit Amount",
                min_value=10000.0,
                value=300000.0,
                step=10000.0
            )

        with col2:

            annuity = st.number_input(
                "Loan Annuity",
                min_value=1000.0,
                value=30000.0,
                step=1000.0
            )

            goods_price = st.number_input(
                "Goods Price",
                min_value=10000.0,
                value=250000.0,
                step=10000.0
            )
        
        st.divider()

        st.markdown("### 💼 Employment Information")

        col1, col2 = st.columns(2)

        with col1:

            employment_years = st.number_input(
                "Years Employed",
                min_value=0.0,
                max_value=60.0,
                value=5.0,
                step=1.0
            )

            income_type = st.selectbox(
                "Income Type",
                [
                    "Working",
                    "Commercial associate",
                    "Pensioner",
                    "State servant",
                    "Student",
                    "Unemployed"
                ]
            )

        with col2:

            occupation = st.selectbox(
                "Occupation",
                [
                    "Core staff",
                    "Laborers",
                    "Managers",
                    "Drivers",
                    "Sales staff",
                    "Medicine staff",
                    "High skill tech staff",
                    "Security staff",
                    "Cleaning staff",
                    "Cooking staff",
                    "HR staff",
                    "IT staff",
                    "Low-skill Laborers",
                    "Private service staff",
                    "Realty agents",
                    "Secretaries",
                    "Waiters/barmen staff"
                ]
            )

        st.divider()

        st.markdown("### 🏠 Housing Information")

        col1, col2 = st.columns(2)

        with col1:

            owns_car = st.checkbox("Owns a Car")

            owns_realty = st.checkbox("Owns Real Estate")

        with col2:

            housing_type = st.selectbox(
                "Housing Type",
                [
                    "House / apartment",
                    "Municipal apartment",
                    "Office apartment",
                    "Rented apartment",
                    "With parents"
                ]
            )
            
        st.divider()

        st.markdown("### 📈 Credit Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            ext_source_1 = st.number_input(
                "External Credit Score 1",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01,
                format="%.2f"
            )

        with col2:
            ext_source_2 = st.number_input(
                "External Credit Score 2",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01,
                format="%.2f"
            )

        with col3:
            ext_source_3 = st.number_input(
                "External Credit Score 3",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01,
                format="%.2f"
            )


        submitted = st.form_submit_button("🔍 Predict Credit Risk", use_container_width=True)

        
    if submitted:

        sample = create_empty_dataframe(manual_features)

        # =============================
        # Numerical Features
        # =============================

        sample.loc[0, "AGE_YEARS"] = age
        sample.loc[0, "EMPLOYMENT_YEARS"] = employment_years

        sample.loc[0, "CNT_CHILDREN"] = children
        sample.loc[0, "CNT_FAM_MEMBERS"] = family_members

        sample.loc[0, "AMT_INCOME_TOTAL"] = income
        sample.loc[0, "AMT_CREDIT"] = credit
        sample.loc[0, "AMT_ANNUITY"] = annuity
        sample.loc[0, "AMT_GOODS_PRICE"] = goods_price

        sample.loc[0, "EXT_SOURCE_1"] = ext_source_1
        sample.loc[0, "EXT_SOURCE_2"] = ext_source_2
        sample.loc[0, "EXT_SOURCE_3"] = ext_source_3

        # =============================
        # Binary Features
        # =============================

        sample.loc[0, "CODE_GENDER_M"] = (
            1 if gender == "Male" else 0
        )

        sample.loc[0, "FLAG_OWN_CAR_Y"] = (
            1 if owns_car else 0
        )

        sample.loc[0, "FLAG_OWN_REALTY_Y"] = (
            1 if owns_realty else 0
        )

        # =============================
        # Engineered Features
        # =============================

        sample = calculate_ratios(sample)
        sample = encode_categorical_features(
            sample,
            education,
            income_type,
            family_status,
            occupation,
            housing_type
        )
    
        # =============================
        # Prediction
        # =============================

        (
            probability,
            risk_level,
            top_features
        ) = predict_credit_risk(
            manual_model,
            manual_explainer,
            sample
        )
        
        # =============================
        # Save Prediction
        # =============================

        st.session_state.manual_prediction = True
        st.session_state.manual_sample = sample
        st.session_state.manual_probability = probability
        st.session_state.manual_risk_level = risk_level
        st.session_state.manual_top_features = top_features

        with st.spinner("Generating AI credit risk assessment..."):

            report = generate_credit_risk_report(
                retriever=retriever,
                sample=sample,
                probability=probability,
                risk_level=risk_level,
                top_features=top_features
            )

        st.session_state.manual_report = report

        st.divider()
    
    # =============================
    # Display Prediction
    # =============================

    if st.session_state.manual_prediction:

        sample = st.session_state.manual_sample
        probability = st.session_state.manual_probability
        risk_level = st.session_state.manual_risk_level
        top_features = st.session_state.manual_top_features

        display_prediction_summary(
            probability,
            risk_level
        )

        display_applicant_information(sample)

        display_explainability(top_features)

        if st.session_state.manual_report:

            display_ai_report(
                sample=sample,
                probability=probability,
                risk_level=risk_level,
                top_features=top_features,
                report=st.session_state.manual_report,
                file_name="Manual_Credit_Risk_Report.pdf"
            )
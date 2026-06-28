import shap
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from utils.feature_engineering import (
    create_empty_dataframe,
    calculate_ratios,
    encode_categorical_features
)
from utils.shap_utils import (
    get_top_shap_features,
    plot_shap_bar
)

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import streamlit as st
import joblib
import pandas as pd


from google import genai
from config import GEMINI_API_KEY

client = genai.Client(
    api_key=GEMINI_API_KEY
)

st.set_page_config(
    page_title="Explainable Credit Risk System",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_dataset_model():
    return joblib.load(
        "C:\\Users\\Asus\\OneDrive\\Desktop\\Explainable-Credit-Risk-System\\models\\best_xgb_model.joblib"
    )


@st.cache_resource
def load_manual_model():
    return joblib.load(
        "C:\\Users\\Asus\\OneDrive\\Desktop\\Explainable-Credit-Risk-System\\models\\manual_entry_xgb_model.joblib"
    )


@st.cache_resource
def load_manual_features():
    return joblib.load(
        "C:\\Users\\Asus\\OneDrive\\Desktop\\Explainable-Credit-Risk-System\\models\\manual_entry_features.joblib"
    )

@st.cache_data
def load_data():
    return pd.read_csv(
        "C:\\Users\\Asus\\OneDrive\\Desktop\\Explainable-Credit-Risk-System\\data\\processed\\preprocessed_v2_encoded.csv"
    )

dataset_model = load_dataset_model()

manual_model = load_manual_model()

manual_features = load_manual_features()

dataset_explainer = shap.TreeExplainer(dataset_model)

manual_explainer = shap.TreeExplainer(manual_model)

df = load_data()

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=str(
            project_root / "rag" / "vectordb"
        ),
        embedding_function=embeddings
    )

    return vectorstore


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

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

st.success("Model Loaded Successfully")

st.divider()

if input_mode == "Dataset Applicant":

    st.subheader("Select Applicant")

    applicant_index = st.number_input(
        "Applicant Index",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    st.write(
        "Selected Applicant Index:",
        applicant_index
    )

    st.divider()

    sample = df.drop(
        "TARGET",
        axis=1
    ).iloc[[int(applicant_index)]]

    shap_values = dataset_explainer.shap_values(sample)

    probability = dataset_model.predict_proba(
        sample
    )[0][1]

    if probability < 0.20:
        risk_level = "Low Risk"
    elif probability < 0.50:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            label="Default Probability",
            value=f"{probability * 100:.2f}%"
        )

    with metric_col2:
        st.metric(
            label="Risk Level",
            value=risk_level
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Applicant Features")

        feature_cols = [
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO",
            "AGE_YEARS",
            "EMPLOYMENT_YEARS"
        ]

        feature_df = sample[feature_cols].T
        feature_df.columns = ["Value"]

        st.dataframe(
            feature_df,
            use_container_width=True
        )

    with col2:

        st.subheader("SHAP Explanation")

        shap_df = pd.DataFrame({
            "Feature": sample.columns,
            "SHAP Value": shap_values[0]
        })

        shap_df["Impact"] = shap_df["SHAP Value"].abs()

        top_features = (
            shap_df
            .sort_values(
                "Impact",
                ascending=False
            )
            .head(5)
        )

        top_features["Direction"] = top_features[
            "SHAP Value"
        ].apply(
            lambda x:
            "↑ Increases Risk"
            if x > 0
            else "↓ Decreases Risk"
        )

        display_shap = top_features[
            ["Feature", "SHAP Value", "Direction"]
        ].copy()

        display_shap["SHAP Value"] = (
            display_shap["SHAP Value"]
            .round(3)
        )

        st.dataframe(
            display_shap,
            use_container_width=True,
            hide_index=True
        )
    
    plot_shap_bar(top_features)

    st.info(
        """
        SHAP Interpretation

        • Positive SHAP values increase default risk.

        • Negative SHAP values decrease default risk.

        • Larger absolute SHAP values have a stronger influence on the prediction.
        """
    )

    st.divider()

    if st.button("Generate Credit Risk Report"):

        query = f"""
        Applicant has:

        - Default probability: {probability*100:.2f}%
        - Risk level: {risk_level}
        - Credit income ratio: {sample['CREDIT_INCOME_RATIO'].values[0]:.2f}
        - Annuity income ratio: {sample['ANNUITY_INCOME_RATIO'].values[0]:.2f}
        - Age: {sample['AGE_YEARS'].values[0]:.1f}
        - Employment duration: {sample['EMPLOYMENT_YEARS'].values[0]:.1f}

        What lending policies and risk rules apply?
        """

        results = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        top_shap_text = "\n".join(
            [
                f"- {row.Feature}: SHAP={row['SHAP Value']:.3f}"
                for _, row in top_features.head(5).iterrows()
            ]
        )
        
        prompt = f"""
        You are an expert Credit Risk Analyst.

        Applicant Information:

        Default Probability:
        {probability*100:.2f}%

        Risk Level:
        {risk_level}

        Credit Income Ratio:
        {sample['CREDIT_INCOME_RATIO'].values[0]:.2f}

        Annuity Income Ratio:
        {sample['ANNUITY_INCOME_RATIO'].values[0]:.2f}

        Age:
        {sample['AGE_YEARS'].values[0]:.1f}

        Employment Duration:
        {sample['EMPLOYMENT_YEARS'].values[0]:.1f}

        Relevant Policies:

        {context}

        Top Model Factors (SHAP):

        {top_shap_text}

        IMPORTANT FEATURE DEFINITIONS

        - AGE_YEARS represents applicant age.
        - EMPLOYMENT_YEARS represents employment duration.
        - Risk Level is produced by the machine learning model.
        - CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL
        - ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL

        Do NOT interpret CREDIT_INCOME_RATIO as Debt-to-Income Ratio.
        Do NOT invent Applicant IDs.
        Do NOT invent dates.
        Do NOT invent missing information.
        Use only the supplied applicant features.

        Generate a professional banking-style credit risk assessment report.

        Use ONLY the provided applicant information and retrieved policies.

        IMPORTANT:
        - Do not invent dates.
        - Do not invent applicant IDs.
        - Do not invent financial information.
        - Keep the report concise and professional.
        - Use bullet points where appropriate.

        Report Format:

        ## Executive Summary

        Provide a 2-3 sentence summary of the applicant's risk profile.

        ## Risk Classification

        - Predicted Default Probability
        - Risk Level

        ## Key Positive Indicators

        List positive factors that support loan approval.

        ## Key Risk Indicators

        List factors that increase lending risk.

        ## Relevant Lending Policies

        Mention only the retrieved policies that directly apply.

        ## Recommendation

        Provide one of:
        - Approve
        - Approve with Review
        - Manual Review Required
        - Reject

        Include a short justification.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        st.divider()

        st.subheader("Credit Risk Assessment Report")

        st.markdown(response.text)

else:
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
        sample = create_empty_dataframe(
            manual_features
        )

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

        probability = manual_model.predict_proba(sample)[0][1]

        if probability < 0.20:
            risk_level = "Low Risk"
        elif probability < 0.50:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Default Probability",
                f"{probability*100:.2f}%"
            )

        with metric_col2:
            st.metric(
                "Risk Level",
                risk_level
            )
        
        st.divider()

        st.subheader("SHAP Explanation")

        manual_shap_values = manual_explainer.shap_values(sample)

        top_features = get_top_shap_features(
            manual_shap_values,
            sample
        )

        st.dataframe(
            top_features[
                ["Feature", "SHAP Value"]
            ],
            use_container_width=True
        )

        st.divider()

        plot_shap_bar(top_features)
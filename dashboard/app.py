import shap
import matplotlib.pyplot as plt
import sys
from pathlib import Path

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
def load_model():
    return joblib.load(
        "C:\\Users\\Asus\\OneDrive\\Desktop\\Explainable-Credit-Risk-System\\models\\best_xgb_model.joblib"
    )

@st.cache_data
def load_data():
    return pd.read_csv(
        "C:\\Users\\Asus\\OneDrive\\Desktop\\Explainable-Credit-Risk-System\\data\\processed\\preprocessed_v2_encoded.csv"
    )

model = load_model()
explainer = shap.TreeExplainer(model)

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

st.success("✅ XGBoost Model Loaded Successfully")

st.divider()

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

shap_values = explainer.shap_values(sample)

probability = model.predict_proba(
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

fig, ax = plt.subplots(figsize=(6, 3.5))

chart_data = top_features.sort_values("SHAP Value")

bars = ax.barh(
    chart_data["Feature"],
    chart_data["SHAP Value"]
)

# Smaller fonts
ax.set_title(
    "Top SHAP Feature Contributions",
    fontsize=11,
    pad=10
)

ax.set_xlabel(
    "SHAP Value",
    fontsize=9
)

ax.tick_params(
    axis="both",
    labelsize=8
)

# Cleaner look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Light grid
ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

st.pyplot(fig)

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
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
df = load_data()

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="rag/vectordb",
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
    max_value=len(df)-1,
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

probability = model.predict_proba(
    sample
)[0][1]

if probability < 0.20:
    risk_level = "Low Risk"
elif probability < 0.50:
    risk_level = "Medium Risk"
else:
    risk_level = "High Risk"

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Default Probability",
        f"{probability * 100:.2f}%"
    )

with col2:
    st.metric(
        "Risk Level",
        risk_level
    )

st.divider()

st.subheader("Applicant Features")

feature_cols = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS"
]

st.dataframe(
    sample[feature_cols].T.rename(
        columns={sample.index[0]: "Value"}
    )
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

    IMPORTANT FEATURE DEFINITIONS

    - CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL
    - ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL

    Do NOT interpret CREDIT_INCOME_RATIO as Debt-to-Income Ratio.
    Do NOT invent Applicant IDs.
    Do NOT invent dates.
    Do NOT invent missing information.
    Use only the supplied applicant features.

    Generate a professional credit risk assessment report.

    Include:

    1. Risk Level
    2. Prediction Summary
    3. Positive Risk Indicators
    4. Negative Risk Indicators
    5. Relevant Lending Policies
    6. Final Recommendation
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.divider()

    st.subheader("Credit Risk Assessment Report")

    st.markdown(response.text)
from google import genai
from pathlib import Path

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=GEMINI_API_KEY
)

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please check your .env file."
    )


project_root = Path(__file__).resolve().parent.parent.parent

@st.cache_resource
def load_retriever():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=str(
            project_root / "rag" / "vectordb"
        ),
        embedding_function=embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

def generate_credit_risk_report(
    retriever,
    sample,
    probability,
    risk_level,
    top_features
):
    """
    Generates an AI-powered credit risk assessment report
    using Gemini + ChromaDB.
    """

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

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        error_message = str(e)

        if "RESOURCE_EXHAUSTED" in error_message or "429" in error_message:

            return """
                # ⚠️ AI Report Unavailable

                The Gemini API quota for this project has been reached.

                The credit risk prediction, SHAP explanation and PDF generation are still available.

                Please try again later or use another Gemini API key.
                """

        return f""" # ⚠️ AI Report Error

                An unexpected error occurred while generating the AI assessment.

                Error:

                {error_message}
                """
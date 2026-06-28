import pandas as pd

def create_empty_dataframe(feature_list):
    """
    Creates a single-row dataframe with all model features initialized to 0.
    """

    return pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_list,
        dtype=float
    )

def calculate_ratios(df):
    """
    Calculate engineered features.
    """

    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_INCOME_TOTAL"]
    )

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] /
        df["AMT_INCOME_TOTAL"]
    )

    df["EMPLOYMENT_AGE_RATIO"] = (
        df["EMPLOYMENT_YEARS"] /
        df["AGE_YEARS"]
    )

    return df

def encode_categorical_features(
    df,
    education,
    income_type,
    family_status,
    occupation,
    housing_type
):
    """
    One-hot encode all categorical features.
    """

    education_column = f"NAME_EDUCATION_TYPE_{education}"
    if education_column in df.columns:
        df.loc[0, education_column] = 1

    income_column = f"NAME_INCOME_TYPE_{income_type}"
    if income_column in df.columns:
        df.loc[0, income_column] = 1

    family_column = f"NAME_FAMILY_STATUS_{family_status}"
    if family_column in df.columns:
        df.loc[0, family_column] = 1

    occupation_column = f"OCCUPATION_TYPE_{occupation}"
    if occupation_column in df.columns:
        df.loc[0, occupation_column] = 1

    housing_column = f"NAME_HOUSING_TYPE_{housing_type}"
    if housing_column in df.columns:
        df.loc[0, housing_column] = 1

    return df
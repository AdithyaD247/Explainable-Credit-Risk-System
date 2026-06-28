import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st


def get_top_shap_features(shap_values, sample, top_n=10):
    """
    Returns the top N SHAP features for a single prediction.
    """

    shap_df = pd.DataFrame({
        "Feature": sample.columns,
        "SHAP Value": shap_values[0]
    })

    shap_df["Impact"] = shap_df["SHAP Value"].abs()

    return (
        shap_df
        .sort_values("Impact", ascending=False)
        .head(top_n)
    )


def plot_shap_bar(top_features):
    """
    Plot a horizontal SHAP contribution chart for a single applicant.
    """

    fig, ax = plt.subplots(figsize=(7, 3.8))

    chart_data = top_features.sort_values("SHAP Value")

    colors = [
        "#d62728" if value > 0 else "#2ca02c"
        for value in chart_data["SHAP Value"]
    ]

    ax.barh(
        chart_data["Feature"],
        chart_data["SHAP Value"],
        color=colors
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1
    )

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

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)
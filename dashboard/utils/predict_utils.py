from utils.shap_utils import get_top_shap_features


def predict_credit_risk(
    model,
    explainer,
    sample
):
    """
    Runs credit risk prediction and SHAP analysis.
    """

    probability = model.predict_proba(sample)[0][1]

    if probability < 0.20:
        risk_level = "Low Risk"
    elif probability < 0.50:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    shap_values = explainer.shap_values(sample)

    top_features = get_top_shap_features(
        shap_values,
        sample
    )

    return (
        probability,
        risk_level,
        top_features
    )
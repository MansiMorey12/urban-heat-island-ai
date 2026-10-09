"""
============================================================
explainable_mitigation.py
============================================================
Explainable-AI-driven mitigation selection using SHAP.

This module replaces "pick mitigation strategies from a fixed
list keyed only by the predicted class" with "pick mitigation
strategies based on WHICH SPECIFIC FEATURES actually drove THIS
prediction," using SHAP (SHapley Additive exPlanations) values.

Tested and verified to run against random_forest_model.pkl,
feature_names.pkl, and label_encoder.pkl from this project.

  pip install shap
============================================================
"""

import numpy as np
import shap


# ── Feature -> actionable mitigation strategy map ─────────────
# Only features a city/planner can actually DO something about
# are included here. Purely meteorological features (Temperature,
# Humidity, Wind Speed, Solar Radiation, Elevation, Rainfall as a
# driver) are left out on purpose — they are not actionable levers,
# only the structural/urban-form features are.
FEATURE_MITIGATION_MAP = {
    "Green_Cover_%": (
        "🌳", "Increase Urban Green Cover",
        "Plant trees and expand parks in this area; low green cover is "
        "a top driver of this prediction."
    ),
    "Building_Density": (
        "🏢", "Cool / Green Roofs",
        "High building density is trapping heat; retrofit roofs with "
        "reflective or vegetated coatings."
    ),
    "NDVI": (
        "🌱", "Vegetation Restoration",
        "Low vegetation health detected; prioritize planting and "
        "irrigation programs."
    ),
    "Albedo": (
        "🎨", "High-Albedo Surfaces",
        "Dark road/roof surfaces are absorbing heat; switch to "
        "reflective, light-colored materials."
    ),
    "Traffic_Density": (
        "🚦", "Traffic & Emissions Management",
        "Heavy traffic is a contributing factor; consider EV "
        "incentives or traffic calming measures."
    ),
    "Industrial_Proximity_km": (
        "🏭", "Industrial Buffer Greenbelt",
        "Proximity to industry is contributing to local heat; add a "
        "buffer greenbelt between industrial and residential zones."
    ),
    "Air_Quality_Index": (
        "😷", "Air Quality Improvement",
        "Poor air quality is compounding heat stress; pursue "
        "emissions controls and cleaner-fuel policy."
    ),
    "Population_Density_per_km2": (
        "👥", "Urban Density Management",
        "High population density adds to the local heat load; "
        "consider decentralization or density-management policy."
    ),
}


def get_shap_explainer(model):
    """Build (and cache) a SHAP TreeExplainer for the trained Random Forest.
    Call this once (e.g. with @st.cache_resource in the Streamlit app)
    since building the explainer has a small fixed cost."""
    return shap.TreeExplainer(model)


def explain_and_select_mitigation(model, explainer, input_df, feature_names,
                                   le, predicted_class_idx, top_n=3):
    """
    Compute SHAP values for a single prediction and select mitigation
    strategies based on which actionable features actually pushed the
    prediction toward its predicted class — this is the explainable-AI
    layer that makes mitigation choice model-driven rather than a
    static class -> strategy-list lookup.

    Parameters
    ----------
    model : trained RandomForestClassifier
    explainer : shap.TreeExplainer (from get_shap_explainer)
    input_df : pandas.DataFrame, single row, already scaled — the exact
        feature vector that was passed to model.predict()
    feature_names : list[str], matching input_df's column order
    le : fitted LabelEncoder for Heat_Level
    predicted_class_idx : int, the model's predicted class index
        (i.e. model.predict(input_df)[0])
    top_n : how many SHAP-driven strategies to return

    Returns
    -------
    dict with:
        'predicted_class': str
        'contributing_factors': list of (feature_name, shap_value) sorted
            by |shap_value| descending, for ALL features (for display in
            an updated "Why This Prediction?" panel)
        'shap_mitigation': list of (icon, title, description) — the
            SHAP-selected, per-prediction mitigation strategies
    """
    predicted_class = le.inverse_transform([predicted_class_idx])[0]

    shap_values = explainer.shap_values(input_df)
    # shap_values shape: (n_samples, n_features, n_classes) for this
    # shap/sklearn version combination — take sample 0, this class
    class_shap = shap_values[0, :, predicted_class_idx]

    feat_shap_pairs = list(zip(feature_names, class_shap))
    feat_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    if predicted_class == "Low":
        # IMPORTANT: for a Low (good) prediction, a feature with a
        # positive SHAP value toward "Low" is HELPING keep heat down
        # (e.g. good green cover). It would be backwards to recommend
        # "fixing" it. So for Low predictions we do not mine SHAP for
        # problems to fix — we just reinforce the low-effort baseline.
        shap_mitigation = [
            ("🌳", "Maintain Existing Green Cover",
             "This area is already at low heat risk; focus on "
             "preserving the conditions responsible for that."),
            ("💧", "Basic Rainwater Harvesting",
             "A low-cost, no-regrets measure that supports continued "
             "vegetation health."),
        ]
    else:
        # For Medium/High (bad) predictions, we want the SHAP values
        # for THIS class where a positive value means "this feature's
        # current value is pushing the prediction toward Medium/High,"
        # i.e. it is genuinely part of the problem and worth targeting.
        actionable = [
            (f, v) for f, v in feat_shap_pairs
            if f in FEATURE_MITIGATION_MAP and v > 0
        ]
        actionable.sort(key=lambda x: abs(x[1]), reverse=True)

        shap_mitigation = [
            FEATURE_MITIGATION_MAP[f] for f, _ in actionable[:top_n]
        ]

        # Fallback: a Medium/High prediction with no positive actionable
        # driver among the mapped features (e.g. driven mainly by pure
        # weather, which isn't actionable) still needs *something* shown.
        if not shap_mitigation:
            shap_mitigation = [
                ("🌳", "Expand Tree Canopy Coverage",
                 "No single actionable structural feature dominates this "
                 "prediction; general green-infrastructure investment is "
                 "still recommended given the elevated heat level."),
                ("🎨", "Cool-Roof Coatings",
                 "A broadly effective, low-regret measure for elevated "
                 "heat risk."),
            ]

    return {
        "predicted_class": predicted_class,
        "contributing_factors": feat_shap_pairs,
        "shap_mitigation": shap_mitigation,
    }

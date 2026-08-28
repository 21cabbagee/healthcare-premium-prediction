"""
app.py — Streamlit front-end for the Healthcare Premium Prediction project.

Loads the trained sklearn Pipeline (preprocessing + model bundled together),
lets a user describe a hypothetical policyholder, predicts their annual
premium, and explains the prediction with feature importances plus the
model comparison metrics from training.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("reports/metrics.json")

# --- Chart palette (validated categorical/sequential slots — see dataviz skill) ---
BLUE = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

st.set_page_config(
    page_title="Healthcare Premium Predictor",
    page_icon="🏥",
    layout="centered",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


def age_group(age: int) -> str:
    if age < 30:
        return "young_adult"
    if age < 45:
        return "adult"
    if age < 60:
        return "middle_aged"
    return "senior"


st.title("🏥 Healthcare Premium Predictor")
st.markdown(
    "Estimate an individual's **annual health insurance premium** from a handful "
    "of demographic and lifestyle details, using a model trained on real-world "
    "insurance charges data."
)

model = load_model()
metrics = load_metrics()

# --- Inputs -----------------------------------------------------------------
st.header("Policyholder details")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", min_value=18, max_value=64, value=35)
    bmi = st.slider("BMI (Body Mass Index)", min_value=15.0, max_value=54.0, value=27.0, step=0.1)
    children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
with col2:
    sex = st.radio("Sex", options=["male", "female"], horizontal=True)
    smoker = st.radio("Smoker", options=["no", "yes"], horizontal=True)
    region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

# --- Build a single-row feature frame matching training features ------------
smoker_flag = 1 if smoker == "yes" else 0
smoker_bmi_interaction = smoker_flag * bmi

input_row = pd.DataFrame(
    [
        {
            "age": age,
            "bmi": bmi,
            "children": children,
            "smoker_bmi_interaction": smoker_bmi_interaction,
            "sex": sex,
            "smoker": smoker,
            "region": region,
            "bmi_category": bmi_category(bmi),
            "age_group": age_group(age),
        }
    ]
)

prediction = model.predict(input_row)[0]

st.divider()
st.header("Estimated annual premium")
st.metric(label="Predicted premium", value=f"${prediction:,.0f}")
st.caption(
    "Estimate only — trained on a public sample dataset for demonstration purposes, "
    "not intended for real underwriting decisions."
)

if smoker == "yes":
    st.warning(
        "This estimate is sharply higher because the policyholder smokes — "
        "smoking dramatically amplifies the cost impact of BMI (see below)."
    )

# --- What's driving this estimate -------------------------------------------
st.divider()
st.header("What's driving this estimate")
st.markdown(
    "The chart below shows overall **feature importance** from the trained model — "
    "i.e. which inputs matter most across all predictions, not just this one."
)

best_model_step = model.named_steps["model"]
preprocessor = model.named_steps["preprocessor"]
feature_names = preprocessor.get_feature_names_out()

if hasattr(best_model_step, "feature_importances_"):
    importances = best_model_step.feature_importances_
elif hasattr(best_model_step, "coef_"):
    importances = np.abs(best_model_step.coef_)
else:
    importances = None

if importances is not None:
    imp_series = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(8)
        .iloc[::-1]  # ascending for horizontal bar (largest at top)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    bars = ax.barh(imp_series.index, imp_series.values, color=BLUE, height=0.6)

    # value labels at bar ends
    max_val = imp_series.values.max()
    for bar, val in zip(bars, imp_series.values):
        ax.text(
            val + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color=INK_PRIMARY,
        )

    ax.set_xlabel("Relative importance", color=INK_MUTED, fontsize=10)
    ax.tick_params(axis="y", colors=INK_PRIMARY, labelsize=9)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max_val * 1.18)

    fig.tight_layout()
    st.pyplot(fig, transparent=True)
else:
    st.info("Feature importances are not available for this model type.")

st.caption(
    "Key insight: the **smoker × BMI interaction** consistently ranks as the single "
    "strongest driver of predicted cost — smoking doesn't just add a flat surcharge, "
    "it multiplies the effect of a high BMI on premiums."
)

# --- Model performance --------------------------------------------------------
st.divider()
with st.expander("📊 Model performance (training-time comparison)"):
    results_df = pd.DataFrame(metrics["results"])
    best_model_name = metrics["best_model"]

    display_df = results_df.rename(
        columns={
            "model": "Model",
            "rmse": "RMSE ($)",
            "mae": "MAE ($)",
            "r2": "R²",
            "cv_rmse_mean": "CV RMSE (mean)",
            "cv_rmse_std": "CV RMSE (std)",
        }
    ).copy()
    display_df["Model"] = display_df["Model"].apply(
        lambda m: f"⭐ **{m}**" if m == best_model_name else m
    )

    st.dataframe(display_df, hide_index=True, use_container_width=True)
    st.caption(
        f"**{best_model_name}** was selected as the final model based on the lowest "
        "test-set RMSE, and is the model powering the prediction above."
    )

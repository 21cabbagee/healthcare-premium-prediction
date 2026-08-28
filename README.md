# 🏥 Healthcare Premium Prediction

Predicting annual health insurance premiums from demographic and lifestyle data — a compact, end-to-end regression project with feature engineering, model comparison, and a deployed Streamlit app.

**🔗 Live Demo:** _[will be added after deployment]_

## Problem Statement

Insurers need to estimate an individual's expected healthcare cost (premium) from a small set of easily-collected attributes — age, sex, BMI, number of children, smoking status, and region — in order to price policies and assess risk. This project builds and compares several regression models to produce that estimate, and ships the best one behind an interactive app so a user can explore how each factor moves the predicted premium.

## Dataset

- **Source:** [stedy/Machine-Learning-with-R-datasets — `insurance.csv`](https://github.com/stedy/Machine-Learning-with-R-datasets)
- **Rows:** 1,338 individuals
- **Columns:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges` (target — annual medical costs billed by health insurance, in USD)

## Approach

1. **EDA** — checked correlations, and grouped charges by `smoker` and `region`. Key findings:
   - Smokers pay ~4x more on average than non-smokers ($32,050 vs $8,434).
   - Region differences are small (a few hundred dollars) relative to smoker/age/BMI effects.
   - Raw `bmi` alone correlates weakly with charges (~0.20), but **splitting by smoker status reveals BMI's real effect is conditional on smoking** — obese non-smokers pay only slightly more than non-obese non-smokers, while obese smokers pay dramatically more than non-obese smokers.
2. **Feature engineering** — one-hot encoded `sex`, `smoker`, `region`; added a `smoker_bmi_interaction` term (smoker flag × BMI), plus `bmi_category` and `age_group` buckets so tree models can split on clinically meaningful thresholds.
3. **Feature selection** — confirmed with both a correlation check and `RandomForestRegressor.feature_importances_` that `smoker_bmi_interaction` (correlation ≈ 0.85 with charges) and `age` dominate predictive signal, while `children` and `region` contribute little but are kept for realism at negligible cost.
4. **Model comparison** — built an sklearn `Pipeline` (`ColumnTransformer` + estimator) so preprocessing ships with the model, trained on an 80/20 split (`random_state=42`), validated with 5-fold CV on the training set, and compared `LinearRegression`, `Ridge`, `RandomForestRegressor`, and `GradientBoostingRegressor` on held-out test RMSE/MAE/R².
5. **Selection** — picked the lowest test-RMSE model and persisted the fitted pipeline with `joblib` for the app to load directly.

## Results

| Model | RMSE ($) | MAE ($) | R² |
|---|---|---|---|
| **GradientBoostingRegressor** | **4409.83** | **2498.85** | **0.8747** |
| Ridge | 4517.64 | 2738.27 | 0.8685 |
| LinearRegression | 4529.04 | 2729.15 | 0.8679 |
| RandomForestRegressor | 4630.65 | 2544.68 | 0.8619 |

**GradientBoostingRegressor** was selected as the final model — lowest test RMSE and highest R² among all candidates. Full metrics (including 5-fold CV RMSE) are in [`reports/metrics.json`](reports/metrics.json) and are also rendered inside the app.

## 💡 Key Insight

**Smoking doesn't just add a flat surcharge — it multiplies the cost impact of BMI.** Obese smokers pay roughly 2x more than non-obese smokers, while obese non-smokers pay barely more than non-obese non-smokers. This interaction effect (`smoker_bmi_interaction`) was, by a wide margin, the single most important feature across every model tested — a textbook case where a simple engineered interaction term captures far more signal than either raw feature alone.

## Tech Stack

- Python
- pandas / numpy — data loading & feature engineering
- scikit-learn — pipelines, preprocessing, model training & evaluation
- matplotlib / seaborn — feature importance visualization
- Streamlit — interactive web app
- joblib — model persistence

## Run locally

```bash
git clone https://github.com/uurbanbuddha/healthcare-premium-prediction.git
cd healthcare-premium-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (optional) retrain the model from scratch
python train.py

# launch the app
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Deploying to Streamlit Community Cloud

- **Main file path:** `app.py`
- **Repo root:** this repository's root directory
- `requirements.txt` pins the exact package versions used to train and pickle the model, to avoid scikit-learn version mismatches when unpickling on the cloud.
- `runtime.txt` targets Python 3.11 for Streamlit Cloud compatibility. Note: this project was developed locally against newer package versions (see `requirements.txt`); if Streamlit Cloud's Python 3.11 build can't resolve one of those exact pins, relax that single pin (e.g. drop the patch/minor version) and re-run `train.py` in that environment before redeploying, so the pickled model always matches the installed scikit-learn version.

## Project Structure

```
healthcare-premium-prediction/
├── app.py                # Streamlit app
├── train.py               # EDA, feature engineering, model training/eval
├── data/insurance.csv     # dataset
├── models/model.joblib    # trained pipeline (preprocessing + model)
├── reports/metrics.json   # model comparison metrics
├── requirements.txt
├── runtime.txt
└── README.md
```

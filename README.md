# Predictive Analytics System — Infrastructure & Project Delay Detection

An AI-powered system that analyzes project data to forecast land acquisition or infrastructure delays, assigns risk scores, and surfaces insights via an executive dashboard.

## Stack
- **Models**: XGBoost, LightGBM
- **Explainability**: SHAP, LIME
- **Imbalance Handling**: SMOTE, class weights
- **Dashboard**: Streamlit + Plotly

## Setup

```bash
# From the repository root — any OS
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

All scripts resolve paths from the repository root automatically — no need to `cd` into a specific directory first.

```bash
# Step 1: Generate synthetic data
python generate_data.py

# Step 2: Train models, evaluate (incl. 5-fold CV), and generate SHAP + LIME outputs
python train.py

# Step 3: Launch executive dashboard
streamlit run dashboard/app.py
```

## Project Structure

```
├── data/               # raw + processed datasets
├── src/                # preprocessing, feature engineering, models, explainability
├── dashboard/          # Streamlit app
├── outputs/            # metrics, SHAP plots, LIME reports, saved models
├── generate_data.py    # synthetic data generator
├── train.py            # end-to-end training pipeline
└── requirements.txt
```

## Risk Tiers

| Score | Tier     |
|-------|----------|
| 0–30  | 🟢 Low      |
| 31–60 | 🟡 Medium   |
| 61–80 | 🟠 High     |
| 81–100| 🔴 Critical |
# Predictive-Analytics-System-for-Infrastructure-Project-Delay-Detection

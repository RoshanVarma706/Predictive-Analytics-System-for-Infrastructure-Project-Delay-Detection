# Predictive Analytics System — Infrastructure & Project Delay Detection

An AI-powered system that analyzes project data to forecast land acquisition or infrastructure delays, assigns risk scores, and surfaces insights via an executive dashboard.

## Stack
- **Models**: XGBoost, LightGBM
- **Explainability**: SHAP, LIME
- **Imbalance Handling**: SMOTE, class weights
- **Dashboard**: Streamlit + Plotly

## Setup

```powershell
cd "E:\Predictive Project"
pip install -r requirements.txt
```

## Run

```powershell
# Step 1: Generate synthetic data
python generate_data.py

# Step 2: Train models, evaluate, and generate SHAP outputs
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

"""
Main training pipeline — orchestrates:
  1. Data loading & feature engineering
  2. Preprocessing & splitting
  3. XGBoost + LightGBM training
  4. Evaluation (test metrics + cross-validation)
  5. SHAP analysis & plot generation
  6. Risk score computation
  7. Saving all outputs
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Run from the project root so all relative input/output paths resolve,
# regardless of the directory this script is invoked from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

from generate_data import generate_data
from src.feature_engineering import engineer_features
from src.data_preprocessing import load_raw, preprocess, split_data
from src.models.gradient_boosting import XGBoostTrainer, LightGBMTrainer
from src.evaluation import evaluate, cross_validate, save_metrics
from src.risk_scoring import build_risk_dataframe
from src.explainability.shap_analysis import (
    compute_shap_values, plot_shap_summary, plot_shap_waterfall,
    save_shap_artifacts, get_top_drivers,
)
from src.explainability.lime_analysis import (
    build_lime_explainer, explain_instance, plot_lime_explanation,
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/shap", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def run_pipeline():
    print("\n" + "="*60)
    print("  PREDICTIVE ANALYTICS — INFRASTRUCTURE DELAY DETECTION")
    print("="*60)

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    raw_path = "data/raw/project_data.csv"
    if not os.path.exists(raw_path):
        print("\n[Pipeline] Generating synthetic data...")
        generate_data(output_path=raw_path)

    print("\n[Pipeline] Loading & engineering features...")
    df_raw = load_raw(raw_path)
    df = engineer_features(df_raw)

    # ------------------------------------------------------------------ #
    # 2. Preprocess & split
    # ------------------------------------------------------------------ #
    print("\n[Pipeline] Preprocessing...")
    X, y, preprocessor, feature_names = preprocess(df, fit=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Save processed data
    X.join(y).to_csv("data/processed/processed_data.csv", index=False)
    print(f"[Pipeline] Processed data saved → data/processed/processed_data.csv")

    # ------------------------------------------------------------------ #
    # 3. Train XGBoost
    # ------------------------------------------------------------------ #
    print("\n[Pipeline] Training XGBoost...")
    xgb_model = XGBoostTrainer(n_estimators=400, use_smote=True)
    xgb_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    xgb_model.save("outputs/models/xgboost_model.pkl")

    # ------------------------------------------------------------------ #
    # 4. Train LightGBM
    # ------------------------------------------------------------------ #
    print("\n[Pipeline] Training LightGBM...")
    lgbm_model = LightGBMTrainer(n_estimators=400)
    lgbm_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    lgbm_model.save("outputs/models/lightgbm_model.pkl")

    # ------------------------------------------------------------------ #
    # 5. Evaluate
    # ------------------------------------------------------------------ #
    print("\n[Pipeline] Evaluating models on test set...")
    xgb_metrics  = evaluate(xgb_model,  X_test, y_test, "XGBoost")
    lgbm_metrics = evaluate(lgbm_model, X_test, y_test, "LightGBM")

    # Save feature importances
    fi_df = pd.DataFrame({
        "feature":           feature_names,
        "xgb_importance":    xgb_model.feature_importances_,
        "lgbm_importance":   lgbm_model.feature_importances_,
    }).sort_values("xgb_importance", ascending=False)
    fi_df.to_csv("outputs/feature_importance.csv", index=False)
    print(f"[Pipeline] Feature importance saved → outputs/feature_importance.csv")

    # ------------------------------------------------------------------ #
    # 5b. Cross-validation (stratified 5-fold, train set only)
    # ------------------------------------------------------------------ #
    print("\n[Pipeline] Running 5-fold stratified cross-validation...")
    xgb_cv = cross_validate(
        XGBoostTrainer, {"n_estimators": 200, "use_smote": True},
        X_train, y_train, n_splits=5, model_name="XGBoost",
    )
    lgbm_cv = cross_validate(
        LightGBMTrainer, {"n_estimators": 200},
        X_train, y_train, n_splits=5, model_name="LightGBM",
    )
    save_metrics([xgb_cv, lgbm_cv], path="outputs/cv_metrics_report.json")

    # ------------------------------------------------------------------ #
    # 6. SHAP (XGBoost — primary model)
    # ------------------------------------------------------------------ #
    print("\n[Pipeline] Computing SHAP values (XGBoost)...")
    shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
    explainer, shap_vals = compute_shap_values(xgb_model, shap_sample, "XGBoost")
    plot_shap_summary(shap_vals, shap_sample, "XGBoost")
    plot_shap_waterfall(explainer, shap_vals, shap_sample, idx=0, model_name="XGBoost")
    shap_importance = save_shap_artifacts(shap_vals, shap_sample, "XGBoost")

    print("\n[Pipeline] Computing SHAP values (LightGBM)...")
    explainer_lgbm, shap_vals_lgbm = compute_shap_values(lgbm_model, shap_sample, "LightGBM")
    plot_shap_summary(shap_vals_lgbm, shap_sample, "LightGBM")
    save_shap_artifacts(shap_vals_lgbm, shap_sample, "LightGBM")

    # ------------------------------------------------------------------ #
    # 6b. LIME — local explanations for a sample of projects
    # ------------------------------------------------------------------ #
    print("\n[Pipeline] Computing LIME explanations (XGBoost)...")
    lime_explainer = build_lime_explainer(np.array(X_train), feature_names)
    n_lime = min(10, len(shap_sample))
    lime_frames = []
    for i in range(n_lime):
        instance = shap_sample.iloc[i].values
        project_id = df_raw.loc[shap_sample.index[i], "project_id"]
        contrib = explain_instance(lime_explainer, xgb_model, instance, num_features=10)
        contrib.insert(0, "sample_idx", i)
        contrib.insert(0, "project_id", project_id)
        lime_frames.append(contrib)
        plot_lime_explanation(
            lime_explainer, xgb_model, instance,
            project_id=project_id, num_features=10,
        )
    lime_df = pd.concat(lime_frames, ignore_index=True)
    os.makedirs("outputs/lime", exist_ok=True)
    lime_df.to_csv("outputs/lime/lime_contributions.csv", index=False)
    print(f"[Pipeline] LIME explanations saved → outputs/lime/lime_contributions.csv ({n_lime} projects)")

    # ------------------------------------------------------------------ #
    # 7. Risk scores
    # ------------------------------------------------------------------ #
    print("\n[Pipeline] Computing risk scores...")
    # Recover original metadata for risk scoring
    df_test_orig = df_raw.iloc[X_test.index].reset_index(drop=True)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    risk_df = build_risk_dataframe(
        project_ids=df_test_orig["project_id"].values,
        delay_proba=xgb_proba,
        project_types=df_test_orig["project_type"].values,
        project_budgets=df_test_orig["project_budget_musd"].values,
    )
    risk_df["actual_delayed"] = y_test.reset_index(drop=True)
    risk_df.to_csv("outputs/risk_scores.csv", index=False)
    print(f"[Pipeline] Risk scores saved → outputs/risk_scores.csv")
    print("\nRisk Tier Distribution:")
    print(risk_df["risk_tier"].value_counts().to_string())

    # ------------------------------------------------------------------ #
    # 8. Save SHAP + raw test data for dashboard
    # ------------------------------------------------------------------ #
    shap_sample.to_csv("outputs/shap_sample_X.csv", index=False)
    np.save("outputs/shap_values_xgb.npy", shap_vals)
    joblib.dump(explainer, "outputs/shap_explainer_xgb.pkl")
    # Save full X_test + y_test for dashboard use
    X_test.reset_index(drop=True).to_csv("outputs/X_test.csv", index=False)
    y_test.reset_index(drop=True).to_csv("outputs/y_test.csv", index=False)
    df_test_orig.to_csv("outputs/df_test_orig.csv", index=False)

    # ------------------------------------------------------------------ #
    # 9. Save consolidated metrics
    # ------------------------------------------------------------------ #
    all_metrics = [xgb_metrics, lgbm_metrics]
    save_metrics(all_metrics)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  XGBoost  AUC-ROC: {xgb_metrics['AUC-ROC']}")
    print(f"  LightGBM AUC-ROC: {lgbm_metrics['AUC-ROC']}")
    print("  All outputs saved to outputs/")
    print("="*60)
    print("\nLaunch dashboard:  streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    run_pipeline()

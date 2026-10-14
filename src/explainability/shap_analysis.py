"""
SHAP-based global and local explainability for gradient boosting models.
Uses TreeExplainer for fast SHAP value computation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib


def compute_shap_values(model, X: pd.DataFrame, model_name: str = "model"):
    """
    Compute SHAP values using TreeExplainer.
    Returns (explainer, shap_values, X).
    """
    raw_model = getattr(model, "model_", model)
    explainer = shap.TreeExplainer(raw_model)
    shap_values = explainer.shap_values(X)

    # LightGBM returns list [neg, pos]; take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(f"[SHAP] Computed SHAP values for {X.shape[0]} samples, {X.shape[1]} features")
    return explainer, shap_values


def plot_shap_summary(shap_values, X: pd.DataFrame, model_name: str,
                      out_dir: str = "outputs/shap"):
    """Global SHAP summary bar + beeswarm plots."""
    os.makedirs(out_dir, exist_ok=True)
    safe = model_name.lower().replace(" ", "_")

    # Bar chart (mean |SHAP|)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False,
                      color="#e74c3c", max_display=15)
    plt.title(f"{model_name} — Global Feature Importance (SHAP)", fontsize=13, pad=12)
    plt.tight_layout()
    bar_path = f"{out_dir}/{safe}_shap_bar.png"
    plt.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close()

    # Beeswarm summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    plt.title(f"{model_name} — SHAP Summary (Beeswarm)", fontsize=13, pad=12)
    plt.tight_layout()
    bee_path = f"{out_dir}/{safe}_shap_beeswarm.png"
    plt.savefig(bee_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"[SHAP] Plots saved → {bar_path}, {bee_path}")
    return bar_path, bee_path


def plot_shap_waterfall(explainer, shap_values, X: pd.DataFrame, idx: int,
                        model_name: str, out_dir: str = "outputs/shap"):
    """Local waterfall plot for a single prediction."""
    os.makedirs(out_dir, exist_ok=True)
    safe = model_name.lower().replace(" ", "_")

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=expected_value,
        data=X.iloc[idx].values,
        feature_names=X.columns.tolist(),
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.title(f"{model_name} — SHAP Waterfall (Sample #{idx})", fontsize=12)
    plt.tight_layout()
    path = f"{out_dir}/{safe}_waterfall_sample{idx}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Waterfall saved → {path}")
    return path


def get_top_drivers(shap_values, X: pd.DataFrame, idx: int, top_n: int = 5) -> pd.DataFrame:
    """Returns top N risk drivers for a specific project."""
    sv = shap_values[idx]
    driver_df = pd.DataFrame({
        "feature": X.columns,
        "shap_value": sv,
        "abs_impact": np.abs(sv),
        "feature_value": X.iloc[idx].values,
    }).sort_values("abs_impact", ascending=False).head(top_n)
    driver_df["direction"] = driver_df["shap_value"].apply(
        lambda x: "↑ Increases Risk" if x > 0 else "↓ Reduces Risk"
    )
    return driver_df.reset_index(drop=True)


def save_shap_artifacts(shap_values, X: pd.DataFrame, model_name: str,
                        out_dir: str = "outputs/shap"):
    """Save SHAP values and feature importance CSV."""
    os.makedirs(out_dir, exist_ok=True)
    safe = model_name.lower().replace(" ", "_")
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    path = f"{out_dir}/{safe}_feature_importance.csv"
    importance_df.to_csv(path, index=False)
    print(f"[SHAP] Feature importance CSV saved → {path}")
    return importance_df

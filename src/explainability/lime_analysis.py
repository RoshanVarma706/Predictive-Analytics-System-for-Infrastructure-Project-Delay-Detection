"""
LIME-based local explanation for individual project predictions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import joblib


def build_lime_explainer(X_train: np.ndarray, feature_names: list,
                         class_names: list = None) -> LimeTabularExplainer:
    """Build a LimeTabularExplainer fitted on training data statistics."""
    if class_names is None:
        class_names = ["On-Time", "Delayed"]
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    print("[LIME] Explainer built.")
    return explainer


def explain_instance(
    lime_explainer: LimeTabularExplainer,
    model,
    X_instance: np.ndarray,
    num_features: int = 10,
) -> dict:
    """
    Generate LIME explanation for a single instance.
    Returns dict with feature contributions (sorted by abs impact).
    """
    predict_fn = lambda x: model.predict_proba(x)
    exp = lime_explainer.explain_instance(
        X_instance,
        predict_fn,
        num_features=num_features,
        top_labels=2,  # explain both classes so label 1 ("Delayed") is always available
    )
    contributions = exp.as_list(label=1)  # contributions for "Delayed" class
    contrib_df = pd.DataFrame(contributions, columns=["feature_condition", "impact"])
    contrib_df["abs_impact"] = contrib_df["impact"].abs()
    contrib_df = contrib_df.sort_values("abs_impact", ascending=False)
    return contrib_df


def plot_lime_explanation(
    lime_explainer: LimeTabularExplainer,
    model,
    X_instance: np.ndarray,
    project_id: str = "Unknown",
    num_features: int = 10,
    out_dir: str = "outputs/lime",
) -> str:
    """Generate LIME bar chart for a project and save to PNG."""
    os.makedirs(out_dir, exist_ok=True)
    contrib_df = explain_instance(lime_explainer, model, X_instance, num_features)

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in contrib_df["impact"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(contrib_df["feature_condition"], contrib_df["impact"],
                   color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="white", lw=1.2, alpha=0.7)
    ax.set_xlabel("LIME Contribution (toward Delay)", fontsize=11)
    ax.set_title(f"LIME Local Explanation — Project {project_id}", fontsize=12, pad=10)
    ax.invert_yaxis()

    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values(): spine.set_edgecolor("#333366")

    plt.tight_layout()
    safe_id = str(project_id).replace("-", "_")
    path = f"{out_dir}/lime_{safe_id}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"[LIME] Plot saved → {path}")
    return path

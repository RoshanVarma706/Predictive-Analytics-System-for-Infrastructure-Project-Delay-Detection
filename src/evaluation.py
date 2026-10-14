"""
Model evaluation: AUC-ROC, F1, Precision, Recall, MCC, confusion matrix,
precision-recall curve, and stratified k-fold cross-validation.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


def evaluate(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Full evaluation on a held-out test set."""
    os.makedirs("outputs", exist_ok=True)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "AUC-ROC":   round(roc_auc_score(y_test, y_proba), 4),
        "Avg-Precision": round(average_precision_score(y_test, y_proba), 4),
        "F1 (weighted)": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "F1 (macro)":    round(f1_score(y_test, y_pred, average="macro"), 4),
        "Precision":     round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":        round(recall_score(y_test, y_pred), 4),
        "MCC":           round(matthews_corrcoef(y_test, y_pred), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Test Set Metrics")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:20s}: {v}")
    print(classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["On-Time", "Delayed"],
                yticklabels=["On-Time", "Delayed"])
    axes[0].set_title(f"{model_name} — Confusion Matrix")
    axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    axes[1].plot(rec, prec, lw=2, color="#e74c3c",
                 label=f"AP={metrics['Avg-Precision']:.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{model_name} — Precision-Recall Curve")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    safe_name = model_name.lower().replace(" ", "_")
    out_path = f"outputs/{safe_name}_evaluation.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Plot saved → {out_path}")

    return metrics


def cross_validate(model_cls, model_kwargs: dict, X: np.ndarray, y: np.ndarray,
                   n_splits: int = 5, model_name: str = "Model") -> dict:
    """Stratified K-Fold CV — returns mean ± std metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, f1s, aps = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        m = model_cls(**model_kwargs)
        m.fit(
            X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx],
            y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx],
        )
        X_v = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_v = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]
        proba = m.predict_proba(X_v)[:, 1]
        pred  = m.predict(X_v)
        aucs.append(roc_auc_score(y_v, proba))
        f1s.append(f1_score(y_v, pred, average="weighted"))
        aps.append(average_precision_score(y_v, proba))
        print(f"  Fold {fold}: AUC={aucs[-1]:.4f}  F1={f1s[-1]:.4f}")

    cv_metrics = {
        "model": model_name,
        "CV_AUC_mean": round(np.mean(aucs), 4),
        "CV_AUC_std":  round(np.std(aucs), 4),
        "CV_F1_mean":  round(np.mean(f1s), 4),
        "CV_F1_std":   round(np.std(f1s), 4),
        "CV_AP_mean":  round(np.mean(aps), 4),
    }
    print(f"\n[CV {model_name}] AUC: {cv_metrics['CV_AUC_mean']} ± {cv_metrics['CV_AUC_std']}")
    return cv_metrics


def save_metrics(metrics_list: list, path: str = "outputs/metrics_report.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"[Evaluation] Metrics saved → {path}")

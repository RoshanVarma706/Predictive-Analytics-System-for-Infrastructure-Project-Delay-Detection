"""
Risk scoring matrix: converts model probability outputs to 0–100 risk scores
and assigns tiers based on probability × impact severity.
"""

import numpy as np
import pandas as pd


# Risk tier boundaries
RISK_TIERS = {
    "Low":      (0,  30),
    "Medium":   (31, 60),
    "High":     (61, 80),
    "Critical": (81, 100),
}

# Impact severity weights by project type (budget-adjusted later)
PROJECT_TYPE_IMPACT = {
    "road":     1.0,
    "bridge":   1.4,
    "pipeline": 1.2,
    "building": 0.9,
    "rail":     1.5,
}


def compute_risk_scores(
    delay_proba: np.ndarray,
    project_types: np.ndarray = None,
    project_budgets: np.ndarray = None,
) -> np.ndarray:
    """
    Convert delay probabilities → risk scores (0–100).
    Optionally amplifies by impact (project_type + budget).

    Args:
        delay_proba: shape (n,) probabilities of delay (class 1)
        project_types: array of project type strings (optional)
        project_budgets: array of budget in M USD (optional)

    Returns:
        risk_scores: shape (n,) float array in [0, 100]
    """
    base_score = delay_proba * 100  # scale to 0–100

    if project_types is not None and project_budgets is not None:
        # Impact severity: type weight × log-normalized budget
        type_weights = np.array([
            PROJECT_TYPE_IMPACT.get(t, 1.0) for t in project_types
        ])
        budget_norm = np.log1p(project_budgets) / np.log1p(500)  # normalize to [0,1]
        impact_multiplier = 0.7 + 0.3 * (type_weights * budget_norm)
        risk_scores = np.clip(base_score * impact_multiplier, 0, 100)
    else:
        risk_scores = np.clip(base_score, 0, 100)

    return risk_scores


def assign_risk_tier(score: float) -> str:
    """Return tier label for a single risk score."""
    if score <= 30: return "Low"
    elif score <= 60: return "Medium"
    elif score <= 80: return "High"
    else:            return "Critical"


def build_risk_dataframe(
    project_ids: np.ndarray,
    delay_proba: np.ndarray,
    project_types: np.ndarray = None,
    project_budgets: np.ndarray = None,
) -> pd.DataFrame:
    """Produce a risk summary DataFrame for the dashboard."""
    scores = compute_risk_scores(delay_proba, project_types, project_budgets)
    tiers = [assign_risk_tier(s) for s in scores]

    return pd.DataFrame({
        "project_id":    project_ids,
        "delay_prob":    np.round(delay_proba, 4),
        "risk_score":    np.round(scores, 1),
        "risk_tier":     tiers,
    })

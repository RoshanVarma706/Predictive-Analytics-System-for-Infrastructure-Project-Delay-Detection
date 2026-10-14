"""
Feature engineering: creates domain-specific interaction features and risk tiers.
"""

import numpy as np
import pandas as pd


CONTRACTOR_TIER_MAP = {"Tier1": 0, "Tier2": 1, "Tier3": 2}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features to raw dataframe (before preprocessing).
    Call this BEFORE the preprocessor fit/transform.
    """
    df = df.copy()

    # --- Ratio features ---
    df["permit_ratio"] = df["permit_processing_days"] / (df["planned_duration_days"] + 1)
    df["start_delay_ratio"] = df["actual_start_delay_days"] / (df["planned_duration_days"] + 1)

    # --- Interaction features ---
    df["land_reg_interaction"] = df["land_acquisition_issues"] * df["regulatory_hurdles"]
    df["budget_workforce_interaction"] = (
        df["budget_utilization_pct"] / 100 * df["workforce_availability_pct"] / 100
    )

    # --- Contractor risk tier (numeric) ---
    df["contractor_risk_tier"] = df["contractor_tier"].map(CONTRACTOR_TIER_MAP).fillna(1)

    print(f"[Feature Engineering] Added 5 engineered features → total columns: {df.shape[1]}")
    return df

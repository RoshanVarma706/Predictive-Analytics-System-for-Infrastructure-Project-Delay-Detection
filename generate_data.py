"""
Synthetic data generator for the Predictive Analytics System.
Generates ~5,000 realistic infrastructure project records with realistic
delay patterns, land acquisition issues, and resource constraints.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

def generate_data(n: int = N, output_path: str = "data/raw/project_data.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Categorical features ---
    project_types = np.random.choice(
        ["road", "bridge", "pipeline", "building", "rail"],
        size=n, p=[0.30, 0.15, 0.20, 0.25, 0.10]
    )
    regions = np.random.choice(
        ["North", "South", "East", "West", "Central"],
        size=n, p=[0.20, 0.22, 0.18, 0.20, 0.20]
    )
    contractor_tiers = np.random.choice(
        ["Tier1", "Tier2", "Tier3"],
        size=n, p=[0.25, 0.45, 0.30]
    )

    # --- Numerical schedule features ---
    planned_duration_days = np.random.randint(90, 1500, size=n)
    actual_start_delay_days = np.clip(
        np.random.exponential(scale=15, size=n).astype(int), 0, 180
    )
    permit_processing_days = np.clip(
        np.random.exponential(scale=20, size=n).astype(int), 1, 240
    )

    # --- Resource features ---
    budget_utilization_pct = np.clip(np.random.normal(78, 18, n), 30, 130)
    workforce_availability_pct = np.clip(np.random.normal(72, 15, n), 20, 100)
    equipment_availability_pct = np.clip(np.random.normal(75, 14, n), 20, 100)

    # --- Risk events ---
    land_acquisition_issues = np.random.binomial(1, 0.28, size=n)
    regulatory_hurdles = np.random.poisson(1.2, size=n)
    weather_events = np.random.poisson(0.8, size=n)
    stakeholder_conflicts = np.random.binomial(1, 0.20, size=n)

    # --- Historical performance ---
    contractor_past_delays = np.random.poisson(1.5, size=n)
    similar_project_avg_delay_days = np.clip(
        np.random.exponential(30, size=n).astype(int), 0, 300
    )

    # --- Budget & scope ---
    project_budget_musd = np.round(np.random.uniform(0.5, 500, size=n), 2)

    # --- Target: binary delay flag (class imbalance ~35% positive) ---
    # Build a latent score based on realistic risk drivers
    risk_score_latent = (
        0.25 * land_acquisition_issues
        + 0.15 * (regulatory_hurdles / 5)
        + 0.10 * (actual_start_delay_days / 180)
        + 0.12 * (permit_processing_days / 240)
        + 0.10 * np.where(contractor_tiers == "Tier3", 1, np.where(contractor_tiers == "Tier2", 0.5, 0))
        + 0.08 * stakeholder_conflicts
        + 0.05 * (weather_events / 5)
        + 0.08 * (budget_utilization_pct / 130)
        + 0.07 * (contractor_past_delays / 8)
        + np.random.uniform(0, 0.15, size=n)   # noise
    )
    delay_probability = 1 / (1 + np.exp(-8 * (risk_score_latent - 0.45)))
    delayed = (delay_probability > np.random.uniform(0, 1, size=n)).astype(int)

    df = pd.DataFrame({
        "project_id": [f"PRJ-{i:05d}" for i in range(n)],
        "project_type": project_types,
        "region": regions,
        "contractor_tier": contractor_tiers,
        "planned_duration_days": planned_duration_days,
        "actual_start_delay_days": actual_start_delay_days,
        "permit_processing_days": permit_processing_days,
        "budget_utilization_pct": budget_utilization_pct.round(1),
        "workforce_availability_pct": workforce_availability_pct.round(1),
        "equipment_availability_pct": equipment_availability_pct.round(1),
        "land_acquisition_issues": land_acquisition_issues,
        "regulatory_hurdles": regulatory_hurdles,
        "weather_events": weather_events,
        "stakeholder_conflicts": stakeholder_conflicts,
        "contractor_past_delays": contractor_past_delays,
        "similar_project_avg_delay_days": similar_project_avg_delay_days,
        "project_budget_musd": project_budget_musd,
        "delayed": delayed,
    })

    df.to_csv(output_path, index=False)
    print(f"[Data Generator] Saved {len(df):,} records → {output_path}")
    print(f"  Delay rate: {delayed.mean():.1%}")
    return df


if __name__ == "__main__":
    generate_data()

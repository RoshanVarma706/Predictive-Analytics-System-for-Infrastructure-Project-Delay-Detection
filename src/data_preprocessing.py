"""
Data preprocessing pipeline: loading, cleaning, encoding, scaling, and splitting.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib


CATEGORICAL_FEATURES = ["project_type", "region", "contractor_tier"]
NUMERIC_FEATURES = [
    "planned_duration_days", "actual_start_delay_days", "permit_processing_days",
    "budget_utilization_pct", "workforce_availability_pct", "equipment_availability_pct",
    "land_acquisition_issues", "regulatory_hurdles", "weather_events",
    "stakeholder_conflicts", "contractor_past_delays", "similar_project_avg_delay_days",
    "project_budget_musd",
    # Engineered features added later
    "permit_ratio", "start_delay_ratio", "land_reg_interaction",
    "budget_workforce_interaction", "contractor_risk_tier",
]
TARGET = "delayed"
DROP_COLS = ["project_id"]


def load_raw(path: str = "data/raw/project_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[Preprocessing] Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def build_preprocessor(feature_names: list) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for numeric + categorical features."""
    num_feats = [f for f in NUMERIC_FEATURES if f in feature_names]
    cat_feats = [f for f in CATEGORICAL_FEATURES if f in feature_names]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, num_feats),
        ("cat", categorical_pipeline, cat_feats),
    ], remainder="drop")
    return preprocessor


def preprocess(
    df: pd.DataFrame,
    preprocessor=None,
    fit: bool = True,
    save_path: str = "outputs/preprocessor.pkl",
) -> tuple:
    """
    Returns (X_transformed_df, y, preprocessor, feature_names).
    If fit=True, fits and saves the preprocessor.
    If fit=False, uses the provided preprocessor.
    """
    os.makedirs("outputs", exist_ok=True)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # y = df[TARGET].astype(int)
    # X = df.drop(columns=[TARGET])

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET])

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    feature_names = X.columns.tolist()

    if fit:
        preprocessor = build_preprocessor(feature_names)
        X_transformed = preprocessor.fit_transform(X)
        joblib.dump(preprocessor, save_path)
        print(f"[Preprocessing] Preprocessor saved → {save_path}")
    else:
        X_transformed = preprocessor.transform(X)

    # Reconstruct column names
    num_feats = [f for f in NUMERIC_FEATURES if f in feature_names]
    cat_feats = [f for f in CATEGORICAL_FEATURES if f in feature_names]
    out_cols = num_feats + cat_feats

    X_out = pd.DataFrame(X_transformed, columns=out_cols, index=X.index)
    return X_out, y, preprocessor, out_cols


def split_data(X: pd.DataFrame, y: pd.Series, val_size: float = 0.15, test_size: float = 0.15):
    """70 / 15 / 15 stratified split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), stratify=y, random_state=42
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, stratify=y_temp, random_state=42
    )
    print(f"[Split] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test

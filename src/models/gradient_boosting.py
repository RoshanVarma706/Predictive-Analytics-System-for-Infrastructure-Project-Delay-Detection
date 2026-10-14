"""
Gradient Boosting models: XGBoost and LightGBM trainers with imbalance handling.
Both implement a unified BaseModel interface.
"""

import os
import numpy as np
import pandas as pd
import joblib
import json
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    def predict_proba(self, X): raise NotImplementedError
    def save(self, path): joblib.dump(self, path); print(f"[Model] Saved → {path}")

    @classmethod
    def load(cls, path):
        return joblib.load(path)


# ---------------------------------------------------------------------------
# XGBoost Trainer
# ---------------------------------------------------------------------------

class XGBoostTrainer(BaseModel):
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        use_smote: bool = True,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.use_smote = use_smote
        self.random_state = random_state
        self.model_ = None
        self.feature_names_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else None

        X_arr = np.array(X)
        y_arr = np.array(y)

        if self.use_smote:
            smote = SMOTE(random_state=self.random_state)
            X_arr, y_arr = smote.fit_resample(X_arr, y_arr)
            print(f"[XGBoost] After SMOTE: {np.bincount(y_arr)}")

        pos_weight = (y_arr == 0).sum() / max((y_arr == 1).sum(), 1)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(np.array(X_val), np.array(y_val))]

        model_kwargs = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        # Early stopping requires a validation dataset (not used in CV folds)
        if eval_set is not None:
            model_kwargs["early_stopping_rounds"] = 30
        self.model_ = xgb.XGBClassifier(**model_kwargs)

        self.model_.fit(
            X_arr, y_arr,
            eval_set=eval_set,
            verbose=False,
        )
        print(f"[XGBoost] Trained. Best iteration: {getattr(self.model_, 'best_iteration', 'n/a')}")
        return self

    def predict(self, X):
        return self.model_.predict(np.array(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.array(X))

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


# ---------------------------------------------------------------------------
# LightGBM Trainer
# ---------------------------------------------------------------------------

class LightGBMTrainer(BaseModel):
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = -1,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model_ = None
        self.feature_names_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else None

        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            is_unbalance=True,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

        callbacks = [lgb.log_evaluation(period=-1)]
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(np.array(X_val), np.array(y_val))]
            # Only apply early stopping when an eval set exists
            callbacks.insert(0, lgb.early_stopping(30, verbose=False))

        self.model_.fit(
            np.array(X), np.array(y),
            eval_set=eval_set,
            callbacks=callbacks,
        )
        print(f"[LightGBM] Trained. Best iteration: {getattr(self.model_, 'best_iteration_', 'n/a')}")
        return self

    def predict(self, X):
        return self.model_.predict(np.array(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(np.array(X))

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_

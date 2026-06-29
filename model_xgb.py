"""
--------------------------------------------
SIGNAL PIPELINE - MODEL A: XGBOOST
--------------------------------------------

Gradient-boosted trees on the WIDE point-in-time feature set. Trees prune
noisy features themselves, so we feed the full set and let SHAP tell us what
mattered. Runs on CPU - no CUDA, no GPU headaches.

Outputs a win probability (0..1) per (ticker, date).
"""

import numpy as np
import pandas as pd

import config


def prepare_tabular(df: pd.DataFrame, feature_cols: list[str]):
    """Return (X, y): one feature row per day, label from labels.py."""
    raise NotImplementedError


def train_xgb(X_train, y_train, scale_pos_weight: float | None = None):
    """
    Train an XGBoost classifier.
    scale_pos_weight handles class imbalance (winning labels are the minority).
    Returns the fitted model.
    """
    import xgboost as xgb   # lazy import - module stays importable before xgboost is installed
    raise NotImplementedError


def predict_xgb(model, X) -> np.ndarray:
    """Return win-probability (0..1) for each row."""
    raise NotImplementedError

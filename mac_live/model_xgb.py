"""
--------------------------------------------
SIGNAL PIPELINE - MODEL A: XGBOOST
--------------------------------------------

Gradient-boosted trees on the WIDE feature set (config.WIDE_FEATURES). Trees prune
noisy features themselves, so we feed the full set. CPU only - no CUDA.

Input frame = features.py + labels.add_target (has the WIDE cols + Target_Label).
Outputs an (uncalibrated) win probability per row; calibration happens in ensemble.py.
"""

import numpy as np
import pandas as pd

import config


# ----------------------------------
# DATA PREP
# ----------------------------------

def prepare_tabular(df: pd.DataFrame,
                    feature_cols: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    # feature/label frame -> (X, y) arrays for XGBoost
    cols = feature_cols or config.WIDE_FEATURES
    X = df[cols].to_numpy(dtype="float32")
    y = df["Target_Label"].to_numpy(dtype="int8")
    return X, y


# ----------------------------------
# TRAIN / PREDICT
# ----------------------------------

def train_xgb(X_train, y_train, scale_pos_weight: float | None = None):
    """
    Modest, regularised trees. scale_pos_weight balances the ~1.7:1 loss:win skew;
    if not given it's derived from the training labels.
    """
    import xgboost as xgb

    if scale_pos_weight is None:
        pos = max(int((y_train == 1).sum()), 1)
        scale_pos_weight = float((y_train == 0).sum()) / pos

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_xgb(model, X) -> np.ndarray:
    # win probability in [0, 1] (uncalibrated - ensemble.py calibrates)
    return model.predict_proba(X)[:, 1]

"""
--------------------------------------------
SIGNAL PIPELINE - ENSEMBLE / COMBINER
--------------------------------------------

Combines the two base-model OOF probabilities (XGBoost + CNN-LSTM) into one final
score. The models are diverse (measured corr ~0.42), so combining is both more
accurate and steadier than either alone.

Discipline: calibrators + meta-learner are FIT on some OOF folds and evaluated on
LATER, unseen folds - never fit and scored on the same rows (that would leak).
"""

import numpy as np


# ----------------------------------
# CALIBRATION  (Platt / logistic)
# ----------------------------------

def fit_calibrator(p_raw, y):
    # Platt scaling: a 1-D logistic maps a raw score -> an honest probability
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression().fit(np.asarray(p_raw).reshape(-1, 1), np.asarray(y))


def apply_calibrator(clf, p_raw) -> np.ndarray:
    return clf.predict_proba(np.asarray(p_raw).reshape(-1, 1))[:, 1]


# ----------------------------------
# COMBINERS
# ----------------------------------

def weighted_blend(p1, p2, w1: float = 0.5, w2: float = 0.5) -> np.ndarray:
    return w1 * np.asarray(p1) + w2 * np.asarray(p2)


def agreement_gate(p1, p2, threshold: float = 0.75) -> np.ndarray:
    # 1 only when BOTH models clear the threshold (high-precision selectivity)
    return ((np.asarray(p1) >= threshold) & (np.asarray(p2) >= threshold)).astype("int8")


def train_meta_learner(P, y):
    # low-capacity logistic stacker on [p_xgb, p_lstm] (+ optional regime cols)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression().fit(np.asarray(P), np.asarray(y))


def predict_meta(clf, P) -> np.ndarray:
    return clf.predict_proba(np.asarray(P))[:, 1]

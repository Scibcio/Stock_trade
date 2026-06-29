"""
--------------------------------------------
SIGNAL PIPELINE - ENSEMBLE / COMBINER
--------------------------------------------

Combines the two base-model probabilities (XGBoost + CNN-LSTM) into one final
score. Because the models are very different (tabular trees vs temporal net),
their mistakes are decorrelated - so combining them is more accurate and
steadier than either alone.

Start simple (agreement gate + weighted blend). The optional logistic
meta-learner MUST train on OUT-OF-FOLD predictions (the walk-forward test
predictions) or it leaks the future and lies to you.
"""

import numpy as np
import pandas as pd


def weighted_blend(p_xgb: np.ndarray,
                   p_lstm: np.ndarray,
                   w_xgb: float = 0.5,
                   w_lstm: float = 0.5) -> np.ndarray:
    """Weighted average of the two probabilities -> one final score per row."""
    raise NotImplementedError


def agreement_gate(p_xgb: np.ndarray,
                   p_lstm: np.ndarray,
                   threshold: float = 0.75) -> np.ndarray:
    """
    Return a 0/1 'take the trade' mask: 1 only when BOTH models clear `threshold`.
    Fewer trades, higher precision - matches the selective trading style.
    """
    raise NotImplementedError


def train_meta_learner(oof_preds: pd.DataFrame, y: np.ndarray):
    """
    Optional combiner: logistic regression on [p_xgb, p_lstm], trained ONLY on
    out-of-fold base predictions to avoid leakage. Returns the fitted combiner.
    Keep it simple - a complex stacker on top overfits fast.
    """
    raise NotImplementedError

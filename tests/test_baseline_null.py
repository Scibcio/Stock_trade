"""
Gate test for baseline_null.py - the fast rank-AUC must match sklearn exactly.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import baseline_null


def test_fast_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < 0.4).astype(int)
    ranks = rankdata(p)
    pos_idx = np.where(y == 1)[0]
    fast = baseline_null._fast_auc(ranks, pos_idx, int(y.sum()), len(y))
    assert abs(fast - roc_auc_score(y, p)) < 1e-9

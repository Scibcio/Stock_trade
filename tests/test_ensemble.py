"""
Gate tests for ensemble.py (Phase 4).
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.metrics import roc_auc_score

import ensemble


def test_weighted_blend():
    assert np.allclose(ensemble.weighted_blend([0.2, 0.8], [0.4, 0.6]), [0.3, 0.7])


def test_agreement_gate():
    g = ensemble.agreement_gate([0.8, 0.8, 0.4], [0.8, 0.4, 0.9], threshold=0.75)
    assert list(g) == [1, 0, 0]


def test_calibrator_outputs_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    c = ensemble.apply_calibrator(ensemble.fit_calibrator(p, y), p)
    assert c.min() >= 0.0 and c.max() <= 1.0


def test_meta_learner_beats_singles():
    # two weak, INDEPENDENT signals -> the stacker should beat the better single
    rng = np.random.default_rng(1)
    n = 6000
    y = rng.integers(0, 2, n)
    p1 = np.clip(y * 0.3 + rng.uniform(0, 0.7, n), 0, 1)
    p2 = np.clip(y * 0.3 + rng.uniform(0, 0.7, n), 0, 1)
    tr, te = slice(0, 4500), slice(4500, n)
    m = ensemble.train_meta_learner(np.c_[p1[tr], p2[tr]], y[tr])
    pm = ensemble.predict_meta(m, np.c_[p1[te], p2[te]])
    best = max(roc_auc_score(y[te], p1[te]), roc_auc_score(y[te], p2[te]))
    assert roc_auc_score(y[te], pm) > best

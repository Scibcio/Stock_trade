"""
Gate tests for model_xgb.py (Phase 3, Model A).
Proves the model LEARNS (beats random on held-out data), not just runs.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
from sklearn.metrics import roc_auc_score

import config
import features
import labels
import model_xgb


def test_learns_on_signal():
    # a learnable dataset -> held-out AUC must clearly beat random
    rng = np.random.default_rng(0)
    n = 4000
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=0.5, size=n) > 0).astype(int)
    X = np.column_stack([signal, rng.normal(size=n)])          # 1 real + 1 noise feature
    model = model_xgb.train_xgb(X[:3000], y[:3000])
    auc = roc_auc_score(y[3000:], model_xgb.predict_xgb(model, X[3000:]))
    assert auc > 0.7


def test_predictions_are_probabilities():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 4))
    y = (X[:, 0] > 0).astype(int)
    p = model_xgb.predict_xgb(model_xgb.train_xgb(X, y), X)
    assert p.min() >= 0.0 and p.max() <= 1.0


@pytest.mark.skipif(not config.DB_PATH.exists(), reason="trading.db not present")
def test_runs_on_real_db():
    # end-to-end on pooled dev tickers, time-split; sane AUC band (not inverted, not leaking)
    conn = sqlite3.connect(config.DB_PATH)
    df = pd.concat(
        [labels.add_target(features.build_feature_table(t, conn)) for t in config.DEV_UNIVERSE[:6]],
        ignore_index=True,
    )
    conn.close()
    train, test = df[df["date"] < "2023-01-01"], df[df["date"] >= "2023-01-01"]
    model = model_xgb.train_xgb(*model_xgb.prepare_tabular(train))
    Xte, yte = model_xgb.prepare_tabular(test)
    auc = roc_auc_score(yte, model_xgb.predict_xgb(model, Xte))
    assert 0.45 < auc < 0.95      # sane: not label-inverted, not suspiciously perfect (leakage)

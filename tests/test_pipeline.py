"""
Gate tests for pipeline.py (Phase 3 walk-forward).
Purge logic + a small real-data smoke of the fold loop.
"""

import sqlite3

import pytest

pytest.importorskip("xgboost")

import config
import pipeline


def test_purge_cutoff_is_before_train_end():
    cut = pipeline._purge_cutoff("2014-01-01")
    assert cut < "2014-01-01"


@pytest.mark.skipif(not config.DB_PATH.exists(), reason="trading.db not present")
def test_walk_forward_smoke():
    # 3 dev tickers through the real fold loop
    conn = sqlite3.connect(config.DB_PATH)
    pooled = pipeline.build_dataset(config.DEV_UNIVERSE[:3], conn)
    conn.close()

    oof, metrics = pipeline.walk_forward(pooled)
    assert len(oof) > 0
    assert oof["p_xgb"].between(0, 1).all()
    assert metrics["auc"].notna().all()
    # OOF must never contain a training-period row (test windows only)
    assert (oof["date"] >= config.FOLDS[0]["test_start"]).all()

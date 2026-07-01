"""
Gate tests for labels.py (Phase 2).
Hand-built series with known outcomes + a real-data class-balance sanity check.
"""

import sqlite3

import pandas as pd
import pytest

import config
import features
import labels


def _df(closes):
    return pd.DataFrame({"close": closes})


def test_win_label():
    # +3% reached before any -1% -> win
    s = labels.triple_barrier(_df([100, 101, 103.5, 100]), 0.03, -0.01, 3)
    assert s.iloc[0] == 1


def test_loss_label():
    # -1% hit first -> loss, even though +5% comes later
    s = labels.triple_barrier(_df([100, 98.5, 105, 100]), 0.03, -0.01, 3)
    assert s.iloc[0] == 0


def test_timeout_label():
    # neither barrier touched inside the window -> loss
    s = labels.triple_barrier(_df([100, 100.4, 100.2, 100.5]), 0.03, -0.01, 3)
    assert s.iloc[0] == 0


def test_tail_is_unlabelled():
    # last hold_days rows have no forward window -> NaN
    s = labels.triple_barrier(_df([100] * 10), 0.03, -0.01, 3)
    assert s.iloc[-3:].isna().all()


@pytest.mark.skipif(not config.DB_PATH.exists(), reason="trading.db not present")
def test_labels_real_db():
    # win rate on real data should be a sane fraction (FYP anchor ~0.37)
    conn = sqlite3.connect(config.DB_PATH)
    df = labels.add_target(features.build_feature_table(config.DEV_UNIVERSE[0], conn))
    conn.close()
    assert set(df["Target_Label"].unique()) <= {0, 1}
    assert 0.0 < df["Target_Label"].mean() < 1.0

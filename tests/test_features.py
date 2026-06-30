"""
Gate tests for features.py (Phase 1).
The leakage test (test_no_lookahead) is the one that matters most.
"""

import sqlite3

import pandas as pd
import pytest

import config
import features


def test_builds_expected_columns(sample):
    # every declared feature is produced
    ohlcv, baselines = sample
    out = features.compute_features(ohlcv, baselines)
    assert set(config.WIDE_FEATURES).issubset(out.columns)


def test_no_nan_in_features(sample):
    # no NaNs survive after warmup dropna
    ohlcv, baselines = sample
    out = features.compute_features(ohlcv, baselines)
    assert len(out) > 0
    assert not out[config.WIDE_FEATURES].isna().any().any()


def test_daily_return_known_value(sample):
    # spot-check one feature by hand: Daily_Return = close[t]/close[t-1] - 1
    ohlcv, baselines = sample
    out = features.compute_features(ohlcv, baselines)
    row = out.iloc[50]
    prev_close = ohlcv.loc[ohlcv["date"] == out["date"].iloc[49], "close"].iloc[0]
    expected = row["close"] / prev_close - 1
    assert row["Daily_Return"] == pytest.approx(expected, abs=1e-9)


def test_no_lookahead(sample):
    # THE leakage test: a day's features must not change if future data is removed.
    ohlcv, baselines = sample
    cut = 350
    target = ohlcv["date"].iloc[cut]

    full = features.compute_features(ohlcv, baselines)
    part = features.compute_features(
        ohlcv.iloc[:cut + 1],
        {k: v[v["date"] <= target] for k, v in baselines.items()},
    )

    cols = config.WIDE_FEATURES
    full_row = full.loc[full["date"] == target, cols].reset_index(drop=True)
    part_row = part.loc[part["date"] == target, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_row, part_row, check_exact=False, atol=1e-9)


@pytest.mark.skipif(not config.DB_PATH.exists(), reason="trading.db not present")
def test_real_db_smoke():
    # end-to-end on a real dev-universe ticker straight from trading.db
    conn = sqlite3.connect(config.DB_PATH)
    out = features.build_feature_table(config.DEV_UNIVERSE[0], conn)
    conn.close()
    assert len(out) > 0
    assert not out[config.WIDE_FEATURES].isna().any().any()

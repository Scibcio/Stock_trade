"""
Gate tests for threshold_analysis.py (Phase 5).
"""

import numpy as np
import pandas as pd

import threshold_analysis as ta


def test_ev_breakeven_and_scale():
    assert abs(ta.ev(0.25)) < 1e-9       # 25% win rate = breakeven at 3:1
    assert abs(ta.ev(0.50) - 1.0) < 1e-9  # 50% win rate = +1 per trade


def test_regime_rules():
    assert ta.label_regime(100, 110, 15) == "bear"       # below 200MA
    assert ta.label_regime(120, 110, 15) == "bull"       # above 200MA, calm VIX
    assert ta.label_regime(120, 110, 30) == "sideways"   # above 200MA, high VIX
    assert ta.label_regime(120, float("nan"), 15) == "warmup"


def test_sweep_trades_decrease_with_selectivity():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"blend": rng.uniform(0, 1, n),
                       "Target_Label": rng.integers(0, 2, n),
                       "regime": "bull"})
    rows = ta.sweep(df, top_pcts=(100, 50, 10))
    allrows = rows[rows["regime"] == "ALL"].sort_values("top_%", ascending=False)
    assert allrows["trades"].is_monotonic_decreasing

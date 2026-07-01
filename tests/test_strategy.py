"""
Gate tests for strategy.py (Phase 5).
"""

import numpy as np
import pandas as pd

import strategy


def test_edge_ev_formula():
    # 50% win at 3:1, no cost -> EV = 0.5*3 - 0.5 = +1.0 risk-units
    trades = pd.DataFrame({"Target_Label": [1, 0, 1, 0]})
    cfg = strategy.StrategyConfig(cost=0.0)
    assert abs(strategy.edge(trades, cfg)["ev_per_trade"] - 1.0) < 1e-9


def test_select_top_per_day_caps_k():
    df = pd.DataFrame({
        "date": ["2020-01-01"] * 8 + ["2020-01-02"] * 4,
        "ticker": list("ABCDEFGH") + list("WXYZ"),
        "blend": np.linspace(0, 1, 12),
        "Target_Label": [0, 1] * 6,
    })
    sel = strategy.select_top_per_day(df, k=3)
    assert (sel.groupby("date").size() <= 3).all()


def test_monte_carlo_edge_grows_and_ruin_bounds():
    cfg = strategy.StrategyConfig()
    good = strategy.monte_carlo(0.45, cfg, years=5, n_sims=300)
    bad = strategy.monte_carlo(0.10, cfg, years=5, n_sims=300)
    assert good["median_mult"] > 1.0          # a real edge grows
    assert bad["risk_of_50pct_dd"] > good["risk_of_50pct_dd"]   # a bad edge ruins more
    assert 0.0 <= good["risk_of_50pct_dd"] <= 1.0

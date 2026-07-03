"""
Gate tests for the event-driven trailing engine (backtest_trailing) - the
next-open fill discipline (F1) on synthetic paths with known outcomes.
"""

import numpy as np
import pandas as pd

from backtest_trailing import simulate_trailing

DATES = [f"2024-01-{d:02d}" for d in range(1, 9)]


def _world(opens, closes):
    prices = {"AAA": pd.DataFrame({"open": opens, "close": closes}, index=DATES)}
    df = pd.DataFrame({"date": DATES, "ticker": "AAA", "sector": "S",
                       "regime": "bull", "prob": 0.9})
    return df, prices


def test_entry_fills_at_next_session_open():
    # decision on day 0 (close 100), overnight gap -> day 1 opens at 110.
    # the position must cost 110, not the untradeable 100.
    opens = [100, 110, 112, 112, 112, 112, 112, 112]
    closes = [100, 112, 113, 114, 115, 116, 117, 118]
    df, prices = _world(opens, closes)
    curve, trades, cash = simulate_trailing(df, prices, signal_col="prob",
                                            n_slots=1, trail=0.99, max_hold=60, cost=0.001)
    assert len(trades) == 1
    tr = trades.iloc[0]
    assert tr["entry_date"] == DATES[1]                     # filled the day AFTER the decision
    assert np.isclose(tr["return"], 118 / 110 * 0.999 - 1)  # entry at the 110 open, not 100
    assert np.isclose(cash, 1 * (118 / 110) * 0.999)        # cash conserves the same trade


def test_entry_day_close_can_stop_out():
    # fills at 110, same session closes at 100 (-9.1%) -> initial stop exits day 1
    opens = [100, 110, 100, 100, 100, 100, 100, 100]
    closes = [100, 100, 100, 100, 100, 100, 100, 100]
    df, prices = _world(opens, closes)
    _, trades, _ = simulate_trailing(df, prices, signal_col="prob",
                                     n_slots=1, trail=0.10, max_hold=60, cost=0.001)
    tr = trades.iloc[0]
    assert tr["entry_date"] == DATES[1] and tr["exit_date"] == DATES[1]
    assert tr["days"] == 1
    assert np.isclose(tr["return"], 100 / 110 * 0.999 - 1)

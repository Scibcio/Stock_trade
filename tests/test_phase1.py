"""
Phase-1 correctness fixes (handoff F1-F8) - one gate test per fix.
"""

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtest
import config
import predict_live
import strategy

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------
# F1 - next-open entries
# ----------------------------------

def test_next_open_entry_uses_gap_price():
    # signal on day 0's close (100); overnight gap -> day 1 opens at 110.
    # A close-entry would book 100 (untradeable); the honest entry is 110.
    open_ = np.array([100.0, 110.0, 110.0, 110.0, 110.0, 110.0])
    close = np.array([100.0, 110.0, 110.0, 110.0, 110.0, 121.0])
    r = backtest.realized_return_series(open_, close, take_profit=None,
                                        stop_loss=None, hold=4)
    assert np.isclose(r[0], 121.0 / 110.0 - 1)      # +10%, NOT +21% from the stale close
    assert np.isnan(r[1:]).all()                    # tail (hold+1 rows) is untradeable


def test_next_open_barriers_start_at_entry_session():
    # entry day's own close can hit a barrier (entry = that morning's open)
    open_ = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    close = np.array([100.0, 104.0, 100.0, 100.0, 100.0, 100.0])
    r = backtest.realized_return_series(open_, close, take_profit=0.03,
                                        stop_loss=-0.03, hold=4)
    assert np.isclose(r[0], 0.04)                   # TP touched on the entry session itself


def test_fill_pending_uses_next_session_open():
    conn = sqlite3.connect(":memory:")
    predict_live.migrate(conn)
    conn.execute("CREATE TABLE daily_prices (ticker TEXT, date TEXT, open REAL, close REAL)")
    conn.execute("INSERT INTO paper_trades (pick_date, ticker, entry_close, status) "
                 "VALUES ('2020-01-01', 'AAA', 100, 'pending')")
    conn.execute("INSERT INTO daily_prices VALUES ('AAA', '2020-01-02', 110, 111)")
    assert predict_live.fill_pending(conn) == 1
    row = conn.execute("SELECT entry_open, status FROM paper_trades").fetchone()
    assert row == (110.0, "open")
    conn.close()


# ----------------------------------
# F2 - constants live in config only
# ----------------------------------

def test_strategy_constants_defined_once():
    pattern = re.compile(r"^\s*(TOP_K|MAX_PER_SECTOR|REGIME_EXPOSURE)\s*=(?!\s*config\.)", re.M)
    offenders = []
    for py in ROOT.glob("*.py"):
        if py.name == "config.py":
            continue
        for m in pattern.finditer(py.read_text(encoding="utf-8")):
            offenders.append(f"{py.name}: {m.group(0).strip()}")
    assert not offenders, f"strategy constants redefined outside config.py: {offenders}"


# ----------------------------------
# F3 - bear regime = 100% cash
# ----------------------------------

def _synthetic_day(n=30):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "ticker": [f"T{i:02d}" for i in range(n)],
        "sector": [f"S{i % 5}" for i in range(n)],
        "NATR_14": rng.uniform(1, 8, n),
        "prob": rng.uniform(0.3, 0.9, n),
        "regime": "bull",
        "close": rng.uniform(20, 500, n),
    })


def test_bear_regime_selects_nothing():
    assert config.REGIME_EXPOSURE["bear"] == 0.0
    assert strategy.select_cohort(_synthetic_day(), "prob", "bear") is None


# ----------------------------------
# F4 - schema migration
# ----------------------------------

def test_migration_upgrades_legacy_schema():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE paper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, pick_date TEXT NOT NULL,
        ticker TEXT NOT NULL, entry_close REAL, prob REAL, regime TEXT,
        natr REAL, status TEXT DEFAULT 'open', UNIQUE(pick_date, ticker))""")
    conn.execute("INSERT INTO paper_trades (pick_date, ticker, entry_close) "
                 "VALUES ('2020-01-01', 'AAA', 100)")
    predict_live.migrate(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)").fetchall()}
    assert {"weight", "exposure", "cohort_id", "entry_open",
            "exit_date", "exit_price", "realized_ret"} <= cols
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 1  # data kept
    conn.close()


# ----------------------------------
# F5 - cohort cadence in trading sessions
# ----------------------------------

def test_cohort_cadence_counts_sessions_not_calendar():
    conn = sqlite3.connect(":memory:")
    predict_live.migrate(conn)
    conn.execute("CREATE TABLE daily_prices (ticker TEXT, date TEXT, close REAL)")
    assert predict_live.cohort_due(conn)                        # empty book -> due
    conn.execute("INSERT INTO cohorts VALUES ('2020-01-01', '2020-01-01', 'bull', 1.0, 15)")
    for i in range(config.HOLD_DAYS - 1):                       # 9 sessions elapsed
        conn.execute("INSERT INTO daily_prices VALUES ('AAA', ?, 100)", (f"2020-01-{i + 2:02d}",))
    assert not predict_live.cohort_due(conn)
    conn.execute("INSERT INTO daily_prices VALUES ('AAA', '2020-02-01', 100)")  # 10th session
    assert predict_live.cohort_due(conn)
    conn.close()


# ----------------------------------
# F8 - backtest and live share one cohort picker
# ----------------------------------

def test_cohort_parity_backtest_vs_live():
    day = _synthetic_day()
    via_backtest = backtest.pick_cohort(day, "prob")            # the sim's path
    via_live = strategy.select_cohort(day, "prob", "bull")      # predict_live's path
    assert list(via_backtest["ticker"]) == list(via_live["ticker"])
    assert np.allclose(via_backtest["weight"], via_live["weight"])
    assert len(via_live) == config.TOP_K
    assert via_live.groupby("sector").size().max() <= config.MAX_PER_SECTOR
    assert np.isclose(via_live["weight"].sum(), config.REGIME_EXPOSURE["bull"])

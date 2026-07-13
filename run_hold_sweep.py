"""
HOLD-LENGTH SWEEP - does a shorter hold (e.g. Mon->Fri, 5 sessions) beat the
current 10-day hold? Same picks, same ±3% exits, next-open entries, 10 bps.
Only the hold window (and the matching rebalance cadence) changes.

Caveat: the model is TRAINED on a 10-day label, so shorter holds exit before
the predicted move completes - a horizon mismatch. This measures the cost.

Run:  python run_hold_sweep.py
"""

import sqlite3
import warnings

import numpy as np
import pandas as pd

import backtest
import config
from backtest import load_signals, realized_return_series, simulate, summarize

warnings.filterwarnings("ignore")

TP, SL = config.EXIT_TAKE_PROFIT, config.EXIT_STOP_LOSS
HOLDS = [5, 7, 10]


def realized_frame(prices, hold):
    frames = []
    for t, panel in prices.items():
        o = panel["open"].to_numpy(dtype=float)
        c = panel["close"].to_numpy(dtype=float)
        r = realized_return_series(o, c, TP, SL, hold)
        frames.append(pd.DataFrame({"ticker": t, "date": panel.index.to_numpy(), "realized": r}))
    return pd.concat(frames, ignore_index=True).dropna(subset=["realized"])


def main():
    conn = sqlite3.connect(config.DB_PATH)
    print("\n" + "=" * 74)
    print("  HOLD-LENGTH SWEEP  (±3% exits, next-open, 10 bps)")
    print("=" * 74)
    df, prices = load_signals(conn)
    spy = pd.read_sql_query(
        "SELECT close FROM market_baselines WHERE symbol='SPY' AND date>=? AND date<=? ORDER BY date",
        conn, params=(df["date"].min(), df["date"].max()))
    spy_total = spy["close"].iloc[-1] / spy["close"].iloc[0] - 1
    conn.close()

    rows = []
    for hold in HOLDS:
        backtest.REBALANCE = hold                       # one book per `hold` sessions
        backtest.PERIODS_PER_YEAR = 252 / hold          # annualise Sharpe on the new grid
        rr = realized_frame(prices, hold)
        sub = df.drop(columns=["realized"]).merge(rr, on=["ticker", "date"], how="inner")
        eq, tr = simulate(sub, signal_col="p_xgb")
        s = summarize(eq, tr)
        rows.append((hold, s, float((tr["return"] > 0).mean())))

    hdr = f"  {'hold (sessions)':<18}{'trades':>8}{'pos%':>7}{'net/tr':>9}{'total':>9}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for hold, s, pos in rows:
        tag = f"{hold}  (current)" if hold == config.HOLD_DAYS else str(hold)
        print(f"  {tag:<18}{s['trades']:>8,}{pos:>6.1%}{s['avg_return']:>+9.2%}"
              f"{s['total_return']:>+9.0%}{s['cagr']:>+8.1%}{s['max_drawdown']:>8.1%}{s['sharpe']:>8.2f}")
    print(f"  {'SPY buy & hold':<18}{'-':>8}{'-':>7}{'-':>9}{spy_total:>+9.0%}")
    print("\n  Shorter hold exits before the 10-day predicted move completes.")


if __name__ == "__main__":
    main()

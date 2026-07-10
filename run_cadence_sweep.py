"""
--------------------------------------------
CADENCE SWEEP - does making picks MORE often help?
--------------------------------------------

The live system runs ONE 15-name book per 10 trading sessions. The question:
would entering more often (staggered, overlapping books) make more money, or
just churn the same capital at higher cost?

Same everything (honest next-open entries, ±3% / 10-day exit, 10 bps/position,
bear-scaled exposure, top-15 sector-capped by p_xgb) — only the ENTRY CADENCE
changes. At cadence N we keep K = round(10/N) overlapping books, so total slots
= 15·K sized to stay ~fully invested; the sector cap scales 3·K to match. This
is exactly the same capital deployed, just entered in more/smaller tranches, so
any difference is signal-freshness + smoothing vs turnover cost — net of costs.

Survivorship-biased (levels inflated, comparisons fair). Run:  python run_cadence_sweep.py
"""

import sqlite3
import warnings

import pandas as pd

import config
from backtest import load_signals
from backtest_trailing import curve_stats, simulate_trailing, spy_daily

warnings.filterwarnings("ignore")

TP, SL, HOLD = config.EXIT_TAKE_PROFIT, config.EXIT_STOP_LOSS, config.HOLD_DAYS
NAMES = config.TOP_K

CONFIGS = [
    ("1 book / 10 sessions (current)", 10),
    ("new book every 5 sessions", 5),
    ("new book every 2 sessions", 2),
    ("new book every session (old daily)", 1),
]


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    print("\n" + "=" * 80)
    print("  CADENCE SWEEP  (±3% / 10-day exit, next-open, 10 bps, same capital)")
    print("=" * 80 + "\n")

    df, prices = load_signals(conn)
    start, end = df["date"].min(), df["date"].max()

    rows = []
    for label, n in CONFIGS:
        k = max(1, round(HOLD / n))
        slots = NAMES * k
        curve, trades, cash = simulate_trailing(
            df, prices, signal_col="p_xgb", n_slots=slots, trail=1.0,
            max_hold=HOLD, init_stop=SL, take_profit=TP, cost=config.COST_PER_TRADE,
            entry_every=n, sector_cap=config.MAX_PER_SECTOR * k)
        st = curve_stats(label, curve, trades, cash)
        st["slots"] = slots
        rows.append(st)
    spy = spy_daily(conn, start, end)
    conn.close()

    print(f"  Period {start} -> {end}\n")
    hdr = (f"  {'cadence':<36}{'slots':>6}{'trades':>8}{'win%':>7}"
           f"{'total':>9}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['name']:<36}{r['slots']:>6}{r['trades']:>8,}{r['win_rate']:>6.1%}"
              f"{r['total']:>+9.0%}{r['cagr']:>+8.1%}{r['maxdd']:>8.1%}{r['sharpe']:>8.2f}")
    print(f"  {'SPY buy & hold':<36}{'-':>6}{'-':>8}{'-':>7}"
          f"{spy['total']:>+9.0%}{spy['cagr']:>+8.1%}{spy['maxdd']:>8.1%}{spy['sharpe']:>8.2f}")
    print("\n  More trades = more cost drag (already netted). Higher Sharpe at faster")
    print("  cadence = signal-freshness + smoothing beat the churn; lower = churn wins.")


if __name__ == "__main__":
    main()

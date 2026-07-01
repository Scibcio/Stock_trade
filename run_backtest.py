"""
--------------------------------------------
RUN PHASE 5 - BACKTEST / RISK STRESS-TEST
--------------------------------------------

Selects the top-K-per-day signals from the combined OOF, measures the realised
edge, Monte-Carlo stress-tests a 5-year book (growth + risk of ruin), and
benchmarks against buy-and-hold SPY over the same period.
"""

import sqlite3
import warnings

import numpy as np
import pandas as pd

import config
import threshold_analysis as ta
import strategy

warnings.filterwarnings("ignore")


def spy_cagr(conn, start: str, end: str) -> float:
    spy = pd.read_sql_query(
        "SELECT date, close FROM market_baselines WHERE symbol='SPY' AND date>=? AND date<=? ORDER BY date",
        conn, params=(start, end))
    yrs = (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25
    return (spy["close"].iloc[-1] / spy["close"].iloc[0]) ** (1 / yrs) - 1


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    df = ta.load_blended_oof(conn)
    cfg = strategy.StrategyConfig(top_k_per_day=5, risk_per_trade=0.01)
    trades = strategy.select_top_per_day(df, cfg.top_k_per_day)
    e = strategy.edge(trades, cfg)

    start, end = trades["date"].min(), trades["date"].max()
    yrs = (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25
    spy = spy_cagr(conn, start, end)
    conn.close()

    mc = strategy.monte_carlo(e["win_rate"], cfg, years=5)

    print("\nBacktest / risk stress-test (Phase 5) :\n")
    print(f"  strategy   : top {cfg.top_k_per_day}/day, risk {cfg.risk_per_trade:.0%}/trade, "
          f"{cfg.reward_risk:.0f}:1, cost {cfg.cost:.1%}")
    print(f"  period     : {start} -> {end}  ({yrs:.1f} yrs)")
    print(f"  trades     : {e['trades']:,}  ({e['trades']/yrs:,.0f}/yr)")
    print(f"  win rate   : {e['win_rate']:.1%}   EV/trade : {e['ev_per_trade']:+.3f} risk-units\n")
    print(f"  Monte Carlo (5-year book, 2000 sims) :")
    print(f"    median growth   : {mc['median_mult']:.1f}x   (P5 {mc['p5_mult']:.1f}x .. P95 {mc['p95_mult']:.1f}x)")
    print(f"    median max DD   : {mc['median_max_dd']:.0%}")
    print(f"    risk of -50% DD : {mc['risk_of_50pct_dd']:.1%}\n")
    print(f"  Benchmark  : SPY buy-and-hold {spy:.1%}/yr  ->  {(1+spy)**5:.1f}x over 5 yrs")


if __name__ == "__main__":
    main()

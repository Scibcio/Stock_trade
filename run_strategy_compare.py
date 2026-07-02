"""
--------------------------------------------
STRATEGY COMPARISON - do the risk controls pay off?
--------------------------------------------

We built a sector-diversification cap + regime exposure scaling, but never measured
whether they actually improve RISK-ADJUSTED returns. This backtests two strategies on
the same OOF predictions:

  NAIVE        : top-K by confidence each day, equal weight, always full exposure
  RISK-MANAGED : top-K sector-capped (max 3/sector) + regime exposure scaling

...and compares Sharpe + max drawdown. (Absolute returns are survivorship-inflated -
the RELATIVE comparison is what's meaningful. Sharpe is approximate: overlapping 10-day
trades attributed to entry date.)
"""

import sqlite3
import warnings

import numpy as np
import pandas as pd

import config
import threshold_analysis as ta

warnings.filterwarnings("ignore")

K = 10
MAX_PER_SECTOR = 3
REGIME_EXPOSURE = {"bull": 1.0, "sideways": 0.6, "bear": 0.3}
TP, SL, RISK = 3.0, -1.0, 0.01                       # risk units + 1% risk/trade


def _pick(day: pd.DataFrame, sector_cap: bool) -> pd.DataFrame:
    day = day.sort_values("blend", ascending=False)
    if not sector_cap:
        return day.head(K)
    chosen, per = [], {}
    for _, r in day.iterrows():
        if per.get(r["sector"], 0) >= MAX_PER_SECTOR:
            continue
        chosen.append(r)
        per[r["sector"]] = per.get(r["sector"], 0) + 1
        if len(chosen) >= K:
            break
    return pd.DataFrame(chosen)


def daily_returns(df: pd.DataFrame, sector_cap: bool, regime_scale: bool) -> pd.Series:
    out = {}
    for d, g in df.groupby("date"):
        picks = _pick(g, sector_cap)
        if picks.empty:
            continue
        w = 1.0 / len(picks)
        exposure = REGIME_EXPOSURE.get(picks["regime"].iloc[0], 0.5) if regime_scale else 1.0
        trade_ret = np.where(picks["Target_Label"] == 1, TP, SL) * RISK
        out[d] = (trade_ret * w).sum() * exposure
    return pd.Series(out).sort_index()


def metrics(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    return {"sharpe": sharpe, "max_dd": dd, "total": eq.iloc[-1] - 1, "win_day": (r > 0).mean()}


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    df = ta.load_blended_oof(conn)
    df["sector"] = df["ticker"].map(dict(conn.execute("SELECT ticker, sector FROM stocks").fetchall()))
    conn.close()

    naive = metrics(daily_returns(df, sector_cap=False, regime_scale=False))
    managed = metrics(daily_returns(df, sector_cap=True, regime_scale=True))

    print("\nStrategy comparison (top-10/day, 3:1, 1% risk) :\n")
    print(f"  {'metric':<14}{'NAIVE':>12}{'RISK-MANAGED':>15}")
    print(f"  {'Sharpe (approx)':<14}{naive['sharpe']:>12.2f}{managed['sharpe']:>15.2f}")
    print(f"  {'max drawdown':<14}{naive['max_dd']:>12.1%}{managed['max_dd']:>15.1%}")
    print(f"  {'win-day rate':<14}{naive['win_day']:>12.1%}{managed['win_day']:>15.1%}")
    print(f"\n  verdict: risk-managed {'IMPROVES' if managed['sharpe'] >= naive['sharpe'] and managed['max_dd'] >= naive['max_dd'] else 'trades some'} "
          f"risk-adjusted profile (higher/equal Sharpe AND shallower drawdown = win).")


if __name__ == "__main__":
    main()

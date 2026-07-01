"""
--------------------------------------------
SIGNAL PIPELINE - THRESHOLD + REGIME ANALYSIS (Phase 5)
--------------------------------------------

Answers two questions on the combined OOF predictions:
  1. How SELECTIVE should we be? (trade only the most confident slice)
  2. Does the edge SURVIVE per market regime? (bull / sideways / bear)

Blends the two base-model OOF preds, tags each row with a leakage-free market
regime (SPY vs 200-day MA + VIX, a backward-looking per-date state), then sweeps
selectivity and reports win rate + expected value at the 3:1 payoff.
"""

import sqlite3
import warnings

import numpy as np
import pandas as pd

import config

warnings.filterwarnings("ignore")
LSTM_OOF = config.HERE / "walk_forward_oof_lstm.csv"

REWARD_RISK = config.TAKE_PROFIT / abs(config.STOP_LOSS)   # 0.03 / 0.01 = 3


def ev(win_rate: float) -> float:
    # expected value per trade at the reward:risk payoff (win=+rr, loss=-1)
    return win_rate * (REWARD_RISK + 1) - 1                 # breakeven = 1/(rr+1) = 25%


def label_regime(spy_close: float, spy_200: float, vix: float) -> str:
    if np.isnan(spy_200):
        return "warmup"
    if spy_close < spy_200:
        return "bear"
    return "bull" if vix <= 20 else "sideways"


def compute_regimes(conn) -> pd.DataFrame:
    spy = pd.read_sql_query("SELECT date, close FROM market_baselines WHERE symbol='SPY' ORDER BY date", conn)
    vix = pd.read_sql_query("SELECT date, close AS vix FROM market_baselines WHERE symbol='^VIX' ORDER BY date", conn)
    spy["spy_200"] = spy["close"].rolling(config.SPY_TREND_WINDOW).mean()
    m = spy.merge(vix, on="date", how="left")
    m["vix"] = m["vix"].ffill()
    m["regime"] = [label_regime(c, s, v) for c, s, v in zip(m["close"], m["spy_200"], m["vix"])]
    return m[["date", "regime"]]


def load_blended_oof(conn) -> pd.DataFrame:
    xgb = pd.read_csv(config.OOF_PATH)
    lstm = pd.read_csv(LSTM_OOF)[["date", "ticker", "p_lstm"]]
    df = xgb.merge(lstm, on=["date", "ticker"], how="inner").dropna()
    df["blend"] = 0.5 * df["p_xgb"] + 0.5 * df["p_lstm"]
    return df.merge(compute_regimes(conn), on="date", how="left")


def sweep(df: pd.DataFrame, top_pcts=(100, 50, 25, 10, 5)) -> pd.DataFrame:
    scopes = [("ALL", df)] + [(r, df[df["regime"] == r]) for r in ("bull", "sideways", "bear")]
    rows = []
    for name, sub in scopes:
        if len(sub) == 0:
            continue
        for pct in top_pcts:
            cut = sub["blend"].quantile(1 - pct / 100)
            sel = sub[sub["blend"] >= cut]
            prec = float(sel["Target_Label"].mean())
            rows.append({"regime": name, "top_%": pct, "trades": len(sel),
                         "win_rate": round(prec, 3), "EV/trade": round(ev(prec), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    df = load_blended_oof(conn)
    conn.close()

    print("\nThreshold + regime analysis (Phase 5) :\n")
    print(f"  blended OOF preds : {len(df):,}   base win rate {df['Target_Label'].mean():.1%}")
    print(f"  breakeven win rate at {REWARD_RISK:.0f}:1 = {1 / (REWARD_RISK + 1):.0%}")
    print(f"  regime mix        : {df['regime'].value_counts().to_dict()}\n")
    print(sweep(df).to_string(index=False))


if __name__ == "__main__":
    main()

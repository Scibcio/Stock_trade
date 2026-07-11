"""
--------------------------------------------
U5 - META-LABELING  (take BETTER trades, not more)
--------------------------------------------

A second-stage model that, GIVEN the primary signal fired, predicts
P(this pick ends net-positive under the adopted ±3% exit). Used as a filter:
skip picks whose meta-p is low.

Strict separation (prime directive: never fit and score the same rows):
  * primary p_xgb is already out-of-fold (walk-forward)
  * candidate pool = top-30 by p_xgb each day; meta label = realised(±3%) > 0
  * meta model fits on folds 1-8, is evaluated ONLY on folds 9-12
  * point-in-time features: p_xgb, NATR, VIX level/change, SPY 200-dist, regime,
    cohort breadth (how many strong setups that day)

The decisive test is NOT "does meta-p rank winners" (p_xgb already does) - it is
whether meta-SELECTIVITY beats P_XGB-SELECTIVITY on the same held-out picks. If
dropping low-meta-p picks helps no more than just keeping higher-p_xgb picks,
the meta adds nothing and is kill-listed.

Run:  python run_meta_label.py
"""

import sqlite3
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import backtest
import config
import strategy

warnings.filterwarnings("ignore")

FEATS = ["p_xgb", "NATR_14", "VIX_Level", "VIX_Change", "SPY_200_dist",
         "breadth", "reg_bull", "reg_side"]
POOL_N = 30                    # candidate pool = top-30 by p_xgb each day
BREADTH_BAR = 0.55


def date_to_fold(d: str):
    for f in config.FOLDS:
        if f["test_start"] <= d <= f["test_end"]:
            return f["fold"]
    return None


def market_context(conn) -> pd.DataFrame:
    spy = pd.read_sql_query(
        "SELECT date, close FROM market_baselines WHERE symbol='SPY' ORDER BY date", conn)
    spy["SPY_200_dist"] = spy["close"] / spy["close"].rolling(200).mean() - 1
    vix = pd.read_sql_query(
        "SELECT date, close AS VIX_Level FROM market_baselines WHERE symbol='^VIX' ORDER BY date", conn)
    vix["VIX_Change"] = vix["VIX_Level"].pct_change()
    return spy[["date", "SPY_200_dist"]].merge(vix[["date", "VIX_Level", "VIX_Change"]], on="date")


def build(conn) -> pd.DataFrame:
    df, _ = backtest.load_signals(conn)                       # date,ticker,p_xgb,realized,NATR_14,sector,regime
    df["fold"] = df["date"].map(date_to_fold)
    df = df.dropna(subset=["fold"])
    df = df.merge(market_context(conn), on="date", how="left")

    breadth = (df.assign(hot=df["p_xgb"] >= BREADTH_BAR)
                 .groupby("date")["hot"].sum().rename("breadth").reset_index())
    df = df.merge(breadth, on="date", how="left")
    df["reg_bull"] = (df["regime"] == "bull").astype(int)
    df["reg_side"] = (df["regime"] == "sideways").astype(int)

    df["rank"] = df.groupby("date")["p_xgb"].rank(ascending=False, method="first")
    pool = df[df["rank"] <= POOL_N].copy()
    pool["meta_y"] = (pool["realized"] > 0).astype(int)
    return pool.dropna(subset=FEATS + ["meta_y"])


def selectivity(picks: pd.DataFrame, by: str) -> list:
    # keep the top q% of picks by column `by`; report win rate + avg realised
    rows = []
    for keep in (1.0, 0.8, 0.6, 0.4):
        cut = picks[by].quantile(1 - keep)
        sel = picks[picks[by] >= cut]
        rows.append((keep, len(sel), float((sel["realized"] > 0).mean()), float(sel["realized"].mean())))
    return rows


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    print("\n" + "=" * 74)
    print("  U5 - META-LABELING  (fit folds 1-8, evaluate folds 9-12)")
    print("=" * 74)

    pool = build(conn)
    train = pool[pool["fold"] <= 8]
    test = pool[pool["fold"] >= 9].copy()
    print(f"\n  pool {len(pool):,}  |  train {len(train):,} (folds 1-8)  "
          f"test {len(test):,} (folds 9-12)  |  base net-positive rate {pool['meta_y'].mean():.1%}")

    meta = XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         random_state=42, n_jobs=0)
    meta.fit(train[FEATS], train["meta_y"])
    test["meta_p"] = meta.predict_proba(test[FEATS])[:, 1]

    meta_auc = roc_auc_score(test["meta_y"], test["meta_p"])
    prim_auc = roc_auc_score(test["meta_y"], test["p_xgb"])
    print(f"\n  Predicting net-positive on HELD-OUT folds 9-12:")
    print(f"    meta-model AUC : {meta_auc:.4f}")
    print(f"    p_xgb-only AUC : {prim_auc:.4f}   (the bar to beat - meta must exceed this)")

    # decisive test: on the ACTUAL strategy picks (test folds), does meta
    # selectivity beat p_xgb selectivity?
    rebal = sorted(test["date"].unique())[::backtest.REBALANCE]
    picks = []
    for d in rebal:
        day = test[test["date"] == d]
        cohort = strategy.select_cohort(day, "p_xgb", day["regime"].iloc[0])
        if cohort is not None:                                # cohort rows already carry meta_p
            picks.append(cohort)
    picks = pd.concat(picks, ignore_index=True)

    print(f"\n  On the strategy's {len(picks):,} held-out picks - keep the best q% by:\n")
    print(f"  {'keep':>6}  {'--- by META-p ---':>28}   {'--- by p_xgb ---':>28}")
    print(f"  {'':>6}  {'trades':>7}{'win%':>8}{'avg/tr':>9}   {'trades':>7}{'win%':>8}{'avg/tr':>9}")
    print("  " + "-" * 70)
    m = {k: (n, w, a) for k, n, w, a in selectivity(picks, "meta_p")}
    p = {k: (n, w, a) for k, n, w, a in selectivity(picks, "p_xgb")}
    for keep in (1.0, 0.8, 0.6, 0.4):
        mn, mw, ma = m[keep]
        pn, pw, pa = p[keep]
        print(f"  {keep:>5.0%}  {mn:>7,}{mw:>8.1%}{ma:>+9.2%}   {pn:>7,}{pw:>8.1%}{pa:>+9.2%}")

    conn.close()
    print("\n  Verdict: meta earns its keep only if the META columns beat the p_xgb")
    print("  columns at the same keep-rate (higher win% / avg per trade).")


if __name__ == "__main__":
    main()

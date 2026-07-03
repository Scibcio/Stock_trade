"""
--------------------------------------------
RETRAIN ON THE TRAILING-STOP LABEL  (Phase 6c)
--------------------------------------------

The decisive question: if the model is trained to select stocks that TREND
(labels.trailing_stop_label) instead of ones that pop +3% in 10 days, does the
let-winners-run strategy finally beat SPY?

  1. rebuild the pooled dataset with the trailing label (trail=0.20, max_hold=90,
     matching the best exit from backtest_trailing.py),
  2. walk-forward XGBoost -> out-of-fold p_xgb (saved to walk_forward_oof_trail.csv);
     report OOF AUC — FINDINGS warns longer horizons hurt ranking, so we check,
  3. replay the SAME event-driven trailing portfolio ranking by the NEW model, and
     compare to the old (10-day-trained) ranker and to SPY.

Survivorship-biased, XGB-only, 0.1% costs — same caveats as backtest_trailing.py.
Run:  python run_trail_label.py
"""

import sqlite3
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import features
import labels
import model_xgb
import pipeline
from backtest import load_signals
from backtest_trailing import curve_stats, simulate_trailing, spy_daily

warnings.filterwarnings("ignore")

TRAIL, MAX_HOLD = 0.20, 90
TRAIL_OOF = config.HERE / "walk_forward_oof_trail.csv"


def build_trail_dataset(conn) -> pd.DataFrame:
    baselines = features.load_baselines(conn)
    frames = []
    tickers = pipeline.load_universe(conn)
    for i, t in enumerate(tickers, 1):
        raw = features.load_stock(t, conn)
        if raw.empty:
            continue
        feat = features.compute_features(raw, baselines)
        if feat.empty:
            continue
        feat["Target_Label"] = labels.trailing_stop_label(feat, TRAIL, MAX_HOLD)
        feat = feat.dropna(subset=["Target_Label"])
        feat["Target_Label"] = feat["Target_Label"].astype("int8")
        feat.insert(1, "ticker", t)
        frames.append(feat)
        if i % 50 == 0:
            print(f"  label {i}/{len(tickers)}")
    return pd.concat(frames, ignore_index=True)


def walk_forward(pooled: pd.DataFrame, feats: list) -> pd.DataFrame:
    oof = []
    for f in config.FOLDS:
        cut = (date.fromisoformat(f["train_end"]) - timedelta(days=config.PURGE_DAYS)).isoformat()
        tr = pooled[pooled["date"] <= cut]
        te = pooled[(pooled["date"] >= f["test_start"]) & (pooled["date"] <= f["test_end"])]
        if tr.empty or te.empty:
            continue
        m = model_xgb.train_xgb(tr[feats].to_numpy("float32"), tr["Target_Label"].to_numpy("int8"))
        blk = te[["date", "ticker", "Target_Label"]].copy()
        blk["fold"] = f["fold"]
        blk["p_xgb"] = model_xgb.predict_xgb(m, te[feats].to_numpy("float32"))
        oof.append(blk)
    return pd.concat(oof, ignore_index=True)


def _print_table(rows) -> None:
    hdr = (f"  {'variant':<32}{'trades':>7}{'win%':>7}{'avg/tr':>9}{'med/tr':>9}"
           f"{'total':>9}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        wr = "   -  " if pd.isna(r["win_rate"]) else f"{r['win_rate']:>5.1%}"
        av = "     - " if pd.isna(r["avg"]) else f"{r['avg']:>+7.2%}"
        md = "     - " if pd.isna(r["median"]) else f"{r['median']:>+7.2%}"
        tr_ = "   -  " if pd.isna(r["trades"]) else f"{int(r['trades']):>6}"
        print(f"  {r['name']:<32}{tr_:>7}{wr:>7}{av:>9}{md:>9}"
              f"{r['total']:>+9.0%}{r['cagr']:>+8.1%}{r['maxdd']:>8.1%}{r['sharpe']:>8.2f}")


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    print("\n" + "=" * 78)
    print(f"  RETRAIN ON TRAILING LABEL  (trail {TRAIL:.0%} / {MAX_HOLD}d)")
    print("=" * 78)

    print("\n  [1/3] labeling with the trailing-stop target ...")
    pooled = build_trail_dataset(conn)
    print(f"        trailing-win base rate: {pooled['Target_Label'].mean():.1%}   rows {len(pooled):,}")

    print("\n  [2/3] walk-forward XGBoost on the new label ...")
    trail_oof = walk_forward(pooled, config.WIDE_FEATURES)
    trail_oof.to_csv(TRAIL_OOF, index=False)
    auc = roc_auc_score(trail_oof["Target_Label"], trail_oof["p_xgb"])
    print(f"        trailing-label OOF AUC: {auc:.4f}   (the 10-day +3% label scored ~0.561)")

    print("\n  [3/3] replaying the event-driven trailing portfolio ...")
    df, prices = load_signals(conn)
    old_c, old_t, old_cash = simulate_trailing(df, prices, signal_col="p_xgb", trail=TRAIL, max_hold=MAX_HOLD)
    dfn = df.merge(trail_oof[["date", "ticker", "p_xgb"]].rename(columns={"p_xgb": "p_trail"}),
                   on=["date", "ticker"], how="inner")
    dfn["p_xgb"] = dfn["p_trail"]
    new_c, new_t, new_cash = simulate_trailing(dfn, prices, signal_col="p_xgb", trail=TRAIL, max_hold=MAX_HOLD)
    spy = spy_daily(conn, df["date"].min(), df["date"].max())
    conn.close()

    print()
    _print_table([
        curve_stats(f"OLD ranker (10d +3%) + trail", old_c, old_t, old_cash),
        curve_stats(f"NEW ranker (trailing label) + trail", new_c, new_t, new_cash),
        spy,
    ])
    verdict = "BEATS SPY" if (new_cash - 1) > spy["total"] else "still < SPY"
    print(f"\n  Verdict: retraining on the trailing label -> {verdict}")


if __name__ == "__main__":
    main()

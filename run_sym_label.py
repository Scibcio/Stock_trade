"""
--------------------------------------------
U3 (part 1) - RETRAIN ON THE SYMMETRIC +/-3% LABEL
--------------------------------------------

The strategy now EXECUTES symmetric +/-3% exits, but the ranker is still trained
on the old 3:1 (+3/-1) target. Does teaching the model the geometry it actually
trades improve the ranking and the book?

  1. relabel with labels.triple_barrier(take_profit=+3%, stop_loss=-3%),
  2. 12-fold purged walk-forward XGBoost -> walk_forward_oof_sym.csv (+ AUC),
  3. replay the cohort book (honest next-open, +/-3% exits) ranking by the
     NEW sym-label model vs the CURRENT 3:1-label model,
  4. null-certify the new label's OOF (prime directive: label changed).

Run:  python run_sym_label.py
"""

import sqlite3
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import backtest
import config
import features
import labels
import model_xgb
import pipeline
from baseline_null import _fast_auc

warnings.filterwarnings("ignore")

SYM_TP, SYM_SL = 0.03, -0.03
SYM_OOF = config.HERE / "walk_forward_oof_sym.csv"


def build_sym_dataset(conn) -> pd.DataFrame:
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
        feat = labels.add_target(feat, take_profit=SYM_TP, stop_loss=SYM_SL)
        if feat.empty:
            continue
        feat.insert(1, "ticker", t)
        frames.append(feat)
        if i % 100 == 0:
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
        blk["p_sym"] = model_xgb.predict_xgb(m, te[feats].to_numpy("float32"))
        oof.append(blk)
        print(f"  fold {f['fold']}/12 done")
    return pd.concat(oof, ignore_index=True)


def certify(oof: pd.DataFrame, draws: int = 1000) -> str:
    df = oof.dropna()
    y = df["Target_Label"].to_numpy()
    p = df["p_sym"].to_numpy()
    n, n_pos = len(y), int(y.sum())
    rng = np.random.default_rng(12345)
    ranks = rankdata(p)
    real = roc_auc_score(y, p)
    null = np.array([_fast_auc(ranks, rng.choice(n, n_pos, replace=False), n_pos, n)
                     for _ in range(draws)])
    z = (real - null.mean()) / null.std()
    verdict = "CERTIFIED REAL" if real > null.max() else "NOT CERTIFIED"
    return (f"real AUC {real:.4f} | null {null.mean():.4f}+/-{null.std():.4f} "
            f"(best of {draws}: {null.max():.4f}) | z {z:+.0f} sigma -> {verdict}")


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    print("\n" + "=" * 70)
    print(f"  U3.1  SYMMETRIC-LABEL RETRAIN  (+/-{SYM_TP:.0%} barriers, {config.HOLD_DAYS}d)")
    print("=" * 70)

    print("\n  [1/4] relabeling ...")
    pooled = build_sym_dataset(conn)
    print(f"        sym base rate: {pooled['Target_Label'].mean():.1%}   rows {len(pooled):,}")

    print("\n  [2/4] purged walk-forward XGBoost ...")
    oof = walk_forward(pooled, config.WIDE_FEATURES)
    oof.to_csv(SYM_OOF, index=False)
    auc = roc_auc_score(oof["Target_Label"], oof["p_sym"])
    print(f"        sym-label OOF AUC: {auc:.4f}   (3:1 label scores ~0.561 on its own target)")

    print("\n  [3/4] cohort replay: sym-trained vs 3:1-trained ranker (same +/-3% exits) ...")
    df, _ = backtest.load_signals(conn)                       # realized = adopted +/-3% exits
    dfn = df.merge(oof[["date", "ticker", "p_sym"]], on=["date", "ticker"], how="inner")
    rows = []
    for name, sig in (("3:1-trained (current)", "p_xgb"), ("sym-trained (U3)", "p_sym")):
        eq, tr = backtest.simulate(dfn, signal_col=sig)
        s = backtest.summarize(eq, tr)
        rows.append((name, s, float((tr["return"] > 0).mean())))
    hdr = f"  {'ranker':<24}{'pos%':>7}{'net/tr':>9}{'total':>9}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, s, pos in rows:
        print(f"  {name:<24}{pos:>6.1%}{s['avg_return']:>+9.2%}{s['total_return']:>+9.0%}"
              f"{s['cagr']:>+8.1%}{s['max_drawdown']:>8.1%}{s['sharpe']:>8.2f}")

    print("\n  [4/4] null certification of the sym label ...")
    print("        " + certify(oof))
    conn.close()


if __name__ == "__main__":
    main()

"""
--------------------------------------------
CROSS-SECTIONAL + OWN-TREND LIFT TEST
--------------------------------------------

Tests the most evidence-backed accuracy candidates on the FULL universe:
  - SMA_200_Dist        : the stock's OWN long-term trend (the Losing Loonies gap)
  - *_xrank             : cross-sectional percentile rank vs the universe each day
                          (momentum / RSI / vol-surge / relative strength)

Runs the purged walk-forward with XGBoost on WIDE vs WIDE+NEW and compares OOF AUC.
Cross-sectional rank needs the whole universe, so this runs on all ~500 names.
"""

import sqlite3
import warnings
from datetime import date, timedelta

import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import features
import labels
import model_xgb
import pipeline

warnings.filterwarnings("ignore")


def build_dataset(conn) -> pd.DataFrame:
    baselines = features.load_baselines(conn)
    frames = []
    for i, t in enumerate(pipeline.load_universe(conn), 1):
        raw = features.load_stock(t, conn)
        if raw.empty:
            continue
        df = labels.add_target(features.compute_features(raw, baselines))
        if df.empty:
            continue
        df.insert(1, "ticker", t)
        frames.append(df)
        if i % 100 == 0:
            print(f"    features [{i}]")
    return features.add_cross_sectional(pd.concat(frames, ignore_index=True))


def wf_auc(pooled: pd.DataFrame, feats: list) -> float:
    oof = []
    for f in config.FOLDS:
        cut = (date.fromisoformat(f["train_end"]) - timedelta(days=config.PURGE_DAYS)).isoformat()
        tr = pooled[pooled["date"] <= cut]
        te = pooled[(pooled["date"] >= f["test_start"]) & (pooled["date"] <= f["test_end"])]
        if tr.empty or te.empty:
            continue
        m = model_xgb.train_xgb(tr[feats].to_numpy("float32"), tr["Target_Label"].to_numpy("int8"))
        blk = te[["Target_Label"]].copy()
        blk["p"] = model_xgb.predict_xgb(m, te[feats].to_numpy("float32"))
        oof.append(blk)
    oof = pd.concat(oof, ignore_index=True)
    return roc_auc_score(oof["Target_Label"], oof["p"])


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    print("\nCross-sectional lift test (full universe) :\n")
    pooled = build_dataset(conn)
    conn.close()
    print(f"\n  dataset: {len(pooled):,} rows\n")

    base = wf_auc(pooled, config.WIDE_FEATURES)
    plus = wf_auc(pooled, config.WIDE_FEATURES + config.NEW_FEATURES)
    print(f"  {'WIDE':<16}{base:.4f}")
    print(f"  {'WIDE + new':<16}{plus:.4f}")
    print(f"  {'lift':<16}{plus - base:+.4f}   ({'KEEP - real lift' if plus - base > 0.002 else 'skip - within noise'})")
    print(f"\n  new features tested: {', '.join(config.NEW_FEATURES)}")


if __name__ == "__main__":
    main()

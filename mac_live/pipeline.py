"""
--------------------------------------------
SIGNAL PIPELINE - WALK-FORWARD ORCHESTRATOR
--------------------------------------------

Runs the real 12-fold walk-forward for the base model(s): train on the expanding
window up to each fold's train_end (PURGED near the boundary), predict the
out-of-sample test year, and collect out-of-fold (OOF) predictions.

The OOF predictions are the honest, leakage-free substrate that calibration
(ensemble.py), the ensemble, and the strategy/threshold analysis all consume.

Reads trading.db via features.py + labels.py. Model A = model_xgb.py.
Run after database.py has built/updated trading.db.
"""

import sqlite3
import time
from datetime import date, timedelta

import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import features
import labels
import model_xgb


# ----------------------------------
# UNIVERSE + DATASET
# ----------------------------------

def load_universe(conn: sqlite3.Connection) -> list[str]:
    # ML-ready tickers (enough history) = the full aim-big universe
    rows = conn.execute(
        "SELECT ticker FROM stocks WHERE ml_ready = 1 ORDER BY ticker"
    ).fetchall()
    return [r[0] for r in rows]


def build_dataset(tickers: list[str], conn: sqlite3.Connection) -> pd.DataFrame:
    # features + label per ticker, pooled into one frame; baselines loaded ONCE
    baselines = features.load_baselines(conn)
    frames = []
    for i, t in enumerate(tickers, 1):
        raw = features.load_stock(t, conn)
        if raw.empty:
            continue
        df = labels.add_target(features.compute_features(raw, baselines))
        if df.empty:
            continue
        df.insert(1, "ticker", t)
        frames.append(df)
        if i % 50 == 0:
            print(f"    features [{i}/{len(tickers)}]")
    return pd.concat(frames, ignore_index=True)


# ----------------------------------
# WALK-FORWARD
# ----------------------------------

def _purge_cutoff(train_end: str) -> str:
    # drop the last PURGE_DAYS before train_end so no training label straddles the boundary
    return (date.fromisoformat(train_end) - timedelta(days=config.PURGE_DAYS)).isoformat()


def walk_forward(pooled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train Model A per fold on the purged expanding window, predict the test year.
    Returns (oof_predictions, per_fold_metrics).
    """
    feat = config.WIDE_FEATURES
    oof, metrics = [], []

    for f in config.FOLDS:
        cut = _purge_cutoff(f["train_end"])
        train = pooled[pooled["date"] <= cut]
        test = pooled[(pooled["date"] >= f["test_start"]) & (pooled["date"] <= f["test_end"])]
        if train.empty or test.empty:
            continue

        model = model_xgb.train_xgb(train[feat].to_numpy("float32"),
                                    train["Target_Label"].to_numpy("int8"))
        prob = model_xgb.predict_xgb(model, test[feat].to_numpy("float32"))

        block = test[["date", "ticker", "Target_Label"]].copy()
        block["fold"] = f["fold"]
        block["p_xgb"] = prob
        oof.append(block)

        auc = roc_auc_score(test["Target_Label"], prob)
        metrics.append({
            "fold": f["fold"],
            "test": f"{f['test_start'][:7]}..{f['test_end'][:7]}",
            "train_rows": len(train), "test_rows": len(test),
            "win_rate": round(float(test["Target_Label"].mean()), 3),
            "auc": round(float(auc), 4),
        })
        print(f"    fold {f['fold']:>2}  {metrics[-1]['test']}  "
              f"train {len(train):>9,}  test {len(test):>7,}  AUC {auc:.4f}")

    return pd.concat(oof, ignore_index=True), pd.DataFrame(metrics)


# ----------------------------------
# ENTRY POINT
# ----------------------------------

def run() -> None:
    t0 = time.time()
    print("\nWalk-forward :\n")

    conn = sqlite3.connect(config.DB_PATH)
    tickers = load_universe(conn)
    print(f"  universe : {len(tickers)} ml-ready tickers")
    pooled = build_dataset(tickers, conn)
    conn.close()
    print(f"  dataset  : {len(pooled):,} rows x {len(config.WIDE_FEATURES)} features\n")

    oof, metrics = walk_forward(pooled)
    oof.to_csv(config.OOF_PATH, index=False)

    overall = roc_auc_score(oof["Target_Label"], oof["p_xgb"])
    print(f"\n  OOF saved     : {config.OOF_PATH.name}  ({len(oof):,} predictions)")
    print(f"  mean fold AUC : {metrics['auc'].mean():.4f}  +/- {metrics['auc'].std():.4f}")
    print(f"  pooled OOF AUC: {overall:.4f}")
    print(f"  duration      : {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()

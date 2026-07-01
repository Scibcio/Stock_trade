"""
--------------------------------------------
RUN MODEL B - LSTM WALK-FORWARD + ENSEMBLE CHECK
--------------------------------------------

Runs the 12-fold CNN-LSTM walk-forward on the full universe, saves its OOF
predictions, then compares them to the XGBoost OOF (config.OOF_PATH) - the
diversity go/no-go: if corr(p_xgb, p_lstm) is too high the ensemble adds nothing.

Run after pipeline.py has produced the XGBoost OOF. GPU-accelerated (RTX 5070).
"""

import sqlite3
import time

import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import pipeline
import model_lstm

LSTM_OOF = config.HERE / "walk_forward_oof_lstm.csv"


def main() -> None:
    t0 = time.time()
    print("\nModel B - LSTM walk-forward :\n")

    conn = sqlite3.connect(config.DB_PATH)
    tickers = pipeline.load_universe(conn)
    print(f"  universe : {len(tickers)} tickers")
    pooled = pipeline.build_dataset(tickers, conn)
    conn.close()
    print(f"  dataset  : {len(pooled):,} rows\n")

    oof, metrics = model_lstm.walk_forward(pooled)
    oof.to_csv(LSTM_OOF, index=False)
    print(f"\n  LSTM mean fold AUC : {metrics['auc'].mean():.4f} +/- {metrics['auc'].std():.4f}")
    print(f"  LSTM pooled OOF AUC: {roc_auc_score(oof['Target_Label'], oof['p_lstm']):.4f}")

    # Ensemble diversity check vs XGBoost OOF
    if config.OOF_PATH.exists():
        xgb = pd.read_csv(config.OOF_PATH)[["date", "ticker", "p_xgb"]]
        merged = oof.merge(xgb, on=["date", "ticker"], how="inner")
        corr = merged["p_xgb"].corr(merged["p_lstm"])
        print(f"\n  Ensemble check :\n")
        print(f"    aligned preds       : {len(merged):,}")
        print(f"    corr(p_xgb, p_lstm) : {corr:.3f}")
        print(f"    verdict : {'GOOD - diverse, ensemble should help' if corr < 0.9 else 'TOO HIGH - ensemble adds little'}")

    print(f"\n  duration : {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

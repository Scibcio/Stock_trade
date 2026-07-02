"""
--------------------------------------------
BASELINE NULL / PERMUTATION TEST
--------------------------------------------

Certify the edge is REAL, not noise ("beat N random baselines", borrowed from the
Losing Loonies channel). Pins the model's OOF results against empirical nulls:

  [1] AUC vs a SHUFFLED-LABEL null (does the ranking beat random label assignment?)
  [2] Top-5% win rate vs a RANDOM-SELECTION null of the same trade count

If the real result sits far in the right tail (tiny p-value), the edge isn't luck.

Caveat: our triple-barrier labels OVERLAP (10-day windows), so the effective sample
size is far below the row count - the naive permutation UNDERSTATES the null variance
and OVERSTATES significance. Read the verdict as "clearly real", not "N sigma exactly".
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import config

warnings.filterwarnings("ignore")
LSTM_OOF = config.HERE / "walk_forward_oof_lstm.csv"
N = 2000


def _fast_auc(ranks: np.ndarray, pos_idx: np.ndarray, n_pos: int, n: int) -> float:
    # rank-based AUC (Mann-Whitney): O(n_pos) given precomputed ranks
    return (ranks[pos_idx].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (n - n_pos))


def main() -> None:
    xgb = pd.read_csv(config.OOF_PATH)
    lstm = pd.read_csv(LSTM_OOF)[["date", "ticker", "p_lstm"]]
    df = xgb.merge(lstm, on=["date", "ticker"], how="inner").dropna()
    df["blend"] = 0.5 * df["p_xgb"] + 0.5 * df["p_lstm"]

    y = df["Target_Label"].to_numpy()
    p = df["blend"].to_numpy()
    n, n_pos = len(y), int(y.sum())
    rng = np.random.default_rng(12345)

    print("\nBaseline null / permutation test :\n")
    print(f"  ensemble OOF preds : {n:,}   base win rate {y.mean():.1%}   ({N:,} random baselines)\n")

    # [1] AUC vs shuffled-label null
    ranks = rankdata(p)
    real_auc = roc_auc_score(y, p)
    null_auc = np.array([_fast_auc(ranks, rng.choice(n, n_pos, replace=False), n_pos, n) for _ in range(N)])
    print("  [1] AUC vs shuffled-label null")
    print(f"      real AUC : {real_auc:.4f}")
    print(f"      null AUC : {null_auc.mean():.4f} +/- {null_auc.std():.4f}  (best of {N:,}: {null_auc.max():.4f})")
    print(f"      -> real is {(real_auc - null_auc.mean()) / null_auc.std():.0f} sigma above null; "
          f"p < {1/(N+1):.1e}   {'CERTIFIED REAL' if real_auc > null_auc.max() else 'not beaten'}\n")

    # [2] Top-5% win rate vs random-selection null
    k = int(0.05 * n)
    real_wr = y[p >= np.quantile(p, 0.95)].mean()
    null_wr = np.array([y[rng.choice(n, k, replace=False)].mean() for _ in range(N)])
    print("  [2] Top-5% win rate vs random-selection null")
    print(f"      real WR  : {real_wr:.1%}  (top 5% by confidence = {k:,} trades)")
    print(f"      null WR  : {null_wr.mean():.1%} +/- {null_wr.std():.1%}  (best of {N:,}: {null_wr.max():.1%})")
    print(f"      -> real is {(real_wr - null_wr.mean()) / null_wr.std():.0f} sigma above random; "
          f"p < {1/(N+1):.1e}   {'CERTIFIED REAL' if real_wr > null_wr.max() else 'not beaten'}\n")

    ok = (real_auc > null_auc.max()) and (real_wr > null_wr.max())
    print(f"  VERDICT: {'EDGE IS REAL - beats every random baseline on both tests.' if ok else 'NOT certified.'}")


if __name__ == "__main__":
    main()

"""
--------------------------------------------
U8 (dev) - DE-BETA THE LABEL
--------------------------------------------

The 3:1 fixed-% label mechanically rewards high-beta names (top-15 book beta
~1.62) - beta pays the label, not selection skill, and it dies in bears
(fold-9 AUC 0.490). Test labels whose difficulty is uniform across vol/beta:

  base 3:1 (+3%/-1%)   - current, the control
  vol-sym  (+-2xATR)   - handoff U8a: barriers scaled per stock's own ATR
  vol-asym (+1.5/-0.5xATR) - same vol scaling but keeps the 3:1 asymmetry
                             (U3 lesson: the asymmetry IS the selection edge)
  residual (r - B*rSPY > 0, 10d) - handoff U8b: label on beta-adjusted return

DEV UNIVERSE first (15 tickers - fast, per handoff): report pooled AUC,
fold-9 (2022 bear) AUC, and the top-5% slice's mean Beta_20 / NATR (the
beta-tilt proxy). Winner graduates to the full universe.

Run:  python run_debeta.py
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

warnings.filterwarnings("ignore")

HOLD = config.HOLD_DAYS
LABELS = ["base31", "volsym", "volasym", "resid"]


def vol_barrier(close: np.ndarray, natr: np.ndarray,
                k_tp: float, k_sl: float, hold: int = HOLD) -> np.ndarray:
    # first-touch triple barrier with PER-ENTRY barriers = +-k x ATR%(entry)
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(n - hold):
        entry, tp, sl = close[i], k_tp * natr[i] / 100, -k_sl * natr[i] / 100
        out[i] = 0.0
        for px in close[i + 1: i + 1 + hold]:
            ret = (px - entry) / entry
            if ret <= sl:
                break
            if ret >= tp:
                out[i] = 1.0
                break
    return out


def residual_label(close: np.ndarray, beta: np.ndarray,
                   spy_fwd: np.ndarray, hold: int = HOLD) -> np.ndarray:
    # 1 if the 10d return BEYOND beta x SPY is positive (selection, not beta)
    n = len(close)
    out = np.full(n, np.nan)
    fwd = np.full(n, np.nan)
    fwd[:n - hold] = close[hold:] / close[:n - hold] - 1
    resid = fwd - beta * spy_fwd
    out[:n - hold] = (resid[:n - hold] > 0).astype(float)
    return out


def build(conn) -> pd.DataFrame:
    baselines = features.load_baselines(conn)
    spy = baselines["SPY"].copy()
    spy["spy_fwd"] = spy["close"].shift(-HOLD) / spy["close"] - 1
    spy_fwd_by_date = dict(zip(spy["date"], spy["spy_fwd"]))

    frames = []
    for t in config.DEV_UNIVERSE:
        feat = features.compute_features(features.load_stock(t, conn), baselines)
        if feat.empty:
            continue
        close = feat["close"].to_numpy(dtype=float)
        natr = feat["NATR_14"].to_numpy(dtype=float)
        beta = feat["Beta_20"].to_numpy(dtype=float)
        spy_fwd = feat["date"].map(spy_fwd_by_date).to_numpy(dtype=float)

        feat["y_base31"] = labels.triple_barrier(feat)
        feat["y_volsym"] = vol_barrier(close, natr, 2.0, 2.0)
        feat["y_volasym"] = vol_barrier(close, natr, 1.5, 0.5)
        feat["y_resid"] = residual_label(close, beta, spy_fwd)
        feat.insert(1, "ticker", t)
        frames.append(feat)
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=[f"y_{L}" for L in LABELS]).reset_index(drop=True)


def walk_forward_auc(pooled: pd.DataFrame, ycol: str, feats: list) -> dict:
    oof = []
    for f in config.FOLDS:
        cut = (date.fromisoformat(f["train_end"]) - timedelta(days=config.PURGE_DAYS)).isoformat()
        tr = pooled[pooled["date"] <= cut]
        te = pooled[(pooled["date"] >= f["test_start"]) & (pooled["date"] <= f["test_end"])]
        if tr.empty or te.empty or tr[ycol].nunique() < 2:
            continue
        m = model_xgb.train_xgb(tr[feats].to_numpy("float32"),
                                tr[ycol].to_numpy(dtype="int8"))
        blk = te[["date", "ticker", ycol, "Beta_20", "NATR_14"]].copy()
        blk["fold"] = f["fold"]
        blk["p"] = model_xgb.predict_xgb(m, te[feats].to_numpy("float32"))
        oof.append(blk)
    df = pd.concat(oof, ignore_index=True).dropna()
    f9 = df[df["fold"] == 9]
    top = df[df["p"] >= df["p"].quantile(0.95)]
    return {
        "base_rate": float(df[ycol].mean()),
        "auc": float(roc_auc_score(df[ycol], df["p"])),
        "auc_bear": float(roc_auc_score(f9[ycol], f9["p"])) if f9[ycol].nunique() > 1 else float("nan"),
        "top5_beta": float(top["Beta_20"].mean()),
        "top5_natr": float(top["NATR_14"].mean()),
    }


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    print("\n" + "=" * 74)
    print("  U8 (dev universe) - DE-BETA THE LABEL")
    print("=" * 74)
    df = build(conn)
    conn.close()
    print(f"\n  dev rows: {len(df):,}  tickers: {df['ticker'].nunique()}\n")

    hdr = (f"  {'label':<12}{'base%':>7}{'AUC':>8}{'bear AUC':>10}"
           f"{'top5 beta':>11}{'top5 NATR':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for L in LABELS:
        r = walk_forward_auc(df, f"y_{L}", config.WIDE_FEATURES)
        print(f"  {L:<12}{r['base_rate']:>6.1%}{r['auc']:>8.4f}{r['auc_bear']:>10.4f}"
              f"{r['top5_beta']:>11.2f}{r['top5_natr']:>11.2f}")
    print("\n  hypothesis: de-beta'd labels keep pooled AUC while (a) fixing the bear")
    print("  fold and (b) cutting the top-slice beta/NATR tilt. Winner -> full universe.")


if __name__ == "__main__":
    main()

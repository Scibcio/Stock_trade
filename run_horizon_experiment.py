"""
--------------------------------------------
HORIZON EXPERIMENT
--------------------------------------------

Root-cause test: both insider AND sentiment edges are long-horizon signals, but
our target is a 10-day swing. Does a LONGER horizon predict better - and does it
revive the insider edge?

Relabels the dev universe at 10/20/40/60-day horizons (vol-scaled barriers, same
3:1 risk:reward), re-runs the XGBoost walk-forward, and reports OOF AUC + win rate
per horizon. At the longest horizon it also re-tests WIDE vs WIDE+insider.
"""

import math
import sqlite3
import warnings
from datetime import date, timedelta

import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import features
import labels
import model_xgb

warnings.filterwarnings("ignore")

BASE_TP, BASE_SL, BASE_HOLD = 0.03, 0.01, 10
HORIZONS = [10, 20, 40, 60]


def barriers(hold: int) -> tuple[float, float]:
    f = math.sqrt(hold / BASE_HOLD)                       # vol scales with sqrt(time)
    return round(BASE_TP * f, 4), round(-BASE_SL * f, 4)


def build_feature_frames(conn) -> dict:
    baselines = features.load_baselines(conn)
    frames = {}
    for t in config.DEV_UNIVERSE:
        df = features.compute_features(features.load_stock(t, conn), baselines)
        df = features.merge_insider(df, features.load_insider(t, conn))
        df["ticker"] = t
        frames[t] = df
    return frames


def label_and_pool(frames: dict, tp: float, sl: float, hold: int) -> pd.DataFrame:
    return pd.concat([labels.add_target(df, take_profit=tp, stop_loss=sl, hold_days=hold)
                      for df in frames.values()], ignore_index=True)


def wf_auc(pooled: pd.DataFrame, feats: list, hold: int) -> tuple[float, float]:
    purge = int(hold * 1.6) + 4                           # purge must cover the label horizon
    oof = []
    for f in config.FOLDS:
        cut = (date.fromisoformat(f["train_end"]) - timedelta(days=purge)).isoformat()
        tr = pooled[pooled["date"] <= cut]
        te = pooled[(pooled["date"] >= f["test_start"]) & (pooled["date"] <= f["test_end"])]
        if tr.empty or te.empty:
            continue
        m = model_xgb.train_xgb(tr[feats].to_numpy("float32"), tr["Target_Label"].to_numpy("int8"))
        blk = te[["Target_Label"]].copy()
        blk["p"] = model_xgb.predict_xgb(m, te[feats].to_numpy("float32"))
        oof.append(blk)
    oof = pd.concat(oof, ignore_index=True)
    return roc_auc_score(oof["Target_Label"], oof["p"]), float(oof["Target_Label"].mean())


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    frames = build_feature_frames(conn)
    conn.close()

    print("\nHorizon experiment (dev universe) :\n")
    print(f"  {'horizon':>8}{'barriers':>16}{'win rate':>10}{'OOF AUC':>10}")
    best = None
    for hold in HORIZONS:
        tp, sl = barriers(hold)
        pooled = label_and_pool(frames, tp, sl, hold)
        auc, wr = wf_auc(pooled, config.WIDE_FEATURES, hold)
        print(f"  {hold:>6}d{f'+{tp:.1%}/{sl:.1%}':>16}{wr:>10.1%}{auc:>10.4f}")
        best = (hold, tp, sl)

    # does the insider edge revive at the longest horizon?
    hold, tp, sl = best
    pooled = label_and_pool(frames, tp, sl, hold)
    base, _ = wf_auc(pooled, config.WIDE_FEATURES, hold)
    plus, _ = wf_auc(pooled, config.WIDE_FEATURES + config.INSIDER_FEATURES, hold)
    print(f"\n  Insider lift @ {hold}d : WIDE {base:.4f} -> WIDE+insider {plus:.4f}  ({plus-base:+.4f})")


if __name__ == "__main__":
    main()

"""
--------------------------------------------
INSIDER LIFT TEST (edge #1)
--------------------------------------------

The honest question: does insider flow actually improve the model?
Builds the dev-universe feature+label set WITH point-in-time insider features,
runs the walk-forward with XGBoost on WIDE vs WIDE+INSIDER, and compares OOF AUC
overall and on the insider-RICH names (where any lift should concentrate).
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

warnings.filterwarnings("ignore")
RICH = ["BAC", "XOM", "HD", "CAT", "NVDA", "BA"]     # names with real insider buying


def build_dataset(conn) -> pd.DataFrame:
    baselines = features.load_baselines(conn)
    frames = []
    for t in config.DEV_UNIVERSE:
        raw = features.load_stock(t, conn)
        df = labels.add_target(features.compute_features(raw, baselines))
        df = features.merge_insider(df, features.load_insider(t, conn))
        df.insert(1, "ticker", t)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def walk_forward_oof(pooled: pd.DataFrame, feats: list) -> pd.DataFrame:
    oof = []
    for f in config.FOLDS:
        cut = (date.fromisoformat(f["train_end"]) - timedelta(days=config.PURGE_DAYS)).isoformat()
        tr = pooled[pooled["date"] <= cut]
        te = pooled[(pooled["date"] >= f["test_start"]) & (pooled["date"] <= f["test_end"])]
        if tr.empty or te.empty:
            continue
        m = model_xgb.train_xgb(tr[feats].to_numpy("float32"), tr["Target_Label"].to_numpy("int8"))
        blk = te[["date", "ticker", "Target_Label"]].copy()
        blk["p"] = model_xgb.predict_xgb(m, te[feats].to_numpy("float32"))
        oof.append(blk)
    return pd.concat(oof, ignore_index=True)


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    pooled = build_dataset(conn)
    conn.close()

    base = walk_forward_oof(pooled, config.WIDE_FEATURES)
    plus = walk_forward_oof(pooled, config.WIDE_FEATURES + config.INSIDER_FEATURES)

    def auc(df):
        return roc_auc_score(df["Target_Label"], df["p"])

    print("\nInsider lift test (dev universe) :\n")
    print(f"  dataset : {len(pooled):,} rows,  insider-buy rows: "
          f"{int((pooled['Has_Insider_Buy'] == 1).sum()):,}\n")
    ab, ap = auc(base), auc(plus)
    print(f"  {'scope':<18}{'WIDE':>9}{'WIDE+insider':>15}{'lift':>9}")
    print(f"  {'ALL dev universe':<18}{ab:>9.4f}{ap:>15.4f}{ap-ab:>+9.4f}")
    br = base[base["ticker"].isin(RICH)]
    pr = plus[plus["ticker"].isin(RICH)]
    abr, apr = auc(br), auc(pr)
    print(f"  {'insider-rich only':<18}{abr:>9.4f}{apr:>15.4f}{apr-abr:>+9.4f}   (BAC/XOM/HD/CAT/NVDA/BA)")


if __name__ == "__main__":
    main()

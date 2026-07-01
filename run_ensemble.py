"""
--------------------------------------------
RUN PHASE 4 - ENSEMBLE EVALUATION
--------------------------------------------

Combines Model A + Model B on their OOF predictions and proves (honestly) whether
the ensemble beats the better single model: calibrators + meta-learner are fit on
folds 1-8 and evaluated on the unseen folds 9-12. No fold is scored on rows the
combiner trained on.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
import ensemble

warnings.filterwarnings("ignore")
LSTM_OOF = config.HERE / "walk_forward_oof_lstm.csv"


def main() -> None:
    xgb = pd.read_csv(config.OOF_PATH)                       # date, ticker, Target_Label, fold, p_xgb
    lstm = pd.read_csv(LSTM_OOF)[["date", "ticker", "p_lstm"]]
    df = xgb.merge(lstm, on=["date", "ticker"], how="inner").dropna()
    print(f"\nEnsemble (Phase 4) :\n")
    print(f"  aligned OOF preds : {len(df):,}")

    # Honest split: fit calibration + meta on folds 1-8, evaluate on 9-12
    train = df[df["fold"] <= 8].copy()
    test = df[df["fold"] >= 9].copy()

    cal_x = ensemble.fit_calibrator(train["p_xgb"], train["Target_Label"])
    cal_l = ensemble.fit_calibrator(train["p_lstm"], train["Target_Label"])
    for d in (train, test):
        d["cx"] = ensemble.apply_calibrator(cal_x, d["p_xgb"])
        d["cl"] = ensemble.apply_calibrator(cal_l, d["p_lstm"])

    meta = ensemble.train_meta_learner(train[["cx", "cl"]].to_numpy(), train["Target_Label"].to_numpy())
    y = test["Target_Label"].to_numpy()
    auc_x = roc_auc_score(y, test["cx"])
    auc_l = roc_auc_score(y, test["cl"])
    auc_blend = roc_auc_score(y, ensemble.weighted_blend(test["cx"], test["cl"]))
    auc_meta = roc_auc_score(y, ensemble.predict_meta(meta, test[["cx", "cl"]].to_numpy()))

    print(f"\n  Held-out folds 9-12 (incl. 2022 bear) AUC:")
    print(f"    XGBoost (calibrated) : {auc_x:.4f}")
    print(f"    LSTM (calibrated)    : {auc_l:.4f}")
    print(f"    Blend 50/50          : {auc_blend:.4f}")
    print(f"    Meta-learner         : {auc_meta:.4f}")

    best_single = max(auc_x, auc_l)
    lift = auc_meta - best_single
    print(f"\n    best single  : {best_single:.4f}")
    print(f"    ensemble lift: {lift:+.4f}   ({'PASS - ensemble helps' if lift > 0 else 'no AUC lift'})")
    print(f"    meta weights : xgb {meta.coef_[0][0]:+.2f}, lstm {meta.coef_[0][1]:+.2f}")


if __name__ == "__main__":
    main()

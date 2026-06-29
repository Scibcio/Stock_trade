"""
--------------------------------------------
SIGNAL PIPELINE - ORCHESTRATOR
--------------------------------------------

Wires the whole signal layer together, the way run() does in database.py.
Run this AFTER trading.db has been built/updated by database.py.

Flow:

    database.py  ->  features  ->  labels
                          |
                 +--------+--------+
                 |                 |
            model_xgb         model_lstm        (same walk-forward folds)
                 |  P_xgb          |  P_lstm
                 +--------+--------+
                          |
                      ensemble  ->  strategy  ->  evaluation

Every step is a stub right now. We fill them in one at a time, starting
with features.py.
"""

import config
import features
import labels
import model_xgb
import model_lstm
import ensemble
import strategy


def run() -> None:
    """
    Full signal pipeline.

    1. Build features + labels for every ML-ready stock in trading.db.
    2. Walk-forward: for each fold in config.FOLDS, train XGBoost and the
       CNN-LSTM on the train window, then predict the out-of-sample test
       year -> collect P_xgb and P_lstm.
    3. Combine the two with ensemble.py (agreement gate / blend / meta-learner).
    4. Turn the final scores into trades with strategy.py.
    5. Evaluate (reuse the FYP eval suite: ROC, walk-forward equity, Monte Carlo).
    """
    raise NotImplementedError("Pipeline scaffolded - fill in features.py first.")


if __name__ == "__main__":
    run()

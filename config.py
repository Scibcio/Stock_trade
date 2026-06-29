"""
--------------------------------------------
SIGNAL PIPELINE - SHARED CONFIG
--------------------------------------------

One place for every constant the pipeline shares, so the feature code,
both models, and the walk-forward loop never disagree. Edit here once.
"""

from pathlib import Path


# ----------------------------------
# PATHS
# ----------------------------------
HERE    = Path(__file__).parent
DB_PATH = HERE / "trading.db"          # the database built by database.py


# ----------------------------------
# LABELING (triple barrier)
# ----------------------------------
TAKE_PROFIT = 0.03      # +3% upper barrier (a "win")
STOP_LOSS   = -0.01     # -1% lower barrier (a "loss")
HOLD_DAYS   = 10        # vertical / time barrier (how many days we wait)


# ----------------------------------
# MODELS
# ----------------------------------
SEQ_LENGTH = 60         # CNN-LSTM input length: 60-day sequences

# XGBoost uses the WIDE feature set; the CNN-LSTM uses the LEAN set.
# Filled in once features.py exists.
LEAN_FEATURES: list[str] = []   # TODO: V4-style ~8 features for the LSTM
WIDE_FEATURES: list[str] = []   # TODO: full ~40 feature set for XGBoost


# ----------------------------------
# WALK-FORWARD FOLDS
# ----------------------------------
# Expanding training window, ~1-year out-of-sample test per fold.
# Each fold trains on everything up to train_end, then predicts the test year.
FOLDS = [
    {"fold": 1,  "train_end": "2013-12-31", "test_start": "2014-04-01", "test_end": "2015-03-31"},
    {"fold": 2,  "train_end": "2014-12-31", "test_start": "2015-04-01", "test_end": "2016-03-31"},
    {"fold": 3,  "train_end": "2015-12-31", "test_start": "2016-04-01", "test_end": "2017-03-31"},
    {"fold": 4,  "train_end": "2016-12-31", "test_start": "2017-04-01", "test_end": "2018-03-31"},
    {"fold": 5,  "train_end": "2017-12-31", "test_start": "2018-04-01", "test_end": "2019-03-31"},
    {"fold": 6,  "train_end": "2018-12-31", "test_start": "2019-04-01", "test_end": "2020-03-31"},
    {"fold": 7,  "train_end": "2019-12-31", "test_start": "2020-04-01", "test_end": "2021-03-31"},
    {"fold": 8,  "train_end": "2020-12-31", "test_start": "2021-04-01", "test_end": "2022-03-31"},
    {"fold": 9,  "train_end": "2021-12-31", "test_start": "2022-04-01", "test_end": "2023-03-31"},
    {"fold": 10, "train_end": "2022-12-31", "test_start": "2023-04-01", "test_end": "2024-03-31"},
    {"fold": 11, "train_end": "2023-12-31", "test_start": "2024-04-01", "test_end": "2025-03-31"},
    {"fold": 12, "train_end": "2024-12-31", "test_start": "2025-04-01", "test_end": "2026-03-31"},
]

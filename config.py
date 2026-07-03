"""
--------------------------------------------
SIGNAL PIPELINE - SHARED CONFIG
--------------------------------------------

One place for every constant the pipeline shares, so features.py, both models,
and the walk-forward loop never disagree. Edit here once.
"""

from pathlib import Path


# ----------------------------------
# PATHS
# ----------------------------------
HERE     = Path(__file__).parent
DB_PATH  = HERE / "trading.db"            # built by database.py
OOF_PATH = HERE / "walk_forward_oof.csv"  # out-of-fold predictions from pipeline.py


# ----------------------------------
# DEV UNIVERSE  (Phase 0)
# ----------------------------------
# 15 liquid, sector-diverse S&P 500 names (all full-history in trading.db).
# Develop/test on these; scale to the full universe only once proven.
DEV_UNIVERSE = [
    "AAPL", "MSFT", "NVDA",          # tech / semis
    "JPM", "BAC",                    # financials
    "XOM", "CVX",                    # energy
    "JNJ", "UNH",                    # health
    "PG", "KO", "WMT", "HD",         # staples / retail
    "CAT", "BA",                     # industrials
]


# ----------------------------------
# LABELING  (triple barrier - labels.py)
# ----------------------------------
# These define the TRAINING TARGET only. Execution exits live in the STRATEGY
# section below so exit geometry can be tuned without silently changing what
# the model is trained to rank (U3 retrains the label when geometry is adopted).
TAKE_PROFIT = 0.03      # +3% upper barrier (a "win")
STOP_LOSS   = -0.01     # -1% lower barrier (a "loss")
HOLD_DAYS   = 10        # vertical / time barrier
PURGE_DAYS  = 14        # calendar days purged before each fold's train_end (covers HOLD_DAYS)


# ----------------------------------
# STRATEGY  (shared by backtest.py + predict_live.py — one definition, F2)
# ----------------------------------
TOP_K          = 15     # names per cohort
MAX_PER_SECTOR = 3      # diversification cap
COST_PER_TRADE = 0.001  # round-trip cost per position (10 bps): spread + impact on liquid S&P names

# Gross exposure by market regime. Bear = 0.0: fold 9 (2022 bear) OOF AUC was
# 0.490 — the model cannot rank in bears, so bear exposure is edge-less risk (F3).
REGIME_EXPOSURE = {"bull": 1.0, "sideways": 0.6, "bear": 0.0}

# Execution exits (what the strategy actually trades — may differ from the label
# barriers above until the U3 retrain confirms a geometry change end-to-end).
# Symmetric +/-3% adopted PROVISIONALLY from the next-open exit sweep
# (run_exit_sweep.py, §2 decision rule): Sharpe 0.52 vs 0.37, total +72% vs +36%,
# DD within the 1.2x gate. The -1% stop whipsawed out of picks that recover.
# Must still survive the U3 symmetric-label retrain before it is final.
EXIT_TAKE_PROFIT = 0.03
EXIT_STOP_LOSS   = -0.03


# ----------------------------------
# INDICATOR WINDOWS  (features.py)
# ----------------------------------
SMA_WINDOWS      = [5, 10, 20, 50, 100, 200]   # 200 = the stock's own long-term trend
MOM_WINDOWS      = [5, 10, 20]
VOL_WINDOWS      = [5, 10, 20]
ZSCORE_WINDOWS   = [20, 60]
EMA_SPANS        = (12, 26)     # MACD fast / slow
EMA_DIST_SPAN    = 9            # short-term EMA distance
RSI_WINDOW       = 14
BB_WINDOW        = 20
BB_STD           = 2
ATR_WINDOW       = 14
BETA_WINDOW      = 20
VWAP_WINDOW      = 20
VOL_SURGE_WINDOW = 20
SPY_TREND_WINDOW = 200          # SPY 200MA regime


# ----------------------------------
# FEATURE SETS
# ----------------------------------
# LEAN -> CNN-LSTM (model_lstm.py); WIDE -> XGBoost (model_xgb.py).
# All scale-free / regime features; raw price levels are excluded on purpose
# (the FYP "toxic absolute value" lesson - V4 beat V1 by dropping them).
LEAN_FEATURES = [
    "Daily_Return", "SPY_Return", "Beta_20", "NATR_14",
    "VWAP_Dist", "BB_Width", "Vol_Surge", "EMA_9_Dist",
]

WIDE_FEATURES = LEAN_FEATURES + [
    "Relative_Strength", "Market_Bullish",
    "VIX_Level", "VIX_Change", "HYG_Return", "TNX_Change",
    "RSI_14", "MACD_Norm", "Boll_Pos",
    "SMA_5_Dist", "SMA_10_Dist", "SMA_20_Dist", "SMA_50_Dist", "SMA_100_Dist",
    "Mom_5", "Mom_10", "Mom_20",
    "Vol_5", "Vol_10", "Vol_20",
    "ZScore_20", "ZScore_60",
]

# Insider-flow features (edge #1, from insider_flow table). Tested ON TOP of WIDE
# for the lift test; XGBoost only (sparse/intermittent - would pollute the LSTM).
INSIDER_WINDOW = 20
INSIDER_FEATURES = ["Insider_Buys_20d", "Insider_Sells_20d", "Has_Insider_Buy"]

# Cross-sectional + own-trend candidates (lift-tested on top of WIDE). Cross-rank =
# a stock's percentile standing vs the whole universe that day (relative, not absolute).
CROSS_FEATURES = ["Mom_20_xrank", "RSI_14_xrank", "Vol_Surge_xrank", "Relative_Strength_xrank"]
NEW_FEATURES = ["SMA_200_Dist"] + CROSS_FEATURES


# ----------------------------------
# MODELS
# ----------------------------------
SEQ_LENGTH = 60         # CNN-LSTM input length: 60-day sequences


# ----------------------------------
# WALK-FORWARD FOLDS
# ----------------------------------
# Expanding training window, ~1-year out-of-sample test per fold.
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

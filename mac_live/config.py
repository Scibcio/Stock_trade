"""
--------------------------------------------
SHARED CONFIG  —  "MAX" AGGRESSIVE PROFILE  (Mac paper account)
--------------------------------------------

Same certified model as the main system, but tuned for MAXIMUM RETURN with risk
deliberately ignored — a paper-money sandbox to watch how much of the huge
backtest numbers survive live (spoiler: not much; that's the experiment).

Differences from the conservative live config:
  * TOP_K = 5          (max concentration — K=5 was the highest backtest return)
  * exits UNCAPPED     (let winners run; only a −15% catastrophe stop + time exit)
  * EXEC_HOLD_DAYS = 20 (~1 month hold; label stays 10-day = the certified ranker)
  * fully invested     (100% in every regime, bear included)
  * PORTFOLIO_MODE pure (no SPY core — 100% strategy)
  * PAPER_EQUITY_CAP 100k (deploy the whole paper account)

The TRAINING LABEL is unchanged (10-day 3:1) — we keep the certified ranker and
only change how its picks are TRADED.
"""

from pathlib import Path


# ----------------------------------
# PATHS
# ----------------------------------
HERE     = Path(__file__).parent
DB_PATH  = HERE / "trading.db"
OOF_PATH = HERE / "walk_forward_oof.csv"

DEV_UNIVERSE = ["AAPL", "MSFT", "NVDA", "JPM", "BAC", "XOM", "CVX",
                "JNJ", "UNH", "PG", "KO", "WMT", "HD", "CAT", "BA"]


# ----------------------------------
# LABELING  (the TRAINING TARGET — unchanged, the certified 10-day ranker)
# ----------------------------------
TAKE_PROFIT = 0.03
STOP_LOSS   = -0.01
HOLD_DAYS   = 10        # LABEL horizon — do NOT change (longer labels decay AUC)
PURGE_DAYS  = 14


# ----------------------------------
# STRATEGY  —  MAX AGGRESSION (risk ignored, paper only)
# ----------------------------------
TOP_K          = 5      # max concentration
MAX_PER_SECTOR = 5      # no sector cap for a 5-name book — take the raw top 5
COST_PER_TRADE = 0.001  # models spread/impact; the live account measures the real number

EXEC_HOLD_DAYS = 20     # EXECUTION hold + cohort cadence (~1 month); label stays 10-day

# Fully invested in every regime — no bear-cash, chase everything.
REGIME_EXPOSURE = {"bull": 1.0, "sideways": 1.0, "bear": 1.0}

EARNINGS_BLACKOUT = 0

# UNCAPPED exits: let winners run (no take-profit); only a wide catastrophe stop
# and the EXEC_HOLD_DAYS time exit close a position.
EXIT_TAKE_PROFIT = 99.0     # effectively no upper cap
EXIT_STOP_LOSS   = -0.15    # catastrophe stop only


# ----------------------------------
# INDICATOR WINDOWS  (features.py)
# ----------------------------------
SMA_WINDOWS      = [5, 10, 20, 50, 100, 200]
MOM_WINDOWS      = [5, 10, 20]
VOL_WINDOWS      = [5, 10, 20]
ZSCORE_WINDOWS   = [20, 60]
EMA_SPANS        = (12, 26)
EMA_DIST_SPAN    = 9
RSI_WINDOW       = 14
BB_WINDOW        = 20
BB_STD           = 2
ATR_WINDOW       = 14
BETA_WINDOW      = 20
VWAP_WINDOW      = 20
VOL_SURGE_WINDOW = 20
SPY_TREND_WINDOW = 200


# ----------------------------------
# FEATURE SETS
# ----------------------------------
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
INSIDER_WINDOW = 20
INSIDER_FEATURES = ["Insider_Buys_20d", "Insider_Sells_20d", "Has_Insider_Buy"]
CROSS_FEATURES = ["Mom_20_xrank", "RSI_14_xrank", "Vol_Surge_xrank", "Relative_Strength_xrank"]
NEW_FEATURES = ["SMA_200_Dist"] + CROSS_FEATURES


# ----------------------------------
# PAPER TRADING  (Alpaca DEMO account — real money out of scope, forever)
# ----------------------------------
PAPER_EQUITY_CAP  = 100_000       # deploy the whole paper account (max profit)
PORTFOLIO_MODE    = "pure"        # 100% strategy, no SPY core
SATELLITE_WEIGHT  = 0.25          # unused in pure mode (kept for compatibility)
MAX_ORDER_NOTIONAL = 50_000       # per-order cap; a 5-name book puts ~$20k/name
MIN_ORDER_NOTIONAL = 1.0
HALT_FILE = HERE / "HALT"         # kill switch — create this file to block all new orders
SUBMIT_MAX_HOURS  = 30            # defer entries when the next session is >30h away


# ----------------------------------
# MODELS / WALK-FORWARD FOLDS
# ----------------------------------
SEQ_LENGTH = 60
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

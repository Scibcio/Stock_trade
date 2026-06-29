"""
--------------------------------------------
SIGNAL PIPELINE - FEATURE ENGINEERING
--------------------------------------------

Turns the raw OHLCV + baselines in trading.db into a feature table
(one row per stock per day). Both models read from this same table:

  - XGBoost   uses the WIDE column set (a point-in-time snapshot per day)
  - CNN-LSTM  uses the LEAN column set (fed as 60-day sequences)

Stubs only for now - we fill these in first, porting the proven maths
from the FYP Feature_code_V4 / V1-2-3 files.
"""

import sqlite3

import numpy as np
import pandas as pd

import config


# ----------------------------------
# LOADERS
# ----------------------------------

def load_stock(ticker: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Load one stock's OHLCV history from daily_prices, oldest date first."""
    raise NotImplementedError("TODO: SELECT date, open, high, low, close, volume FROM daily_prices")


def load_baseline(symbol: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Load one baseline's history from market_baselines (e.g. SPY)."""
    raise NotImplementedError("TODO: SELECT date, close FROM market_baselines WHERE symbol = ?")


# ----------------------------------
# FEATURE COMPUTATION
# ----------------------------------

def compute_features(df: pd.DataFrame,
                     baselines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Add technical + macro features to one stock's OHLCV frame.
    Returns the frame with feature columns added (rolling-window NaNs dropped).

    TODO - port from FYP:
      LEAN (V4):  Daily_Return, SPY_Return, Beta_20, NATR_14, VWAP_Dist,
                  BB_Width, Vol_Surge, EMA_9_Dist
      WIDE (V1-2-3 + macro): SMAs + dist, RSI, MACD, Bollinger, momentum,
                  z-scores, and VIX / HYG / TNX / UUP / RSP derived features
    """
    raise NotImplementedError


def build_feature_table(ticker: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Full per-ticker step: load OHLCV + baselines, then compute_features()."""
    raise NotImplementedError

"""
--------------------------------------------
SIGNAL PIPELINE - FEATURE ENGINEERING
--------------------------------------------

Reads OHLCV + baselines from trading.db (built by database.py) and produces one
feature row per stock per day. Models pick columns via config:
    config.LEAN_FEATURES -> model_lstm.py     config.WIDE_FEATURES -> model_xgb.py

No look-ahead: every feature uses past data only (rolling / ewm / shift / diff).
The forward-looking target lives in labels.py, never here.
"""

import sqlite3

import numpy as np
import pandas as pd

import config


# ----------------------------------
# LOADERS  (from trading.db)
# ----------------------------------

def load_stock(ticker: str, conn: sqlite3.Connection) -> pd.DataFrame:
    # daily_prices -> OHLCV, oldest first
    return pd.read_sql_query(
        "SELECT date, open, high, low, close, volume "
        "FROM daily_prices WHERE ticker = ? ORDER BY date",
        conn, params=(ticker,),
    )


def load_baseline(symbol: str, conn: sqlite3.Connection) -> pd.DataFrame:
    # market_baselines -> date + close, oldest first
    return pd.read_sql_query(
        "SELECT date, close FROM market_baselines WHERE symbol = ? ORDER BY date",
        conn, params=(symbol,),
    )


def load_baselines(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    # the macro-context instruments features use
    return {s: load_baseline(s, conn) for s in ("SPY", "^VIX", "HYG", "^TNX")}


# ----------------------------------
# INDICATOR HELPERS
# ----------------------------------

def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-8)))


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# ----------------------------------
# FEATURE COMPUTATION
# ----------------------------------

def compute_features(df: pd.DataFrame,
                     baselines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Add all engineered features to one stock's OHLCV frame.
    `baselines` keyed by symbol (SPY, ^VIX, HYG, ^TNX); missing ones skipped.
    Returns the frame with feature columns, warmup NaNs dropped.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    # returns & momentum
    df["Daily_Return"] = close.pct_change()
    for w in config.MOM_WINDOWS:
        df[f"Mom_{w}"] = close.pct_change(w)
    for w in config.VOL_WINDOWS:
        df[f"Vol_{w}"] = df["Daily_Return"].rolling(w).std()

    # moving-average distances (scale-free "rubber band")
    for w in config.SMA_WINDOWS:
        sma = close.rolling(w).mean()
        df[f"SMA_{w}_Dist"] = (close - sma) / sma
    ema_dist = close.ewm(span=config.EMA_DIST_SPAN, adjust=False).mean()
    df["EMA_9_Dist"] = close / ema_dist - 1

    # MACD, normalised by price to stay scale-free
    fast = close.ewm(span=config.EMA_SPANS[0], adjust=False).mean()
    slow = close.ewm(span=config.EMA_SPANS[1], adjust=False).mean()
    df["MACD_Norm"] = (fast - slow) / close

    # z-scores
    for w in config.ZSCORE_WINDOWS:
        m, s = close.rolling(w).mean(), close.rolling(w).std()
        df[f"ZScore_{w}"] = (close - m) / (s + 1e-8)

    # RSI
    df["RSI_14"] = _rsi(close, config.RSI_WINDOW)

    # Bollinger: width (lean) + position in band (wide)
    mid = close.rolling(config.BB_WINDOW).mean()
    std = close.rolling(config.BB_WINDOW).std()
    upper, lower = mid + config.BB_STD * std, mid - config.BB_STD * std
    df["BB_Width"] = (upper - lower) / mid
    df["Boll_Pos"] = (close - lower) / (upper - lower + 1e-8)

    # volatility & volume
    df["NATR_14"] = _atr(df, config.ATR_WINDOW) / close * 100
    df["Vol_Surge"] = vol / vol.rolling(config.VOL_SURGE_WINDOW).mean()

    # VWAP distance (20-day rolling)
    typ = (high + low + close) / 3
    vwap = (typ * vol).rolling(config.VWAP_WINDOW).sum() / vol.rolling(config.VWAP_WINDOW).sum()
    df["VWAP_Dist"] = close / vwap - 1

    # macro context (past values only - ffill never bfill)
    df = _merge_macro(df, baselines)

    return df.dropna().reset_index(drop=True)


def _merge_macro(df: pd.DataFrame,
                 baselines: dict[str, pd.DataFrame]) -> pd.DataFrame:
    spy = baselines.get("SPY")
    if spy is not None and not spy.empty:
        df = df.merge(spy.rename(columns={"close": "_spy"}), on="date", how="left")
        df["_spy"] = df["_spy"].ffill()
        df["SPY_Return"] = df["_spy"].pct_change()
        df["Relative_Strength"] = df["Daily_Return"] - df["SPY_Return"]
        df["Market_Bullish"] = (df["_spy"] > df["_spy"].rolling(config.SPY_TREND_WINDOW).mean()).astype(int)
        cov = df["Daily_Return"].rolling(config.BETA_WINDOW).cov(df["SPY_Return"])
        var = df["SPY_Return"].rolling(config.BETA_WINDOW).var()
        df["Beta_20"] = (cov / (var + 1e-12)).replace([np.inf, -np.inf], np.nan)
        df.drop(columns="_spy", inplace=True)

    vix = baselines.get("^VIX")
    if vix is not None and not vix.empty:
        df = df.merge(vix.rename(columns={"close": "VIX_Level"}), on="date", how="left")
        df["VIX_Level"] = df["VIX_Level"].ffill()
        df["VIX_Change"] = df["VIX_Level"].pct_change()

    hyg = baselines.get("HYG")
    if hyg is not None and not hyg.empty:
        df = df.merge(hyg.rename(columns={"close": "_hyg"}), on="date", how="left")
        df["_hyg"] = df["_hyg"].ffill()
        df["HYG_Return"] = df["_hyg"].pct_change()
        df.drop(columns="_hyg", inplace=True)

    tnx = baselines.get("^TNX")
    if tnx is not None and not tnx.empty:
        df = df.merge(tnx.rename(columns={"close": "_tnx"}), on="date", how="left")
        df["_tnx"] = df["_tnx"].ffill()
        df["TNX_Change"] = df["_tnx"].diff()
        df.drop(columns="_tnx", inplace=True)

    return df


def build_feature_table(ticker: str, conn: sqlite3.Connection) -> pd.DataFrame:
    # full per-ticker step: load OHLCV + baselines -> compute_features
    df = load_stock(ticker, conn)
    if df.empty:
        return df
    return compute_features(df, load_baselines(conn))

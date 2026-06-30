"""
Shared pytest fixtures. Synthetic OHLCV + baselines so feature tests are fast,
deterministic, and don't depend on trading.db or the network.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project modules (config.py, features.py) importable from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _series(rng: np.random.Generator, start: float, n: int) -> pd.DataFrame:
    # geometric random walk -> always-positive OHLCV
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    close = start * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "date":   dates,
        "open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":   close * (1 + rng.uniform(0, 0.01, n)),
        "low":    close * (1 - rng.uniform(0, 0.01, n)),
        "close":  close,
        "volume": rng.integers(1_000_000, 5_000_000, n),
    })


@pytest.fixture
def sample():
    """(ohlcv, baselines) - 400 business days, enough to clear the 200-day warmup."""
    rng = np.random.default_rng(42)
    n = 400
    ohlcv = _series(rng, 100, n)
    baselines = {
        "SPY":  _series(rng, 400, n)[["date", "close"]],
        "^VIX": _series(rng, 20, n)[["date", "close"]],
        "HYG":  _series(rng, 80, n)[["date", "close"]],
        "^TNX": _series(rng, 15, n)[["date", "close"]],
    }
    return ohlcv, baselines

"""
--------------------------------------------
SIGNAL PIPELINE - STRATEGY LAYER
--------------------------------------------

Turns a final probability/score into actual trades. Kept SEPARATE from the
models on purpose: you can change confidence, position sizing, and
reward:risk WITHOUT retraining anything. This is where "different investing
strategies / risk profiles" live.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    confidence: float       = 0.75    # minimum score to take a trade
    risk_per_trade: float   = 50.0    # $ risked per trade
    reward_risk: float      = 3.0     # 3:1 reward-to-risk
    use_regime_filter: bool = True    # only trade when SPY > 200-day MA


def apply_strategy(predictions: pd.DataFrame,
                   cfg: StrategyConfig) -> pd.DataFrame:
    """Filter scored predictions into trades according to cfg. Returns a trades frame."""
    raise NotImplementedError


def simulate_portfolio(trades: pd.DataFrame,
                       start_balance: float = 1000.0) -> pd.DataFrame:
    """
    Walk the trades chronologically, compounding a wallet. Returns an equity
    curve. (Port the wallet sim from the FYP Quant_dashboard.)
    """
    raise NotImplementedError

"""
--------------------------------------------
SIGNAL PIPELINE - LABELING
--------------------------------------------

Builds the prediction target the models learn.

Current target = Triple Barrier: for each day, look forward HOLD_DAYS and
label 1 if price reaches +TAKE_PROFIT before -STOP_LOSS, otherwise 0.

The "let winners run" variants (phase 2) are stubbed at the bottom - those
are how we eventually catch big movers instead of capping every win at +3%.
"""

import numpy as np
import pandas as pd

import config


# ----------------------------------
# CURRENT TARGET (fixed triple barrier)
# ----------------------------------

def triple_barrier(df: pd.DataFrame,
                   take_profit: float = config.TAKE_PROFIT,
                   stop_loss: float = config.STOP_LOSS,
                   hold_days: int = config.HOLD_DAYS) -> pd.Series:
    """
    Return a 0/1 label per row.
    1 = take-profit hit before stop-loss within hold_days, else 0.
    (Port the forward-window loop from FYP Feature_code_V4.)
    """
    raise NotImplementedError


# ----------------------------------
# FUTURE: "let winners run" labels (phase 2)
# ----------------------------------

def trailing_stop_label(df: pd.DataFrame,
                        trail_pct: float,
                        max_hold: int) -> pd.Series:
    """
    TODO (phase 2): label trades with a TRAILING stop instead of a fixed +3%
    cap, so the model can learn to ride large moves (e.g. an IPO that runs
    150 -> 300) rather than selling at the first +3%.
    """
    raise NotImplementedError

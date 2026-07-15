"""
--------------------------------------------
SIGNAL PIPELINE - LABELING
--------------------------------------------

Builds the prediction target from the close prices of a feature frame
(features.py). Current target = Triple Barrier: for each day, look forward
HOLD_DAYS and label 1 if +TAKE_PROFIT is reached before -STOP_LOSS, else 0.
Barriers are checked on close (matches FYP V4).

The final HOLD_DAYS rows have no complete forward window, so they are left
unlabelled (NaN) and dropped - we never fake a label we can't actually know.

The "let winners run" variant (phase 2) is stubbed at the bottom.
"""

import numpy as np
import pandas as pd

import config


# ----------------------------------
# CURRENT TARGET (triple barrier)
# ----------------------------------

def triple_barrier(df: pd.DataFrame,
                   take_profit: float = config.TAKE_PROFIT,
                   stop_loss: float = config.STOP_LOSS,
                   hold_days: int = config.HOLD_DAYS) -> pd.Series:
    """
    0/1 label per row; NaN for the final hold_days (incomplete forward window).
    The first barrier touched inside the window decides the outcome.
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    out = np.full(n, np.nan)

    for i in range(n - hold_days):
        entry = close[i]
        out[i] = 0.0
        for px in close[i + 1: i + 1 + hold_days]:
            ret = (px - entry) / entry
            if ret <= stop_loss:        # stop-loss first -> stays a loss
                break
            if ret >= take_profit:      # take-profit first -> win
                out[i] = 1.0
                break

    return pd.Series(out, index=df.index, name="Target_Label")


def add_target(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # attach the label and drop the unlabelled tail; label as int 0/1
    df = df.copy()
    df["Target_Label"] = triple_barrier(df, **kwargs)
    df = df.dropna(subset=["Target_Label"]).reset_index(drop=True)
    df["Target_Label"] = df["Target_Label"].astype("int8")
    return df


# ----------------------------------
# "LET WINNERS RUN" label (phase 2)
# ----------------------------------

def trailing_stop_label(df: pd.DataFrame,
                        trail_pct: float,
                        max_hold: int,
                        stop_loss: float = config.STOP_LOSS) -> pd.Series:
    """
    0/1 label under a TRAILING stop instead of a fixed +TP, so a win can ride a
    large move (150 -> 300) rather than capping at +3%. From entry we hold an
    initial stop at `stop_loss`; once price runs up, the stop ratchets to
    peak * (1 - trail_pct). Exit at the first close through the stop, else at
    `max_hold`. Label = 1 if that exit is profitable. NaN for the final max_hold
    rows (incomplete forward window). Close-based, no look-ahead.
    """
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    out = np.full(n, np.nan)

    for i in range(n - max_hold):
        entry = close[i]
        peak = entry
        exit_ret = (close[i + max_hold] - entry) / entry      # default: time-barrier exit
        for px in close[i + 1: i + 1 + max_hold]:
            peak = max(peak, px)
            stop = max(entry * (1 + stop_loss), peak * (1 - trail_pct))
            if px <= stop:                                    # trailing / initial stop hit
                exit_ret = (px - entry) / entry
                break
        out[i] = 1.0 if exit_ret > 0 else 0.0

    return pd.Series(out, index=df.index, name="Target_Label")

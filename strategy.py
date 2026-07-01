"""
--------------------------------------------
SIGNAL PIPELINE - STRATEGY / BACKTEST (Phase 5)
--------------------------------------------

Turns the combined signal into a tradeable strategy and stress-tests it.
Selection = the top-K most-confident signals PER DAY (a real, capacity-limited
book, ~K concurrent positions). Each trade resolves via its triple-barrier label
(+reward_risk risk-units on a win, -1 on a loss), fixed-fractional sizing, costs.

Because our labels are OUTCOMES (not price paths), we validate the EDGE and the
RISK OF RUIN via Monte Carlo - not a false-precision equity curve. Kept separate
from the models so risk / selectivity change without retraining.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    top_k_per_day: int = 5        # concurrent positions (highest-confidence signals each day)
    risk_per_trade: float = 0.01  # fraction of equity risked per trade (the -1% stop)
    reward_risk: float = 3.0      # +3% / -1%
    cost: float = 0.001           # round-trip cost as a fraction of risk
    start_capital: float = 1000.0
    turnovers_per_year: int = 25  # ~10-day holds -> ~25 book turnovers a year


def select_top_per_day(df: pd.DataFrame, k: int) -> pd.DataFrame:
    # the k highest-confidence signals each day
    return (df.sort_values(["date", "blend"], ascending=[True, False])
              .groupby("date").head(k).reset_index(drop=True))


def edge(trades: pd.DataFrame, cfg: StrategyConfig) -> dict:
    p = float(trades["Target_Label"].mean())
    ev = p * cfg.reward_risk - (1 - p) - cfg.cost      # expected risk-units per trade
    return {"trades": len(trades), "win_rate": p, "ev_per_trade": ev}


def monte_carlo(win_rate: float, cfg: StrategyConfig, years: int = 5,
                n_sims: int = 2000, seed: int = 12345) -> dict:
    # compound a K-position book period-by-period; measure growth + downside
    rng = np.random.default_rng(seed)
    K, f, rr, cost = cfg.top_k_per_day, cfg.risk_per_trade, cfg.reward_risk, cfg.cost
    n_periods = years * cfg.turnovers_per_year
    finals, max_dds, ruined = [], [], 0
    for _ in range(n_sims):
        eq, peak, mdd, blew = cfg.start_capital, cfg.start_capital, 0.0, False
        for _ in range(n_periods):
            wins = rng.random(K) < win_rate
            period_ret = float(np.sum(np.where(wins, rr * f, -f)) - K * cost * f)
            eq *= (1 + period_ret)
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1)
            if eq < cfg.start_capital * 0.5:
                blew = True
        finals.append(eq)
        max_dds.append(mdd)
        ruined += blew
    finals = np.array(finals)
    return {"years": years,
            "median_mult": float(np.median(finals) / cfg.start_capital),
            "p5_mult": float(np.percentile(finals, 5) / cfg.start_capital),
            "p95_mult": float(np.percentile(finals, 95) / cfg.start_capital),
            "risk_of_50pct_dd": ruined / n_sims,
            "median_max_dd": float(np.median(max_dds))}

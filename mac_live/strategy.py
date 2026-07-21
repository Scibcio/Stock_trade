"""
--------------------------------------------
SIGNAL PIPELINE - STRATEGY LAYER
--------------------------------------------

The ONE place cohort selection lives (F8): sector-capped top-K by probability,
inverse-NATR weights, regime-scaled gross exposure. backtest.py and
predict_live.py both import select_cohort, so the simulation and the live
system agree by construction, not by discipline.

Below it: the original Monte-Carlo edge/risk-of-ruin study (research artifact).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


# ----------------------------------
# COHORT SELECTION  (shared: backtest.py + predict_live.py)
# ----------------------------------

def select_cohort(day: pd.DataFrame, prob_col: str, regime: str,
                  equal_weight: bool = False) -> pd.DataFrame | None:
    """
    One day's tradeable book from candidate rows (needs prob_col, sector, NATR_14).
    Top-K by prob_col with a per-sector cap; weights = inverse NATR normalised to
    the regime's gross exposure (rest is cash). None if regime exposure is 0 or
    no candidates (bear = stay in cash, F3).
    """
    exposure = config.REGIME_EXPOSURE.get(regime, 0.5)
    if exposure <= 0 or day.empty:
        return None

    chosen, per_sector = [], {}
    for _, r in day.sort_values(prob_col, ascending=False).iterrows():
        if per_sector.get(r["sector"], 0) >= config.MAX_PER_SECTOR:
            continue
        chosen.append(r)
        per_sector[r["sector"]] = per_sector.get(r["sector"], 0) + 1
        if len(chosen) >= config.TOP_K:
            break
    if not chosen:
        return None

    picks = pd.DataFrame(chosen)
    if equal_weight:
        w = np.ones(len(picks)) / len(picks)
    else:
        inv = 1.0 / picks["NATR_14"].clip(lower=0.1)
        w = (inv / inv.sum()).to_numpy()
    picks["weight"] = w * exposure
    picks["regime_exposure"] = exposure
    return picks


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

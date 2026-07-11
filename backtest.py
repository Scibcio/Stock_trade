"""
--------------------------------------------
BACKTEST / SIMULATION  (Phase 6)
--------------------------------------------

Replays the AI's *out-of-sample* picks day-by-day across history and asks the
question you actually care about: "if I had traded this, what would have happened?"

For every rebalance date it runs the SAME code path predict_live.py trades
(strategy.select_cohort — parity by construction, F8):
  * rank by probability, sector-cap, top-K, inverse-NATR weights, regime exposure
    (bear = 0.0 -> all cash, F3),
  * ENTER AT THE NEXT SESSION'S OPEN (F1) — the signal prints on the close, so
    that close is untradeable; exits via config.EXIT_* barriers on closes,
  * charge a round-trip transaction cost on every position,
then compounds the cohorts into an equity curve and reports the real win rate,
expectancy, drawdown, Sharpe, and a per-year / per-regime breakdown.

TWO SIGNALS, reported side by side (a reviewer caught the live system and an
earlier draft disagreeing on which one to trade):
  * LIVE (XGB only)   - exactly what predict_live.py ranks by today. This is the
                        honest "what my deployed system does" number.
  * BLEND (XGB+LSTM)  - what you'd get IF you added the LSTM to the live picker.
                        Shown only to answer "is upgrading live worth it?".

Signals are the walk-forward OOF predictions (pipeline.py / run_lstm.py): each was
made by a model that never saw its own test window, so this is a genuine replay.

BENCHMARK: SPY buy-and-hold on the identical rebalance grid, with its OWN Sharpe
and drawdown, so the risk comparison is apples-to-apples (no cherry-picking).

!!  SURVIVORSHIP CAVEAT  !!
The universe is TODAY's S&P 500 survivors, so these figures are optimistically
biased (dead/delisted names are absent). Treat this as a study of the strategy's
*behaviour* — the bias-free proof is the FORWARD paper-trade (predict_live.py).

Run:  python backtest.py            (full OOF span, ~2014-2026)
      python backtest.py 2024-01-01 (only from a start date)
"""

import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd

import config
import features
import strategy
import threshold_analysis as ta

warnings.filterwarnings("ignore")

REBALANCE = config.HOLD_DAYS                                   # non-overlapping cohorts, one per hold window
PERIODS_PER_YEAR = 252 / REBALANCE                             # ~25 cohorts / year, for annualising Sharpe
COST_PER_TRADE = config.COST_PER_TRADE                         # single definition in config (F2)

EQUITY_CSV = config.HERE / "backtest_equity.csv"
TRADES_CSV = config.HERE / "backtest_trades.csv"


# ==================================================
# REALISED RETURNS  (triple barrier on close, honest gap-through)
# ==================================================

def realized_return_series(open_: np.ndarray,
                           close: np.ndarray,
                           take_profit: float = config.EXIT_TAKE_PROFIT,
                           stop_loss: float = config.EXIT_STOP_LOSS,
                           hold: int = config.HOLD_DAYS) -> np.ndarray:
    """
    Per-signal realised % return with HONEST timing (F1): the signal prints on
    day i's close, so the entry fills at the NEXT session's open (open[i+1]) —
    day i's close was untradeable. Barriers are checked on closes from the entry
    session onward; time barrier = close[i+1+hold]. Exit at the actual close
    (a gap can win >TP or lose <SL). take_profit/stop_loss of None disables that
    barrier (plain-hold / catastrophe-stop variants in the exit sweep).
    NaN for the final hold+1 rows (no complete forward window -> untradeable).
    """
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(n - hold - 1):
        entry = open_[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        r = (close[i + 1 + hold] - entry) / entry              # default: time-barrier (timeout) exit
        for j in range(i + 1, i + 2 + hold):
            ret = (close[j] - entry) / entry
            if ((stop_loss is not None and ret <= stop_loss)
                    or (take_profit is not None and ret >= take_profit)):
                r = ret                                        # first barrier touched decides the exit
                break
        out[i] = r
    return out


# ==================================================
# SIGNAL TABLE  (OOF preds + regime + close/NATR/realised, per date+ticker)
# ==================================================

def load_signals(conn) -> pd.DataFrame:
    oof = ta.load_blended_oof(conn)                            # date, ticker, Target_Label, p_xgb, p_lstm, blend, regime
    oof = oof[["date", "ticker", "Target_Label", "p_xgb", "p_lstm", "blend", "regime"]]

    sectors = dict(conn.execute("SELECT ticker, sector FROM stocks").fetchall())
    baselines = features.load_baselines(conn)

    frames = []
    tickers = sorted(oof["ticker"].unique())
    for n, t in enumerate(tickers, 1):
        feat = features.compute_features(features.load_stock(t, conn), baselines)
        if feat.empty:
            continue
        feat = feat[["date", "open", "close", "NATR_14", "Beta_20"]].copy()   # Beta_20 for U7 hedge
        feat["realized"] = realized_return_series(feat["open"].to_numpy(dtype=float),
                                                  feat["close"].to_numpy(dtype=float))
        feat["ticker"] = t
        frames.append(feat)
        if n % 50 == 0:
            print(f"  features {n}/{len(tickers)}")

    market = pd.concat(frames, ignore_index=True)
    prices = {t: g.set_index("date")[["open", "close"]].sort_index()   # full path per ticker
              for t, g in market.groupby("ticker")}                    # (trailing exits + next-open fills)
    df = oof.merge(market, on=["date", "ticker"], how="inner")
    df["sector"] = df["ticker"].map(sectors).fillna("Unknown")
    df = df.dropna(subset=["realized", "NATR_14"]).sort_values("date").reset_index(drop=True)
    return df, prices


# ==================================================
# ONE COHORT  — the SAME code path predict_live trades (strategy.select_cohort, F8)
# ==================================================

def pick_cohort(day: pd.DataFrame, signal_col: str, equal_weight: bool = False):
    if day.empty:
        return None
    return strategy.select_cohort(day, signal_col, day["regime"].iloc[0], equal_weight)


# ==================================================
# SIMULATE  (one signal, net of costs)
# ==================================================

def simulate(df: pd.DataFrame, signal_col: str = "p_xgb",
             cost: float = COST_PER_TRADE, equal_weight: bool = False):
    dates = sorted(df["date"].unique())
    rebal_dates = dates[::REBALANCE]                          # non-overlapping: one cohort per hold window

    equity, peak, rows, trades = 1.0, 1.0, [], []
    for d in rebal_dates:
        day = df[df["date"] == d]
        picks = pick_cohort(day, signal_col, equal_weight)
        if picks is None:                                     # bear/empty -> all cash, flat period (F3)
            if not day.empty:
                rows.append({"date": d, "regime": day["regime"].iloc[0], "n": 0, "exposure": 0.0,
                             "cohort_return": 0.0, "equity": equity, "drawdown": equity / peak - 1})
            continue
        net = picks["realized"] - cost                        # each position pays a round-trip cost
        cohort_ret = float((picks["weight"] * net).sum())     # cash portion contributes 0
        equity *= (1 + cohort_ret)
        peak = max(peak, equity)
        rows.append({"date": d, "regime": picks["regime"].iloc[0], "n": len(picks),
                     "exposure": picks["regime_exposure"].iloc[0],
                     "cohort_return": cohort_ret, "equity": equity, "drawdown": equity / peak - 1})
        for _, r in picks.iterrows():
            trades.append({"date": d, "ticker": r["ticker"], "sector": r["sector"], "regime": r["regime"],
                           "prob": r[signal_col], "win": int(r["Target_Label"]),
                           "return": float(r["realized"] - cost), "gross_return": float(r["realized"])})
    return pd.DataFrame(rows), pd.DataFrame(trades)


def summarize(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}
    wins = trades_df["win"] == 1                              # barrier win = hit +3% before -1% (cost-agnostic)
    gains = trades_df.loc[trades_df["return"] > 0, "return"]  # net-of-cost P&L
    losses = trades_df.loc[trades_df["return"] <= 0, "return"]

    span_days = (pd.to_datetime(equity_df["date"].iloc[-1]) - pd.to_datetime(equity_df["date"].iloc[0])).days
    years = max(span_days / 365.25, 1e-9)
    final = equity_df["equity"].iloc[-1]
    cohort = equity_df["cohort_return"]

    by_year = (trades_df.assign(year=pd.to_datetime(trades_df["date"]).dt.year)
               .groupby("year").agg(trades=("win", "size"), win_rate=("win", "mean"), avg_return=("return", "mean")))
    by_regime = (trades_df.groupby("regime")
                 .agg(trades=("win", "size"), win_rate=("win", "mean"), avg_return=("return", "mean")))

    return {
        "trades": len(trades_df), "cohorts": len(equity_df),
        "win_rate": float(wins.mean()),
        "avg_return": float(trades_df["return"].mean()),        # net expectancy per trade
        "avg_gross": float(trades_df["gross_return"].mean()),   # before costs, for reference
        "avg_win": float(gains.mean()) if len(gains) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(gains.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
        "total_return": float(final - 1),
        "cagr": float(final ** (1 / years) - 1),
        "max_drawdown": float(equity_df["drawdown"].min()),
        "sharpe": float(cohort.mean() / cohort.std() * np.sqrt(PERIODS_PER_YEAR)) if cohort.std() else 0.0,
        "years": years, "start": equity_df["date"].iloc[0], "end": equity_df["date"].iloc[-1],
        "by_year": by_year, "by_regime": by_regime,
    }


def spy_stats(conn, rebal_dates) -> dict:
    # SPY buy-and-hold on the IDENTICAL rebalance grid, with its own Sharpe + drawdown
    spy = pd.read_sql_query(
        "SELECT date, close FROM market_baselines WHERE symbol='SPY' ORDER BY date", conn)
    spy = spy[spy["date"].isin(rebal_dates)].sort_values("date").reset_index(drop=True)
    if len(spy) < 2:
        return {}
    per = spy["close"].pct_change().dropna()                  # same 10-trading-day period returns
    eq = spy["close"] / spy["close"].iloc[0]
    return {
        "total_return": float(eq.iloc[-1] - 1),
        "sharpe": float(per.mean() / per.std() * np.sqrt(PERIODS_PER_YEAR)) if per.std() else 0.0,
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
    }


# ==================================================
# REPORT
# ==================================================

def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else None
    conn = sqlite3.connect(config.DB_PATH)
    print("\n" + "=" * 64)
    print("  BACKTEST / SIMULATION" + (f"   (from {start})" if start else "   (full OOF span)"))
    print("=" * 64)
    print(f"  costs: {COST_PER_TRADE:.2%} round-trip / position   |   ! survivorship-biased")
    print("  (today's survivors - a behaviour study, not the bias-free proof)\n")

    df, _prices = load_signals(conn)
    if start:
        df = df[df["date"] >= start]
    rebal_dates = sorted(df["date"].unique())[::REBALANCE]
    spy = spy_stats(conn, rebal_dates)

    runs = {}
    for name, sig in (("LIVE  (XGB only)", "p_xgb"), ("BLEND (XGB+LSTM)", "blend")):
        eq, tr = simulate(df, signal_col=sig)
        runs[name] = (eq, tr, summarize(eq, tr))

    # the LIVE-parity run is what the dashboard shows (it is what predict_live actually trades)
    eq, tr, live = runs["LIVE  (XGB only)"]
    eq.to_csv(EQUITY_CSV, index=False)
    tr.to_csv(TRADES_CSV, index=False)
    conn.close()

    s = live
    print(f"  Period : {s['start']}  ->  {s['end']}   ({s['years']:.1f} yrs, {s['cohorts']} cohorts)\n")
    hdr = f"  {'signal':<18}{'win%':>7}{'net/trade':>11}{'PF':>6}{'total':>9}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, (_, _, st) in runs.items():
        print(f"  {name:<18}{st['win_rate']:>6.1%}{st['avg_return']:>+11.2%}{st['profit_factor']:>6.2f}"
              f"{st['total_return']:>+9.0%}{st['cagr']:>+8.1%}{st['max_drawdown']:>8.1%}{st['sharpe']:>8.2f}")
    if spy:
        print(f"  {'SPY buy&hold':<18}{'-':>7}{'-':>11}{'-':>6}"
              f"{spy['total_return']:>+9.0%}{'-':>8}{spy['max_drawdown']:>8.1%}{spy['sharpe']:>8.2f}")

    print(f"\n  Net expectancy incl. costs: {s['avg_return']:+.2%}/trade "
          f"(gross {s['avg_gross']:+.2%}, win {s['avg_win']:+.2%} / loss {s['avg_loss']:+.2%})")
    print("\n  LIVE (XGB only) by year :\n")
    yr = s["by_year"].copy()
    yr["win_rate"] = (yr["win_rate"] * 100).round(1)
    yr["avg_return"] = (yr["avg_return"] * 100).round(2)
    print(yr.to_string())
    print("\n  LIVE (XGB only) by regime :\n")
    rg = s["by_regime"].copy()
    rg["win_rate"] = (rg["win_rate"] * 100).round(1)
    rg["avg_return"] = (rg["avg_return"] * 100).round(2)
    print(rg.to_string())
    print(f"\n  Saved LIVE equity -> {EQUITY_CSV.name}   trades -> {TRADES_CSV.name}")


if __name__ == "__main__":
    main()

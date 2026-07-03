"""
--------------------------------------------
LET-WINNERS-RUN EXPERIMENT  (Phase 6b)
--------------------------------------------

The baseline backtest (backtest.py) diagnosed the bottleneck: the +3% take-profit caps
every winner, so a long-only book cannot out-compound a historic bull market. This tests
the fix — replace the fixed +TP with a TRAILING STOP that lets winners ride.

Variable, longer holds mean positions overlap, so the 10-day non-overlapping cohort model
no longer applies. This is a proper EVENT-DRIVEN portfolio:
  * up to N concurrent slots, sized 1/N of equity, marked-to-market daily,
  * enter on the same rebalance cadence (top probability, sector-capped, regime-scaled slots),
  * exit each name on a trailing stop: initial stop at -1%, ratcheting to peak*(1-trail)
    once in profit, or a hard time-barrier at max_hold,
  * round-trip transaction cost per position.

It uses the EXISTING XGB ranking (no retrain) on purpose: that isolates the effect of the
PAYOFF STRUCTURE from the effect of the label horizon (longer horizons hurt AUC — FINDINGS).
If a trailing exit beats SPY here, the next step is to retrain on labels.trailing_stop_label.

Same survivorship caveat as backtest.py: today's survivors, a behaviour study.

Run:  python backtest_trailing.py
"""

import sqlite3
import warnings

import numpy as np
import pandas as pd

import config
from backtest import (COST_PER_TRADE, MAX_PER_SECTOR, REBALANCE, REGIME_EXPOSURE,
                      TOP_K, load_signals, simulate, summarize)

warnings.filterwarnings("ignore")

EQUITY_CSV = config.HERE / "backtest_trailing_equity.csv"


def _asof(panel_entry, t):
    # last close at or before date t (ISO strings sort correctly); nan if t precedes history
    dates, closes = panel_entry
    idx = np.searchsorted(dates, t, side="right") - 1
    return closes[idx] if idx >= 0 else np.nan


# ==================================================
# EVENT-DRIVEN PORTFOLIO with trailing-stop exits
# ==================================================

def simulate_trailing(df, prices, signal_col="p_xgb", n_slots=TOP_K,
                      trail=0.10, max_hold=60, init_stop=config.STOP_LOSS,
                      cost=COST_PER_TRADE):
    panel = {t: (s.index.to_numpy(), s.to_numpy(dtype=float)) for t, s in prices.items()}
    dates = sorted(df["date"].unique())
    by_date = {d: g.sort_values(signal_col, ascending=False)[["ticker", "sector", "close"]].to_numpy()
               for d, g in df.groupby("date")}
    regime_of = dict(zip(df["date"], df["regime"]))

    open_pos, cash, trades, curve = [], 1.0, [], []

    def close_out(p, px, exit_date):
        nonlocal cash
        cash += p["invested"] * (px / p["entry"]) * (1 - cost)   # exit proceeds, net of cost
        trades.append({"entry_date": p["entry_date"], "exit_date": exit_date, "ticker": p["ticker"],
                       "sector": p["sector"], "days": p["days"],
                       "return": px / p["entry"] * (1 - cost) - 1})   # net return, same haircut as cash

    for i, t in enumerate(dates):
        # --- 1. process exits + mark-to-market ---
        still, mtm = [], 0.0
        for p in open_pos:
            px = _asof(panel[p["ticker"]], t)
            if np.isnan(px):
                still.append(p)
                continue
            p["peak"] = max(p["peak"], px)
            p["days"] += 1
            stop = max(p["entry"] * (1 + init_stop), p["peak"] * (1 - trail))
            if px <= stop or p["days"] >= max_hold:
                close_out(p, px, t)
            else:
                still.append(p)
                mtm += p["invested"] * (px / p["entry"])
        open_pos = still
        equity = cash + mtm

        # --- 2. entries on the rebalance cadence, filling free slots ---
        if i % REBALANCE == 0 and t in by_date:
            target = int(round(n_slots * REGIME_EXPOSURE.get(regime_of.get(t, "bull"), 0.5)))
            held = {p["ticker"] for p in open_pos}
            sec = {}
            for p in open_pos:
                sec[p["sector"]] = sec.get(p["sector"], 0) + 1
            for ticker, sector, close_t in by_date[t]:
                if len(open_pos) >= target:
                    break
                if ticker in held or sec.get(sector, 0) >= MAX_PER_SECTOR:
                    continue
                invest = min(equity / n_slots, cash)
                if invest <= 1e-6:
                    break
                cash -= invest
                open_pos.append({"ticker": ticker, "entry": float(close_t), "peak": float(close_t),
                                 "invested": invest, "days": 0, "sector": sector, "entry_date": t})
                held.add(ticker)
                sec[sector] = sec.get(sector, 0) + 1

        curve.append({"date": t, "equity": equity, "cash": cash, "open": len(open_pos)})

    # --- 3. close anything still open at the last price ---
    last = dates[-1]
    for p in open_pos:
        px = _asof(panel[p["ticker"]], last)
        close_out(p, px if not np.isnan(px) else p["entry"], last)
    return pd.DataFrame(curve), pd.DataFrame(trades), cash


def curve_stats(name, curve_df, trades_df, final_cash):
    eq = curve_df["equity"].reset_index(drop=True)
    if len(curve_df):
        eq.iloc[-1] = final_cash                                  # after final close, equity = cash
    span = (pd.to_datetime(curve_df["date"].iloc[-1]) - pd.to_datetime(curve_df["date"].iloc[0])).days
    years = max(span / 365.25, 1e-9)
    daily = eq.pct_change().dropna()
    ret = trades_df["return"] if len(trades_df) else pd.Series(dtype=float)
    return {
        "name": name,
        "total": final_cash - 1,
        "cagr": final_cash ** (1 / years) - 1,
        "maxdd": float((eq / eq.cummax() - 1).min()),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() else 0.0,
        "trades": len(trades_df),
        "win_rate": float((ret > 0).mean()) if len(ret) else float("nan"),
        "avg": float(ret.mean()) if len(ret) else float("nan"),
        "median": float(ret.median()) if len(ret) else float("nan"),
        "hold": float(trades_df["days"].mean()) if len(trades_df) else float("nan"),
    }


def spy_daily(conn, start, end):
    spy = pd.read_sql_query(
        "SELECT date, close FROM market_baselines WHERE symbol='SPY' AND date>=? AND date<=? ORDER BY date",
        conn, params=(start, end))
    eq = spy["close"] / spy["close"].iloc[0]
    daily = spy["close"].pct_change().dropna()
    years = max((pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25, 1e-9)
    return {"name": "SPY buy&hold", "total": float(eq.iloc[-1] - 1),
            "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
            "maxdd": float((eq / eq.cummax() - 1).min()),
            "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)),
            "trades": np.nan, "win_rate": np.nan, "avg": np.nan, "median": np.nan, "hold": np.nan}


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    print("\n" + "=" * 78)
    print("  LET-WINNERS-RUN EXPERIMENT  (trailing stop vs the +3% cap)")
    print("=" * 78)
    print(f"  costs {COST_PER_TRADE:.2%} round-trip | XGB-only ranking (no retrain) | survivorship-biased\n")

    df, prices = load_signals(conn)
    start, end = df["date"].min(), df["date"].max()

    rows = []

    # reference: the current strategy (fixed +3% / -1% / 10-day cohort model)
    eq0, tr0 = simulate(df, signal_col="p_xgb")
    s0 = summarize(eq0, tr0)
    rows.append({"name": "+3% cap / 10d (current)", "total": s0["total_return"], "cagr": s0["cagr"],
                 "maxdd": s0["max_drawdown"], "sharpe": s0["sharpe"], "trades": s0["trades"],
                 "win_rate": s0["win_rate"], "avg": s0["avg_return"],
                 "median": float(tr0["return"].median()), "hold": float(REBALANCE)})

    # the experiment: several trailing-stop configurations
    configs = [
        ("trail 8% / 40d", dict(trail=0.08, max_hold=40)),
        ("trail 12% / 60d", dict(trail=0.12, max_hold=60)),
        ("trail 20% / 90d", dict(trail=0.20, max_hold=90)),
        ("-1% stop only / 60d", dict(trail=0.99, max_hold=60)),
    ]
    best = None
    for name, kw in configs:
        curve, trades, cash = simulate_trailing(df, prices, signal_col="p_xgb", **kw)
        st = curve_stats(name, curve, trades, cash)
        rows.append(st)
        if best is None or st["total"] > best[0]:
            best = (st["total"], name, curve)

    rows.append(spy_daily(conn, start, end))
    conn.close()

    if best is not None:
        best[2].to_csv(EQUITY_CSV, index=False)

    print(f"  Period {start} -> {end}\n")
    hdr = (f"  {'variant':<24}{'trades':>7}{'win%':>7}{'avg/tr':>9}{'med/tr':>9}{'hold':>7}"
           f"{'total':>9}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        wr = "   -  " if pd.isna(r["win_rate"]) else f"{r['win_rate']:>5.1%}"
        av = "     - " if pd.isna(r["avg"]) else f"{r['avg']:>+7.2%}"
        md = "     - " if pd.isna(r["median"]) else f"{r['median']:>+7.2%}"
        hd = "    -" if pd.isna(r["hold"]) else f"{r['hold']:>5.0f}"
        tr_ = "   -  " if pd.isna(r["trades"]) else f"{int(r['trades']):>6}"
        print(f"  {r['name']:<24}{tr_:>7}{wr:>7}{av:>9}{md:>9}{hd:>7}"
              f"{r['total']:>+9.0%}{r['cagr']:>+8.1%}{r['maxdd']:>8.1%}{r['sharpe']:>8.2f}")
    print(f"\n  Saved best-total equity curve -> {EQUITY_CSV.name}")
    print("  (Uses the existing 10-day-trained ranker; if a trailing exit wins, retrain on")
    print("   labels.trailing_stop_label to let the model select for the longer hold.)")


if __name__ == "__main__":
    main()

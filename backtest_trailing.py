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
  * decide on the rebalance cadence (top probability, sector-capped, regime-scaled slots)
    and FILL AT THE NEXT SESSION'S OPEN (F1 honesty - the decision close is untradeable),
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
from backtest import COST_PER_TRADE, REBALANCE, load_signals, simulate, summarize

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

def simulate_trailing(df, prices, signal_col="p_xgb", n_slots=config.TOP_K,
                      trail=0.10, max_hold=60, init_stop=config.STOP_LOSS,
                      cost=COST_PER_TRADE, take_profit=None, entry_every=None,
                      sector_cap=None):
    # take_profit / entry_every / sector_cap generalize the engine for the
    # cadence sweep (run_cadence_sweep.py). Defaults preserve trailing behaviour.
    entry_every = entry_every or REBALANCE
    sector_cap = sector_cap or config.MAX_PER_SECTOR
    panel = {t: (p.index.to_numpy(), p["close"].to_numpy(dtype=float),
                 p["open"].to_numpy(dtype=float)) for t, p in prices.items()}
    dates = sorted(df["date"].unique())
    by_date = {d: g.sort_values(signal_col, ascending=False)[["ticker", "sector"]].to_numpy()
               for d, g in df.groupby("date")}
    regime_of = dict(zip(df["date"], df["regime"]))

    open_pos, cash, trades, curve = [], 1.0, [], []
    scheduled = {}                                             # fill_date -> entries decided earlier
    pending = set()                                            # tickers awaiting their fill
    equity = 1.0

    def close_out(p, px, exit_date):
        nonlocal cash
        cash += p["invested"] * (px / p["entry"]) * (1 - cost)   # exit proceeds, net of cost
        trades.append({"entry_date": p["entry_date"], "exit_date": exit_date, "ticker": p["ticker"],
                       "sector": p["sector"], "days": p["days"],
                       "return": px / p["entry"] * (1 - cost) - 1})   # net return, same haircut as cash

    for i, t in enumerate(dates):
        # --- 1. fill entries scheduled for today at TODAY'S OPEN (F1 honesty:
        #        the decision-day close was untradeable) ---
        for e in scheduled.pop(t, []):
            pending.discard(e["ticker"])
            invest = min(equity / n_slots, cash)               # equity as of yesterday's marks
            if invest <= 1e-6:
                continue
            cash -= invest
            open_pos.append({"ticker": e["ticker"], "sector": e["sector"],
                             "entry": e["open"], "peak": e["open"], "invested": invest,
                             "days": 0, "entry_date": t})

        # --- 2. exits + mark-to-market on today's close (a new position's first
        #        check is its own entry-session close, matching the label) ---
        still, mtm = [], 0.0
        for p in open_pos:
            px = _asof(panel[p["ticker"]][:2], t)
            if np.isnan(px):
                still.append(p)
                continue
            p["peak"] = max(p["peak"], px)
            p["days"] += 1
            stop = max(p["entry"] * (1 + init_stop), p["peak"] * (1 - trail))
            hit_tp = take_profit is not None and px >= p["entry"] * (1 + take_profit)
            if px <= stop or hit_tp or p["days"] >= max_hold:
                close_out(p, px, t)
            else:
                still.append(p)
                mtm += p["invested"] * (px / p["entry"])
        open_pos = still
        equity = cash + mtm

        # --- 3. decisions on the rebalance cadence: rank at today's close,
        #        schedule each fill for the ticker's NEXT session open ---
        if i % entry_every == 0 and t in by_date:
            target = int(round(n_slots * config.REGIME_EXPOSURE.get(regime_of.get(t, "bull"), 0.5)))
            held = {p["ticker"] for p in open_pos} | pending
            sec = {}
            for p in open_pos:
                sec[p["sector"]] = sec.get(p["sector"], 0) + 1
            n_book = len(open_pos) + len(pending)
            for ticker, sector in by_date[t]:
                if n_book >= target:
                    break
                if ticker in held or sec.get(sector, 0) >= sector_cap:
                    continue
                tdates, _, topens = panel[ticker]
                j = np.searchsorted(tdates, t, side="right")   # first bar strictly after t
                if j >= len(tdates) or not np.isfinite(topens[j]) or topens[j] <= 0:
                    continue
                scheduled.setdefault(tdates[j], []).append(
                    {"ticker": ticker, "sector": sector, "open": float(topens[j])})
                pending.add(ticker)
                held.add(ticker)
                sec[sector] = sec.get(sector, 0) + 1
                n_book += 1

        curve.append({"date": t, "equity": equity, "cash": cash, "open": len(open_pos)})

    # --- 4. close anything still open at the last price ---
    last = dates[-1]
    for p in open_pos:
        px = _asof(panel[p["ticker"]][:2], last)
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

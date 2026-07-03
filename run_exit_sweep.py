"""
--------------------------------------------
EXIT-GEOMETRY SWEEP  (handoff §2 + addendum §B, on honest next-open entries)
--------------------------------------------

The panel's exit study ran on close entries; F1 moved entries to the next
session's open, so every candidate re-runs here under honest timing before any
geometry is adopted. Candidates:

  * 3:1 (+3% / -1%)          - current config
  * symmetric +/-3%          - panel's provisional winner (§2)
  * symmetric +/-2%          - control (tight exits die to costs/noise)
  * plain 10d hold, -15% cat - addendum A2: exits should be catastrophe-only
                               (WARNING: the most survivorship-inflated variant)

Same book everywhere: strategy.select_cohort picks, XGB-only ranking, 10 bps
costs, bear = cash. Decision rule (§2/§B): best Sharpe subject to
total >= current AND maxDD <= 1.2x current. Survivor universe - relative
comparisons meaningful, absolute levels are not promises.

Run:  python run_exit_sweep.py
"""

import sqlite3
import warnings

import pandas as pd

import backtest
import config
import features
import threshold_analysis as ta

warnings.filterwarnings("ignore")

RULES = {
    "3:1 (+3/-1) current": (0.03, -0.01),
    "sym +/-3%":           (0.03, -0.03),
    "sym +/-2%":           (0.02, -0.02),
    "hold10, cat -15%":    (None, -0.15),
}
BASELINE = "3:1 (+3/-1) current"


def load_signals_multi(conn) -> pd.DataFrame:
    # one slow feature pass; realised next-open returns for EVERY rule at once
    oof = ta.load_blended_oof(conn)[["date", "ticker", "Target_Label", "p_xgb", "regime"]]
    sectors = dict(conn.execute("SELECT ticker, sector FROM stocks").fetchall())
    baselines = features.load_baselines(conn)

    frames = []
    tickers = sorted(oof["ticker"].unique())
    for n, t in enumerate(tickers, 1):
        feat = features.compute_features(features.load_stock(t, conn), baselines)
        if feat.empty:
            continue
        feat = feat[["date", "open", "close", "NATR_14"]].copy()
        o = feat["open"].to_numpy(dtype=float)
        c = feat["close"].to_numpy(dtype=float)
        for name, (tp, sl) in RULES.items():
            feat[f"r::{name}"] = backtest.realized_return_series(o, c, tp, sl)
        feat["ticker"] = t
        frames.append(feat)
        if n % 100 == 0:
            print(f"  features {n}/{len(tickers)}")

    df = oof.merge(pd.concat(frames, ignore_index=True), on=["date", "ticker"], how="inner")
    df["sector"] = df["ticker"].map(sectors).fillna("Unknown")
    rcols = [f"r::{name}" for name in RULES]
    return df.dropna(subset=rcols + ["NATR_14"]).sort_values("date").reset_index(drop=True)


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    print("\n" + "=" * 78)
    print("  EXIT-GEOMETRY SWEEP  (next-open entries, XGB-only, 10 bps, bear = cash)")
    print("=" * 78 + "\n")

    df = load_signals_multi(conn)
    rebal_dates = sorted(df["date"].unique())[::backtest.REBALANCE]
    spy = backtest.spy_stats(conn, rebal_dates)
    conn.close()

    rows = {}
    for name in RULES:
        sub = df.copy()
        sub["realized"] = sub[f"r::{name}"]
        eq, tr = backtest.simulate(sub, signal_col="p_xgb")
        s = backtest.summarize(eq, tr)
        s["pos_rate"] = float((tr["return"] > 0).mean())        # net-positive rate (the honest win%)
        rows[name] = s

    base = rows[BASELINE]
    print(f"  Period {base['start']} -> {base['end']}   |   trades/rule ~{base['trades']:,}\n")
    hdr = (f"  {'exit rule':<22}{'pos%':>7}{'net/tr':>9}{'total':>9}{'CAGR':>8}"
           f"{'maxDD':>8}{'Sharpe':>8}{'  verdict'}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    best, best_s = None, None
    for name, s in rows.items():
        ok = (s["total_return"] >= base["total_return"]
              and s["max_drawdown"] >= 1.2 * base["max_drawdown"])   # DD is negative
        verdict = "eligible" if ok else "fails gate"
        if name == BASELINE:
            verdict = "baseline"
        if ok and (best_s is None or s["sharpe"] > best_s["sharpe"]):
            best, best_s = name, s
        print(f"  {name:<22}{s['pos_rate']:>6.1%}{s['avg_return']:>+9.2%}"
              f"{s['total_return']:>+9.0%}{s['cagr']:>+8.1%}{s['max_drawdown']:>8.1%}"
              f"{s['sharpe']:>8.2f}  {verdict}")
    if spy:
        print(f"  {'SPY buy&hold':<22}{'-':>7}{'-':>9}{spy['total_return']:>+9.0%}"
              f"{'-':>8}{spy['max_drawdown']:>8.1%}{spy['sharpe']:>8.2f}")

    print(f"\n  Decision rule winner: {best}   "
          f"(best Sharpe with total >= baseline and DD <= 1.2x baseline)")
    print("  Reminder: hold10/cat-15% is the most survivorship-inflated variant (A2);")
    print("  adoption still requires the U3 label retrain + the forward paper record.")


if __name__ == "__main__":
    main()

"""
--------------------------------------------
U1 (impact) - DOES THE EARNINGS BLACKOUT PAY?
--------------------------------------------

Replays the adopted book (next-open, +/-3% exits, 10 bps, bear = cash) twice:
with and without the earnings blackout (skip candidates reporting within
EARNINGS_BLACKOUT sessions). The filter's job is the LOSS TAIL - earnings gaps
through the stop - so the acceptance metrics are p5 of trade returns and the
positive rate, not just the total.

Run:  python run_earnings_impact.py      (needs the earnings_dates backfill)
"""

import sqlite3
import warnings

import numpy as np

import backtest
import config
import earnings

warnings.filterwarnings("ignore")


def stats(eq, tr):
    s = backtest.summarize(eq, tr)
    r = tr["return"]
    return {"total": s["total_return"], "sharpe": s["sharpe"], "maxdd": s["max_drawdown"],
            "pos": float((r > 0).mean()), "net": float(r.mean()),
            "p5": float(np.percentile(r, 5)), "worst": float(r.min()), "trades": len(tr)}


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    print("\n" + "=" * 70)
    print(f"  U1 IMPACT - earnings blackout ({config.EARNINGS_BLACKOUT} sessions)")
    print("=" * 70)

    df, _ = backtest.load_signals(conn)
    mask = earnings.blackout_mask(df, conn)
    print(f"\n  candidate rows in blackout: {mask.mean():.1%} of {len(df):,}")

    base = stats(*backtest.simulate(df, signal_col="p_xgb"))
    filt = stats(*backtest.simulate(df[~mask], signal_col="p_xgb"))
    conn.close()

    hdr = (f"  {'book':<22}{'trades':>7}{'pos%':>7}{'net/tr':>9}{'p5':>8}{'worst':>8}"
           f"{'total':>8}{'maxDD':>8}{'Sharpe':>8}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, s in (("no filter (current)", base), ("earnings blackout", filt)):
        print(f"  {name:<22}{s['trades']:>7,}{s['pos']:>6.1%}{s['net']:>+9.2%}"
              f"{s['p5']:>+8.1%}{s['worst']:>+8.1%}{s['total']:>+8.0%}"
              f"{s['maxdd']:>8.1%}{s['sharpe']:>8.2f}")

    d_p5 = (filt["p5"] - base["p5"]) * 100
    d_pos = (filt["pos"] - base["pos"]) * 100
    print(f"\n  delta: pos-rate {d_pos:+.1f}pp | p5 tail {d_p5:+.2f}pp | "
          f"Sharpe {filt['sharpe'] - base['sharpe']:+.2f} | total {(filt['total'] - base['total']) * 100:+.0f}pp")
    print("  (point-in-time note: historical dates are as-known-today - a proxy, flagged in FINDINGS)")


if __name__ == "__main__":
    main()

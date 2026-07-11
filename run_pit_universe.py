"""
--------------------------------------------
U6 - POINT-IN-TIME UNIVERSE  (survivorship attack)
--------------------------------------------

Our tradeable universe is TODAY's 503 S&P 500 survivors, so the backtest has two
survivorship biases:
  (A) ADDITION look-ahead - trading a name in 2014 that only JOINED the index in
      2021 (we "knew" it would become a quality S&P name). FIXABLE here.
  (B) DELISTING survival - names that were in the index but got dropped/delisted
      are absent from our 503 entirely (no prices). UNFIXABLE without their data;
      we can only QUANTIFY the gap.

Uses fja05680/sp500 point-in-time membership (1996-2026, verified source). Builds
index_membership, measures coverage (what % of the historical index we can even
see), and re-runs the backtest trading only members-as-of-each-date.

Run:  python run_pit_universe.py
"""

import bisect
import io
import sqlite3
import warnings

import pandas as pd

import backtest
import config

warnings.filterwarnings("ignore")

# point-in-time S&P 500 membership, 1996-present (verified source)
HIST_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
            "S%26P%20500%20Historical%20Components%20%26%20Changes(08-17-2024).csv")
HIST_CSV = config.HERE / "sp500_membership.csv"           # cached (gitignored)

MEMBER_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_membership (
    ticker TEXT PRIMARY KEY, added TEXT, removed TEXT
);
"""


def norm(t: str) -> str:
    return t.replace(".", "-").strip().upper()


def load_snapshots():
    if not HIST_CSV.exists():
        import requests
        print("  downloading point-in-time membership (fja05680/sp500) ...")
        HIST_CSV.write_bytes(requests.get(HIST_URL, timeout=60).content)
    df = pd.read_csv(HIST_CSV)
    snaps = [(row["date"], frozenset(norm(t) for t in str(row["tickers"]).split(",")))
             for _, row in df.iterrows()]
    snaps.sort(key=lambda x: x[0])
    return [s[0] for s in snaps], [s[1] for s in snaps]


def member_asof(dates, sets, d):
    i = bisect.bisect_right(dates, d) - 1
    return sets[i] if i >= 0 else frozenset()


def store_membership(conn, dates, sets) -> None:
    conn.executescript(MEMBER_SCHEMA)
    first, last, final = {}, {}, sets[-1]
    for d, s in zip(dates, sets):
        for t in s:
            first.setdefault(t, d)
            last[t] = d
    for t, added in first.items():
        removed = None if t in final else last[t]
        conn.execute("INSERT OR REPLACE INTO index_membership VALUES (?,?,?)", (t, added, removed))
    conn.commit()


def _summ(df, tag):
    eq, tr = backtest.simulate(df, signal_col="p_xgb")
    s = backtest.summarize(eq, tr)
    return {"tag": tag, "total": s["total_return"], "cagr": s["cagr"],
            "maxdd": s["max_drawdown"], "sharpe": s["sharpe"], "trades": s["trades"],
            "pos": float((tr["return"] > 0).mean())}


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    print("\n" + "=" * 74)
    print("  U6 - POINT-IN-TIME UNIVERSE  (survivorship attack)")
    print("=" * 74)

    dates, sets = load_snapshots()
    store_membership(conn, dates, sets)

    df, _ = backtest.load_signals(conn)
    lo, hi = df["date"].min(), df["date"].max()

    # (B) coverage: every name in the index during our window vs what we hold prices for
    hist = set()
    for d, s in zip(dates, sets):
        if lo <= d <= hi:
            hist |= s
    hist |= member_asof(dates, sets, lo)                  # membership at the very start
    ours = {norm(t) for (t,) in conn.execute("SELECT ticker FROM stocks").fetchall()}
    covered = hist & ours
    missing = hist - ours
    print(f"\n  Backtest window {lo} -> {hi}")
    print(f"  Names ever in the index this window : {len(hist)}")
    print(f"  ...for which we hold prices          : {len(covered)}  "
          f"({len(covered) / len(hist):.0%} coverage)")
    print(f"  ...MISSING (dropped/delisted, no px) : {len(missing)}  "
          f"<- unfixable survivorship (bias B)")

    # (A) point-in-time filter: trade only members-as-of-date
    udates = sorted(df["date"].unique())
    memb = {d: member_asof(dates, sets, d) for d in udates}
    keep = [norm(t) in memb[d] for t, d in zip(df["ticker"], df["date"])]
    df_pit = df[pd.Series(keep, index=df.index)]
    dropped = 1 - len(df_pit) / len(df)

    base = _summ(df, "survivor universe (current)")
    pit = _summ(df_pit, "point-in-time members only")
    conn.close()

    print(f"\n  (A) addition look-ahead removed: {dropped:.1%} of candidate rows were "
          "not-yet-members\n")
    hdr = f"  {'universe':<32}{'trades':>8}{'pos%':>7}{'total':>9}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in (base, pit):
        print(f"  {r['tag']:<32}{r['trades']:>8,}{r['pos']:>6.1%}{r['total']:>+9.0%}"
              f"{r['cagr']:>+8.1%}{r['maxdd']:>8.1%}{r['sharpe']:>8.2f}")
    print(f"\n  (A) is the fixable part; (B) - the {len(missing)} missing delisted names, mostly")
    print("  losers - stays inflated. Point-in-time is a PARTIAL fix; the paper record is clean.")


if __name__ == "__main__":
    main()

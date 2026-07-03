"""
--------------------------------------------
EARNINGS BLACKOUT  (edge upgrade U1)
--------------------------------------------

A stop-based 10-day hold routinely sits through earnings; the gap through the
stop is unrankable from technical features and fattens the loss tail. This
module collects earnings dates (yfinance, past + upcoming) into the
earnings_dates table and answers one question for the strategy layer:
"does this candidate report within the next EARNINGS_BLACKOUT sessions?"

Point-in-time note: yfinance gives announcement dates as known TODAY. For the
live filter that is exactly right (upcoming dates are known in advance). For
backtests it is a reasonable proxy, flagged as an approximation in FINDINGS.

Run directly to backfill/refresh:  python earnings.py
"""

import sqlite3
import time
from datetime import date, timedelta

import pandas as pd

import config

EARNINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS earnings_dates (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    UNIQUE(ticker, date)
);
"""
FETCH_LIMIT = 60          # quarters requested per ticker (~15 years; Yahoo caps ~100)
RATE_SLEEP = 0.4          # polite gap between yfinance calls


def ensure_schema(conn) -> None:
    conn.executescript(EARNINGS_SCHEMA)
    conn.commit()


def fetch_ticker(ticker: str) -> list[str]:
    # ISO dates (past + future) for one ticker; empty list on any failure
    import yfinance as yf
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=FETCH_LIMIT)
        if df is None or df.empty:
            return []
        return sorted({d.date().isoformat() for d in df.index})
    except Exception:
        return []


def store(conn, ticker: str, dates: list[str]) -> int:
    n = 0
    for d in dates:
        n += conn.execute("INSERT OR IGNORE INTO earnings_dates (ticker, date) VALUES (?,?)",
                          (ticker, d)).rowcount
    conn.commit()
    return n


def refresh(conn, tickers: list[str], only_stale: bool = True) -> dict:
    """
    Fetch + store earnings dates. only_stale=True skips tickers that already
    have a KNOWN FUTURE date (nothing new to learn) - makes reruns incremental
    and the daily hot path cheap.
    """
    ensure_schema(conn)
    today = date.today().isoformat()
    stale = []
    for t in tickers:
        row = conn.execute("SELECT MAX(date) FROM earnings_dates WHERE ticker=?", (t,)).fetchone()
        if not only_stale or row[0] is None or row[0] < today:
            stale.append(t)

    added, failed = 0, 0
    for i, t in enumerate(stale, 1):
        dates = fetch_ticker(t)
        if dates:
            added += store(conn, t, dates)
        else:
            failed += 1
        time.sleep(RATE_SLEEP)
        if i % 25 == 0:
            print(f"  earnings {i}/{len(stale)}  (+{added} rows, {failed} empty)")
    return {"checked": len(stale), "added": added, "empty": failed}


def _next_sessions(pick_date: str, n: int) -> str:
    # end of the blackout window: n WEEKDAYS after pick_date (holiday-agnostic,
    # deliberately slightly conservative)
    d, added = date.fromisoformat(pick_date), 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d.isoformat()


def blackout_tickers(conn, pick_date: str,
                     window: int = None) -> set:
    # tickers reporting within (pick_date, pick_date + window sessions]
    window = config.EARNINGS_BLACKOUT if window is None else window
    end = _next_sessions(pick_date, window)
    rows = conn.execute("SELECT DISTINCT ticker FROM earnings_dates WHERE date > ? AND date <= ?",
                        (pick_date, end)).fetchall()
    return {r[0] for r in rows}


def blackout_mask(df: pd.DataFrame, conn, window: int = None) -> pd.Series:
    # vectorised backtest helper: True where (ticker, date) sits inside a
    # blackout window - i.e. an earnings date falls in (date, date + window]
    window = config.EARNINGS_BLACKOUT if window is None else window
    earn = pd.read_sql_query("SELECT ticker, date FROM earnings_dates", conn)
    if earn.empty:
        return pd.Series(False, index=df.index)
    ends = {d: _next_sessions(d, window) for d in df["date"].unique()}
    edates = earn.groupby("ticker")["date"].apply(sorted).to_dict()

    import bisect
    def hit(t, d):
        ds = edates.get(t)
        if not ds:
            return False
        i = bisect.bisect_right(ds, d)
        return i < len(ds) and ds[i] <= ends[d]

    return pd.Series([hit(t, d) for t, d in zip(df["ticker"], df["date"])], index=df.index)


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    tickers = [r[0] for r in conn.execute(
        "SELECT ticker FROM stocks WHERE ml_ready=1 ORDER BY ticker").fetchall()]
    print(f"\nEarnings backfill : {len(tickers)} tickers (incremental)\n")
    stats = refresh(conn, tickers)
    total = conn.execute("SELECT COUNT(*) FROM earnings_dates").fetchone()[0]
    span = conn.execute("SELECT MIN(date), MAX(date) FROM earnings_dates").fetchone()
    conn.close()
    print(f"\n  checked {stats['checked']}  added {stats['added']}  empty {stats['empty']}")
    print(f"  table now {total:,} rows  spanning {span[0]} -> {span[1]}")


if __name__ == "__main__":
    main()

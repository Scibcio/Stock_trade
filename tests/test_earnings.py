"""
Gate tests for the earnings blackout (U1) - window logic and schema, no network.
"""

import sqlite3

import pandas as pd

import earnings


def _db():
    conn = sqlite3.connect(":memory:")
    earnings.ensure_schema(conn)
    return conn


def test_next_sessions_skips_weekends():
    # Friday + 5 sessions = next Friday (weekend never counts)
    assert earnings._next_sessions("2024-01-05", 5) == "2024-01-12"


def test_blackout_window_is_exclusive_start_inclusive_end():
    conn = _db()
    conn.executemany("INSERT INTO earnings_dates VALUES (?,?)", [
        ("SAME", "2024-01-08"),   # earnings ON pick date -> already public, not blocked
        ("SOON", "2024-01-10"),   # 2 sessions ahead -> blocked
        ("EDGE", "2024-01-15"),   # exactly 5 sessions (Mon) -> blocked (inclusive)
        ("FAR",  "2024-01-16"),   # 6 sessions -> safe
    ])
    blocked = earnings.blackout_tickers(conn, "2024-01-08", window=5)
    assert blocked == {"SOON", "EDGE"}
    conn.close()


def test_blackout_mask_matches_per_row():
    conn = _db()
    conn.execute("INSERT INTO earnings_dates VALUES ('AAA', '2024-01-10')")
    df = pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"],
                       "date": ["2024-01-08", "2024-01-11", "2024-01-08"]})
    mask = earnings.blackout_mask(df, conn, window=5)
    assert mask.tolist() == [True, False, False]   # inside window / after it / no earnings
    conn.close()


def test_store_is_idempotent():
    conn = _db()
    assert earnings.store(conn, "AAA", ["2024-01-10", "2024-04-10"]) == 2
    assert earnings.store(conn, "AAA", ["2024-01-10", "2024-04-10"]) == 0
    conn.close()

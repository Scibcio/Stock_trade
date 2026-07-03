"""
Gate test for the forward paper-trade scorer (predict_live.score_paper_trades).
Exit barriers are pinned so the test checks MECHANICS, independent of whatever
geometry config.EXIT_* currently adopts.
"""

import sqlite3

import config
import predict_live


def _add_trade(conn, ticker, entry, closes):
    conn.execute("INSERT INTO paper_trades (pick_date, ticker, entry_close, prob, regime, natr) "
                 "VALUES ('2020-01-01', ?, ?, 0.7, 'bull', 5)", (ticker, entry))
    for i, px in enumerate(closes, 1):
        conn.execute("INSERT INTO daily_prices (ticker, date, close) VALUES (?, ?, ?)",
                     (ticker, f"2020-01-{i + 1:02d}", px))


def test_scorer_marks_win_loss_and_leaves_open(monkeypatch):
    monkeypatch.setattr(config, "EXIT_TAKE_PROFIT", 0.03)
    monkeypatch.setattr(config, "EXIT_STOP_LOSS", -0.01)
    conn = sqlite3.connect(":memory:")
    conn.executescript(predict_live.PAPER_SCHEMA)
    conn.execute("CREATE TABLE daily_prices (ticker TEXT, date TEXT, close REAL)")

    _add_trade(conn, "WIN", 100, [100.5, 101, 103.5, 100, 100, 100, 100, 100, 100, 100])  # +3% hit
    _add_trade(conn, "LOSS", 100, [98.5, 105, 105, 105, 105, 105, 105, 105, 105, 105])     # -1% first
    _add_trade(conn, "OPEN", 100, [100, 100, 100])                                          # < 10 days
    conn.commit()

    predict_live.score_paper_trades(conn)
    st = dict(conn.execute("SELECT ticker, status FROM paper_trades").fetchall())
    assert st["WIN"] == "win"
    assert st["LOSS"] == "loss"
    assert st["OPEN"] == "open"
    conn.close()

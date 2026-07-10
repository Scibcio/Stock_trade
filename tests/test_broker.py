"""
Gate tests for the Alpaca paper broker (Phase 3, A7) - every safety rail,
fully mocked, zero network.
"""

import sqlite3

import pytest

import broker_alpaca as broker
import config
import paper_trader
import predict_live


def _env(monkeypatch, key="PK_TEST_KEY", secret="s3cret", paper="true"):
    monkeypatch.setattr(broker, "_load_env",
                        lambda: {"key": key, "secret": secret, "paper": paper})


# ----------------------------------
# A1/A4 - the paper-only hard-fail
# ----------------------------------

def test_refuses_without_keys(monkeypatch):
    _env(monkeypatch, key="", secret="")
    with pytest.raises(broker.BrokerSafetyError, match="no API keys"):
        broker.get_client()


def test_refuses_non_paper_flag(monkeypatch):
    _env(monkeypatch, paper="false")
    with pytest.raises(broker.BrokerSafetyError, match="APCA_PAPER"):
        broker.get_client()


def test_refuses_live_looking_key(monkeypatch):
    _env(monkeypatch, key="AKLIVEKEY123")            # live keys start with AK
    with pytest.raises(broker.BrokerSafetyError, match="PAPER key"):
        broker.get_client()


def test_endpoint_check_reads_enum_value():
    # alpaca-py's _base_url is an Enum whose str() is the NAME, not the URL -
    # the check must read .value (this refused a real paper account once)
    class _Enum:
        value = "https://paper-api.alpaca.markets"
        def __str__(self):
            return "BaseURL.TRADING_PAPER"

    class _Client:
        _base_url = _Enum()

    assert broker._paper_endpoint_ok(_Client())

    class _LiveEnum:
        value = "https://api.alpaca.markets"

    class _LiveClient:
        _base_url = _LiveEnum()

    assert not broker._paper_endpoint_ok(_LiveClient())


# ----------------------------------
# A4 - HALT kill switch + notional caps
# ----------------------------------

class _FakeClient:
    def __init__(self):
        self.orders = []

    def submit_order(self, req):
        self.orders.append(req)
        return req


def test_halt_blocks_buys_not_sells(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "HALT_FILE", tmp_path / "HALT")
    config.HALT_FILE.touch()
    c = _FakeClient()
    assert broker.submit_notional(c, "AAPL", 100, "buy", "x") is None     # blocked
    assert broker.submit_notional(c, "SPY", 100, "sell", "y") is not None  # de-risking allowed
    assert len(c.orders) == 1


def test_notional_caps(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "HALT_FILE", tmp_path / "HALT")           # absent = not halted
    c = _FakeClient()
    assert broker.submit_notional(c, "AAPL", 0.5, "buy", "x") is None     # below $1 minimum
    with pytest.raises(broker.BrokerSafetyError, match="MAX_ORDER_NOTIONAL"):
        broker.submit_notional(c, "AAPL", config.MAX_ORDER_NOTIONAL + 1, "buy", "x")
    order = broker.submit_notional(c, "AAPL", 150, "buy", "cohort|AAPL")
    assert order.notional == 150 and str(order.side).lower().find("buy") >= 0


# ----------------------------------
# A2/U9 - sleeve sizing
# ----------------------------------

def test_satellite_sleeve_budget(monkeypatch):
    monkeypatch.setattr(broker, "account_equity", lambda c: 100_000.0)    # $100k paper cash
    monkeypatch.setattr(config, "PORTFOLIO_MODE", "satellite")
    monkeypatch.setattr(config, "SATELLITE_WEIGHT", 0.25)
    # capped at PAPER_EQUITY_CAP first, then the satellite share of it
    assert paper_trader.sleeve_budget(object()) == config.PAPER_EQUITY_CAP * 0.25
    monkeypatch.setattr(config, "PORTFOLIO_MODE", "pure")
    assert paper_trader.sleeve_budget(object()) == config.PAPER_EQUITY_CAP


# ----------------------------------
# A5 - reconcile writes REAL fills into the record
# ----------------------------------

def test_duplicate_order_id_is_noop(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "HALT_FILE", tmp_path / "HALT")

    class _DupClient:
        def submit_order(self, req):
            raise RuntimeError("client_order_id must be unique")

    assert broker.submit_notional(_DupClient(), "AAPL", 100, "buy", "x") == broker.DUPLICATE


def test_submit_cohort_marks_and_isolates(monkeypatch):
    # the marker is submitted_at (never entry_open) and one bad symbol
    # doesn't block the rest - the two bugs the adversarial review caught
    conn = sqlite3.connect(":memory:")
    predict_live.migrate(conn)
    conn.execute("CREATE TABLE daily_prices (ticker TEXT, date TEXT, open REAL, close REAL)")
    for t, status in (("GOOD", "pending"), ("BAD", "pending"), ("BOOT", "open")):
        conn.execute("INSERT INTO paper_trades (pick_date, ticker, status, cohort_id, weight, "
                     "entry_open) VALUES ('2026-07-02', ?, ?, '2026-07-02', 0.06, ?)",
                     (t, status, 100.0 if status == "open" else None))
    monkeypatch.setattr(paper_trader, "sleeve_budget", lambda c: 2500.0)

    def fake_entry(client, symbol, notional, order_id):
        if symbol == "BAD":
            raise RuntimeError("asset not tradable")
        return {"ok": symbol}
    monkeypatch.setattr(paper_trader.broker, "submit_entry", fake_entry)
    monkeypatch.setattr(paper_trader.broker, "halted", lambda: False)

    assert paper_trader._submit_cohort(conn, object()) == 2      # GOOD + BOOT (bootstrap)
    marks = dict(conn.execute("SELECT ticker, submitted_at IS NOT NULL "
                              "FROM paper_trades").fetchall())
    assert marks == {"GOOD": 1, "BOOT": 1, "BAD": 0}             # BAD retries tomorrow
    conn.close()


def test_stale_pending_expires():
    conn = sqlite3.connect(":memory:")
    predict_live.migrate(conn)
    conn.execute("CREATE TABLE daily_prices (ticker TEXT, date TEXT, open REAL, close REAL)")
    conn.execute("INSERT INTO paper_trades (pick_date, ticker, status) "
                 "VALUES ('2026-01-01', 'GHOST', 'pending')")
    for i in range(2, 15):                                       # 13 sessions pass, no fill
        conn.execute("INSERT INTO daily_prices VALUES ('AAA', ?, 1, 1)", (f"2026-01-{i:02d}",))
    assert predict_live.expire_stale_pending(conn) == 1
    assert conn.execute("SELECT status FROM paper_trades").fetchone()[0] == "dead"
    conn.close()


def test_reconcile_applies_fill_price(monkeypatch):
    conn = sqlite3.connect(":memory:")
    predict_live.migrate(conn)
    conn.execute("INSERT INTO paper_trades (pick_date, ticker, entry_close, status, cohort_id) "
                 "VALUES ('2026-07-03', 'AAPL', 200.0, 'pending', '2026-07-03')")
    monkeypatch.setattr(broker, "is_configured", lambda: True)
    monkeypatch.setattr(broker, "get_client", lambda: object())
    monkeypatch.setattr(broker, "dead_order_ids", lambda c: set())
    monkeypatch.setattr(broker, "filled_orders", lambda c: [    # date-scoped id, prefix-matched
        {"order_id": "2026-07-03|AAPL|20260706", "symbol": "AAPL", "side": "OrderSide.BUY",
         "filled_avg_price": 201.5, "filled_at": "2026-07-06T13:30:00Z"}])
    assert paper_trader.reconcile(conn) == 1
    row = conn.execute("SELECT entry_open, status FROM paper_trades").fetchone()
    assert row == (201.5, "open")                                # the FILL is the truth
    slip = conn.execute("SELECT slippage_bps FROM alpaca_fills WHERE side='buy'").fetchone()[0]
    assert abs(slip - 75.0) < 1e-6                               # (201.5/200 - 1) = +75 bps
    conn.close()


def test_reconcile_heals_canceled_order(monkeypatch):
    # F-A: a submitted pick canceled with no fill must be re-queued (submitted_at
    # cleared), not orphaned - and cohort-1 legacy 'open' rows are NOT touched
    conn = sqlite3.connect(":memory:")
    predict_live.migrate(conn)
    conn.execute("INSERT INTO paper_trades (pick_date, ticker, status, cohort_id, submitted_at) "
                 "VALUES ('2026-07-03', 'MU', 'pending', '2026-07-03', '2026-07-04T22:00:00')")
    conn.execute("INSERT INTO paper_trades (pick_date, ticker, status, cohort_id, submitted_at) "
                 "VALUES ('2026-07-02', 'LEG', 'open', '2026-07-02', '2026-07-02T22:00:00')")
    monkeypatch.setattr(broker, "is_configured", lambda: True)
    monkeypatch.setattr(broker, "get_client", lambda: object())
    monkeypatch.setattr(broker, "filled_orders", lambda c: [])
    monkeypatch.setattr(broker, "dead_order_ids", lambda c: {"2026-07-03|MU|20260704"})
    paper_trader.reconcile(conn)
    marks = dict(conn.execute("SELECT ticker, submitted_at IS NULL FROM paper_trades").fetchall())
    assert marks == {"MU": 1, "LEG": 0}                          # MU re-queued, legacy untouched
    conn.close()


def test_reconcile_exit_slippage(monkeypatch):
    conn = sqlite3.connect(":memory:")
    predict_live.migrate(conn)
    conn.execute("INSERT INTO paper_trades (pick_date, ticker, status, exit_date, exit_price) "
                 "VALUES ('2026-06-01', 'MU', 'win', '2026-06-15', 100.0)")
    monkeypatch.setattr(broker, "is_configured", lambda: True)
    monkeypatch.setattr(broker, "get_client", lambda: object())
    monkeypatch.setattr(broker, "dead_order_ids", lambda c: set())
    monkeypatch.setattr(broker, "filled_orders", lambda c: [
        {"order_id": "broker-uuid-xyz", "symbol": "MU", "side": "OrderSide.SELL",
         "filled_avg_price": 99.0, "filled_at": "2026-06-16T13:31:00Z"}])
    paper_trader.reconcile(conn)
    row = conn.execute("SELECT side, slippage_bps FROM alpaca_fills WHERE symbol='MU'").fetchone()
    assert row[0] == "sell" and abs(row[1] - (-100.0)) < 1e-6    # (99/100 - 1) = -100 bps exit slip
    conn.close()


def test_divergence_flags_missing_and_foreign():
    conn = sqlite3.connect(":memory:")
    predict_live.migrate(conn)
    conn.executescript(paper_trader.ALPACA_SCHEMA)
    # AAA: real fill (entry_open set) -> expected at broker; LEG: legacy (NULL) -> not expected
    conn.execute("INSERT INTO paper_trades (pick_date, ticker, status, entry_open) "
                 "VALUES ('2026-07-06', 'AAA', 'open', 150.0)")
    conn.execute("INSERT INTO paper_trades (pick_date, ticker, status, entry_open) "
                 "VALUES ('2026-07-02', 'LEG', 'open', NULL)")
    positions = {"SPY": {}, "ZZZ": {}}                           # AAA missing, ZZZ foreign
    d = paper_trader._divergence(conn, positions)
    assert d == {"missing": ["AAA"], "foreign": ["ZZZ"]}         # LEG not flagged
    conn.close()

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

def test_reconcile_applies_fill_price(monkeypatch):
    conn = sqlite3.connect(":memory:")
    predict_live.migrate(conn)
    conn.execute("INSERT INTO paper_trades (pick_date, ticker, entry_close, status, cohort_id) "
                 "VALUES ('2026-07-03', 'AAPL', 200.0, 'pending', '2026-07-03')")
    monkeypatch.setattr(broker, "is_configured", lambda: True)
    monkeypatch.setattr(broker, "get_client", lambda: object())
    monkeypatch.setattr(broker, "filled_orders", lambda c: [
        {"order_id": "2026-07-03|AAPL", "symbol": "AAPL", "side": "OrderSide.BUY",
         "filled_avg_price": 201.5, "filled_at": "2026-07-06T13:30:00Z"}])
    assert paper_trader.reconcile(conn) == 1
    row = conn.execute("SELECT entry_open, status FROM paper_trades").fetchone()
    assert row == (201.5, "open")                                # the FILL is the truth
    slip = conn.execute("SELECT slippage_bps FROM alpaca_fills").fetchone()[0]
    assert abs(slip - 75.0) < 1e-6                               # (201.5/200 - 1) = +75 bps
    conn.close()

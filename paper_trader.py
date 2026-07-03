"""
--------------------------------------------
PAPER TRADER  (Phase 3, A3/A5 + U9 core-satellite)
--------------------------------------------

Turns the nightly picks into real DEMO orders on Alpaca paper and feeds the
truth (actual fills) back into the record. Two hooks, called by run_daily:

  reconcile(conn)  BEFORE fill_pending: match last night's fills to pending
                   picks -> entry_open = the REAL fill price (A5: fills are
                   truth; the yfinance open is only the fallback), log slippage.
  trade(conn)      AFTER scoring + new picks:
                     1. mirror exits - close any Alpaca position whose
                        paper_trades row has been scored (win/loss)
                     2. U9 core: keep (1-w) x cap in SPY, rebalance on >2% drift
                     3. submit tonight's pending picks as notional market DAY
                        orders (queued -> fill at tomorrow's open)
                     4. snapshot equity/positions to alpaca_state for the
                        dashboard (which stays DB-read-only - no API calls)

Does NOTHING unless .env holds paper keys (broker_alpaca hard-fails on any
non-paper configuration). The HALT file blocks new entries, never exits.
"""

import sqlite3
from datetime import datetime

import broker_alpaca as broker
import config

ALPACA_SCHEMA = """
CREATE TABLE IF NOT EXISTS alpaca_fills (
    order_id  TEXT PRIMARY KEY,
    symbol    TEXT,
    side      TEXT,
    price     REAL,
    filled_at TEXT,
    slippage_bps REAL
);
CREATE TABLE IF NOT EXISTS alpaca_state (
    ts        TEXT PRIMARY KEY,
    equity    REAL,
    spy_value REAL,
    n_sleeve  INTEGER
);
"""

SPY_DRIFT = 0.02                     # rebalance the core when off target by >2%


def _order_id(cohort_id: str, ticker: str) -> str:
    return f"{cohort_id}|{ticker}"[:48]           # deterministic -> idempotent submits


def _cap_equity(client) -> float:
    return min(config.PAPER_EQUITY_CAP, broker.account_equity(client))


def sleeve_budget(client) -> float:
    # the strategy sleeve's capital under the adopted portfolio mode
    cap = _cap_equity(client)
    return cap * config.SATELLITE_WEIGHT if config.PORTFOLIO_MODE == "satellite" else cap


def reconcile(conn) -> int:
    """Write real fill prices into pending picks; returns fills applied."""
    if not broker.is_configured():
        return 0
    client = broker.get_client()
    conn.executescript(ALPACA_SCHEMA)
    fills = {f["order_id"]: f for f in broker.filled_orders(client)}

    applied = 0
    rows = conn.execute("SELECT id, cohort_id, ticker, entry_close FROM paper_trades "
                        "WHERE status='pending'").fetchall()
    for tid, cohort_id, ticker, entry_close in rows:
        f = fills.get(_order_id(cohort_id or "", ticker))
        if f is None or f["side"].lower().find("buy") < 0:
            continue
        px = f["filled_avg_price"]
        conn.execute("UPDATE paper_trades SET entry_open=?, status='open' WHERE id=?", (px, tid))
        slip = (px / entry_close - 1) * 1e4 if entry_close else None
        conn.execute("INSERT OR IGNORE INTO alpaca_fills VALUES (?,?,?,?,?,?)",
                     (f["order_id"], ticker, "buy", px, f["filled_at"], slip))
        applied += 1
    conn.commit()
    if applied:
        print(f"  alpaca: {applied} fills reconciled (real open prices -> record)")
    return applied


def _mirror_exits(conn, client, positions) -> int:
    # close Alpaca positions whose record row is finished (scored win/loss)
    live = {r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM paper_trades WHERE status IN ('pending','open')").fetchall()}
    closed = 0
    for symbol in positions:
        if symbol == "SPY" or symbol in live:
            continue
        broker.close_symbol(client, symbol)
        closed += 1
    if closed:
        print(f"  alpaca: {closed} finished positions closed (market, fills at next open)")
    return closed


def _rebalance_spy(conn, client, positions) -> None:
    if config.PORTFOLIO_MODE != "satellite":
        return
    target = _cap_equity(client) * (1 - config.SATELLITE_WEIGHT)
    current = positions.get("SPY", {}).get("market_value", 0.0)
    diff = target - current
    if target <= 0 or abs(diff) / target <= SPY_DRIFT:
        return
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if diff > 0:
        broker.submit_notional(client, "SPY", min(diff, config.MAX_ORDER_NOTIONAL),
                               "buy", f"core|SPY|{stamp}")
        print(f"  alpaca: SPY core top-up ${min(diff, config.MAX_ORDER_NOTIONAL):,.0f}")
    else:
        broker.submit_notional(client, "SPY", min(-diff, config.MAX_ORDER_NOTIONAL),
                               "sell", f"core|SPY|{stamp}")
        print(f"  alpaca: SPY core trim ${min(-diff, config.MAX_ORDER_NOTIONAL):,.0f}")


def _submit_cohort(conn, client) -> int:
    budget = sleeve_budget(client)
    submitted = 0
    rows = conn.execute("SELECT cohort_id, ticker, weight FROM paper_trades "
                        "WHERE status='pending' AND entry_open IS NULL").fetchall()
    for cohort_id, ticker, weight in rows:
        if weight is None:
            continue
        notional = round(budget * float(weight), 2)
        try:
            order = broker.submit_entry(client, ticker, notional,
                                        _order_id(cohort_id or "", ticker))
        except Exception as e:                                    # duplicate id = already sent
            if "client_order_id" in str(e).lower():
                continue
            raise
        if order is not None:
            submitted += 1
    if submitted:
        print(f"  alpaca: {submitted} entry orders queued for tomorrow's open "
              f"(sleeve ${budget:,.0f})")
    return submitted


def trade(conn) -> dict:
    """The post-scoring broker pass. Returns a status summary."""
    if not broker.is_configured():
        return {"skipped": "no keys"}
    client = broker.get_client()
    conn.executescript(ALPACA_SCHEMA)
    positions = broker.get_positions(client)

    closed = _mirror_exits(conn, client, positions)
    _rebalance_spy(conn, client, positions)
    submitted = _submit_cohort(conn, client)

    equity = broker.account_equity(client)
    spy_val = positions.get("SPY", {}).get("market_value", 0.0)
    n_sleeve = sum(1 for s in positions if s != "SPY")
    conn.execute("INSERT OR REPLACE INTO alpaca_state VALUES (?,?,?,?)",
                 (datetime.now().isoformat(timespec="seconds"), equity, spy_val, n_sleeve))
    conn.commit()
    print(f"  alpaca: equity ${equity:,.0f} | SPY core ${spy_val:,.0f} | "
          f"{n_sleeve} sleeve positions | halted={broker.halted()}")
    return {"closed": closed, "submitted": submitted, "equity": equity}


if __name__ == "__main__":
    c = sqlite3.connect(config.DB_PATH)
    reconcile(c)
    trade(c)
    c.close()

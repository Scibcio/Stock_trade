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
    from predict_live import migrate
    migrate(conn)                                   # schema guard (adds submitted_at etc.)
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
    # close Alpaca positions whose record row is finished (scored win/loss).
    # Only symbols the record has EVER held are ours to touch; a failure on one
    # symbol never blocks the rest. Known limitation: a ticker re-picked while
    # its old lot is still held keeps both lots until the newer row scores.
    ours = {r[0] for r in conn.execute("SELECT DISTINCT ticker FROM paper_trades").fetchall()}
    live = {r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM paper_trades WHERE status IN ('pending','open')").fetchall()}
    closed = 0
    for symbol in positions:
        if symbol == "SPY" or symbol not in ours or symbol in live:
            continue
        try:
            broker.close_symbol(client, symbol)
            closed += 1
        except Exception as e:
            print(f"  alpaca: close {symbol} failed ({e}) - will retry tomorrow")
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
    side = "buy" if diff > 0 else "sell"
    chunk = min(abs(diff), config.MAX_ORDER_NOTIONAL)
    # one deterministic id per side per day -> same-day re-runs are broker no-ops
    order_id = f"core|SPY|{datetime.now():%Y%m%d}|{side}"
    r = broker.submit_notional(client, "SPY", chunk, side, order_id)
    if r == broker.DUPLICATE:
        return
    if r is not None:
        print(f"  alpaca: SPY core {'top-up' if side == 'buy' else 'trim'} ${chunk:,.0f}")


def _whole_share_fallback(conn, client, cohort_id, ticker, notional):
    # non-fractionable asset: buy floor(notional/price) whole shares; a slot too
    # small for one share is skipped (returns None -> row still marked, no retry loop)
    row = conn.execute("SELECT close FROM daily_prices WHERE ticker=? ORDER BY date DESC LIMIT 1",
                       (ticker,)).fetchone()
    if not row or not row[0]:
        return None
    qty = int(notional // row[0])
    if qty < 1:
        print(f"  alpaca: {ticker} not fractionable and slot ${notional:,.0f} < 1 share "
              f"(${row[0]:,.2f}) - skipped")
        return None
    r = broker.submit_qty_buy(client, ticker, qty, row[0], _order_id(cohort_id or "", ticker))
    if r is not None and r != broker.DUPLICATE:
        print(f"  alpaca: {ticker} not fractionable - bought {qty} whole share(s) instead")
    return r


def _submit_cohort(conn, client) -> int:
    """
    Queue every record position the broker doesn't hold yet. The marker is
    submitted_at (NOT entry_open): a failed night can never orphan a pick,
    and existing 'open' rows bootstrap the broker book on first configure.
    Per-row isolation - one bad symbol never blocks the rest.
    """
    budget = sleeve_budget(client)
    submitted = 0
    rows = conn.execute(
        "SELECT id, cohort_id, ticker, weight FROM paper_trades "
        "WHERE submitted_at IS NULL AND exit_date IS NULL "
        "AND status IN ('pending','open') AND weight IS NOT NULL").fetchall()
    now = datetime.now().isoformat(timespec="seconds")
    for tid, cohort_id, ticker, weight in rows:
        notional = round(min(budget * float(weight), config.MAX_ORDER_NOTIONAL), 2)
        try:
            order = broker.submit_entry(client, ticker, notional,
                                        _order_id(cohort_id or "", ticker))
        except Exception as e:
            if "not fractionable" in str(e).lower():             # whole shares or nothing
                order = _whole_share_fallback(conn, client, cohort_id, ticker, notional)
            else:
                print(f"  alpaca: entry {ticker} failed ({e}) - will retry tomorrow")
                continue
        if order is None and broker.halted():
            continue                                   # blocked by HALT -> retry when lifted
        # sent, already-sent (duplicate), or below-minimum: mark so it never re-queues
        conn.execute("UPDATE paper_trades SET submitted_at=? WHERE id=?", (now, tid))
        if order is not None and order != broker.DUPLICATE:
            submitted += 1
    conn.commit()
    if submitted:
        print(f"  alpaca: {submitted} entry orders queued for the next open "
              f"(sleeve ${budget:,.0f})")
    return submitted


def trade(conn) -> dict:
    """The post-scoring broker pass. Returns a status summary."""
    if not broker.is_configured():
        return {"skipped": "no keys"}
    from predict_live import migrate
    migrate(conn)                                   # schema guard (adds submitted_at etc.)
    client = broker.get_client()
    conn.executescript(ALPACA_SCHEMA)
    positions = broker.get_positions(client)

    closed = submitted = 0                          # step isolation: one failure
    try:                                            # never kills the whole pass
        closed = _mirror_exits(conn, client, positions)
    except Exception as e:
        print(f"  alpaca: mirror-exits failed ({e})")
    try:
        _rebalance_spy(conn, client, positions)
    except Exception as e:
        print(f"  alpaca: SPY rebalance failed ({e})")
    try:
        submitted = _submit_cohort(conn, client)
    except Exception as e:
        print(f"  alpaca: cohort submission failed ({e})")

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

"""
--------------------------------------------
PAPER TRADER  (Phase 3, A3/A5 + U9 core-satellite)
--------------------------------------------

Turns the nightly picks into real DEMO orders on Alpaca paper and feeds the
truth (actual fills) back into the record. Two hooks, called by run_daily:

  reconcile(conn)  BEFORE fill_pending:
                     * entry fills  -> entry_open = REAL fill price, log slippage
                     * exit  fills  -> log exit slippage (round-trip cost, Gate B)
                     * dead orders  -> heal: a submitted pick the broker canceled
                       with no fill is re-queued (F-A fix). Fresh date-scoped
                       order ids make the retry a real order, not a duplicate.
  trade(conn)      AFTER scoring + new picks:
                     1. mirror exits (close scored positions)
                     2. divergence check (record vs broker; catches F-A/HON/foreign)
                     3. U9 core: keep (1-w) x cap in SPY, rebalance on >2% drift
                     4. submit tonight's picks - but only if the next session is
                        near (config.SUBMIT_MAX_HOURS), so DAY orders are never
                        swept across a weekend (F-A)
                     5. snapshot equity/positions to alpaca_state (dashboard reads
                        the DB, never the API)

Legacy note: the pre-Phase-3 07-02 cohort is status='open'/entry_open NULL. All
reconcile + healing here is scoped to status='pending', so that cohort is left
record-only on purpose (it can never feed broker slippage - see FINDINGS_SPY).

Does NOTHING unless .env holds paper keys (broker_alpaca hard-fails on any
non-paper configuration). The HALT file blocks new entries, never exits.
"""

import json
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
CREATE TABLE IF NOT EXISTS alpaca_divergence (
    ts      TEXT PRIMARY KEY,
    missing TEXT,
    foreign_syms TEXT
);
"""

SPY_DRIFT = 0.02                     # rebalance the core when off target by >2%


def _order_id(cohort_id: str, ticker: str, day: str = None) -> str:
    # date-scoped -> same-night re-runs are idempotent (same id = broker no-op),
    # but a retry on a LATER night gets a fresh id and actually submits (F-A fix)
    day = day or datetime.now().strftime("%Y%m%d")
    return f"{cohort_id}|{ticker}|{day}"[:48]


def _prefix(cohort_id: str, ticker: str) -> str:
    return f"{cohort_id}|{ticker}|"


def _cap_equity(client) -> float:
    return min(config.PAPER_EQUITY_CAP, broker.account_equity(client))


def sleeve_budget(client) -> float:
    cap = _cap_equity(client)
    return cap * config.SATELLITE_WEIGHT if config.PORTFOLIO_MODE == "satellite" else cap


# ==================================================
# RECONCILE  (fills = truth; heal dead orders; measure round-trip slippage)
# ==================================================

def _reconcile_exits(conn, sell_fills) -> int:
    # match SELL fills (broker-generated ids) to scored rows by symbol -> exit
    # slippage vs the record's exit close. Idempotent on the broker order id.
    n = 0
    for f in sell_fills:
        if conn.execute("SELECT 1 FROM alpaca_fills WHERE order_id=?", (f["order_id"],)).fetchone():
            continue
        row = conn.execute(
            "SELECT exit_price FROM paper_trades WHERE ticker=? AND status IN ('win','loss') "
            "AND exit_price IS NOT NULL AND (exit_date IS NULL OR exit_date<=?) "
            "ORDER BY exit_date DESC LIMIT 1", (f["symbol"], f["filled_at"][:10])).fetchone()
        slip = (f["filled_avg_price"] / row[0] - 1) * 1e4 if row and row[0] else None
        conn.execute("INSERT OR IGNORE INTO alpaca_fills VALUES (?,?,?,?,?,?)",
                     (f["order_id"], f["symbol"], "sell", f["filled_avg_price"], f["filled_at"], slip))
        n += 1
    return n


def reconcile(conn) -> int:
    """Entry fills -> record; exit fills -> slippage; canceled entries -> re-queued."""
    if not broker.is_configured():
        return 0
    from predict_live import migrate
    migrate(conn)                                   # schema guard (adds submitted_at etc.)
    client = broker.get_client()
    conn.executescript(ALPACA_SCHEMA)

    fills = broker.filled_orders(client)
    buys = [f for f in fills if "buy" in f["side"].lower()]
    sells = [f for f in fills if "sell" in f["side"].lower()]
    dead = broker.dead_order_ids(client)

    applied = healed = 0
    rows = conn.execute("SELECT id, cohort_id, ticker, entry_close, submitted_at "
                        "FROM paper_trades WHERE status='pending'").fetchall()
    for tid, cohort_id, ticker, entry_close, submitted_at in rows:
        pref = _prefix(cohort_id or "", ticker)
        f = next((x for x in buys if x["order_id"].startswith(pref)), None)
        if f is not None:                                       # real entry fill = truth
            px = f["filled_avg_price"]
            conn.execute("UPDATE paper_trades SET entry_open=?, status='open' WHERE id=?", (px, tid))
            slip = (px / entry_close - 1) * 1e4 if entry_close else None
            conn.execute("INSERT OR IGNORE INTO alpaca_fills VALUES (?,?,?,?,?,?)",
                         (f["order_id"], ticker, "buy", px, f["filled_at"], slip))
            applied += 1
        elif submitted_at and any(d.startswith(pref) for d in dead):   # F-A heal
            conn.execute("UPDATE paper_trades SET submitted_at=NULL WHERE id=?", (tid,))
            healed += 1

    _reconcile_exits(conn, sells)
    conn.commit()
    if applied:
        print(f"  alpaca: {applied} entry fills reconciled (real open prices -> record)")
    if healed:
        print(f"  alpaca: {healed} canceled-with-no-fill picks re-queued (fresh order ids)")
    return applied


# ==================================================
# BROKER PASS
# ==================================================

def _mirror_exits(conn, client, positions) -> int:
    # close Alpaca positions whose record row is finished (scored win/loss).
    # Only symbols the record has EVER held are ours to touch; per-symbol
    # isolation. A ticker re-picked while its old lot is open keeps both lots
    # until the newer row scores (known, documented in FINDINGS_SPY).
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


def _divergence(conn, positions) -> dict:
    # record vs broker: 'missing' = a real fill (entry_open set) whose position
    # has vanished from the broker; 'foreign' = a broker position never in the
    # record (manual/hand trades). Cohort-1 legacy rows (entry_open NULL) are
    # NOT flagged - they were never expected at the broker.
    expected = {r[0] for r in conn.execute(
        "SELECT ticker FROM paper_trades WHERE status='open' AND entry_open IS NOT NULL").fetchall()}
    held = set(positions) - {"SPY"}
    record_all = {r[0] for r in conn.execute("SELECT DISTINCT ticker FROM paper_trades").fetchall()}
    missing = sorted(expected - held)
    foreign = sorted(held - record_all)
    conn.execute("INSERT OR REPLACE INTO alpaca_divergence VALUES (?,?,?)",
                 (datetime.now().isoformat(timespec="seconds"), json.dumps(missing), json.dumps(foreign)))
    if missing:
        print(f"  alpaca: [DIVERGENCE] {len(missing)} record positions missing at broker: {missing}")
    if foreign:
        print(f"  alpaca: [DIVERGENCE] {len(foreign)} broker positions not in the record "
              f"(manual trades?): {foreign}")
    return {"missing": missing, "foreign": foreign}


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
    order_id = f"core|SPY|{datetime.now():%Y%m%d}|{side}"       # one id per side per day
    r = broker.submit_notional(client, "SPY", chunk, side, order_id)
    if r not in (None, broker.DUPLICATE):
        print(f"  alpaca: SPY core {'top-up' if side == 'buy' else 'trim'} ${chunk:,.0f}")


def _whole_share_fallback(conn, client, cohort_id, ticker, notional):
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
    if r not in (None, broker.DUPLICATE):
        print(f"  alpaca: {ticker} not fractionable - bought {qty} whole share(s) instead")
    return r


def _submit_cohort(conn, client) -> int:
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
            order = broker.submit_entry(client, ticker, notional, _order_id(cohort_id or "", ticker))
        except Exception as e:
            if "not fractionable" in str(e).lower():
                order = _whole_share_fallback(conn, client, cohort_id, ticker, notional)
            else:
                print(f"  alpaca: entry {ticker} failed ({e}) - will retry tomorrow")
                continue
        if order is None and broker.halted():
            continue                                   # blocked by HALT -> retry when lifted
        conn.execute("UPDATE paper_trades SET submitted_at=? WHERE id=?", (now, tid))
        if order not in (None, broker.DUPLICATE):
            submitted += 1
    conn.commit()
    if submitted:
        print(f"  alpaca: {submitted} entry orders queued for the next open (sleeve ${budget:,.0f})")
    return submitted


def trade(conn) -> dict:
    """The post-scoring broker pass. Returns a status summary."""
    if not broker.is_configured():
        return {"skipped": "no keys"}
    from predict_live import migrate
    migrate(conn)
    client = broker.get_client()
    conn.executescript(ALPACA_SCHEMA)
    positions = broker.get_positions(client)

    closed = submitted = 0
    try:
        closed = _mirror_exits(conn, client, positions)
    except Exception as e:
        print(f"  alpaca: mirror-exits failed ({e})")
    try:
        _divergence(conn, positions)
    except Exception as e:
        print(f"  alpaca: divergence check failed ({e})")

    can_submit = broker.market_opens_within(client, config.SUBMIT_MAX_HOURS)
    if can_submit:
        try:
            _rebalance_spy(conn, client, positions)
        except Exception as e:
            print(f"  alpaca: SPY rebalance failed ({e})")
        try:
            submitted = _submit_cohort(conn, client)
        except Exception as e:
            print(f"  alpaca: cohort submission failed ({e})")
    else:
        print(f"  alpaca: next session >{config.SUBMIT_MAX_HOURS}h away - "
              "deferring entries (avoids weekend cancel)")

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

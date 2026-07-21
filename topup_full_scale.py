"""
ONE-OFF: top the CURRENT open cohort up to full scale for tomorrow's open, and
bring the SPY core to its 10% target. TOP-UP only (buy the difference to target);
no sells, so no wash trades and it fits settled cash without margin.

  target(name) = (cap * SATELLITE_WEIGHT) * weight        # 90% sleeve, inverse-NATR
  target(SPY)  =  cap * (1 - SATELLITE_WEIGHT)             # 10% core
  buy(x)       =  max(0, target - current_market_value)    # notional market DAY -> next open

Safe by construction: preview by default (submits NOTHING). Pass --execute to
fire. Paper-locked broker (broker_alpaca hard-fails on anything non-paper).

  python topup_full_scale.py            # preview
  python topup_full_scale.py --execute  # fire the orders
"""

import sqlite3
import sys
from datetime import datetime

import broker_alpaca as broker
import config

EXECUTE = "--execute" in sys.argv
CASH_BUFFER = 250.0     # leave a little settled cash unspent


def last_close(conn, ticker):
    r = conn.execute("SELECT close FROM daily_prices WHERE ticker=? ORDER BY date DESC LIMIT 1",
                     (ticker,)).fetchone()
    return r[0] if r else None


def main():
    if not broker.is_configured():
        print("  no API keys in .env - aborting")
        return
    client = broker.get_client()
    acct = client.get_account()
    equity = float(acct.equity)
    cash = float(acct.cash)
    positions = broker.get_positions(client)

    cap = min(config.PAPER_EQUITY_CAP, equity)
    sleeve = cap * config.SATELLITE_WEIGHT
    spy_target = cap * (1 - config.SATELLITE_WEIGHT)

    conn = sqlite3.connect(config.DB_PATH)
    rows = conn.execute("SELECT cohort_id, ticker, weight FROM paper_trades "
                        "WHERE status='open' AND weight IS NOT NULL ORDER BY weight DESC").fetchall()

    # --- clock ---
    try:
        clock = client.get_clock()
        when = "OPEN NOW" if clock.is_open else f"next open {clock.next_open}"
    except Exception:
        when = "clock unavailable"

    print("\n" + "=" * 78)
    print(f"  TOP-UP TO FULL SCALE  ({'EXECUTE' if EXECUTE else 'PREVIEW - no orders'})")
    print("=" * 78)
    print(f"  equity ${equity:,.2f} | cash ${cash:,.2f} | cap ${cap:,.0f} | "
          f"sleeve ${sleeve:,.0f} (90%) | SPY ${spy_target:,.0f} (10%)")
    print(f"  fills: {when}   |   HALT active: {broker.halted()}\n")

    plan = []   # (ticker, cohort_id, current, target, buy)
    for cohort_id, ticker, weight in rows:
        target = sleeve * float(weight)
        cur = positions.get(ticker, {}).get("market_value", 0.0)
        buy = round(target - cur, 2)
        if buy >= config.MIN_ORDER_NOTIONAL:
            plan.append((ticker, cohort_id, cur, target, buy))
    spy_cur = positions.get("SPY", {}).get("market_value", 0.0)
    spy_buy = round(spy_target - spy_cur, 2)
    if spy_buy >= config.MIN_ORDER_NOTIONAL:
        plan.append(("SPY", None, spy_cur, spy_target, spy_buy))

    total_buy = sum(p[4] for p in plan)

    # --- buying-power guard: scale strategy buys down if we'd overspend cash ---
    avail = cash - CASH_BUFFER
    factor = 1.0
    if total_buy > avail and total_buy > 0:
        factor = avail / total_buy
        print(f"  !! buys ${total_buy:,.0f} > available ${avail:,.0f} -> scaling by {factor:.3f}\n")

    print(f"  {'ticker':<7}{'current':>10}{'target':>10}{'BUY':>10}{'cap-ok':>8}")
    print("  " + "-" * 47)
    submitted = skipped = 0
    for ticker, cohort_id, cur, target, buy in plan:
        buy_s = round(buy * factor, 2)
        capok = buy_s <= config.MAX_ORDER_NOTIONAL
        flag = "" if capok else " CLIP"
        print(f"  {ticker:<7}{cur:>10.2f}{target:>10.2f}{buy_s:>10.2f}{('yes' if capok else 'NO'):>8}{flag}")
        if not EXECUTE:
            continue
        if not capok:
            buy_s = config.MAX_ORDER_NOTIONAL      # rail clip (shouldn't trigger here)
        oid = f"topup|{ticker}|{datetime.now():%Y%m%d}"
        try:
            r = broker.submit_notional(client, ticker, buy_s, "buy", oid)
            if r in (None, broker.DUPLICATE):
                skipped += 1
            else:
                submitted += 1
        except Exception as e:
            if "fractionable" in str(e).lower():
                px = last_close(conn, ticker)
                qty = int(buy_s // px) if px else 0
                if qty >= 1:
                    broker.submit_qty_buy(client, ticker, qty, px, oid)
                    submitted += 1
                    print(f"       {ticker} not fractionable -> {qty} whole shares (~${qty*px:,.0f})")
                else:
                    skipped += 1
                    print(f"       {ticker} not fractionable and < 1 share - skipped")
            else:
                skipped += 1
                print(f"       {ticker} FAILED: {e}")

    print("  " + "-" * 47)
    print(f"  total new buys: ${total_buy * factor:,.2f}   (leaves ~${cash - total_buy*factor:,.0f} cash)")
    if EXECUTE:
        print(f"  submitted {submitted} orders, skipped {skipped} - fill at {when}")
    else:
        print("  PREVIEW only - re-run with --execute to fire these orders")
    conn.close()


if __name__ == "__main__":
    main()

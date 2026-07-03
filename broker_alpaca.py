"""
--------------------------------------------
ALPACA PAPER BROKER  (Phase 3, A1/A2/A4)
--------------------------------------------

The ONLY file that talks to the broker. Paper trading exclusively — by
construction, not by promise:

  * get_client() HARD-FAILS unless APCA_PAPER=true AND the key is a paper key
    ("PK..." prefix; live keys start "AK") AND the SDK is pinned to the paper
    endpoint. There is no code path to a live account.
  * every entry is a NOTIONAL MARKET DAY order submitted after the close — it
    queues overnight and fills at the next open (the backtest's F1 semantics).
    Fractional/notional orders only support DAY (Alpaca constraint), which is
    why exits are managed by the daily loop, not bracket legs — and loop exits
    on closes are what the backtest verified anyway.
  * safety rails: HALT file blocks all NEW orders; per-order notional hard cap;
    $1 Alpaca minimum; deterministic client_order_id (cohort-ticker) makes
    resubmission idempotent; 3-try exponential backoff on transient errors.

Keys live in .env (never in code, never committed):
  APCA_API_KEY_ID=PK...
  APCA_API_SECRET_KEY=...
  APCA_PAPER=true
"""

import os
import time
from pathlib import Path

import config

HERE = Path(__file__).parent
PAPER_HOST = "paper-api.alpaca.markets"


class BrokerSafetyError(RuntimeError):
    """Refusing an unsafe broker configuration or order."""


def _load_env() -> dict:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
    return {"key": os.getenv("APCA_API_KEY_ID", ""),
            "secret": os.getenv("APCA_API_SECRET_KEY", ""),
            "paper": os.getenv("APCA_PAPER", "").strip().lower()}


def is_configured() -> bool:
    env = _load_env()
    return bool(env["key"] and env["secret"])


def get_client():
    """TradingClient locked to the paper endpoint - raises on ANY doubt."""
    env = _load_env()
    if not (env["key"] and env["secret"]):
        raise BrokerSafetyError("no API keys in .env - see README Phase 3 checklist")
    if env["paper"] != "true":
        raise BrokerSafetyError("APCA_PAPER must be exactly 'true' - live trading is out of scope")
    if not env["key"].startswith("PK"):
        raise BrokerSafetyError("key does not look like a PAPER key (PK...) - refusing")

    from alpaca.trading.client import TradingClient
    client = TradingClient(env["key"], env["secret"], paper=True)
    base = str(getattr(client, "_base_url", PAPER_HOST))
    if PAPER_HOST not in base:                                  # belt and braces
        raise BrokerSafetyError(f"client endpoint is not {PAPER_HOST} - refusing")
    return client


def _retry(fn, tries: int = 3):
    for attempt in range(tries):
        try:
            return fn()
        except BrokerSafetyError:
            raise
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def halted() -> bool:
    return config.HALT_FILE.exists()


def account_equity(client) -> float:
    return float(_retry(client.get_account).equity)


def get_positions(client) -> dict:
    # symbol -> {qty, market_value, avg_entry}
    out = {}
    for p in _retry(client.get_all_positions):
        out[p.symbol] = {"qty": float(p.qty), "market_value": float(p.market_value),
                         "avg_entry": float(p.avg_entry_price)}
    return out


def submit_notional(client, symbol: str, notional: float, side: str, order_id: str):
    """
    Notional market order, DAY: submitted post-close -> fills at tomorrow's
    open. Deterministic order_id makes accidental resubmission a broker-side
    no-op. BUYs are blocked by the HALT kill switch; SELLs never are (the
    switch stops NEW risk, it must never trap you in a position).
    Returns the order, or None when skipped (halt / below minimum).
    """
    if side == "buy" and halted():
        print(f"  HALT file present - {symbol} entry blocked")
        return None
    if notional < config.MIN_ORDER_NOTIONAL:
        return None
    if notional > config.MAX_ORDER_NOTIONAL:
        raise BrokerSafetyError(f"{symbol}: ${notional:,.2f} exceeds MAX_ORDER_NOTIONAL "
                                f"(${config.MAX_ORDER_NOTIONAL:,})")

    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    req = MarketOrderRequest(symbol=symbol, notional=round(notional, 2),
                             side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                             time_in_force=TimeInForce.DAY, client_order_id=order_id)
    return _retry(lambda: client.submit_order(req))


def submit_entry(client, symbol: str, notional: float, order_id: str):
    # a cohort entry = notional market BUY (see submit_notional)
    return submit_notional(client, symbol, notional, "buy", order_id)


def close_symbol(client, symbol: str):
    # market close of the whole position (exits are allowed even under HALT -
    # the kill switch stops NEW risk, it never traps you in a position)
    return _retry(lambda: client.close_position(symbol))


def filled_orders(client, limit: int = 200) -> list:
    # recent CLOSED orders -> [{order_id, symbol, side, filled_avg_price, filled_at}]
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    orders = _retry(lambda: client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=limit)))
    out = []
    for o in orders:
        if o.filled_avg_price is None:
            continue
        out.append({"order_id": o.client_order_id, "symbol": o.symbol,
                    "side": str(o.side), "filled_avg_price": float(o.filled_avg_price),
                    "filled_at": str(o.filled_at)})
    return out

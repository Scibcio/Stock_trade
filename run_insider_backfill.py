"""
--------------------------------------------
RUN INSIDER BACKFILL (dev universe)
--------------------------------------------

Pulls + parses Form 4 filings for the 15-stock DEV_UNIVERSE and stores the daily
net-insider-buying signal into the insider_flow table in trading.db. Rate-limited
per SEC rules, so it runs for a while - launch in the background.

Scope note: uses the submissions "recent" window (last ~1000 filings, i.e. the
recent years) - enough to validate the signal on dev before the full 500-stock
backfill. Aggregated by FILING date (point-in-time; no look-ahead).
"""

import sqlite3
import time

import config
import insider


def main() -> None:
    t0 = time.time()
    print("\nInsider backfill (dev universe) :\n")

    conn = sqlite3.connect(config.DB_PATH)
    conn.executescript(insider.INSIDER_SCHEMA)
    cikmap = insider.load_cik_map()

    for i, tkr in enumerate(config.DEV_UNIVERSE, 1):
        cik = cikmap.get(tkr)
        if not cik:
            print(f"  [{i:>2}/{len(config.DEV_UNIVERSE)}] {tkr:<6} no CIK - skipped")
            continue
        try:
            filings = insider.get_form4_filings(cik)
            daily = insider.collect_insider_daily(cik, filings)
            n = insider.store_insider(daily, tkr, conn)
            buys = int(daily["n_buys"].sum()) if len(daily) else 0
            print(f"  [{i:>2}/{len(config.DEV_UNIVERSE)}] {tkr:<6} {len(filings):>4} filings "
                  f"-> {n:>4} daily rows, {buys:>3} open-market BUYS  "
                  f"(elapsed {(time.time()-t0)/60:.1f} min)")
        except Exception as exc:
            print(f"  [{i:>2}/{len(config.DEV_UNIVERSE)}] {tkr:<6} ERROR: {exc}")

    conn.close()
    print(f"\n  done in {(time.time()-t0)/60:.1f} min -> insider_flow table in trading.db")


if __name__ == "__main__":
    main()

"""
--------------------------------------------
DAILY CYCLE  (schedule after US market close)
--------------------------------------------

The whole system in one command:
  1. refresh trading.db (database.py) with the latest close
  2. generate + log today's diversified picks (predict_live.py -> paper_trades)
  3. score any paper trades that have now matured (10 trading days elapsed)

Prints the running forward track record - the survivorship-free proof, building
itself day by day. Wire this to Windows Task Scheduler to run ~1h after the close.
"""

import sqlite3

import config
import database
import predict_live


def main() -> None:
    database.run()               # 1. refresh data
    predict_live.main()          # 2. today's picks -> paper_trades

    conn = sqlite3.connect(config.DB_PATH)   # 3. score matured trades
    n = predict_live.score_paper_trades(conn)
    rec = predict_live.paper_track_record(conn)
    conn.close()

    wr = "n/a" if rec["win_rate"] is None else f"{rec['win_rate']:.0%}"
    print("\nForward paper-trade record :\n")
    print(f"  closed : {rec['win']}W / {rec['loss']}L   win rate {wr}   (+{n} scored today)")
    print(f"  open   : {rec['open']}")


if __name__ == "__main__":
    main()

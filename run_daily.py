"""
--------------------------------------------
DAILY CYCLE  (schedule after US market close)
--------------------------------------------

The whole system in one resilient command:
  1. refresh trading.db (database.py) with the latest close
  2. fill yesterday's PENDING picks at today's open (honest next-open entries, F1)
  3. score any paper trades that have matured
  4. if a new cohort is due (every HOLD_DAYS sessions, F5), generate + log picks

Built for UNATTENDED operation:
  - every run is teed to a dated file in logs/ so you can check what happened
  - each step is isolated: a failure in one is logged and does not kill the others
  - a freshness guard warns if the price data did not actually update

Wired to Windows Task Scheduler (StockAI_Daily) to run ~1h after the close.
"""

import sqlite3
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

import config
import database
import predict_live

HERE = Path(__file__).parent
LOG_DIR = HERE / "logs"


class _Tee:
    # write to the console AND the daily log file
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _latest_data_date():
    conn = sqlite3.connect(config.DB_PATH)
    try:
        return conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()[0]
    finally:
        conn.close()


def _weekdays_since(latest: str | None) -> int:
    # freshness gap in WEEKDAYS (approx. trading sessions) so a long weekend or
    # holiday never false-warns the way calendar days did
    if latest is None:
        return 999
    d, today, n = date.fromisoformat(latest), date.today(), 0
    while d < today:
        d = date.fromordinal(d.toordinal() + 1)
        if d.weekday() < 5:
            n += 1
    return n


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logf = open(LOG_DIR / f"daily_{date.today()}.log", "a", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, logf)
    status = {"data": "skip", "fills": "skip", "score": "skip", "picks": "skip"}

    try:
        print(f"\n{'=' * 60}\n  DAILY RUN  {datetime.now():%Y-%m-%d %H:%M:%S}\n{'=' * 60}")

        try:                                                 # 1. refresh data
            database.run()
            status["data"] = "ok"
        except Exception:
            status["data"] = "FAILED"
            traceback.print_exc()

        latest = _latest_data_date()                         # freshness guard (sessions, not
        gap = _weekdays_since(latest)                        #  calendar days - holiday-safe)
        if gap > 2:
            print(f"\n  [WARN] latest price data is {latest} ({gap} sessions old) - picks may be stale.")

        try:                                                 # 2. fill pending entries at today's open
            conn = sqlite3.connect(config.DB_PATH)
            predict_live.migrate(conn)
            n_fill = predict_live.fill_pending(conn)
            conn.close()
            if n_fill:
                print(f"\n  Filled {n_fill} pending picks at the next session's open.")
            status["fills"] = "ok"
        except Exception:
            status["fills"] = "FAILED"
            traceback.print_exc()

        try:                                                 # 3. score matured paper trades
            conn = sqlite3.connect(config.DB_PATH)
            n = predict_live.score_paper_trades(conn)
            rec = predict_live.paper_track_record(conn)
            conn.close()
            wr = "n/a" if rec["win_rate"] is None else f"{rec['win_rate']:.0%}"
            print(f"\n  Paper-trade record: {rec['win']}W / {rec['loss']}L (win {wr}), "
                  f"{rec['open']} open, {rec['pending']} pending   (+{n} scored today)")
            status["score"] = "ok"
        except Exception:
            status["score"] = "FAILED"
            traceback.print_exc()

        try:                                                 # 4. new cohort if due (F5 cadence)
            predict_live.main()
            status["picks"] = "ok"
        except Exception:
            status["picks"] = "FAILED"
            traceback.print_exc()

        print(f"\n  STATUS: data={status['data']}  fills={status['fills']}  "
              f"score={status['score']}  picks={status['picks']}")
    finally:
        sys.stdout = real_stdout
        logf.close()


if __name__ == "__main__":
    main()

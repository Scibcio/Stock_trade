"""
--------------------------------------------
LIVE PICKS + FORWARD PAPER-TRADE
--------------------------------------------

Trains XGBoost on all ml-ready history, predicts the LATEST date for every stock,
and — when a new cohort is due (F5: every HOLD_DAYS trading sessions, matching the
backtest cadence) — logs a sector-capped, vol-weighted, regime-scaled book to the
paper_trades table. Selection is strategy.select_cohort, the SAME code path the
backtest replays (F8).

HONEST TIMING (F1): a pick decided on day D's close cannot be bought at that
close. Picks are logged as status='pending' with no entry price; the next daily
run fills entry_open from D+1's open and flips them to 'open'. Scoring uses
entry_open, so the forward record measures a price you could actually have paid.

Run after database.py has updated trading.db (i.e. after market close).
"""

import sqlite3
import warnings

import pandas as pd

import config
import earnings
import features
import labels
import model_xgb
import pipeline
import strategy
import threshold_analysis as ta

warnings.filterwarnings("ignore")

PAPER_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    pick_date    TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    entry_close  REAL,
    prob         REAL,
    regime       TEXT,
    natr         REAL,
    status       TEXT DEFAULT 'open',
    weight       REAL,
    exposure     REAL,
    cohort_id    TEXT,
    entry_open   REAL,
    exit_date    TEXT,
    exit_price   REAL,
    realized_ret REAL,
    UNIQUE(pick_date, ticker)
);
CREATE TABLE IF NOT EXISTS cohorts (
    cohort_id  TEXT PRIMARY KEY,
    pick_date  TEXT NOT NULL,
    regime     TEXT,
    exposure   REAL,
    n_picks    INTEGER
);
"""

# columns added by the F4 schema upgrade — backfilled as NULL on legacy rows
_MIGRATE_COLS = {"weight": "REAL", "exposure": "REAL", "cohort_id": "TEXT",
                 "entry_open": "REAL", "exit_date": "TEXT", "exit_price": "REAL",
                 "realized_ret": "REAL"}


def migrate(conn) -> None:
    conn.executescript(PAPER_SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)").fetchall()}
    for col, typ in _MIGRATE_COLS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {col} {typ}")
    conn.commit()


def build_train_and_latest(conn):
    baselines = features.load_baselines(conn)
    train, latest = [], []
    for t in pipeline.load_universe(conn):
        feat = features.compute_features(features.load_stock(t, conn), baselines)
        if feat.empty:
            continue
        labeled = labels.add_target(feat)
        if not labeled.empty:
            labeled["ticker"] = t
            train.append(labeled)
        row = feat.iloc[[-1]].copy()
        row["ticker"] = t
        latest.append(row)
    return pd.concat(train, ignore_index=True), pd.concat(latest, ignore_index=True)


def load_sectors(conn) -> dict:
    return dict(conn.execute("SELECT ticker, sector FROM stocks").fetchall())


def cohort_due(conn) -> bool:
    # F5: one cohort per HOLD_DAYS trading sessions (parity with the backtest),
    # counted in sessions (distinct price dates), never calendar days
    last = conn.execute("SELECT MAX(pick_date) FROM cohorts").fetchone()[0]
    if last is None:                                            # legacy rows predate the cohorts table
        last = conn.execute("SELECT MAX(pick_date) FROM paper_trades").fetchone()[0]
    if last is None:
        return True
    sessions = conn.execute("SELECT COUNT(DISTINCT date) FROM daily_prices WHERE date > ?",
                            (last,)).fetchone()[0]
    return sessions >= config.HOLD_DAYS


def fill_pending(conn) -> int:
    # F1: fill yesterday's pending picks at the first session open AFTER pick_date
    filled = 0
    for tid, pick_date, ticker in conn.execute(
            "SELECT id, pick_date, ticker FROM paper_trades WHERE status='pending'").fetchall():
        row = conn.execute(
            "SELECT open FROM daily_prices WHERE ticker=? AND date>? AND open IS NOT NULL "
            "ORDER BY date LIMIT 1", (ticker, pick_date)).fetchone()
        if row is None:
            continue                                            # entry session hasn't traded yet
        conn.execute("UPDATE paper_trades SET entry_open=?, status='open' WHERE id=?",
                     (float(row[0]), tid))
        filled += 1
    conn.commit()
    return filled


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    migrate(conn)

    if not cohort_due(conn):
        print("\n  Cohort not due yet (one book per "
              f"{config.HOLD_DAYS} trading sessions) - scoring only today.")
        conn.close()
        return

    train, latest = build_train_and_latest(conn)
    feats = config.WIDE_FEATURES
    model = model_xgb.train_xgb(train[feats].to_numpy("float32"),
                                train["Target_Label"].to_numpy("int8"))

    pick_date = latest["date"].max()
    latest = latest[latest["date"] == pick_date].copy()          # only stocks with fresh data
    latest["prob"] = model_xgb.predict_xgb(model, latest[feats].to_numpy("float32"))
    regime = ta.compute_regimes(conn).set_index("date").loc[pick_date, "regime"]
    latest["sector"] = latest["ticker"].map(load_sectors(conn)).fillna("Unknown")

    earnings.ensure_schema(conn)                                 # U1: earnings blackout -
    earnings.refresh(conn, latest["ticker"].tolist())            # incremental, cheap when current
    blocked = earnings.blackout_tickers(conn, pick_date)
    n_blocked = latest["ticker"].isin(blocked).sum()
    latest = latest[~latest["ticker"].isin(blocked)]
    if n_blocked:
        print(f"\n  earnings blackout: {n_blocked} candidates report within "
              f"{config.EARNINGS_BLACKOUT} sessions - excluded.")

    picks = strategy.select_cohort(latest, "prob", regime)       # F8: the backtested code path
    exposure = config.REGIME_EXPOSURE.get(regime, 0.5)

    conn.execute("INSERT OR IGNORE INTO cohorts (cohort_id, pick_date, regime, exposure, n_picks) "
                 "VALUES (?,?,?,?,?)",
                 (pick_date, pick_date, regime, exposure, 0 if picks is None else len(picks)))

    if picks is None:                                            # F3: bear -> 100% cash, on record
        conn.commit()
        conn.close()
        print("\n" + "-" * 60)
        print(f"  AI PICKS - {pick_date}   regime: {regime.upper()}   exposure: 0%")
        print("-" * 60)
        print("  BEAR regime - the model cannot rank in bears (fold-9 AUC 0.490).")
        print("  No entries this cohort; 100% cash. Logged to cohorts.")
        return

    for _, r in picks.iterrows():
        conn.execute("INSERT OR IGNORE INTO paper_trades "
                     "(pick_date, ticker, entry_close, prob, regime, natr, status, "
                     " weight, exposure, cohort_id) "
                     "VALUES (?,?,?,?,?,?,'pending',?,?,?)",
                     (pick_date, r["ticker"], float(r["close"]), float(r["prob"]), regime,
                      float(r["NATR_14"]), float(r["weight"]), exposure, pick_date))
    conn.commit()
    conn.close()

    print("\n" + "-" * 60)
    print(f"  AI PICKS - {pick_date}   regime: {regime.upper()}   exposure: {exposure:.0%}")
    print("-" * 60)
    if regime != "bull":
        print(f"  ! {regime.upper()} regime - edge is weaker; gross exposure scaled to {exposure:.0%}.\n")
    print(f"  {'#':>2}  {'ticker':<7}{'sector':<24}{'conf':>6}{'wt':>7}   signal close")
    for i, (_, r) in enumerate(picks.iterrows(), 1):
        print(f"  {i:>2}  {r['ticker']:<7}{str(r['sector'])[:22]:<24}"
              f"{r['prob']:>6.0%}{r['weight']:>7.1%}   ${r['close']:.2f}")
    print(f"\n  {len(picks)} picks / {picks['sector'].nunique()} sectors | "
          f"{picks['weight'].sum():.0%} invested, {1 - picks['weight'].sum():.0%} cash")
    print("  logged as PENDING - they fill at tomorrow's open (a price you can actually get),")
    print("  then score forward to prove the edge is real.")


def score_paper_trades(conn) -> int:
    """
    Mark matured open trades via the EXIT barriers on forward closes.
    Entry = entry_open (F1). Legacy pre-migration rows (entry_open NULL) keep
    their original close-entry scoring so the early record stays comparable.
    Win = net-positive exit (realized_ret > 0); realized_ret is stored either way.
    """
    scored = 0
    for tid, pick_date, ticker, entry_close, entry_open in conn.execute(
            "SELECT id, pick_date, ticker, entry_close, entry_open FROM paper_trades "
            "WHERE status='open'").fetchall():
        legacy = entry_open is None
        entry = entry_close if legacy else entry_open
        if not entry:
            continue
        fwd = conn.execute(
            "SELECT date, close FROM daily_prices WHERE ticker=? AND date>? ORDER BY date LIMIT ?",
            (ticker, pick_date, config.HOLD_DAYS + 1)).fetchall()
        window = fwd[:config.HOLD_DAYS] if legacy else fwd       # F1 window includes the entry session
        need = config.HOLD_DAYS if legacy else config.HOLD_DAYS + 1
        if len(window) < need:
            continue                                             # not matured yet
        exit_date, exit_price = window[-1]                       # default: time-barrier exit
        for d, px in window:
            ret = (px - entry) / entry
            if ret <= config.EXIT_STOP_LOSS or ret >= config.EXIT_TAKE_PROFIT:
                exit_date, exit_price = d, px
                break
        realized = (exit_price - entry) / entry
        conn.execute("UPDATE paper_trades SET status=?, exit_date=?, exit_price=?, realized_ret=? "
                     "WHERE id=?",
                     ("win" if realized > 0 else "loss", exit_date, float(exit_price),
                      float(realized), tid))
        scored += 1
    conn.commit()
    return scored


def paper_track_record(conn) -> dict:
    d = dict(conn.execute("SELECT status, COUNT(*) FROM paper_trades GROUP BY status").fetchall())
    closed = d.get("win", 0) + d.get("loss", 0)
    return {"open": d.get("open", 0), "pending": d.get("pending", 0),
            "win": d.get("win", 0), "loss": d.get("loss", 0),
            "win_rate": (d.get("win", 0) / closed) if closed else None}


if __name__ == "__main__":
    main()

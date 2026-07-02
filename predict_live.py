"""
--------------------------------------------
LIVE PICKS + FORWARD PAPER-TRADE
--------------------------------------------

Trains XGBoost on all ml-ready history, predicts the LATEST date for every stock,
applies the regime read + volatility-based sizing, prints today's top picks, and
logs them to the paper_trades table so we can score them FORWARD - the only
survivorship-free proof that the edge is real.

Run after database.py has updated trading.db (i.e. after market close).
"""

import sqlite3
import warnings

import pandas as pd

import config
import features
import labels
import model_xgb
import pipeline
import threshold_analysis as ta

warnings.filterwarnings("ignore")
TOP_K = 15

PAPER_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pick_date   TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    entry_close REAL,
    prob        REAL,
    regime      TEXT,
    natr        REAL,
    status      TEXT DEFAULT 'open',
    UNIQUE(pick_date, ticker)
);
"""


def build_train_and_latest(conn):
    baselines = features.load_baselines(conn)
    train, latest = [], []
    for t in pipeline.load_universe(conn):
        feat = features.merge_insider(
            features.compute_features(features.load_stock(t, conn), baselines),
            features.load_insider(t, conn))
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


def main() -> None:
    conn = sqlite3.connect(config.DB_PATH)
    conn.executescript(PAPER_SCHEMA)

    train, latest = build_train_and_latest(conn)
    feats = config.WIDE_FEATURES
    model = model_xgb.train_xgb(train[feats].to_numpy("float32"),
                                train["Target_Label"].to_numpy("int8"))

    pick_date = latest["date"].max()
    latest = latest[latest["date"] == pick_date].copy()          # only stocks with fresh data
    latest["prob"] = model_xgb.predict_xgb(model, latest[feats].to_numpy("float32"))
    regime = ta.compute_regimes(conn).set_index("date").loc[pick_date, "regime"]

    picks = latest.sort_values("prob", ascending=False).head(TOP_K).copy()
    inv = 1.0 / picks["NATR_14"].clip(lower=0.1)                  # vol-targeting: size ~ 1/vol
    picks["weight"] = inv / inv.sum()

    # log to paper_trades for forward scoring
    for _, r in picks.iterrows():
        conn.execute("INSERT OR IGNORE INTO paper_trades (pick_date, ticker, entry_close, prob, regime, natr) "
                     "VALUES (?,?,?,?,?,?)",
                     (pick_date, r["ticker"], float(r["close"]), float(r["prob"]), regime, float(r["NATR_14"])))
    conn.commit()
    conn.close()

    print("\n" + "=" * 56)
    print(f"  AI PICKS - {pick_date}   market regime: {regime.upper()}")
    print("=" * 56)
    if regime == "bear":
        print("  ! BEAR regime - model edge is weak here; size DOWN or sit out.\n")
    print(f"  {'#':>2}  {'ticker':<7}{'conf':>7}{'NATR%':>8}{'weight':>9}   entry")
    for i, (_, r) in enumerate(picks.iterrows(), 1):
        print(f"  {i:>2}  {r['ticker']:<7}{r['prob']:>7.1%}{r['NATR_14']:>8.1f}{r['weight']:>8.1%}   ${r['close']:.2f}")
    print("\n  logged to paper_trades -> score forward to prove the edge is real.")


if __name__ == "__main__":
    main()

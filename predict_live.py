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
MAX_PER_SECTOR = 3                                              # diversification cap
REGIME_EXPOSURE = {"bull": 1.0, "sideways": 0.6, "bear": 0.3}  # gross exposure by regime

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


def load_sectors(conn) -> dict:
    return dict(conn.execute("SELECT ticker, sector FROM stocks").fetchall())


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
    latest["sector"] = latest["ticker"].map(load_sectors(conn)).fillna("Unknown")

    # sector-capped selection: no more than MAX_PER_SECTOR names from one sector
    chosen, per_sector = [], {}
    for _, r in latest.sort_values("prob", ascending=False).iterrows():
        if per_sector.get(r["sector"], 0) >= MAX_PER_SECTOR:
            continue
        chosen.append(r)
        per_sector[r["sector"]] = per_sector.get(r["sector"], 0) + 1
        if len(chosen) >= TOP_K:
            break
    picks = pd.DataFrame(chosen)

    # vol-target weights, scaled down in weaker regimes (rest stays in cash)
    exposure = REGIME_EXPOSURE.get(regime, 0.5)
    inv = 1.0 / picks["NATR_14"].clip(lower=0.1)
    picks["weight"] = inv / inv.sum() * exposure

    for _, r in picks.iterrows():
        conn.execute("INSERT OR IGNORE INTO paper_trades (pick_date, ticker, entry_close, prob, regime, natr) "
                     "VALUES (?,?,?,?,?,?)",
                     (pick_date, r["ticker"], float(r["close"]), float(r["prob"]), regime, float(r["NATR_14"])))
    conn.commit()
    conn.close()

    print("\n" + "-" * 60)
    print(f"  AI PICKS - {pick_date}   regime: {regime.upper()}   exposure: {exposure:.0%}")
    print("-" * 60)
    if regime != "bull":
        print(f"  ! {regime.upper()} regime - edge is weaker; gross exposure scaled to {exposure:.0%}.\n")
    print(f"  {'#':>2}  {'ticker':<7}{'sector':<24}{'conf':>6}{'wt':>7}   entry")
    for i, (_, r) in enumerate(picks.iterrows(), 1):
        print(f"  {i:>2}  {r['ticker']:<7}{str(r['sector'])[:22]:<24}{r['prob']:>6.0%}{r['weight']:>7.1%}   ${r['close']:.2f}")
    print(f"\n  {len(picks)} picks / {picks['sector'].nunique()} sectors | "
          f"{picks['weight'].sum():.0%} invested, {1 - picks['weight'].sum():.0%} cash")
    print("  logged to paper_trades -> score forward to prove the edge is real.")


if __name__ == "__main__":
    main()

"""
--------------------------------------------
STOCK_TRADE DASHBOARD (Streamlit)
--------------------------------------------

A local, READ-ONLY view of the system: today's picks, the forward paper-trade
record, and model performance.

SECURITY POSTURE (see SECURITY.md before any public deploy):
  - opens trading.db strictly READ-ONLY (mode=ro) - the app can never modify data
  - every query is parameterized; there is NO free-text user input into SQL
  - result sizes are bounded; loaders are cached
  - all data access is wrapped - users see a friendly message, never a stack trace
  - a financial disclaimer is shown on every page
  - DO NOT ship the raw trading.db to a public host (see SECURITY.md -> snapshot)

Run:  streamlit run dashboard.py
"""

from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
DB_PATH = HERE / "trading.db"
OOF_XGB = HERE / "walk_forward_oof.csv"
OOF_LSTM = HERE / "walk_forward_oof_lstm.csv"
MAX_ROWS = 1000                                     # hard cap on any table returned

st.set_page_config(page_title="Stock_trade", page_icon="📈", layout="wide")

DISCLAIMER = (
    "**Educational / research tool — NOT financial advice.** These are outputs of an "
    "experimental ML model with a small, honestly-characterized edge whose *live* "
    "performance is unproven. Nothing here is a recommendation to buy or sell any security. "
    "Backtested figures are survivorship-biased and do not predict future returns. Do your "
    "own research; the authors accept no liability for any decision or loss."
)


# ----------------------------------
# DATA ACCESS (read-only, parameterized, cached, guarded)
# ----------------------------------

def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)   # READ-ONLY
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()                                    # never leak internals
    finally:
        conn.close()


@st.cache_data(ttl=300)
def latest_picks() -> pd.DataFrame:
    return _query(
        "SELECT p.pick_date, p.ticker, s.sector, p.prob, p.regime, p.natr, p.status "
        "FROM paper_trades p LEFT JOIN stocks s ON p.ticker = s.ticker "
        "WHERE p.pick_date = (SELECT MAX(pick_date) FROM paper_trades) "
        "ORDER BY p.prob DESC LIMIT ?", (MAX_ROWS,))


@st.cache_data(ttl=300)
def paper_record() -> dict:
    df = _query("SELECT status, COUNT(*) AS n FROM paper_trades GROUP BY status")
    d = dict(zip(df.get("status", []), df.get("n", []))) if not df.empty else {}
    closed = d.get("win", 0) + d.get("loss", 0)
    return {"open": d.get("open", 0), "win": d.get("win", 0), "loss": d.get("loss", 0),
            "win_rate": (d.get("win", 0) / closed) if closed else None}


@st.cache_data(ttl=300)
def scored_trades() -> pd.DataFrame:
    return _query(
        "SELECT pick_date, ticker, prob, regime, status FROM paper_trades "
        "WHERE status != 'open' ORDER BY pick_date DESC LIMIT ?", (MAX_ROWS,))


@st.cache_data(ttl=600)
def model_perf():
    if not (OOF_XGB.exists() and OOF_LSTM.exists()):
        return None
    try:
        from sklearn.metrics import roc_auc_score
        xgb = pd.read_csv(OOF_XGB)
        lstm = pd.read_csv(OOF_LSTM)[["date", "ticker", "p_lstm"]]
        df = xgb.merge(lstm, on=["date", "ticker"], how="inner").dropna()
        df["blend"] = 0.5 * df["p_xgb"] + 0.5 * df["p_lstm"]
        curve = []
        for pct in (100, 50, 25, 10, 5):
            sel = df[df["blend"] >= df["blend"].quantile(1 - pct / 100)]
            curve.append({"top_% by confidence": pct, "win_rate": round(sel["Target_Label"].mean(), 3)})
        return {"auc": roc_auc_score(df["Target_Label"], df["blend"]),
                "n": len(df), "base": df["Target_Label"].mean(),
                "curve": pd.DataFrame(curve).set_index("top_% by confidence")}
    except Exception:
        return None


# ----------------------------------
# UI
# ----------------------------------

st.title("📈 Stock_trade — live picks & forward proof")
st.caption(DISCLAIMER)

if not DB_PATH.exists():
    st.info("No `trading.db` found. Run `python database.py` then `python predict_live.py` to populate it.")
    st.stop()

tab_picks, tab_paper, tab_model = st.tabs(["Today's picks", "Forward paper-trade", "Model performance"])

with tab_picks:
    picks = latest_picks()
    if picks.empty:
        st.info("No picks yet — run `python predict_live.py` (or wait for the daily job).")
    else:
        st.subheader(f"Picks for {picks['pick_date'].iloc[0]}  ·  regime: {str(picks['regime'].iloc[0]).upper()}")
        show = picks[["ticker", "sector", "prob", "natr", "status"]].rename(
            columns={"prob": "confidence", "natr": "NATR %"})
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("Sector spread (diversification cap = max 3 per sector):")
        st.bar_chart(picks["sector"].value_counts())

with tab_paper:
    rec = paper_record()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open", rec["open"])
    c2.metric("Wins", rec["win"])
    c3.metric("Losses", rec["loss"])
    c4.metric("Win rate", "—" if rec["win_rate"] is None else f"{rec['win_rate']:.0%}")
    st.caption("The survivorship-free proof — fills in as trades mature (10 trading days).")
    trades = scored_trades()
    if trades.empty:
        st.info("No matured trades yet. The forward record builds over the coming weeks.")
    else:
        st.dataframe(trades, use_container_width=True, hide_index=True)

with tab_model:
    perf = model_perf()
    if perf is None:
        st.info("Run `python pipeline.py` and `python run_lstm.py` to generate out-of-fold predictions.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("OOF AUC (ensemble)", f"{perf['auc']:.4f}")
        c2.metric("Base win rate", f"{perf['base']:.1%}")
        c3.metric("OOF predictions", f"{perf['n']:,}")
        st.caption("Selectivity: win rate rises as we keep only the most-confident picks.")
        st.line_chart(perf["curve"])
        st.caption("Edge certified real vs 2,000 random baselines — see FINDINGS.md.")

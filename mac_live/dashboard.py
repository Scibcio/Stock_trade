"""
--------------------------------------------
DASHBOARD — MAX aggressive paper profile (Mac, live only, NO backtesting)
--------------------------------------------

A clean control room for the live model: strategy, picks + result, model
quality, the live Alpaca paper account, the database, and run logs — plus
buttons to run the daily cycle. Localhost only.

Run:  streamlit run dashboard.py
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

import config

HERE = Path(__file__).parent
DB_PATH = HERE / "trading.db"
OOF_XGB = HERE / "walk_forward_oof.csv"
LOG_DIR = HERE / "logs"
MAX_ROWS = 1000

st.set_page_config(page_title="Stock_trade · MAX", page_icon="🕮", layout="wide")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
:root{ --ink:#231f18; --muted:#7c7264; --line:#e7ded0; --accent:#a86f2c; --card:#ffffff; }
html, body, .stApp, [data-testid="stAppViewContainer"], [class*="css"]{ font-family:'IBM Plex Sans', system-ui, sans-serif; color:var(--ink); }
.block-container{ padding-top:2rem; padding-bottom:3rem; max-width:1200px; }
h1,h2,h3,h4{ letter-spacing:-0.015em; font-weight:600; } h2{ font-size:1.15rem; } h3{ font-size:1.02rem; }
.app-head{ border-bottom:2px solid var(--accent); padding:0.1rem 0 0.7rem; margin-bottom:1.3rem; }
.app-head .t{ font-family:'Newsreader', Georgia, serif; font-weight:600; font-size:2rem; letter-spacing:-0.01em; }
.app-head .t b{ color:var(--accent); } .app-head .s{ color:var(--muted); font-size:0.88rem; margin-top:0.15rem; }
[data-testid="stMetric"]{ background:var(--card); border:1px solid var(--line); border-radius:9px; padding:0.7rem 0.85rem; }
[data-testid="stMetricLabel"] p{ font-size:0.68rem; font-weight:500; letter-spacing:0.07em; text-transform:uppercase; color:var(--muted); }
[data-testid="stMetricValue"]{ font-family:'IBM Plex Mono', monospace; font-variant-numeric:tabular-nums; font-weight:600; font-size:1.45rem; }
[data-testid="stMetricDelta"]{ font-family:'IBM Plex Mono', monospace; font-size:0.78rem; }
[data-testid="stTabs"] [role="tablist"]{ gap:1.3rem; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] [role="tab"]{ padding:0.35rem 0; color:var(--muted); font-weight:500; }
[data-testid="stTabs"] [aria-selected="true"]{ color:var(--ink); box-shadow:inset 0 -2px 0 var(--accent); }
.stButton button{ border-radius:7px; border:1px solid var(--line); font-weight:500; background:var(--card); }
.stButton button:hover{ border-color:var(--accent); color:var(--accent); }
[data-testid="stBaseButton-primary"]{ background:var(--accent); border-color:var(--accent); color:#fff; }
[data-testid="stSidebar"]{ border-right:1px solid var(--line); }
[data-testid="stAlert"]{ border-radius:9px; border:1px solid var(--line); background:#fbf7f1; }
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:9px; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

DISCLAIMER = ("**Educational / research — NOT financial advice.** This is the deliberately "
              "aggressive, risk-ignored MAX profile on a **demo** account. Backtested returns "
              "for these settings are the most survivorship-inflated in the project; live will be "
              "a fraction. Paper money only; nothing here is a recommendation.")


# ---------- data access ----------
def _query(sql, params=()):
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data(ttl=120)
def db_stats():
    d = _query("SELECT MAX(date) AS m, COUNT(*) AS n FROM daily_prices")
    r = _query("SELECT COUNT(*) AS n FROM stocks WHERE ml_ready = 1")
    rng = _query("SELECT MIN(date) AS lo, MAX(date) AS hi FROM daily_prices")
    return {"date_max": d["m"].iloc[0] if not d.empty else "—",
            "rows": int(d["n"].iloc[0]) if not d.empty and d["n"].iloc[0] else 0,
            "ml_ready": int(r["n"].iloc[0]) if not r.empty and r["n"].iloc[0] else 0,
            "lo": rng["lo"].iloc[0] if not rng.empty else "—",
            "hi": rng["hi"].iloc[0] if not rng.empty else "—"}


@st.cache_data(ttl=120)
def latest_picks():
    return _query(
        "SELECT p.pick_date, p.ticker, s.sector, p.prob, p.natr, p.entry_close, p.entry_open, "
        "p.weight, p.regime, p.status FROM paper_trades p LEFT JOIN stocks s ON p.ticker = s.ticker "
        "WHERE p.pick_date = (SELECT MAX(pick_date) FROM paper_trades) ORDER BY p.prob DESC LIMIT ?",
        (MAX_ROWS,))


@st.cache_data(ttl=120)
def cohort_results():
    picks = _query("SELECT ticker, entry_close, prob FROM paper_trades "
                   "WHERE pick_date=(SELECT MAX(pick_date) FROM paper_trades) ORDER BY prob DESC")
    if picks.empty or picks["entry_close"].isna().all():
        return None
    pdate = _query("SELECT MAX(pick_date) AS d FROM paper_trades")["d"].iloc[0]
    tk = list(picks["ticker"])
    fwd = _query(f"SELECT ticker, date, close FROM daily_prices WHERE ticker IN ({','.join('?'*len(tk))}) "
                 "AND date>? ORDER BY date", tuple(tk) + (pdate,))
    if fwd.empty:
        return None
    TP, SL, HOLD = config.EXIT_TAKE_PROFIT, config.EXIT_STOP_LOSS, config.EXEC_HOLD_DAYS
    rows = []
    for _, p in picks.iterrows():
        entry = p["entry_close"]
        f = fwd[fwd["ticker"] == p["ticker"]].head(HOLD)
        if pd.isna(entry) or f.empty:
            continue
        status, exit_px, ret = "open", f["close"].iloc[-1], f["close"].iloc[-1] / entry - 1
        for _, r in f.iterrows():
            rr = r["close"] / entry - 1
            if rr <= SL:
                status, exit_px, ret = "loss", r["close"], rr
                break
            if rr >= TP:
                status, exit_px, ret = "win", r["close"], rr
                break
        rows.append({"ticker": p["ticker"], "entry": entry, "exit": exit_px, "ret": ret, "status": status})
    if not rows:
        return None
    r = pd.DataFrame(rows)
    return {"table": r, "pdate": pdate, "n": len(r),
            "wins": int((r["status"] == "win").sum()), "losses": int((r["status"] == "loss").sum()),
            "open_": int((r["status"] == "open").sum()), "green": int((r["ret"] > 0).sum()),
            "avg": float(r["ret"].mean())}


@st.cache_data(ttl=120)
def paper_record():
    df = _query("SELECT status, COUNT(*) AS n FROM paper_trades GROUP BY status")
    d = dict(zip(df["status"], df["n"])) if not df.empty else {}
    closed = d.get("win", 0) + d.get("loss", 0)
    return {"open": d.get("open", 0), "pending": d.get("pending", 0), "win": d.get("win", 0),
            "loss": d.get("loss", 0), "win_rate": (d.get("win", 0) / closed) if closed else None}


@st.cache_data(ttl=300)
def model_perf():
    if not OOF_XGB.exists():
        return None
    try:
        from sklearn.metrics import roc_auc_score
        df = pd.read_csv(OOF_XGB).dropna()
        sig = df["p_xgb"]
        curve = [{"top % kept": pct,
                  "win rate %": round(df[sig >= sig.quantile(1 - pct / 100)]["Target_Label"].mean() * 100, 1)}
                 for pct in (100, 50, 25, 10, 5)]
        return {"auc": roc_auc_score(df["Target_Label"], sig), "base": df["Target_Label"].mean(),
                "n": len(df), "curve": pd.DataFrame(curve).set_index("top % kept")}
    except Exception:
        return None


@st.cache_data(ttl=120)
def alpaca_snapshots():
    state = _query("SELECT ts, equity, spy_value, n_sleeve FROM alpaca_state ORDER BY ts")
    fills = _query("SELECT filled_at, symbol, side, price, slippage_bps FROM alpaca_fills ORDER BY filled_at DESC LIMIT ?", (MAX_ROWS,))
    return {"state": state, "fills": fills}


def fetch_alpaca_now():
    try:
        import broker_alpaca as broker
        if not broker.is_configured():
            return {"error": "no paper keys in .env"}
        client = broker.get_client()
        acct = client.get_account()
        pos = broker.get_positions(client)
        rows = [{"Symbol": s, "Qty": round(p["qty"], 3), "Market value": p["market_value"],
                 "Unrealized P&L": p["market_value"] - p["avg_entry"] * p["qty"],
                 "P&L %": (p["market_value"] / (p["avg_entry"] * p["qty"]) - 1) if p["avg_entry"] * p["qty"] else 0.0}
                for s, p in sorted(pos.items())]
        return {"equity": float(acct.equity), "cash": float(acct.cash),
                "buying_power": float(acct.buying_power), "status": str(acct.status),
                "positions": pd.DataFrame(rows)}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=60)
def latest_log():
    logs = sorted(LOG_DIR.glob("daily_*.log")) if LOG_DIR.exists() else []
    if not logs:
        return None
    try:
        return logs[-1].name, logs[-1].read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def run_script(script, timeout=1800):
    proc = subprocess.run([sys.executable, str(HERE / script)], cwd=str(HERE),
                          capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")


# ---------- control panel ----------
st.sidebar.markdown("#### Control panel")
st.sidebar.caption("Run the live model — no terminal needed.")
for label, script, help_ in [
    ("Run full daily cycle", "run_daily.py", "~4 min · data → fills → score → picks → broker"),
    ("Generate picks", "predict_live.py", "~3 min · fresh picks from the latest data"),
    ("Update data only", "database.py", "~3 min · pull the latest close"),
]:
    if st.sidebar.button(label, use_container_width=True, help=help_):
        with st.spinner(f"Running {script} — please wait…"):
            try:
                rc, out = run_script(script)
                st.session_state.update(run_out=out, run_ok=(rc == 0), run_name=script)
            except subprocess.TimeoutExpired:
                st.session_state.update(run_out=f"{script} timed out.", run_ok=False, run_name=script)
        st.cache_data.clear()
        st.rerun()
st.sidebar.divider()
st.sidebar.caption("⚠️ MAX / risk-ignored profile · demo money · local only · not financial advice.")


# ---------- page ----------
st.markdown('<div class="app-head"><div class="t">Stock<b>_</b>trade &nbsp;·&nbsp; MAX</div>'
            '<div class="s">Aggressive paper profile — top 5, uncapped winners, ~1-month hold, '
            'fully invested. Live on Alpaca demo.</div></div>', unsafe_allow_html=True)

if st.session_state.get("run_out"):
    ok = st.session_state.get("run_ok")
    (st.success if ok else st.error)(f"{'✅' if ok else '❌'} `{st.session_state.get('run_name')}` "
                                     f"{'finished' if ok else 'failed'}.")
    with st.expander("Show run output", expanded=not ok):
        st.code(st.session_state["run_out"][-4000:])

with st.expander("ℹ️  Disclaimer — please read"):
    st.warning(DISCLAIMER)

if not DB_PATH.exists():
    st.info("No database yet. Click **Update data only** in the sidebar to build it (~3 min).")
    st.stop()

tab_over, tab_picks, tab_model, tab_live, tab_db, tab_logs = st.tabs(
    ["Overview", "Picks", "Model", "Live paper", "Database", "Run log"])

with tab_over:
    st.subheader("Strategy — MAX (risk ignored)")
    c = st.columns(4)
    c[0].metric("Names", config.TOP_K)
    c[1].metric("Hold", f"{config.EXEC_HOLD_DAYS} sessions", help="~1 month; model label stays 10-day")
    c[2].metric("Upside cap", "none", help="let winners run")
    c[3].metric("Stop", f"{config.EXIT_STOP_LOSS:.0%}", help="catastrophe stop only")
    st.caption(f"Fully invested every regime · portfolio mode `{config.PORTFOLIO_MODE}` · "
               f"paper cap ${config.PAPER_EQUITY_CAP:,}. The most survivorship-inflated settings — "
               "live returns will be far below backtest.")

    rec, perf = paper_record(), model_perf()
    st.subheader("Forward record (the clean, survivorship-free verdict)")
    c = st.columns(5)
    c[0].metric("Pending", rec["pending"])
    c[1].metric("Open", rec["open"])
    c[2].metric("Wins", rec["win"])
    c[3].metric("Losses", rec["loss"])
    c[4].metric("Win rate", "—" if rec["win_rate"] is None else f"{rec['win_rate']:.0%}")
    if perf:
        st.caption(f"Model: out-of-sample AUC {perf['auc']:.3f} (certified) · base win {perf['base']:.0%} · "
                   f"{perf['n']:,} predictions. The ranker is the same certified 10-day model.")

with tab_picks:
    picks = latest_picks()
    if picks.empty:
        st.info("No picks yet — click **Generate picks** in the sidebar.")
    else:
        st.subheader(f"Picks for {picks['pick_date'].iloc[0]}")
        st.markdown(f"**Regime:** `{str(picks['regime'].iloc[0]).upper()}` · **{len(picks)} names**")
        st.dataframe(pd.DataFrame({
            "Ticker": picks["ticker"].values,
            "Sector": picks["sector"].fillna("—").values,
            "Confidence": [f"{p:.0%}" for p in picks["prob"]],
            "Weight": ["—" if pd.isna(w) else f"{w:.1%}" for w in picks["weight"]],
            "Signal close": [f"${e:,.2f}" for e in picks["entry_close"]],
            "Fill (open)": ["pending" if pd.isna(o) else f"${o:,.2f}" for o in picks["entry_open"]],
            "Status": picks["status"].values,
        }), use_container_width=True, hide_index=True)

        res = cohort_results()
        if res:
            st.subheader(f"Cohort result — mark-to-market · {res['pdate']}")
            st.caption("Would-have paper result (close entry, uncapped / −15% / ~1-month). Not the clean broker record.")
            c = st.columns(4)
            c[0].metric("Green", f"{res['green']} / {res['n']}")
            c[1].metric("Wins / Losses", f"{res['wins']}W / {res['losses']}L")
            c[2].metric("Still open", res["open_"])
            c[3].metric("Avg pick", f"{res['avg']:+.2%}")
            st.dataframe(pd.DataFrame({
                "Ticker": res["table"]["ticker"].values,
                "Entry": [f"${e:,.2f}" for e in res["table"]["entry"]],
                "Now / exit": [f"${e:,.2f}" for e in res["table"]["exit"]],
                "Return": [f"{r:+.1%}" for r in res["table"]["ret"]],
                "Status": res["table"]["status"].str.upper().values,
            }), use_container_width=True, hide_index=True)

with tab_model:
    perf = model_perf()
    if not perf:
        st.info("Model stats file (`walk_forward_oof.csv`) not found in this folder.")
    else:
        st.subheader("The model")
        st.markdown("XGBoost ranker (300 trees, depth 4) on ~30 leak-free technical features, "
                    "trained per fold on a purged 12-fold walk-forward. Predicts P(+3% before −1% in 10 days).")
        c = st.columns(3)
        c[0].metric("Out-of-sample AUC", f"{perf['auc']:.3f}", help="0.50 = coin flip; ~0.56 = small real edge")
        c[1].metric("Base win rate", f"{perf['base']:.0%}")
        c[2].metric("Predictions", f"{perf['n']:,}")
        st.subheader("Selectivity — win rate rises with confidence")
        st.line_chart(perf["curve"])
        st.caption("The edge is in being selective (the top slice), not raw accuracy.")

with tab_live:
    st.subheader("Alpaca paper account")
    st.caption("Demo money. This is the clean, survivorship-free verdict.")
    rec = paper_record()
    c = st.columns(5)
    c[0].metric("Pending", rec["pending"]); c[1].metric("Open", rec["open"])
    c[2].metric("Wins", rec["win"]); c[3].metric("Losses", rec["loss"])
    c[4].metric("Win rate", "—" if rec["win_rate"] is None else f"{rec['win_rate']:.0%}")

    if st.button("Refresh from Alpaca  ·  read-only", type="primary"):
        st.session_state["alpaca_now"] = fetch_alpaca_now()
    snap = st.session_state.get("alpaca_now")
    if snap and "error" in snap:
        st.info(f"Live pull unavailable ({snap['error']}). Add your **paper** keys to `.env`.")
    elif snap:
        st.markdown(f"**Live now · account {snap['status']}**")
        c = st.columns(4)
        c[0].metric("Equity", f"${snap['equity']:,.2f}"); c[1].metric("Cash", f"${snap['cash']:,.2f}")
        c[2].metric("Buying power", f"${snap['buying_power']:,.2f}"); c[3].metric("Positions", len(snap["positions"]))
        if not snap["positions"].empty:
            p = snap["positions"]
            st.dataframe(pd.DataFrame({
                "Symbol": p["Symbol"], "Qty": p["Qty"],
                "Market value": [f"${v:,.2f}" for v in p["Market value"]],
                "Unrealized P&L": [f"${v:+,.2f}" for v in p["Unrealized P&L"]],
                "P&L %": [f"{v:+.1%}" for v in p["P&L %"]],
            }), use_container_width=True, hide_index=True)

    snaps = alpaca_snapshots()
    if not snaps["state"].empty:
        st.divider()
        st.markdown("**Nightly equity (from the daily run)**")
        st.line_chart(snaps["state"].set_index("ts")["equity"], height=240)
        if len(snaps["fills"]):
            st.caption("Recent fills")
            st.dataframe(snaps["fills"], use_container_width=True, hide_index=True)

with tab_db:
    s = db_stats()
    st.subheader("Database")
    c = st.columns(4)
    c[0].metric("Data through", s["date_max"]); c[1].metric("Stocks ready", s["ml_ready"])
    c[2].metric("Price rows", f"{s['rows']:,}"); c[3].metric("History", f"{s['lo']} → {s['hi']}")
    st.caption("Built and updated by `database.py` from yfinance. Rebuild any time with **Update data only**.")
    counts = _query("SELECT status, COUNT(*) AS n FROM paper_trades GROUP BY status")
    if not counts.empty:
        st.caption("paper_trades by status:")
        st.dataframe(counts, use_container_width=True, hide_index=True)

with tab_logs:
    lg = latest_log()
    if not lg:
        st.info("No run logs yet — run the daily cycle from the sidebar.")
    else:
        name, text = lg
        st.subheader(f"Latest run log — {name}")
        st.code(text[-8000:])

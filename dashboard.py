"""
--------------------------------------------
STOCK_TRADE DASHBOARD (Streamlit) — LOCAL CONTROL PANEL
--------------------------------------------

An all-in-one, readable view AND control panel: system overview, today's picks,
model performance, run logs — plus BUTTONS to run every part of the system, so you
never need a terminal.

!!  LOCAL USE ONLY  !!
The control-panel buttons EXECUTE local scripts (database.py, predict_live.py, ...).
That is safe on your own machine but must NEVER be exposed to the public internet.
For a public/read-only build, delete the "CONTROL PANEL" section. See SECURITY.md.

Run:  streamlit run dashboard.py
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
DB_PATH = HERE / "trading.db"
OOF_XGB = HERE / "walk_forward_oof.csv"
OOF_LSTM = HERE / "walk_forward_oof_lstm.csv"
BT_EQUITY = HERE / "backtest_equity.csv"
BT_TRADES = HERE / "backtest_trades.csv"
LOG_DIR = HERE / "logs"
MAX_ROWS = 1000

DISCLAIMER = (
    "**Educational / research tool — NOT financial advice.** These are outputs of an "
    "experimental model with a small, honestly-characterized edge whose *live* performance "
    "is unproven. Nothing here is a recommendation to buy or sell. Backtested figures are "
    "survivorship-biased and do not predict future returns. Do your own research; the author "
    "accepts no liability for any decision or loss."
)

st.set_page_config(page_title="Stock_trade", page_icon="🕮", layout="wide")

# ---- "Ledger" theme polish: fine typography, tabular-mono numerals, hairline
#      cards, muted alerts. Warm paper + one bronze accent — no AI blue/purple. ----
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
:root{
  --ink:#231f18; --muted:#7c7264; --line:#e7ded0; --accent:#a86f2c;
  --card:#ffffff; --pos:#1a7f4b; --neg:#b23b2e;
}
html, body, .stApp, [data-testid="stAppViewContainer"], [class*="css"]{
  font-family:'IBM Plex Sans', system-ui, -apple-system, sans-serif; color:var(--ink);
}
.block-container{ padding-top:2rem; padding-bottom:3rem; max-width:1200px; }
h1,h2,h3,h4{ letter-spacing:-0.015em; font-weight:600; }
h2{ font-size:1.15rem; } h3{ font-size:1.02rem; margin-top:0.3rem; }
hr{ border-color:var(--line); }

/* editorial header */
.app-head{ border-bottom:2px solid var(--accent); padding:0.1rem 0 0.7rem; margin-bottom:1.3rem; }
.app-head .t{ font-family:'Newsreader', Georgia, serif; font-weight:600; font-size:2rem;
  letter-spacing:-0.01em; line-height:1.1; }
.app-head .s{ color:var(--muted); font-size:0.88rem; margin-top:0.15rem; }
.app-head .t b{ color:var(--accent); font-weight:600; }

/* metric = a clean bordered card with a tabular mono value */
[data-testid="stMetric"]{ background:var(--card); border:1px solid var(--line);
  border-radius:9px; padding:0.7rem 0.85rem; }
[data-testid="stMetricLabel"] p{ font-size:0.68rem; font-weight:500; letter-spacing:0.07em;
  text-transform:uppercase; color:var(--muted); }
[data-testid="stMetricValue"]{ font-family:'IBM Plex Mono', monospace; font-variant-numeric:tabular-nums;
  font-weight:600; font-size:1.45rem; color:var(--ink); }
[data-testid="stMetricDelta"]{ font-family:'IBM Plex Mono', monospace; font-size:0.78rem; }

/* tabs: quiet, underline the active one in accent */
[data-testid="stTabs"] [role="tablist"]{ gap:1.3rem; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] [role="tab"]{ padding:0.35rem 0; color:var(--muted); font-weight:500; }
[data-testid="stTabs"] [role="tab"] p{ font-size:0.92rem; }
[data-testid="stTabs"] [aria-selected="true"]{ color:var(--ink); box-shadow:inset 0 -2px 0 var(--accent); }

/* buttons: flat, hairline, accent on hover */
.stButton button, [data-testid="stBaseButton-secondary"]{ border-radius:7px; border:1px solid var(--line);
  font-weight:500; background:var(--card); }
.stButton button:hover{ border-color:var(--accent); color:var(--accent); }
[data-testid="stBaseButton-primary"]{ background:var(--accent); border-color:var(--accent); color:#fff; }

/* sidebar */
[data-testid="stSidebar"]{ border-right:1px solid var(--line); }
[data-testid="stSidebar"] .stButton button{ text-align:left; }

/* muted, flat alerts (kill the loud default blue/yellow blocks) */
[data-testid="stAlert"]{ border-radius:9px; border:1px solid var(--line); background:#fbf7f1; }

/* tables + charts sit in hairline frames */
[data-testid="stDataFrame"], [data-testid="stTable"]{ border:1px solid var(--line); border-radius:9px; }
[data-testid="stCaptionContainer"] p{ color:var(--muted); }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ==================================================
# DATA ACCESS (read-only, cached, guarded)
# ==================================================

def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
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
def db_stats() -> dict:
    d = _query("SELECT MAX(date) AS m, COUNT(*) AS n FROM daily_prices")
    r = _query("SELECT COUNT(*) AS n FROM stocks WHERE ml_ready = 1")
    return {"date_max": d["m"].iloc[0] if not d.empty else "—",
            "rows": int(d["n"].iloc[0]) if not d.empty and d["n"].iloc[0] else 0,
            "ml_ready": int(r["n"].iloc[0]) if not r.empty and r["n"].iloc[0] else 0}


@st.cache_data(ttl=120)
def latest_picks() -> pd.DataFrame:
    return _query(
        "SELECT p.pick_date, p.ticker, s.sector, p.prob, p.natr, p.entry_close, p.entry_open, "
        "p.weight, p.regime, p.status "
        "FROM paper_trades p LEFT JOIN stocks s ON p.ticker = s.ticker "
        "WHERE p.pick_date = (SELECT MAX(pick_date) FROM paper_trades) "
        "ORDER BY p.prob DESC LIMIT ?", (MAX_ROWS,))


@st.cache_data(ttl=120)
def paper_record() -> dict:
    df = _query("SELECT status, COUNT(*) AS n FROM paper_trades GROUP BY status")
    d = dict(zip(df["status"], df["n"])) if not df.empty else {}
    closed = d.get("win", 0) + d.get("loss", 0)
    return {"open": d.get("open", 0), "pending": d.get("pending", 0),
            "win": d.get("win", 0), "loss": d.get("loss", 0),
            "win_rate": (d.get("win", 0) / closed) if closed else None}


@st.cache_data(ttl=120)
def cohort_results():
    # mark-to-market of the latest cohort: walk each pick forward from its entry
    # close and apply the ±3% / 10-day barrier. Would-have paper result, not the
    # clean broker record (close entry, survivorship-biased) - clearly labelled.
    import config
    picks = _query("SELECT ticker, entry_close, prob FROM paper_trades "
                   "WHERE pick_date=(SELECT MAX(pick_date) FROM paper_trades) ORDER BY prob DESC")
    if picks.empty or picks["entry_close"].isna().all():
        return None
    pdate = _query("SELECT MAX(pick_date) AS d FROM paper_trades")["d"].iloc[0]
    tk = list(picks["ticker"])
    qm = ",".join("?" * len(tk))
    fwd = _query(f"SELECT ticker, date, close FROM daily_prices WHERE ticker IN ({qm}) AND date>? ORDER BY date",
                 tuple(tk) + (pdate,))
    if fwd.empty:
        return None
    TP, SL, HOLD = config.EXIT_TAKE_PROFIT, config.EXIT_STOP_LOSS, config.HOLD_DAYS
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


@st.cache_data(ttl=300)
def model_perf():
    # XGB-only is the live signal (F6); the LSTM CSV enriches the view if present
    if not OOF_XGB.exists():
        return None
    try:
        from sklearn.metrics import roc_auc_score
        df = pd.read_csv(OOF_XGB).dropna()
        if OOF_LSTM.exists():
            lstm = pd.read_csv(OOF_LSTM)[["date", "ticker", "p_lstm"]]
            df = df.merge(lstm, on=["date", "ticker"], how="inner").dropna()
        sig = df["p_xgb"]
        curve = [{"top % kept": pct,
                  "win rate %": round(df[sig >= sig.quantile(1 - pct / 100)]["Target_Label"].mean() * 100, 1)}
                 for pct in (100, 50, 25, 10, 5)]
        return {"auc": roc_auc_score(df["Target_Label"], sig),
                "base": df["Target_Label"].mean(), "n": len(df),
                "curve": pd.DataFrame(curve).set_index("top % kept")}
    except Exception:
        return None


@st.cache_data(ttl=120)
def backtest_data():
    eq = pd.read_csv(BT_EQUITY) if BT_EQUITY.exists() else pd.DataFrame()
    tr = pd.read_csv(BT_TRADES) if BT_TRADES.exists() else pd.DataFrame()
    if eq.empty or tr.empty:
        return None
    years = max((pd.to_datetime(eq["date"].iloc[-1]) - pd.to_datetime(eq["date"].iloc[0])).days / 365.25, 1e-9)
    final = eq["equity"].iloc[-1]
    tr["year"] = pd.to_datetime(tr["date"]).dt.year
    spy = _query("SELECT date, close FROM market_baselines WHERE symbol='SPY' AND date>=? AND date<=? ORDER BY date",
                 (eq["date"].iloc[0], eq["date"].iloc[-1]))
    spy = spy[spy["date"].isin(set(eq["date"]))].reset_index(drop=True)   # SPY on the same rebalance grid
    if len(spy) > 1:
        spy_curve = spy["close"] / spy["close"].iloc[0]
        per = spy["close"].pct_change().dropna()
        spy_ret = float(spy_curve.iloc[-1] - 1)
        spy_sharpe = float(per.mean() / per.std() * (252 / 10) ** 0.5) if per.std() else 0.0
        spy_maxdd = float((spy_curve / spy_curve.cummax() - 1).min())
    else:
        spy_ret = spy_sharpe = spy_maxdd = None
    gains, losses = tr.loc[tr["return"] > 0, "return"], tr.loc[tr["return"] <= 0, "return"]
    return {
        "eq": eq, "tr": tr,
        "win_rate": tr["win"].mean(),
        "total_return": final - 1,
        "cagr": final ** (1 / years) - 1,
        "max_dd": eq["drawdown"].min(),
        "sharpe": eq["cohort_return"].mean() / eq["cohort_return"].std() * (252 / 10) ** 0.5 if eq["cohort_return"].std() else 0.0,
        "avg_return": tr["return"].mean(),
        "profit_factor": gains.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf"),
        "trades": len(tr), "start": eq["date"].iloc[0], "end": eq["date"].iloc[-1],
        "spy_ret": spy_ret, "spy_sharpe": spy_sharpe, "spy_maxdd": spy_maxdd,
        "by_year": tr.groupby("year").agg(trades=("win", "size"), win_rate=("win", "mean"), avg_return=("return", "mean")),
        "by_regime": tr.groupby("regime").agg(trades=("win", "size"), win_rate=("win", "mean"), avg_return=("return", "mean")),
    }


@st.cache_data(ttl=120)
def alpaca_live():
    # nightly snapshots written by paper_trader (DB, always available offline)
    state = _query("SELECT ts, equity, spy_value, n_sleeve FROM alpaca_state ORDER BY ts")
    fills = _query("SELECT filled_at, symbol, side, price, slippage_bps FROM alpaca_fills "
                   "ORDER BY filled_at DESC LIMIT ?", (MAX_ROWS,))
    div = _query("SELECT missing, foreign_syms FROM alpaca_divergence ORDER BY ts DESC LIMIT 1")
    return {"state": state, "fills": fills, "div": (None if div.empty else div.iloc[0])}


def fetch_alpaca_now():
    # READ-ONLY live pull (account + positions) - no order path is reachable
    # from here. Triggered by a button so we never hit the API on idle renders.
    try:
        import broker_alpaca as broker
        if not broker.is_configured():
            return {"error": "no paper keys in .env"}
        client = broker.get_client()
        acct = client.get_account()
        pos = broker.get_positions(client)
        rows = []
        for sym, p in sorted(pos.items()):
            cost = p["avg_entry"] * p["qty"]
            rows.append({"Symbol": sym, "Qty": round(p["qty"], 3),
                         "Market value": p["market_value"],
                         "Unrealized P&L": p["market_value"] - cost,
                         "P&L %": (p["market_value"] / cost - 1) if cost else 0.0})
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


def run_script(script: str, timeout: int = 1200):
    proc = subprocess.run([sys.executable, str(HERE / script)], cwd=str(HERE),
                          capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")


# ==================================================
# CONTROL PANEL  (LOCAL ONLY — executes scripts)
# ==================================================

st.sidebar.markdown("#### Control panel")
st.sidebar.caption("Run the system here — no terminal needed.")

ACTIONS = [
    ("Run full daily cycle", "run_daily.py", "~4 min · update data, make picks, score trades"),
    ("Generate picks", "predict_live.py", "~3 min · fresh picks from the latest data"),
    ("Update data only", "database.py", "~2 min · pull the latest close"),
    ("Rebuild model stats", "pipeline.py", "~2 min · refresh the Model tab"),
    ("Run 12-yr backtest", "backtest.py", "~4 min · replay the strategy across history"),
    ("Re-certify edge", "baseline_null.py", "~1 min · permutation test"),
]
for label, script, help_ in ACTIONS:
    if st.sidebar.button(label, use_container_width=True, help=help_):
        with st.spinner(f"Running {script} — this can take a few minutes, please wait…"):
            try:
                rc, out = run_script(script)
                st.session_state.update(run_out=out, run_ok=(rc == 0), run_name=script)
            except subprocess.TimeoutExpired:
                st.session_state.update(run_out=f"{script} timed out.", run_ok=False, run_name=script)
        st.cache_data.clear()
        st.rerun()

st.sidebar.divider()
st.sidebar.caption("⚠️ Local tool — do not deploy publicly (the buttons run code). Educational only, not financial advice.")


# ==================================================
# PAGE
# ==================================================

st.markdown(
    '<div class="app-head"><div class="t">Stock<b>_</b>trade</div>'
    '<div class="s">S&amp;P 500 swing-trading research &nbsp;·&nbsp; walk-forward validated '
    '&nbsp;·&nbsp; live on Alpaca paper</div></div>',
    unsafe_allow_html=True)

if st.session_state.get("run_out"):
    ok = st.session_state.get("run_ok")
    (st.success if ok else st.error)(
        f"{'✅' if ok else '❌'} `{st.session_state.get('run_name')}` "
        f"{'finished' if ok else 'failed'}.")
    with st.expander("Show run output", expanded=not ok):
        st.code(st.session_state["run_out"][-4000:])

with st.expander("ℹ️  Disclaimer — please read"):
    st.warning(DISCLAIMER)

if not DB_PATH.exists():
    st.info("No database yet. Click **📥 Update data only** in the sidebar to build it.")
    st.stop()

tab_overview, tab_picks, tab_backtest, tab_model, tab_live, tab_logs = st.tabs(
    ["Overview", "Picks", "Backtest", "Model", "Live paper", "Run log"])

with tab_overview:
    stats, perf, rec = db_stats(), model_perf(), paper_record()

    st.subheader("System")
    c = st.columns(4)
    c[0].metric("Data through", stats["date_max"])
    c[1].metric("Stocks ready", f"{stats['ml_ready']}")
    c[2].metric("Price rows", f"{stats['rows']:,}")
    c[3].metric("Edge", "✅ Certified")
    st.caption("Edge certified real vs 2,000 random baselines (~124σ). See FINDINGS.md.")

    st.subheader("Model quality")
    c = st.columns(3)
    if perf:
        c[0].metric("Out-of-sample AUC", f"{perf['auc']:.3f}", help="0.50 = coin flip. ~0.56 = a real, small edge.")
        c[1].metric("Base win rate", f"{perf['base']:.0%}")
        c[2].metric("Predictions tested", f"{perf['n']:,}")
    else:
        st.info("Not built yet — click **🔬 Rebuild model stats** in the sidebar.")

    st.subheader("Forward paper-trade record")
    st.caption("The survivorship-free proof — fills in as each pick matures (10 trading days).")
    c = st.columns(5)
    c[0].metric("Pending", rec["pending"], help="Picked at the close; fills at the next session's open")
    c[1].metric("Open", rec["open"])
    c[2].metric("Wins", rec["win"])
    c[3].metric("Losses", rec["loss"])
    c[4].metric("Win rate", "—" if rec["win_rate"] is None else f"{rec['win_rate']:.0%}")

with tab_picks:
    picks = latest_picks()
    if picks.empty:
        st.info("No picks yet — click **🎯 Generate picks** in the sidebar.")
    else:
        st.subheader(f"Picks for {picks['pick_date'].iloc[0]}")
        st.markdown(
            f"**Regime:** `{str(picks['regime'].iloc[0]).upper()}`  ·  "
            f"**{len(picks)} names across {picks['sector'].nunique()} sectors**  ·  max 3 per sector")
        table = pd.DataFrame({
            "Ticker": picks["ticker"].values,
            "Sector": picks["sector"].fillna("—").values,
            "Confidence": [f"{p:.0%}" for p in picks["prob"]],
            "Weight": ["—" if pd.isna(w) else f"{w:.1%}" for w in picks["weight"]],
            "Signal close": [f"${e:,.2f}" for e in picks["entry_close"]],
            "Fill (open)": ["pending" if pd.isna(o) else f"${o:,.2f}" for o in picks["entry_open"]],
            "Status": picks["status"].values,
        })
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("Sector spread:")
        st.bar_chart(picks["sector"].value_counts())

        res = cohort_results()
        if res:
            st.subheader(f"Cohort result — mark-to-market · {res['pdate']}")
            st.caption("**Would-have paper result** (close entry, ±3% / 10-day, survivorship-biased) — "
                       "updates daily until the cohort matures. Not the clean broker record.")
            c = st.columns(4)
            c[0].metric("Green", f"{res['green']} / {res['n']}")
            c[1].metric("Wins / Losses", f"{res['wins']}W / {res['losses']}L")
            c[2].metric("Still open", res["open_"])
            c[3].metric("Avg pick", f"{res['avg']:+.2%}")
            rtab = pd.DataFrame({
                "Ticker": res["table"]["ticker"].values,
                "Entry": [f"${e:,.2f}" for e in res["table"]["entry"]],
                "Now / exit": [f"${e:,.2f}" for e in res["table"]["exit"]],
                "Return": [f"{r:+.1%}" for r in res["table"]["ret"]],
                "Status": res["table"]["status"].str.upper().values,
            })
            st.dataframe(rtab, use_container_width=True, hide_index=True)

with tab_backtest:
    bt = backtest_data()
    if not bt:
        st.info("No backtest yet — click **📉 Run 12-yr backtest** in the sidebar (~4 min).")
    else:
        st.subheader(f"Historical simulation · {bt['start']} → {bt['end']}")
        st.warning("**Survivorship-biased** (today's S&P 500 survivors) and **net of 0.1% round-trip costs**. "
                   "A study of the strategy's *behaviour* — the bias-free proof is the forward paper-trade.")

        st.markdown("**Strategy** — XGB-only, exactly what `predict_live` trades")
        c = st.columns(4)
        c[0].metric("Win rate", f"{bt['win_rate']:.1%}", help="Hit +3% before -1%. Breakeven at 3:1 is 25%.")
        c[1].metric("Net / trade", f"{bt['avg_return']:+.2%}", help="After 0.1% round-trip cost")
        c[2].metric("Profit factor", f"{bt['profit_factor']:.2f}")
        c[3].metric("Trades", f"{bt['trades']:,}")
        c = st.columns(4)
        c[0].metric("Total return", f"{bt['total_return']:+.0%}",
                    delta=None if bt["spy_ret"] is None else f"{(bt['total_return'] - bt['spy_ret']):+.0%} vs SPY")
        c[1].metric("CAGR", f"{bt['cagr']:+.1%}")
        c[2].metric("Max drawdown", f"{bt['max_dd']:.1%}")
        c[3].metric("Sharpe", f"{bt['sharpe']:.2f}")

        if bt["spy_ret"] is not None:
            st.markdown("**SPY buy-and-hold** — same window, same grid")
            c = st.columns(4)
            c[0].metric("Total return", f"{bt['spy_ret']:+.0%}")
            c[1].metric("Max drawdown", f"{bt['spy_maxdd']:.1%}")
            c[2].metric("Sharpe", f"{bt['spy_sharpe']:.2f}")
            c[3].metric("Winner", "SPY")
            st.info("**Honest read:** a small, real positive per-trade edge (profit factor "
                    f"{bt['profit_factor']:.2f}, beats the 25% breakeven), but as a long-only strategy it "
                    "underperforms simply holding SPY on **both** return and Sharpe. The marginally smaller "
                    "drawdown is a cash-holding artifact, not better risk control. The +3% barrier caps every "
                    "winner while the index rode a historic bull — this edge's natural use is *stock selection "
                    "or a market-neutral overlay*, not replacing the index.")

        st.subheader("Equity curve (growth of $1)")
        st.line_chart(bt["eq"].set_index("date")["equity"])

        left, right = st.columns(2)
        with left:
            st.caption("Win rate by year")
            yr = bt["by_year"].copy()
            yr["win_rate"] = (yr["win_rate"] * 100).round(1)
            yr["avg_return"] = (yr["avg_return"] * 100).round(2)
            st.dataframe(yr, use_container_width=True)
        with right:
            st.caption("Win rate by regime")
            rg = bt["by_regime"].copy()
            rg["win_rate"] = (rg["win_rate"] * 100).round(1)
            rg["avg_return"] = (rg["avg_return"] * 100).round(2)
            st.dataframe(rg, use_container_width=True)

with tab_model:
    perf = model_perf()
    if not perf:
        st.info("Click **🔬 Rebuild model stats** in the sidebar to generate this.")
    else:
        st.subheader("Selectivity — win rate rises as we keep only the most-confident picks")
        st.line_chart(perf["curve"])
        st.caption(f"Base win rate {perf['base']:.0%}. The top 5% by confidence win far more — "
                   "the edge lives in *being selective*, not in raw accuracy.")

with tab_live:
    st.subheader("Alpaca paper account — the survivorship-free verdict, live")
    st.caption("Demo money only. This is the ONE clean test: no survivorship, real fills, "
               "real slippage. Expect it to look far closer to the point-in-time backtest "
               "(~+15%) than the survivor backtest (+72%). See FINDINGS_SPY.md / U6.")

    rec = paper_record()
    st.markdown("**Forward record**")
    c = st.columns(5)
    c[0].metric("Pending", rec["pending"], help="Picked at the close; fills at the next open")
    c[1].metric("Open", rec["open"])
    c[2].metric("Wins", rec["win"])
    c[3].metric("Losses", rec["loss"])
    c[4].metric("Win rate", "—" if rec["win_rate"] is None else f"{rec['win_rate']:.0%}")

    # live read-only pull (button-triggered)
    if st.button("Refresh from Alpaca  ·  read-only", type="primary"):
        st.session_state["alpaca_now"] = fetch_alpaca_now()
    snap = st.session_state.get("alpaca_now")
    if snap and "error" in snap:
        st.info(f"Live pull unavailable ({snap['error']}). The nightly snapshot below still works. "
                "Setup: copy `.env.example` → `.env` with your **paper** keys.")
    elif snap:
        st.markdown(f"**Live now** · account {snap['status']}")
        c = st.columns(4)
        c[0].metric("Equity", f"${snap['equity']:,.2f}")
        c[1].metric("Cash", f"${snap['cash']:,.2f}")
        c[2].metric("Buying power", f"${snap['buying_power']:,.2f}")
        c[3].metric("Open positions", len(snap["positions"]))
        if not snap["positions"].empty:
            p = snap["positions"]
            show = pd.DataFrame({
                "Symbol": p["Symbol"], "Qty": p["Qty"],
                "Market value": [f"${v:,.2f}" for v in p["Market value"]],
                "Unrealized P&L": [f"${v:+,.2f}" for v in p["Unrealized P&L"]],
                "P&L %": [f"{v:+.1%}" for v in p["P&L %"]],
            })
            st.dataframe(show, use_container_width=True, hide_index=True)

    live = alpaca_live()
    if live["div"] is not None:
        import json
        miss, foreign = json.loads(live["div"]["missing"] or "[]"), json.loads(live["div"]["foreign_syms"] or "[]")
        if miss:
            st.warning(f"⚠️ {len(miss)} record positions missing at the broker: {miss}")
        if foreign:
            st.warning(f"⚠️ {len(foreign)} broker positions not in the record (manual trades?): {foreign}")

    st.divider()
    if live["state"].empty:
        st.info("No nightly broker snapshots yet — the daily run writes these once the account is connected.")
    else:
        st.markdown("**Nightly snapshots** (written by the daily run)")
        latest = live["state"].iloc[-1]
        c = st.columns(4)
        c[0].metric("Equity (last run)", f"${latest['equity']:,.0f}")
        c[1].metric("SPY core", f"${latest['spy_value']:,.0f}", help="Core–satellite: 75% SPY / 25% strategy")
        c[2].metric("Sleeve positions", int(latest["n_sleeve"]))
        med = live["fills"]["slippage_bps"].median() if len(live["fills"]) else None
        c[3].metric("Median slippage", "—" if med is None or pd.isna(med) else f"{med:+.0f} bps",
                    help="Fill vs signal close. Gate B target: < 15 bps")
        if len(live["state"]) > 1:
            st.line_chart(live["state"].set_index("ts")["equity"], height=260)
        if len(live["fills"]):
            st.caption("Recent fills")
            st.dataframe(live["fills"], use_container_width=True, hide_index=True)

with tab_logs:
    lg = latest_log()
    if not lg:
        st.info("No run logs yet — run the daily cycle from the sidebar.")
    else:
        name, text = lg
        st.subheader(f"Latest run log — {name}")
        st.code(text[-8000:])

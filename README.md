# Stock_trade

A complete, self-running machine-learning system for swing-trading the S&P 500 — from
data collection, to a certified XGBoost ranker, to daily risk-managed picks that grade
themselves forward.

**📖 Full documentation — every file, every concept: [DOCS.md](DOCS.md)**

**Status: built, tested, certified, automated.** The full chain (data → features →
model → strategy → live picks → forward paper-trade) runs end-to-end with 48 passing
tests, CI on every push, and leakage guards at every layer.

**The honest headline:** the model's edge is **certified real** — it beats 2,000 random
baselines by ~124σ — but it is **small** (out-of-sample AUC ≈ 0.56), which is exactly
what an efficient large-cap market allows. The system's real achievement is knowing
*precisely* what it is. Full story: **[FINDINGS.md](FINDINGS.md)**.

---

## How it works

```
database.py     15yr daily OHLCV for ~500 S&P 500 stocks + 6 market baselines + SEC insider flow
      |            -> SQLite trading.db
      v
features.py     ~30 leakage-tested features (momentum, trend, volatility, macro, regime)
labels.py       triple-barrier target: +3% before -1% within 10 days (3:1 reward:risk)
      |
      +----------------------+
      v                      v
model_xgb.py            model_lstm.py
XGBoost (wide feats)    CNN-LSTM (lean 60-day sequences, PyTorch/GPU)
      |  P_xgb                |  P_lstm      (measured prediction corr 0.42 = diverse)
      +----------+-----------+
                 v
           ensemble.py     Platt calibration + logistic meta-learner
                 v
     threshold_analysis.py  confidence selectivity + bull/sideways/bear regime read
                 v
     strategy.py / backtest  fixed-fractional + vol sizing + Monte Carlo risk-of-ruin
                 v
     predict_live.py        daily diversified picks (sector cap + regime scaling)
                 v
     run_daily.py           the whole cycle, scheduled -> logs + scores paper_trades forward
```

Two deliberately different models (tabular trees vs a temporal network) make *different*
mistakes, so combining them is steadier than either alone. The strategy layer is kept
separate from the models so risk/reward changes without retraining.

---

## Setup

Requires **Python 3.11+**. The virtualenv and `trading.db` are gitignored (regenerable /
too large), so recreate them after cloning:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # pandas, yfinance, xgboost, scikit-learn, streamlit, pytest ...
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU build (RTX 50-series)
```

Then build the database (first run downloads ~15yr for 500 stocks, ~5-10 min):

```powershell
python database.py
```

---

## Usage

| Command | What it does |
|---|---|
| `python database.py` | Refresh `trading.db` with the latest close (incremental, ~60-90s) |
| `python pipeline.py` | Full 12-fold walk-forward for the XGBoost model → OOF predictions |
| `python run_lstm.py` | LSTM walk-forward + the XGBoost/LSTM diversity check |
| `python run_ensemble.py` | Calibrate + combine both models; confirm the ensemble beats either alone |
| `python baseline_null.py` | **Certify the edge is real** vs random baselines (permutation test) |
| `python predict_live.py` | Today's diversified, risk-scaled **picks** → logged to `paper_trades` |
| **`python run_daily.py`** | **The daily cycle:** update data → picks → score matured paper trades |

**Automation:** `run_daily.py` is registered with Windows Task Scheduler (`StockAI_Daily`)
to run every weekday at 22:00, so the system builds its own forward track record.
Change/remove it in Task Scheduler or `schtasks /Delete /TN StockAI_Daily`.

---

## What's in `trading.db`

| Table | One row per | Holds |
|---|---|---|
| `stocks` | ticker | company, sector, `ml_ready` flag |
| `daily_prices` | ticker + day | OHLCV |
| `market_baselines` | symbol + day | OHLCV for SPY / VIX / HYG / TNX / UUP / RSP |
| `insider_flow` | ticker + day | SEC Form 4 open-market buys/sells |
| `paper_trades` | pick + day | live picks logged for forward (survivorship-free) scoring |
| `run_log` | run | rows added, duration, size |

`tickers.csv` is the editable universe (add a `watchlist` row for a custom ticker).
Signal-layer constants (barriers, sequence length, 12 walk-forward folds, feature lists)
live in `config.py`.

---

## Testing & rigor

Quant bugs don't crash — they quietly produce great-looking but fake results. Guiding
rule: **a backtest that looks amazing is a leak to hunt, not a win to celebrate.**

- **34 tests**, run with `pytest -q`
- **Leakage guards:** features use past data only; scaler fits on train only per fold;
  time-based **purged + embargoed** 12-fold walk-forward; sequence models never cross
  ticker boundaries; a point-in-time test on every forward-looking feature
- **Edge certified:** `baseline_null.py` shows the model beats the best of 2,000 random
  baselines on both AUC and top-slice win rate — the edge is signal, not luck

**Known limitation — survivorship bias:** the universe is *today's* S&P 500, so delisted
names are missing and every backtest is optimistically biased. This is why the **forward
paper-trade** (`paper_trades`) exists — it's the only survivorship-free proof, and it's
accumulating now. See [FINDINGS.md](FINDINGS.md) for the three things survivorship hid.

---

## Status

- [x] Data pipeline (500 stocks, 15yr OHLCV, baselines, insider flow)
- [x] Feature engineering (leakage-tested)
- [x] Triple-barrier labels
- [x] XGBoost + CNN-LSTM ensemble (calibrated, diverse)
- [x] Purged walk-forward + regime/threshold analysis
- [x] Strategy layer (sizing, sector cap, regime scaling) + Monte Carlo
- [x] **Edge certified real** (permutation test)
- [x] Live pick-generator + forward paper-trade
- [x] Daily automation (scheduled)
- [ ] **Forward validation** — running now; needs weeks of live data
- [ ] Dashboard (optional) · survivorship-free backtest (the open frontier)

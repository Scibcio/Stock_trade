# Stock_trade

Investment-tracking AI to learn market patterns for swing trading.

**Project stage:** Data foundation **done**; signal layer **scaffolded** (module stubs in
place) and being built **test-first**. The ML models, ensemble, and dashboard are not
implemented yet — see the [Roadmap](#roadmap).

---

## What it does today

`database.py` builds and maintains a local SQLite database of historical price data
for the whole S&P 500, ready to train an ML model on later.

- Scrapes the full **S&P 500 (~503 stocks)** from Wikipedia on first run and saves the
  list to `tickers.csv` (editable afterwards — no code changes needed to add/remove stocks).
- Downloads **15 years of daily OHLCV** (since 2010) per stock via the `yfinance` API.
- Also tracks **6 market baselines** for market-wide context (see below).
- **Incremental:** first run downloads full history (~5-10 min); every run after only
  fetches the dates you're missing (~60-90 sec).
- Flags each stock `ml_ready` once it has enough history to train on (~2 years).
- Logs every run and prints a summary.

---

## Project structure

```
database.py      DONE   data collection (OHLCV + baselines -> trading.db)

  -- signal layer (scaffolded, being filled in test-first) --
config.py        shared constants (paths, barriers, SEQ_LENGTH, walk-forward folds)
features.py      trading.db -> feature table
labels.py        triple-barrier target (+ trailing-stop variant for phase 2)
model_xgb.py     Model A: XGBoost on the WIDE feature set (CPU)
model_lstm.py    Model B: CNN-LSTM on LEAN 60-day sequences (PyTorch)
ensemble.py      combine both models' probabilities (gate / blend / meta-learner)
strategy.py      score -> trades (config-driven risk:reward, position sizing)
pipeline.py      orchestrator: wires the whole signal layer together
```

The two models are deliberately different (tabular trees vs temporal network) so their
errors are decorrelated — combining them is more accurate and steadier than either alone.

---

## Setup

Requires **Python 3.11+**. The virtual environment and database are not in git
(they're regenerable / too large), so recreate them after cloning:

```powershell
# from the Stock_trade folder
python -m venv .venv
.venv\Scripts\activate
pip install pandas requests yfinance pytz lxml
```

Extra libraries are added when their stage arrives: `pytest` for tests, then
`xgboost` (Model A) and `torch` (Model B). They're lazy-imported, so the pipeline
imports fine before they're installed.

## Usage

Run after the US market closes (after 4pm ET / ~8pm UK) so the day's candle is final:

```powershell
python database.py
```

That single command does everything: creates the DB if needed, refreshes the ticker
list, downloads any missing data for all stocks and baselines, updates the `ml_ready`
flags, and prints a summary. Safe to run every day — it never duplicates data.

---

## What's in the database (`trading.db`)

| Table | One row per | Holds |
|---|---|---|
| `stocks` | ticker | company name, sector, source, `ml_ready` flag |
| `daily_prices` | ticker + day | OHLCV for each S&P 500 stock |
| `market_baselines` | symbol + day | OHLCV for the market-context instruments |
| `run_log` | script run | rows added, errors, duration, DB size |

`trading.db` grows to ~220 MB with full history, which is why it's gitignored
(GitHub's per-file limit is 100 MB). Anyone can rebuild it by running `database.py`.

---

## Managing your universe

`tickers.csv` is the editable list of what gets tracked. Columns:
`ticker, company_name, sector, source`.

- Add a custom ticker (e.g. a new IPO like SpaceX) by adding a row with
  `source` set to `watchlist` — it survives Wikipedia refreshes.
- Remove a stock by deleting its row.

## Market baselines

These give the future ML model market-wide context (is the market up? is volatility
high?). They live in the `BASELINES` dict in `database.py` — add a valid Yahoo Finance
symbol there and the next run downloads its full history automatically. No other changes
needed.

| Symbol | Meaning |
|---|---|
| `SPY` | S&P 500 ETF (overall market) |
| `^VIX` | Volatility / "fear" index |
| `HYG` | High-yield bond ETF (credit sentiment) |
| `^TNX` | 10-year Treasury yield (rates) |
| `UUP` | US Dollar bullish fund |
| `RSP` | S&P 500 equal-weight ETF |

## Configuration

Top of `database.py`:

- `HISTORY_START` — how far back to pull data (default `2010-01-01`).
- `ML_READY_MIN_ROWS` — minimum daily rows for a stock to count as ML-ready.
  Default `504` ≈ **2 years** of trading days (a year is ~252 trading days, not 365).
  Set higher to demand more history; the flag re-computes every run.

Signal-layer constants live in `config.py` (triple-barrier targets, sequence length,
and the 12 walk-forward folds).

---

## Testing

This is a quant project, so the bugs don't crash — they quietly produce great-looking
but fake results. The guiding rule: **a backtest that looks amazing is a leak to hunt,
not a win to celebrate.** Realistic edges are small.

Each pipeline stage has a **gate** that must pass before the next is built. We develop
and test on a small **~15-stock dev universe** first (seconds per run), then scale to
all ~500. Five things every build must satisfy:

| Goal | How it's verified |
|---|---|
| Runs correctly | Unit test per module + a small-universe smoke run |
| Models combine for real lift | Combined AUC **>** each model alone; predictions are decorrelated; meta-learner trained out-of-fold |
| No data bias / leakage | Features use **past data only**; scaler fits on **train only** (per fold); time-based walk-forward (never random shuffle); purge gap so a label's forward window can't cross the train/test boundary |
| Data full but manageable | Per-ticker coverage report; memory/size budget; dev on a subset, then scale |
| Real understanding, not luck | Beat baselines (coin-flip, majority class, buy-and-hold SPY); consistent across folds/regimes; probability calibration; Monte Carlo significance |

**Known limitation — survivorship bias:** the universe is *today's* S&P 500 (the
survivors), so delisted/dropped names are missing and backtests are optimistically
biased. Documented, not yet corrected.

Run the tests (once `tests/` exists):

```powershell
pip install pytest
pytest -q
```

---

## Roadmap

- [x] Data collection pipeline (S&P 500 OHLCV + market baselines, incremental)
- [x] ML-ready flagging
- [x] Signal-layer **scaffold** (module stubs + shared config + walk-forward folds)
- [ ] `features.py` — port the proven feature maths (test-first, with a leakage test)
- [ ] `labels.py` — triple-barrier target
- [ ] **Model A** — XGBoost on the wide feature set
- [ ] **Model B** — CNN-LSTM (PyTorch) on lean sequences
- [ ] **Ensemble** — combine both models; confirm it beats each alone
- [ ] **Strategy + evaluation** — risk:reward presets, port the FYP backtest suite
- [ ] **Dashboard** — Streamlit UI
- [ ] **Scheduler** — auto-run daily after market close

# Stock_trade

Investment-tracking AI to learn market patterns for swing trading.

**Project stage:** Data foundation. The data-collection pipeline is built and working.
The machine-learning model, trade signals, and dashboard are **not built yet** — see
the [Roadmap](#roadmap).

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

## Setup

Requires **Python 3.11+**. The virtual environment and database are not in git
(they're regenerable / too large), so recreate them after cloning:

```powershell
# from the Stock_trade folder
python -m venv .venv
.venv\Scripts\activate
pip install pandas requests yfinance pytz lxml
```

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

---

## Roadmap

- [x] Data collection pipeline (S&P 500 OHLCV + market baselines, incremental)
- [x] ML-ready flagging
- [ ] **Signal layer** — technical indicators → swing-trade candidates
- [ ] **ML model** — train on the `ml_ready` stocks to predict swing setups
- [ ] **Dashboard** — Streamlit UI (the `progress_fn` hook in `database.py` is already wired for it)
- [ ] **Scheduler** — auto-run daily after market close

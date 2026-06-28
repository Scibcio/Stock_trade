"""
--------------------------------------------
QUANT SYSTEM - DATABASE & DATA COLLECTOR
--------------------------------------------

Run after market close every weekday.

First run  : Downloads 2010 -> today for all stocks (~5-10 min)
Daily runs : Only fetches missing dates (~60-90 seconds)

- Creates a local SQLite file (trading.db) with four tables:
    stocks, daily_prices, market_baselines, run_log.
    Safe to call every run - only creates if not exist.

- On first run: scrapes Wikipedia for the full S&P 500 list and saves it to tickers.csv. 
  On every run after:
  reads that CSV so you can add/remove stocks without touching any code.

Warns you if the market is still open (today's
last candle would be incomplete). Does NOT block.

- Three utility functions used by everything else:
- get_last_stored_date : finds the newest date we already have for a ticker, so we only download what's missing.
- clean_yfinance       : fixes yfinance quirks (multi-index columns, timezone stamps, missing volume, etc.)
- insert_ohlcv         : bulk-writes rows to the DB, silently skipping any date already stored.

- Downloads SPY, VIX, HYG, TNX into market_baselines.
    These are used later as context features for the ML model.

- The main loop. Goes through every ticker, downloads only the dates we're missing, and saves them.
Commits every 20 stocks so a crash doesn't lose everything.

- Marks stocks as ml_ready (have >= 1000 rows of data), logs the run stats, and prints a summary.


run() ties all of the above together in order.
"""

import sqlite3
import time
import warnings
from pathlib import Path
from datetime import date, datetime
from io import StringIO

import pandas as pd
import requests
import yfinance as yf
import pytz

warnings.filterwarnings("ignore")


# ----------------------------------
# 1 - CONFIGURATION
# ----------------------------------
# DB_PATH      : where the SQLite database file lives (same folder as this script)
# TICKERS_CSV  : the ticker list file - auto-created on first run, edit freely after
# HISTORY_START: how far back we pull data (2010 gives ~15 years, plenty for ML)
# BASELINES    : market-wide instruments downloaded alongside individual stocks
# ML_READY_MIN_ROWS: a stock needs at least this many daily rows to be used for ML

DB_PATH       = Path(__file__).parent / "trading.db"
TICKERS_CSV   = Path(__file__).parent / "tickers.csv"
HISTORY_START = "2010-01-01"

BASELINES: dict[str, str] = {
    "SPY"  : "S&P 500 ETF",
    "^VIX" : "Volatility Index",
    "HYG"  : "High-Yield Bond ETF",
    "^TNX" : "10-Year Treasury Yield",
}

ML_READY_MIN_ROWS: int = 504


# ----------------------------------
# 2 - DATABASE SETUP
# ----------------------------------
# Four tables:
#   stocks          - one row per ticker (name, sector, ml_ready flag)
#   daily_prices    - one row per ticker per day (OHLCV)
#   market_baselines- same as daily_prices
#   run_log         - one row per script run
#
# WAL mode  : allows reading the DB while a write is in progress (faster)
# UNIQUE constraints prevent duplicate (ticker, date) rows from being inserted

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS stocks (
    ticker          TEXT    PRIMARY KEY,
    company_name    TEXT,
    sector          TEXT,
    source          TEXT    DEFAULT 'sp500',
    ml_ready        INTEGER DEFAULT 0,
    first_seen      TEXT    DEFAULT (date('now')),
    last_updated    TEXT
);

CREATE TABLE IF NOT EXISTS daily_prices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL,
    volume          INTEGER,
    UNIQUE(ticker, date),
    FOREIGN KEY(ticker) REFERENCES stocks(ticker)
);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
    ON daily_prices (ticker, date);

CREATE TABLE IF NOT EXISTS market_baselines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL, 
    volume          INTEGER,
    UNIQUE(symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_baselines_symbol_date
    ON market_baselines (symbol, date);

CREATE TABLE IF NOT EXISTS run_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at          TEXT    DEFAULT (datetime('now')),
    mode            TEXT,
    stocks_updated  INTEGER DEFAULT 0,
    rows_added      INTEGER DEFAULT 0,
    errors          INTEGER DEFAULT 0,
    duration_sec    REAL,
    db_size_mb      REAL,
    status          TEXT
);
"""

def setup_database() -> None:
    # Creates the .db file and all tables
    # executescript runs the whole SCHEMA string as a batch - safe to call every run.
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print("\nDatabase :\n")
    print(f"  {DB_PATH}")


# ----------------------------------
# TICKER UNIVERSE
# ----------------------------------

def get_sp500_tickers() -> list[dict]:
    """
    Returns the full ticker universe as a list of dicts:
        {ticker, company_name, sector, source}

    Logic:
      - If tickers.csv already exists -> read it directly.
        You can open this file and add/remove rows at any time.
        Add source='watchlist' for custom tickers (e.g. SpaceX after IPO).
      - If tickers.csv does not exist (first run) -> scrape the full S&P 500
        list from Wikipedia and save it as tickers.csv for next time.
    """
    if TICKERS_CSV.exists():
        df = pd.read_csv(TICKERS_CSV, dtype=str).fillna("")
        print(f"  Tickers : loaded from {TICKERS_CSV.name}  ({len(df)} rows)")
        return df.to_dict("records")

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        tables = pd.read_html(StringIO(resp.text), flavor="lxml")

        df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
        df.columns = ["ticker", "company_name", "sector"]

        # yfinance uses dashes not dots  e.g. BRK.B -> BRK-B

        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()
        df["source"] = "sp500"
        df.to_csv(TICKERS_CSV, index=False)

        print(f"  Tickers : saved {len(df)} tickers -> {TICKERS_CSV.name}")
        return df.to_dict("records")

    except Exception as exc:
        print(f"  Tickers : ERROR - could not fetch S&P 500 list: {exc}")
        return []


def save_ticker_universe(rows: list[dict]) -> list[str]:
    """
    Writes every ticker from the CSV into the stocks table in the database.
    Uses INSERT ... ON CONFLICT DO UPDATE so running this multiple times is safe -
    it never deletes a ticker, only updates company name/sector if they changed.
    Returns the plain list of ticker strings to hand to the download loop.
    """
    conn = sqlite3.connect(DB_PATH)

    sp500_count     = 0
    watchlist_count = 0

    for row in rows:
        source = row.get("source", "sp500")
        conn.execute("""
            INSERT INTO stocks (ticker, company_name, sector, source)
            VALUES (:ticker, :company_name, :sector, :source)
            ON CONFLICT(ticker) DO UPDATE SET
                company_name = excluded.company_name,
                sector       = excluded.sector,
                source       = excluded.source
        """,
        {"ticker": row["ticker"],
         "company_name": row.get("company_name", ""),
         "sector": row.get("sector", ""),
         "source": source})

        if source == "sp500":
            sp500_count += 1
        else:
            watchlist_count += 1

    conn.commit()
    conn.close()

    all_tickers = [r["ticker"] for r in rows]
    print(
        f"  S&P 500   = {sp500_count}\n"
        f"  Watchlist = {watchlist_count}\n"
        f"  Total     = {len(all_tickers)}"
    )
    return all_tickers


# ----------------------------------
# MARKET HOURS CHECK
# ----------------------------------

def check_market_status() -> bool:
    """
    Checks the current time in New York to see if NYSE is closed.
    If the market is still open we warn you.
    Never block the script - you might actually want that partial candle.
    Returns True if market is closed (safe to run), False if still open.
    """
    et = datetime.now(pytz.timezone("America/New_York"))
    weekday = et.weekday()

    print("\nMarket :\n")

    if weekday >= 5:
        print("  Closed (weekend)")
        return True

    after_close = et.hour > 16 or (et.hour == 16 and et.minute >= 5)

    if after_close:
        print(f"  Closed ({et.strftime('%H:%M')} ET)")
        return True

    print(f"  WARNING still open ({et.strftime('%H:%M')} ET)")
    return False


# ----------------------------------
# CORE HELPERS
# ----------------------------------

def get_last_stored_date(ticker: str,
                         table: str = "daily_prices",
                         symbol_col: str = "ticker",
                         conn: sqlite3.Connection | None = None) -> str:
    """
    Finds the most recent date we already have in the DB for this ticker.
    Returns that date so yfinance only downloads what's newer.
    If we have nothing yet, returns HISTORY_START (2010-01-01) to get
    the full history on the first download.

    Accepts an optional open connection - the stock loop passes one in so
    we don't open and close a connection for each of 500+ tickers.
    """
    _own_conn = conn is None
    if _own_conn:
        conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        f"SELECT MAX(date) FROM {table} WHERE {symbol_col} = ?", (ticker,)
    ).fetchone()

    if _own_conn:
        conn.close()

    last = row[0] if (row and row[0]) else None
    return last if last else HISTORY_START


def clean_yfinance(raw: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance has several known quirks depending on the ticker and version.
    This function handles all of them and returns a clean, consistent DataFrame
    with exactly these lowercase columns: date | open | high | low | close | volume

    Quirks handled:
      - MultiIndex columns  : yfinance sometimes returns ('Close', 'AAPL') style
                              headers - we flatten them to just 'Close'
      - Duplicate columns   : occasional yfinance bug - we drop the duplicates
      - Timezone in index   : yfinance returns UTC timestamps - we strip the
                              timezone and convert to plain YYYY-MM-DD strings
      - Missing volume      : instruments like ^TNX (Treasury yield) have no
                              trading volume - we fill those with 0
      - Empty result        : if yfinance returned nothing we return an empty
                              DataFrame so the caller can skip gracefully
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.reset_index(inplace=True)

    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[1] for col in df.columns]

    # Remove duplicate column names
    df = df.loc[:, ~df.columns.duplicated()]

    # Find the date column - could be named 'Date' or 'Datetime'
    date_col = next(
        (c for c in df.columns if c.lower() in ("date", "datetime")), None
    )
    if date_col is None:
        return pd.DataFrame()

    # Build the result in a fresh DataFrame to avoid column name collisions
    result = pd.DataFrame()
    result["date"] = (
        pd.to_datetime(df[date_col])
        .dt.tz_localize(None)        # strip UTC timezone
        .dt.strftime("%Y-%m-%d")     # -> plain YYYY-MM-DD string
    )

    col_lower = {c.lower(): c for c in df.columns}
    for target in ("open", "high", "low", "close", "volume"):
        source = col_lower.get(target)
        result[target] = df[source].values if source else None

    result.dropna(subset=["close"], inplace=True)
    result["volume"] = result["volume"].fillna(0).astype("int64")
    return result.reset_index(drop=True)


def insert_ohlcv(ticker: str,
                 df: pd.DataFrame,
                 conn: sqlite3.Connection,
                 table: str = "daily_prices",
                 symbol_col: str = "ticker") -> int:
    """
    Writes a batch of OHLCV rows into the database.
    INSERT OR IGNORE means if a (ticker, date) row already exists it's skipped
    silently - so running this twice never creates duplicates.
    Returns the number of genuinely new rows that were added.
    """
    if df.empty:
        return 0

    before = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {symbol_col} = ?", (ticker,)
    ).fetchone()[0]

    rows = [
        (ticker,
         r["date"],
         r.get("open"),
         r.get("high"),
         r.get("low"),
         r.get("close"),
         int(r.get("volume", 0)))
        for _, r in df.iterrows()
    ]

    conn.executemany(f"""
        INSERT OR IGNORE INTO {table}
            ({symbol_col}, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)

    after = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {symbol_col} = ?", (ticker,)
    ).fetchone()[0]

    return after - before


# ----------------------------------
# BASELINE COLLECTION
# ----------------------------------

def collect_baselines() -> int:
    """
    Downloads the market baseline instruments into market_baselines.
    These are NOT individual stocks - they give the ML model market-wide context:
      SPY  - tracks the whole S&P 500 (is the market up or down overall?)
      ^VIX - the "fear index" (is volatility high? big moves expected?)
      HYG  - high-yield bond ETF (how is credit risk sentiment?)
      ^TNX - 10-year Treasury yield (interest rate environment)

    Same incremental logic as stocks - only downloads dates not already stored.
    Uses a single shared DB connection for the whole loop.
    """
    print("\nBaselines :\n")
    conn      = sqlite3.connect(DB_PATH)
    total_new = 0

    for symbol, name in BASELINES.items():
        last = get_last_stored_date(symbol, "market_baselines", "symbol", conn)
        try:
            df = clean_yfinance(yf.download(symbol, start=last, progress=False, auto_adjust=True))
            added = insert_ohlcv(symbol, df, conn, "market_baselines", "symbol")
            conn.commit()
            total_new += added
            print(f"  {symbol:<6}  {name:<28}  +{added:>5} rows  (from {last})")
        except Exception as exc:
            print(f"  {symbol:<6}  ERROR: {exc}")

    conn.close()
    return total_new


# ----------------------------------
# STOCK COLLECTION
# ----------------------------------

def collect_stocks(tickers: list[str],
                   progress_fn=None) -> tuple[int, int]:
    """
    The main download loop - goes through every ticker in our universe and
    downloads any OHLCV data we're missing.

    Key design decisions:
      - ONE DB connection for the whole loop (500+ tickers would otherwise
        open and close the connection 500+ times)
      - Incremental by design - get_last_stored_date tells us where to start,
        so on daily runs we only pull one new candle per stock
      - Commits every 20 stocks - if the script crashes mid-run we don't lose
        all progress, just the last batch of up to 20
      - 0.1 second sleep between tickers - keeps Yahoo Finance from rate-limiting us
      - progress_fn hook - if a Streamlit dashboard is driving this it can pass
        a callback here to update a live progress bar

    Returns (total_new_rows, error_count).
    """
    total_new = 0
    errors    = 0
    t0        = time.time()

    conn = sqlite3.connect(DB_PATH)
    print(f"\nStocks ({len(tickers)} tickers) :\n")

    for i, ticker in enumerate(tickers, 1):
        try:
            last  = get_last_stored_date(ticker, conn=conn)
            df    = clean_yfinance(yf.download(ticker, start=last, progress=False, auto_adjust=True))
            added = insert_ohlcv(ticker, df, conn)
            total_new += added

            conn.execute(
                "UPDATE stocks SET last_updated = date('now') WHERE ticker = ?",
                (ticker,)
            )

            # Commit and print progress every 20 stocks
            if i % 20 == 0:
                conn.commit()
                elapsed = time.time() - t0
                eta     = (elapsed / i) * (len(tickers) - i)
                print(f"  [{i:>3}/{len(tickers)}]  "
                      f"{ticker:<8}  +{added:>5} rows  "
                      f"  ETA {eta/60:.1f} min")

            if progress_fn:
                progress_fn(i, len(tickers), ticker, added)

        except Exception as exc:
            errors += 1
            print(f"  [!] {ticker}: {exc}")

        time.sleep(0.1)

    conn.commit()
    conn.close()
    return total_new, errors


# ----------------------------------
# POST-COLLECTION TASKS
# ----------------------------------

def mark_ml_ready() -> int:
    """
    After downloading, flags which stocks have enough history to train on.
    A stock needs at least ML_READY_MIN_ROWS daily rows. Each row is one
    TRADING day (~252 per calendar year), so the current 504 is ~2 years.

    Why 2 years and not 1: young stocks (recent IPOs/spin-offs) have only
    lived through a single market regime. A stock like SanDisk that has only
    ever existed during a memory-price boom has a one-directional "up" history;
    training on it teaches the model a biased pattern. Requiring ~2 years keeps
    that data-sparse, regime-biased history out of the training set.

    Recomputes the flag in BOTH directions every run: 1 for stocks that now
    qualify, 0 for any that no longer do. This makes it idempotent and correct
    even if ML_READY_MIN_ROWS is changed later.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"""
        UPDATE stocks SET ml_ready = CASE
            WHEN ticker IN (
                SELECT ticker FROM daily_prices
                GROUP BY ticker
                HAVING COUNT(*) >= {ML_READY_MIN_ROWS}
            ) THEN 1 ELSE 0 END
    """)
    n = conn.execute(
        "SELECT COUNT(*) FROM stocks WHERE ml_ready = 1"
    ).fetchone()[0]
    conn.commit()
    conn.close()
    return n


def get_db_size_mb() -> float:
    return round(DB_PATH.stat().st_size / 1_048_576, 2) if DB_PATH.exists() else 0.0


def log_run(mode: str,
            stocks_updated: int,
            rows_added: int,
            errors: int,
            duration: float,
            status: str) -> None:
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO run_log
            (mode, stocks_updated, rows_added, errors,
             duration_sec, db_size_mb, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    (mode, stocks_updated, rows_added, errors, round(duration, 1), get_db_size_mb(), status))
    conn.commit()
    conn.close()


def print_summary(total_new: int, errors: int, duration: float) -> None:

    conn      = sqlite3.connect(DB_PATH)
    n_stocks  = conn.execute("SELECT COUNT(DISTINCT ticker) FROM daily_prices").fetchone()[0]
    n_rows    = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    date_min  = conn.execute("SELECT MIN(date) FROM daily_prices").fetchone()[0]
    date_max  = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()[0]
    n_ml      = conn.execute("SELECT COUNT(*) FROM stocks WHERE ml_ready = 1").fetchone()[0]
    n_base    = conn.execute("SELECT COUNT(*) FROM market_baselines").fetchone()[0]
    conn.close()

    date_range = f"{date_min}  ->  {date_max}" if date_min else "No data yet"

    print("\n" + "-" * 52)
    print("DATABASE SUMMARY")
    print("-" * 52)
    print(f"Stocks tracked   : {n_stocks}")
    print(f"ML-ready stocks  : {n_ml}  (>= {ML_READY_MIN_ROWS} rows)")
    print(f"Price rows total : {n_rows:,}")
    print(f"Baseline rows    : {n_base:,}")
    print(f"Date range       : {date_range}")
    print(f"New rows added   : +{total_new:,}")
    print(f"Errors           : {errors}")
    print(f"Duration         : {duration:.0f}s")
    print(f"Database size    : {get_db_size_mb()} MB")
    print("-" * 52)


# ----------------------------------
# ENTRY POINT
# ----------------------------------
# run() is the one function you call to do everything.
# It's safe to call every day - all steps are incremental.
# Import it from a dashboard or scheduler, or just run this file directly.

def run(progress_fn=None) -> bool:
    t_start = time.time()

    print("\n" + "-" * 52)
    print("QUANT SYSTEM - DATA COLLECTION")
    print("-" * 52)
    print(f"Date : {date.today()}")

    setup_database()
    check_market_status()

    print("\nUniverse :\n")
    ticker_rows = get_sp500_tickers()
    if not ticker_rows:
        print("  [!] No tickers fetched - check your internet connection")
        return False

    all_tickers   = save_ticker_universe(ticker_rows)
    collect_baselines()
    total_new, errors = collect_stocks(all_tickers, progress_fn)
    n_ready       = mark_ml_ready()
    duration      = time.time() - t_start
    status        = "ok" if errors == 0 else "partial"

    log_run("incremental", len(all_tickers) - errors,
            total_new, errors, duration, status)
    print_summary(total_new, errors, duration)
    print(f"{n_ready} stocks are ML-ready\n")
    return True


if __name__ == "__main__":
    run()

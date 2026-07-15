# Stock_trade — MAX profile · Mac setup

A self-contained live-only package: the certified model, tuned aggressive (top 5,
uncapped winners, ~1-month hold, fully invested, 100% strategy), trading a **second
Alpaca paper account**. No backtesting, no research files — just predict → trade → watch.

> ⚠️ **Demo money only.** These settings are the most survivorship-inflated in the
> project; live returns will be a fraction of the backtest. That's the experiment.

---

## 1. Copy the folder

Copy this whole `mac_live/` folder to your Mac (e.g. `~/stock_max/`).

## 2. Python environment

```bash
cd ~/stock_max                       # wherever you put it
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # ~2 min, no GPU / torch needed
```

## 3. A SECOND Alpaca paper account

Keep this separate from your other account so the experiments never mix.

1. Sign in at **alpaca.markets** → create/select a paper account → **switch to Paper trading** → generate an **API Key ID + Secret**.
2. Create the `.env` file:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and paste your **paper** keys (they start with `PK` — the code refuses anything else):
   ```
   APCA_API_KEY_ID=PK...
   APCA_API_SECRET_KEY=...
   APCA_PAPER=true
   ```

## 4. Build the data (first run only)

```bash
python database.py       # pulls ~500 stocks from yfinance, ~3 min -> trading.db
```

## 5. First live run

```bash
python run_daily.py      # update data -> reconcile -> score -> make picks -> queue orders
```
Picks are queued as market orders that fill at the **next open**. A new 5-name book is
made every ~20 sessions (about monthly); in between, the daily run just scores and reconciles.

## 6. Watch it

```bash
streamlit run dashboard.py    # opens http://localhost:8501
```
Tabs: **Overview** (the strategy + forward record), **Picks** (current book + mark-to-market),
**Model**, **Live paper** (hit *Refresh from Alpaca*), **Database**, **Run log**.

## 7. Automate it (run every weeknight)

Pick ONE. Use `launchd` (Mac-native) or `cron`. Adjust the paths to your folder.

**launchd** — save as `~/Library/LaunchAgents/com.stocktrade.max.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.stocktrade.max</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/YOU/stock_max/.venv/bin/python</string>
    <string>/Users/YOU/stock_max/run_daily.py</string>
  </array>
  <key>WorkingDirectory</key><string>/Users/YOU/stock_max</string>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>17</integer><key>Minute</key><integer>5</integer></dict>
</dict></plist>
```
Then: `launchctl load ~/Library/LaunchAgents/com.stocktrade.max.plist`
(17:05 = 5:05pm local — set it to shortly after the US close in your timezone.)

**cron** alternative — `crontab -e`, add (weekdays 17:05):
```
5 17 * * 1-5  cd /Users/YOU/stock_max && ./.venv/bin/python run_daily.py
```
The Mac must be **awake** at that time for the job to fire.

---

## Safety (built in)

- **Paper-locked by construction** — the broker code raises (won't run) unless the key is a paper key and the endpoint is the paper API. No path to real money.
- **Kill switch** — create a file named `HALT` in this folder to block all new buy orders instantly (exits still allowed). Delete it to resume.
- **Weekend guard** — orders are deferred when the next session is >30h away (no holiday-weekend cancellations).
- **Keep `.env` private** — it's the only secret; never commit or share it.

## What's different from the main system

| | Main (conservative) | This (MAX) |
|---|---|---|
| Names | 15 | **5** |
| Exit | ±3% / 10-day | **uncapped, −15% stop, ~20-day** |
| Exposure | bear = cash | **100% always** |
| Portfolio | 75% SPY + 25% strategy | **100% strategy** |

Same certified 10-day ranker underneath — only how its picks are *traded* changes.

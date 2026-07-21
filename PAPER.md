# Anatomy of a Small Edge: Building and Honestly Testing a Machine-Learning Stock-Trading System for the S&P 500

**Author:** Michal Scibak
**System:** `Stock_trade` — an end-to-end ML research and paper-trading platform
**Date:** June 2026
**Repository:** github.com/Scibcio/Stock_trade

---

## Abstract

I built a system that ranks all ~500 S&P 500 stocks every day, buys the 15 it likes most, and — most importantly — tries hard to *prove itself wrong*. A gradient-boosted model reaches an out-of-sample AUC of **0.5645**, and a permutation test confirms this is a real signal, not luck. But the same rigor that confirms the signal also deflates it. Two deeper tests show the edge is mostly an illusion of two well-known traps: **survivorship bias** (only studying stocks that are still in the index today) explains most of the backtest's profit, and the edge is **market beta, not skill** (the model just picks stocks that rise when the whole market rises). Strip both away and the true, market-independent return is roughly **zero**. The real result isn't a money-maker — it's a repeatable method for telling a *useless* real edge from a *tradeable* one, and a system honest enough to expose its own weakness before any money is at risk.

---

## 1. What this project is

Most retail trading bots show amazing backtests and then lose money live. The usual reasons are known: look-ahead bias, survivorship bias, overfitting, and ignoring trading costs. This project flips the usual goal. Instead of chasing the biggest backtest number, it chases **confidence in what the number actually means** — and treats every big jump in results as a likely bug, not a win.

**The core question:** the S&P 500 is a big, efficient market, so any edge from price data alone must be tiny. The job is not to inflate it but to (1) prove it's real, (2) figure out exactly what it is, and (3) check if it survives honest costs, honest execution, survivorship, and market movement.

---

## 2. Data and method

**Data.** ~1.96M daily price rows for 500 S&P 500 stocks (2010–2026), plus six market-context instruments (S&P index, volatility index, bond and dollar proxies) used only as model *inputs*, never traded. Everything lives in a SQLite database.

**Features.** ~30 signals per stock per day (momentum, moving-average distances, RSI, volatility, beta, etc.), all computed from *past* data only. Leakage is unit-tested.

**Label (what the model predicts).** For each stock-day: will the price rise **+3% before it falls −1%**, within **10 days**? This is a "triple-barrier" label. The asymmetry (needs +3% but only risks −1%) turns out to matter a lot (see §5).

**Validation — the honesty backbone.** I use *purged walk-forward* testing: 12 folds, each trained only on data ending at least two weeks before its test year. So the model is always graded on a future it never saw. Every result below uses these out-of-fold (OOF) predictions.

**Model.** XGBoost (gradient-boosted trees) is the live model. A CNN-LSTM neural net was also built but *dropped* from live trading — it made the top picks slightly worse, so lean won.

**Certification.** To rule out luck, a permutation test shuffles the labels 2,000 times and asks: does the real result beat every fake one? This gate must pass before any signal is trusted.

---

## 3. The strategy

- **Selection:** each 10 trading days, take the top 15 stocks by model score, cap at 3 per sector, and size each position by inverse volatility (so each name risks about the same).
- **Market regime:** tag the market as bull / sideways / bear (from the index vs its 200-day average and the volatility index). Invest 100% / 60% / **0%**. Bear = 0% because the model has *no* ranking skill in bear markets (its bear-fold AUC is 0.49 — a coin flip).
- **Exits:** sell at ±3% or after 10 days (chosen by a formal test; a tight −1% stop kept selling winners too early).
- **Portfolio:** live money is 75% index fund + 25% strategy. This mix is the only one that beat the index on risk-adjusted return.
- **Two non-negotiable honesty rules:** (1) you can't buy at the closing price the signal is computed from, so every trade fills at the *next day's open*; (2) charge 10 bps per round-trip. Both cut the reported returns — and both are kept.

---

## 4. Results

**Is the signal real?** Yes.

| Test | Real result | Best of 2,000 random | Verdict |
|---|---|---|---|
| AUC vs shuffled labels | **0.5645** | 0.5018 | ~124σ — real |
| Top-5% win rate vs random | **44.4%** | 33.5% | ~67σ — real |

AUC of 0.5 is a coin flip; 0.56 is a small but genuine edge. The base win rate is ~33%, but the model's most-confident 5% of picks win ~44% — so being *selective* is where the value is.

**Does it make money?** Backtest 2014–2026 (honest entries, ±3% exits, costs, bear = cash):

| Book | Total return | Yearly (CAGR) | Max drawdown | Sharpe |
|---|---|---|---|---|
| **Strategy** | **+72%** | +4.7% | −20.3% | **0.52** |
| Just buying the S&P 500 | **+322%** | +12.8% | −24.9% | **0.86** |

Each trade wins on average, but the strategy **loses to simply buying the index** — on both return and Sharpe (return per unit of risk). The honesty rules cost a lot: with the old (cheating) same-day entry the backtest showed +77%; fixing it to next-day open cut that roughly in half.

---

## 5. Testing everything (and killing most of it)

I ran ~15 experiments, each with a pass/fail rule set in advance. Most **failed** — and that's the point. A few examples:

- Adding news sentiment, insider trades, longer horizons, cross-sectional ranks → no lift (the model already extracts nearly all the signal in price data).
- A neural net in the live path → made the top slice worse.
- An earnings-avoidance filter → cut returns 21% with no benefit.
- Trading more often → 3–21× the fees for the same or worse result.
- A "meta-model" to filter trades → real on paper, but noise on the actual picks.

Two lessons kept repeating. First, **a higher AUC did not mean a better portfolio** — one model scored *better* on AUC but made *less* money, because the top-15 book, not the average prediction, is what you actually trade. Second, **you can't relabel the edge into something better** — every attempt broke it.

---

## 6. The two findings that changed everything

### 6.1 Survivorship: most of the "profit" was cheating without knowing it

The stock list is *today's* index members. But some of them only joined recently. Trading them in 2014 secretly uses future knowledge (that they'd become winners). Using real, historical index membership:

| Universe | Total return | Sharpe |
|---|---|---|
| Today's survivors (naïve) | +72% | 0.52 |
| **Only real members at the time** | **+15%** | **0.17** |

Fixing just this one bias erased ~80% of the return. And it's the *optimistic* case — 275 delisted stocks (mostly losers) are missing from the data entirely, which would drag it lower still.

### 6.2 Beta vs alpha: the edge is just riding the market

**Beta** = how much a book moves with the market. **Alpha** = return you earn *beyond* the market. If I hedge out the strategy's market exposure correctly:

| Book | Total return | Yearly | Market correlation |
|---|---|---|---|
| Unhedged | +72% | +5.6% | 0.61 |
| **Market-neutral (hedged)** | **−19%** | **−2.1%** | **~0.00** |

Once the market is removed, the return is **negative**. In plain terms: every dollar of profit came from the fact that the market went up and the model picked high-beta stocks that went up with it. The model's real skill is *picking stocks that ride the market*, not beating it. This one fact explains everything — why the signal is real yet loses to the index.

---

## 7. Live deployment

The system runs itself on a broker's **paper** (fake-money) account via a nightly scheduled job. Safety is built in: it can *only* connect to the paper endpoint (the code refuses anything else), it has a kill-switch file, per-order size caps, and a nightly check that the broker's positions match its own records. An independent audit verified the safety rails and caught two real bugs, both fixed. Because this account is survivorship-free and uses real fills, it's the one truly clean test — and it's now collecting data.

---

## 8. What it means, limits, and conclusion

**What it is:** a certified-but-small, survivorship-inflated, beta-driven momentum strategy with essentially **zero market-independent skill**. Honestly, it's best used long-only as a small satellite next to an index fund — never as a market-beating machine.

**Limits:** the survivorship fix is partial (delisted prices are gone); the hedge ratio is measured in-hindsight; and 2014–2026 was one long bull market, which flatters any beta-driven book.

**Conclusion:** the goal was never the biggest number — it was understanding what a real edge on the S&P 500 actually is. The answer, earned through 15 tested-and-mostly-killed ideas and two self-inflicted reality checks, is honest and precise: a tiny, real, but non-tradeable edge. The strategy doesn't beat the market. The **method** — purged testing, statistical certification, a kill-list of everything that failed, and two attacks on the system's own best result — is the actual contribution. It succeeded by proving its own edge illusory *before* risking a cent.

---

*Reproduce: `python pipeline.py` (predictions) → `baseline_null.py` (certify) → `backtest.py` (simulate) → `run_pit_universe.py` (survivorship) → `run_beta_hedge.py` (beta). Full journal: `FINDINGS.md`. 62 automated tests.*

# Anatomy of a Small Edge: A Rigorous, Falsification-First Study of a Machine-Learning Swing-Trading System on the S&P 500

**Author:** Michal Scibak · Final-Year Project
**System:** `Stock_trade` — an end-to-end ML research and paper-trading platform
**Date:** July 2026
**Repository:** github.com/Scibcio/Stock_trade

---

## Abstract

We build a complete machine-learning system that ranks all ~500 constituents of the S&P 500 each trading day, trades the fifteen highest-conviction names as a risk-controlled portfolio, and — critically — subjects its own apparent edge to a battery of falsification tests. Using purged, embargoed walk-forward validation over 2014–2026 (~1.45M out-of-fold predictions), a gradient-boosted model achieves an out-of-sample AUC of **0.5645**, which a permutation test certifies as genuine at ~124σ above a 2,000-sample null. Yet the same discipline that certifies the signal also dismantles the naïve interpretation of it. Two capstone analyses show that (i) **survivorship bias** accounts for the majority of the backtested return — restricting to point-in-time index membership cuts total return from +72% to +15% — and (ii) the edge is **market beta, not alpha**: a correctly-calibrated hedge that neutralises the book's realised beta leaves a market-neutral return of −2%/yr. We conclude that the system is a *smart-beta / momentum* strategy with near-zero market-independent alpha, honestly deployable only long-only as a satellite allocation. The contribution is methodological: a reproducible template for distinguishing a real-but-useless edge from a tradeable one, and a demonstration that a system rigorous enough to *falsify itself* is more valuable than one that merely reports a flattering backtest. The system runs live on broker paper-trading (Alpaca), where a survivorship-free forward record now accumulates as the only clean verdict.

---

## 1. Introduction

Retail algorithmic trading is dominated by backtests that look extraordinary and fail live. The usual culprits — look-ahead bias, survivorship bias, overfitting, and cost-blind execution — are well known yet rarely all controlled at once. This project inverts the usual objective. Rather than maximising a backtest number, it maximises **confidence in what the number means**, and treats every large improvement as a suspected bug rather than a discovery.

**Thesis.** A liquid, efficient market (the S&P 500) permits at most a thin technical edge. The scientific task is not to inflate it but to (a) prove it is real, (b) characterise exactly what it is, and (c) determine whether it is tradeable after honest execution, costs, survivorship, and market-beta are accounted for.

**Contributions.**
1. An end-to-end, leakage-audited pipeline: data → features → labels → purged walk-forward models → statistical certification → strategy → backtest → live paper deployment.
2. A **falsification programme**: fifteen experiments, each pre-registered with an acceptance rule and adjudicated against a null, most of which *failed* and are documented as such.
3. Two capstone analyses that quantify **survivorship inflation** and **beta versus alpha** on the system's own best result.
4. A live, self-healing broker deployment with an independent execution audit.

---

## 2. Data

| Item | Detail |
|---|---|
| Universe | 500 ML-ready S&P 500 constituents (≥504 daily rows) |
| Price history | ~1.96M daily OHLCV rows, 2010-01-04 → 2026-07 |
| Market baselines | SPY, ^VIX, HYG, ^TNX, UUP, RSP (macro context, *not* tradeable) |
| Source | yfinance (equities + baselines), SEC EDGAR (Form 4 insider), fja05680/sp500 (point-in-time membership) |
| Storage | SQLite (`trading.db`, ~220 MB) |

A deliberate distinction is enforced throughout: the **six baseline instruments** are model *inputs*; the **503 equities** are the tradeable *universe*. They live in separate tables and never merge — a common silent source of universe leakage that we explicitly test against.

---

## 3. Methods

### 3.1 Feature engineering

~30 features per stock per day, all strictly backward-looking (rolling / EWM / shifted / differenced): daily and multi-horizon returns; SMA/EMA distances (scale-free "rubber-band"); RSI; Bollinger width and position; MACD (price-normalised); realised-volatility z-scores; normalised ATR (NATR); 20-day rolling beta to SPY; VWAP distance; and macro context (VIX level/change, credit-spread and rate proxies). Raw price *levels* are deliberately excluded — a lesson inherited from prior work in which absolute-value features degraded generalisation. Leakage is unit-tested.

### 3.2 Labelling — the triple barrier

Each stock-day is labelled 1 if price reaches **+3%** before **−1%** within a **10-day** horizon, else 0 (first barrier touched decides; incomplete forward windows are left NaN and dropped, never faked). The asymmetry is intentional and, as §5.3 shows, is itself the source of the selection edge. Reward:risk is 3:1, so breakeven is 25%.

### 3.3 Purged, embargoed walk-forward validation

Twelve expanding folds, each training on data ending ≥14 calendar days before its one-year test window (the purge exceeds the 10-day label horizon, so no training label straddles the boundary). Every prediction is therefore genuinely **out-of-fold (OOF)** — scored on a future the model never saw. All downstream results consume OOF predictions only.

### 3.4 Models

- **Primary (live):** XGBoost classifier (300 trees, depth 4) on the wide feature set, one model per fold. Chosen for speed, robustness, and interpretability.
- **Secondary (research):** a PyTorch CNN-LSTM on 60-day sequences of a lean feature set, per-ticker scaled with train-only fit. It is *excluded* from the live path (§5.3) because it dilutes the traded top-slice.

### 3.5 Statistical certification

To separate signal from luck, a permutation test (`baseline_null.py`) compares the real OOF result against 2,000 shuffled-label / random-selection nulls. Any label change re-runs it. This is the gate every candidate signal must pass.

---

## 4. Strategy construction

### 4.1 Cohort selection

One book of **TOP_K = 15** names is formed per **10 trading sessions** (matching the hold horizon). Selection: rank by model probability, cap at **3 per sector**, size by **inverse-NATR** (volatility targeting → roughly equal risk per name). A single shared function drives both the backtest and the live system, guaranteeing parity by construction rather than by discipline.

### 4.2 Regime detection

Market state is tagged from SPY vs its 200-day moving average and the VIX: **bull / sideways / bear**. Gross exposure scales **100% / 60% / 0%**. Bear = 0% is empirically justified: the model's bear-fold AUC collapses to 0.490 (no ranking skill), so any bear exposure is edge-less risk.

### 4.3 Exit geometry

Adopted after a formal sweep (§5.3): **symmetric ±3%** with a 10-day time barrier. A tight −1% stop was found to whipsaw out of names that subsequently recover.

### 4.4 Portfolio mode — core–satellite

Live capital is 75% SPY core + 25% strategy sleeve, under a $10k notional cap (realistic small-account behaviour). This is the only configuration in testing whose Sharpe exceeds SPY's, via the strategy's ~0.49 correlation to the index.

### 4.5 Execution honesty

Two non-negotiable disciplines: (i) **next-open entries** — a signal computed on day *D*'s close cannot be transacted at that close, so all fills occur at *D+1*'s open; (ii) **costs** — 10 bps round-trip per position, applied everywhere. Both materially reduce reported returns (§5.2) and are treated as ground rules, not options.

---

## 5. Results

### 5.1 Predictive performance and certification

| Test | Real | Best of 2,000 nulls | Verdict |
|---|---|---|---|
| OOF AUC vs shuffled-label null | **0.5645** | 0.5018 | ~124σ — **certified real** |
| Top-5% win rate vs random selection | **44.4%** | 33.5% | ~67σ — **certified real** |

The edge is genuine but small — the honest ceiling for efficient large-cap price data. It lives in *selectivity*: the base win rate is ~33%, but the most-confident 5% of picks win ~44%.

### 5.2 Honest backtest versus benchmark

Deployable strategy, XGB-only, next-open entries, ±3% exits, 10 bps, bear = cash, 2014–2026:

| Book | Pos-rate | Net/trade | Total | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|---|---|
| **Strategy** | 53.7% | +0.32% | **+72%** | +4.7% | −20.3% | **0.52** |
| SPY buy-and-hold | — | — | **+322%** | +12.8% | −24.9% | **0.86** |

The strategy has a positive per-trade expectation (profit factor > 1) but **underperforms the index** on both return and risk-adjusted return. Its marginally shallower drawdown is a cash-holding artifact, not superior risk control. The **execution-honesty tax** is large: booking entries at the untradeable signal-day close would have reported +77% (Sharpe 0.60); the honest next-open figure at the same 3:1 exits is +36% (Sharpe 0.37) — roughly half the headline was an artifact.

### 5.3 The experiment programme (falsification-first)

Fifteen pre-registered experiments; most failed. Each is documented with numbers in the kill-list.

| # | Experiment | Result | Verdict |
|---|---|---|---|
| — | Sentiment / news (LLM/FinBERT) | dead-zone at 10-day horizon | rejected |
| — | Insider flow (SEC Form 4) | −0.002 AUC (horizon mismatch) | rejected |
| — | Longer horizons (20/40/60d) | AUC 0.57 → 0.51 | rejected |
| — | Cross-sectional ranks | +0.001 (noise) | rejected |
| — | CNN-LSTM in live path | dilutes the top slice (45.7%→43.2%) | research only |
| U1 | Earnings blackout | tail unchanged, −21pp total | rejected |
| U2 | Trailing-label retrain | AUC 0.4934 (uncertified) | rejected |
| U3 | Symmetric-label retrain | AUC 0.591 but book +31%/Sh 0.28 | rejected |
| U3 | **Exit geometry sweep** | ±3% best under DD gate | **adopted** |
| U8 | De-beta the label (ATR / residual) | AUC → ~0.51 (edge dies with beta) | rejected |
| U5 | Meta-labeling | +5.6σ on pool, +1.3σ on traded picks | rejected |
| — | Cadence (faster entry) | Sharpe 0.36–0.38 vs 0.50, 3–21× turnover | rejected |
| U9 | **Core–satellite portfolio** | Sharpe > SPY via 0.49 corr | **adopted** |
| U6 | **Point-in-time universe** | +72% → +15% | *capstone* (§6.1) |
| U7 | **Beta-neutralization** | market-neutral −2%/yr | *capstone* (§6.2) |

Two methodological lessons recur. First, **pooled AUC is not the objective**: a symmetric-label model with a *higher* certified AUC (0.591) produced a *worse* traded book (+31% vs +72%), because the 3:1 label's asymmetry is precisely what selects the sharp near-term momentum a top-slice book monetises. Second, the edge cannot be *relabelled* into something better: every attempt to remove the beta tilt from the label destroyed the edge.

---

## 6. Two capstone analyses

### 6.1 Survivorship (U6): most of the return was look-ahead

The tradeable universe is *today's* survivors. Using verified point-in-time membership (fja05680/sp500, 1996–2026):

- **Coverage:** 771 names were index members during 2014–2026; the survivor dataset holds prices for only **496 (64%)**. The **275 missing** names (dropped/delisted, disproportionately losers) inflate the book and are unrecoverable without their price history.
- **Addition look-ahead:** restricting to members-as-of-each-date removes 18.6% of candidate rows and collapses performance:

| Universe | Total | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|
| Survivor (naïve backtest) | +72% | +4.7% | −20.3% | 0.52 |
| **Point-in-time members** | **+15%** | **+1.2%** | −14.4% | **0.17** |

Removing only the *fixable* bias erases ~80% of the return; the remaining 36% of missing (losing) names would depress it further. Mechanism: the names *added* to the index over the window were rising stars, and trading them before they qualified is exactly the look-ahead survivorship injects.

### 6.2 Beta versus alpha (U7): the edge is beta

The book's raw entry-beta averages **1.60**, but the ±3% barriers truncate co-movement, so its *realised* beta is only **0.54**. Hedging the raw 1.60 catastrophically over-hedges; the correctly-calibrated 0.54× SPY short lands market-neutral:

| Book | Total | CAGR | Sharpe | Realised β | Corr(SPY) |
|---|---|---|---|---|---|
| Unhedged (current book) | +72% | +5.6% | 0.57 | 0.54 | 0.61 |
| **Realised-beta hedge (0.54×)** | **−19%** | **−2.1%** | −0.21 | **−0.00** | **−0.00** |

The market-neutral book is genuinely neutral (β and correlation ≈ 0) and its return is **negative**. Decomposition: the beta contribution is positive (it rode a historic bull), the alpha contribution is negative. **Every dollar of the +72% came from beta.** This unifies the whole study: the certified edge is skill at picking stocks that *ride the market up* (high-beta, high-momentum names likelier to hit +3% first), not market-independent selection — which is why the signal certifies yet the book loses to the index, the label rewards beta, and the hedged alpha is negative.

---

## 7. Live deployment

The system runs unattended (Windows Task Scheduler, weeknights) against Alpaca **paper** trading. The broker layer is paper-locked by construction (key-prefix + endpoint assertions that raise, not warn), with a HALT kill-switch, per-order notional caps, idempotent date-scoped order IDs, next-open bracket-free execution matched to the backtest's close-based exits, fill-based slippage measurement, and a nightly broker-vs-record divergence check. An independent five-agent execution audit (documented separately) verified the safety rails live and surfaced two real defects (a holiday-weekend order-cancellation and manual-trade contamination), both fixed. Pre-registered go/no-go gates — Gate A (~500 closed picks: net-positive rate CI above breakeven) and Gate B (~1000 picks: live Sharpe > 0.3, median slippage < 15 bps) — govern any future consideration of real capital; U6 predicts the live record will resemble +15% far more than +72%.

---

## 8. Discussion

The headline result is not a strategy but a **method**. The same rigour that certified the signal (§5.1) is what exposed it as beta and survivorship (§6). This is the intended outcome: a pipeline that can *falsify its own best number* is more trustworthy — and more useful — than one that cannot.

Three transferable lessons:
1. **Optimise the objective, not a proxy.** Higher pooled AUC repeatedly produced worse traded books; the top-K precision the strategy actually consumes is the only metric that matters.
2. **A certified edge can still be useless.** Statistical significance (124σ) and economic value are different questions; beta decomposition and point-in-time testing answer the second.
3. **Improvements are suspected leaks.** Every large jump in this project traced back to a bias (close-entry, survivorship, raw-beta hedge over-fit), not a discovery.

---

## 9. Limitations

- **Survivorship remains partial.** U6 fixes addition look-ahead but cannot recover delisted names' prices; the true survivorship-free number is only obtainable forward (the paper record) or with a paid point-in-time price database.
- **Beta hedge ratio is in-sample.** The 0.54× is fit on the full window; a live hedge needs a rolling estimate, and exit-timing attenuates the estimate — neither changes the sign of the negative alpha.
- **One test decade.** 2014–2026 was an historic bull; a beta-driven book is flattered by it. A sideways or bear decade would tell a different story.
- **Paper, not live.** No real fills, borrow constraints, or market impact beyond the modelled 10 bps.

---

## 10. Conclusion

We set out to determine not how large an edge we could report, but what a real edge on the S&P 500 actually *is*. The answer, earned through fifteen adjudicated experiments and two self-directed falsification attacks, is precise and honest: a **certified, small, survivorship-inflated, beta-driven momentum edge with near-zero market-neutral alpha**, best used long-only as a satellite. The strategy does not beat the index. The *system* — its purged validation, permutation certification, kill-list, capstone survivorship and beta analyses, and audited live deployment — is the contribution, and it succeeded exactly by proving its own edge illusory before a cent of real capital was at risk. The forward paper record, survivorship-free by construction, is the final and only clean verdict, and it is now accumulating.

---

## References

1. López de Prado, M. *Advances in Financial Machine Learning* (2018) — purged/embargoed CV, meta-labelling, triple-barrier.
2. fja05680, *S&P 500 Historical Components & Changes* — point-in-time index membership (1996–present).
3. Chen, T. & Guestrin, C. *XGBoost: A Scalable Tree Boosting System* (2016).
4. SEC EDGAR Form 4 filings — open-market insider transactions.

## Appendix A — Reproducibility

| Stage | Command | Output |
|---|---|---|
| Data | `python database.py` | `trading.db` |
| OOF predictions | `python pipeline.py` | `walk_forward_oof.csv` |
| Certification | `python baseline_null.py` | AUC vs 2,000 nulls |
| Backtest | `python backtest.py` | equity/trades + SPY |
| Survivorship (U6) | `python run_pit_universe.py` | point-in-time comparison |
| Beta (U7) | `python run_beta_hedge.py` | market-neutral decomposition |
| Live picks | `python predict_live.py` | `paper_trades` |
| Daily cycle | `python run_daily.py` | data → fills → score → picks → broker |
| Dashboard | `streamlit run dashboard.py` | overview / picks / backtest / model / live |

Test suite: **62 tests** (`pytest -q`), CI on every push. Full experiment journal: `FINDINGS.md`. Execution audit: `FINDINGS_SPY.md`. File-by-file reference: `DOCS.md`.

# Findings — an honest research journal

What this system actually learned, including every dead end. The discipline here matters
more than any single number: *test cheaply, believe nothing, kill bad ideas fast.*

---

## The headline

**The edge is real, but small.**

- Out-of-sample AUC ≈ **0.5645** on ~1.45M walk-forward predictions (500 stocks, 12 folds).
- Base win rate 32.9%; the **top 5% most-confident picks win 44.4%** — selectivity works.
- At the 3:1 reward:risk barrier, breakeven is 25%, so this is comfortably profitable *in
  expectation* — the question is only whether the magnitude survives (see survivorship).

That ~0.56 sounds unremarkable, and it should: liquid large-cap price data is efficient,
so a *thin* edge is the honest ceiling. The achievement is proving the edge is genuine.

---

## The edge is certified real (not luck)

`baseline_null.py` — a permutation / "beat 2,000 random baselines" test (an idea borrowed
from the Losing Loonies YouTube channel):

| Test | Real | Best of 2,000 random | Verdict |
|---|---|---|---|
| AUC vs shuffled-label null | **0.5645** | 0.5018 | ~124σ above null — **REAL** |
| Top-5% win rate vs random selection | **44.4%** | 33.5% | ~67σ above random — **REAL** |

No random baseline came close. The model finds genuine signal. *(Caveat: overlapping
10-day labels mean the exact σ is overstated, but "beats the best of 2,000" needs no
assumptions.)*

---

## The data ceiling — four experiments, one wall

Every attempt to raise accuracy converged on ~0.56. The model already extracts nearly all
the signal in liquid large-cap price/technical data.

| Experiment | Idea | Result |
|---|---|---|
| News/social sentiment | LLM/FinBERT on headlines | Dead-zone — daily model sits between an intraday signal and a multi-month drift |
| **Insider flow** (SEC Form 4) | follow open-market insider buys | **−0.002** — horizon mismatch (insider alpha is ~12 months; target is 10 days) + sparse on large caps |
| **Longer horizon** (20/40/60d) | maybe the edge lives longer | **Worse** — AUC decayed 0.57 → 0.51; technical features are inherently short-horizon |
| **Cross-sectional + own-trend** | rank each stock vs the universe | **+0.001** (within noise) — the strongest candidate, still not enough |

**Conclusion:** feature-hunting is exhausted. The 10-day horizon was the right call; the
constraint is the *data and market efficiency*, not the features or the model.

---

## What survivorship bias hid (three times)

The universe is *today's* S&P 500 — the survivors. This biased the backtest in three
distinct, instructive ways:

1. **Return magnitude.** The Monte Carlo showed ~33x/5yr vs SPY's 1.8x — a mirage. The
   win rate is inflated because we only see stocks that survived to today.
2. **Bear-regime performance.** Bears showed the *highest* win rates (46%) — but that's
   survivor bounce-backs, and it *contradicts* the model's collapsed AUC (0.497) in 2022.
   The model can't actually rank in bears; the environment just flattered it.
3. **Risk-control value.** A diversified, regime-scaled book looked *slightly worse* than
   a naive concentrated one — because the bias *rewards* the exact risks (sector
   concentration, bear-trading) that prudent controls remove. Live, those controls protect
   you; the biased backtest can't credit them.

**Lesson:** a survivorship-biased backtest cannot validate magnitude, regime behavior, or
risk management. Only forward, out-of-universe data can — which is why the paper-trade exists.

---

## The complete, honest characterization

> **Real. Small. Data-limited. Certified. Forward-pending.**

- **Real** — beats 2,000 nulls, no leakage (34 tests, purged walk-forward).
- **Small** — AUC ~0.56; the edge lives in *selectivity* (top-slice), not raw accuracy.
- **Data-limited** — four experiments hit the same ceiling; more features won't help.
- **Certified** — the signal is genuine, not curve-fitting.
- **Forward-pending** — magnitude and risk-control value can only be proven live.

---

## What's next

Nothing left to *build* for accuracy — the ceiling is reached and the system is automated.
The one open question is answered by **time**: `run_daily.py` logs and scores `paper_trades`
every weekday. In a few weeks, that forward, survivorship-free record is the verdict.

Open frontiers (optional, honest expectations): a **survivorship-free backtest** (needs
delisted/point-in-time data — the real rigor frontier), a **fundamentals/DCF** signal
family (orthogonal but likely another horizon-mismatch), or a **dashboard** to watch the
forward record fill in.

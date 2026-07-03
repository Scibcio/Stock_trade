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

## The full-strategy backtest — a real edge that still loses to the index

`backtest.py` replays the deployable strategy: every 10 trading days form the same
sector-capped, vol-weighted, regime-scaled top-15 book `predict_live` would, exit on the
triple barrier, compound the cohorts. Net of 0.1% round-trip costs, 2014–2026:

| Signal | Win% | Net/trade | PF | Total | Sharpe | Max DD |
|---|---|---|---|---|---|---|
| **LIVE (XGB only)** | 41.1% | +0.24% | 1.13 | **+76%** | 0.59 | −21.5% |
| Blend (XGB+LSTM) | 40.9% | +0.23% | 1.13 | +80% | 0.64 | −26.1% |
| **SPY buy-and-hold** | — | — | — | **+322%** | **0.86** | −24.9% |

Adversarially verified by a 5-agent review (leakage, exit math, portfolio accounting all
verified clean; two real corrections applied afterwards — rank by **XGB-only** to match
live, and compute SPY's **own** Sharpe/drawdown). The corrected verdict:

- **The per-trade edge is real** — +0.24% net, PF 1.13, 41% win at a 3:1 barrier (breakeven
  25%). Positive in expectation, after costs.
- **But the strategy loses to the index** — +76% vs SPY's +322%, and Sharpe 0.59 vs 0.86.
  The slightly shallower drawdown is a cash-holding artifact (~18% avg cash), not skill.
- **The LSTM adds nothing** — the blend barely moves return and *worsens* drawdown. XGB-only
  is the honest, lean choice; live is right to ignore the LSTM. (Bonus: `merge_insider` in
  `predict_live` is dead code — the model trains on `WIDE_FEATURES`, which excludes it.)

**Why it loses:** the +3% barrier caps every winner while 2014–2026 was a historic bull —
you cannot out-compound buy-and-hold by clipping your right tail. The edge is in *ranking*
(which stock over 10 days), not in beating beta long-only.

**The one lever that could change this:** stop capping winners. `labels.trailing_stop_label`
(phase 2, stubbed) lets winners run instead of exiting at +3% — the single highest-value
experiment left. Alternatively, express the ranking edge **market-neutral** (long top /
short bottom) so it stops competing with the index's beta.

---

## The complete, honest characterization

> **Real. Small. Data-limited. Certified. Index-lagging. Forward-pending.**

- **Real** — beats 2,000 nulls, no leakage (34 tests, purged walk-forward).
- **Small** — AUC ~0.56; the edge lives in *selectivity* (top-slice), not raw accuracy.
- **Data-limited** — four experiments hit the same ceiling; more features won't help.
- **Certified** — the signal is genuine, not curve-fitting.
- **Index-lagging** — long-only, it beats random but not buy-and-hold; the +3% cap is the
  bottleneck, not the signal.
- **Forward-pending** — magnitude and risk-control value can only be proven live.

---

## Execution honesty — next-open entries (F1, review-panel fix)

The panel caught the system booking entries at the SAME close the signal was computed
on — a price that had already printed. Entries now fill at the **next session's open**,
in the backtest and in the live paper trail (picks log as `pending`, fill next run).

The honesty tax, measured (3:1 exits, XGB-only, 10 bps, bear = cash):

| Entry timing | Total 2014–26 | Sharpe |
|---|---|---|
| Signal-day close (untradeable) | +77% | 0.60 |
| **Next-session open (honest)** | **+36%** | **0.37** |

Half the old headline return was the untradeable-close artifact. Every number in this
journal from here on is next-open. *(Also from here on: bear regime = 100% cash — fold-9
bear AUC was 0.490, the model cannot rank in bears, so bear exposure was edge-less risk.)*

---

## Exit geometry — the −1% stop was the bottleneck (§2 study, re-run honest)

Same picks, same costs, next-open entries; only the exit rule varies (`run_exit_sweep.py`):

| Exit rule | Net-positive % | Net/trade | Total | CAGR | MaxDD | Sharpe | Verdict |
|---|---|---|---|---|---|---|---|
| 3:1 (+3/−1) — old | 40.0% | +0.17% | +36% | +2.6% | −17.4% | 0.37 | baseline |
| **Symmetric ±3%** | **53.7%** | **+0.32%** | **+72%** | **+4.7%** | −20.3% | **0.52** | **ADOPTED (provisional)** |
| Symmetric ±2% | 53.6% | +0.20% | +52% | +3.6% | −23.6% | 0.47 | rejected (fails DD gate) |
| Plain 10d hold, −15% cat-stop | 54.2% | +1.17% | +482% | +15.9% | −36.1% | 0.81 | fails DD gate; U3 candidate |
| SPY buy & hold | — | — | +322% | — | −24.9% | 0.86 | benchmark |

The −1% stop whipsawed out of picks that recover: the picks carry positive 10-day drift
and the wide symmetric stop lets it accrue. `EXIT_STOP_LOSS = −0.03` is adopted
**provisionally** — it must survive the U3 symmetric-label retrain. The plain-hold row is
the most survivorship-inflated number in the project (no delistings in the data — the
exact events stops exist for); it stays a U3 candidate, not an adoption.

Caveats on record: ranker still trained on the 3:1 label; survivorship flatters wide
stops most; wider stop ≈ 3× risk per position at equal weight; breakeven at ±3% is
~50%+costs (54% clears it, but the margin is thin — the paper trial monitors it).

---

## The hidden beta tilt (panel discovery — the mechanism behind several findings)

Measured on our own OOF picks: the top-15 book runs **β ≈ 1.62** vs SPY (bottom-15:
β ≈ 0.55). The label P(+3% before −1% in 10d) mechanically rewards high-beta names.
This one mechanism explains: the bear-market AUC collapse (high-beta longs are what dies
in bears), why a "market-neutral" long/short book came out at net β ≈ +1.08 (not neutral),
and why SPY-plus-overlay blends acted as hidden leverage. Queued work: **U8** (de-beta the
label: ATR-scaled barriers or residual-return target) then **U7** (true beta-matched
hedging). Until then, every long-only comparison vs SPY partially measures *beta*, not
selection.

---

## Rejected strategies (tested, not vibes)

The kill list. Every future experiment that dies lands here with its numbers.

| Idea | Result | Why it died |
|---|---|---|
| LSTM in the live path | top-5% win 45.7% (XGB) vs 43.2% (blend) | dilutes the selective slice — the only slice traded |
| Tight symmetric exits (±1–2%) | Sharpe 0.34–0.47 | eaten by costs + noise |
| Model + 5-day-reversal combined rank | Sharpe 0.46 vs 0.71 alone | the signals fight |
| Naive long/short (unhedged legs) | net β +1.08, corr 0.63 to SPY | "market-neutral" in name only — revisit after U7 |
| Naive SPY + MN overlay ("portable alpha") | Sharpe falls to 0.82/0.77, DD −39%/−54% | hidden leverage, not alpha |
| SPY 200MA trend-timing standalone | Sharpe 0.80, lags SPY | defensive tool, not a return engine |
| Top-5 concentration (raw) | +1226% but −47% DD | unlivable drawdown — position count is a risk dial (U10) |
| Bear-regime trading at 0.3× | fold-9 AUC 0.490 | no ranking skill in bears — exposure is now 0 |

---

## What's next (updated priority order, per the review panel)

**U1** earnings blackout filter → **U2** trailing-label retrain (`run_trail_label.py`,
ready to run) → **U3** exit finalization (incl. the plain-hold candidate, on next-open
entries) → **U8** de-beta the label → **U7** true beta-neutralization → **U9**
core–satellite mode (50/50 SPY+strategy tested at Sharpe 0.92, DD −18.2% — beats SPY's
0.86 via the 0.49 correlation) → U5 meta-labeling → U10 concentration ladder → U4
learning-to-rank → U6 point-in-time universe → U11 (this registry, continuous).

Then Phase 3: Alpaca paper-trading go-live (owner action required: create the paper
account + `.env` keys). The forward record — `run_daily.py` logging and scoring
`paper_trades` every weekday — remains the only clean verdict.

# Stock_trade — Documentation

One page. Every file, every concept, no fluff. Results live in [FINDINGS.md](FINDINGS.md); how to operate it lives in [README.md](README.md).

---

## What this is

An ML system that ranks all ~500 S&P stocks daily by P(hit +3% before −3% in 10 days), trades the top 15 as a risk-controlled paper portfolio, and — the actual product — **proves honestly whether its edge is real.**

```
database.py → trading.db → features.py → labels.py → pipeline.py (walk-forward XGB)
      → walk_forward_oof.csv → baseline_null.py (certify) → backtest.py (simulate)
      → predict_live.py (daily picks → paper_trades) → dashboard.py (watch)
                     run_daily.py orchestrates the daily loop
```

---

## Files

### Core pipeline
| File | Does |
|---|---|
| `config.py` | Every shared constant: paths, label barriers (3:1), adopted exits (±3%), strategy (TOP_K 15, sector cap 3, regime exposure, costs), feature lists, 12 walk-forward folds. Edit here once. |
| `database.py` | yfinance → `trading.db`: OHLCV for ~503 tickers + 6 market baselines (SPY, VIX, HYG, TNX, UUP, RSP); marks `ml_ready` (≥504 rows). |
| `features.py` | ~30 leak-free features per stock/day (returns, momentum, SMA/EMA distances, RSI, Bollinger, NATR, beta, VWAP dist, macro context). Also insider merge + cross-sectional ranks (research). |
| `labels.py` | Targets: `triple_barrier` (+3%/−1%/10d — the trained label) and `trailing_stop_label` (research; rejected). Incomplete tails = NaN, never faked. |
| `model_xgb.py` | The live model: XGBoost classifier (300 trees, depth 4) on the wide features. |
| `model_lstm.py` | PyTorch CNN-LSTM on 60-day sequences (research artifact — out of the live path; diluted the top slice). |
| `ensemble.py` | Platt calibration, blending, meta-learner (research; blend rejected for live). |
| `pipeline.py` | 12-fold purged walk-forward → out-of-sample predictions for every stock-day → `walk_forward_oof.csv`. |

### Honesty layer
| File | Does |
|---|---|
| `baseline_null.py` | Certification: real AUC + top-5% win rate vs 2,000 shuffled/random nulls. Any label change must re-run this. |
| `threshold_analysis.py` | Regime tagging (SPY vs 200MA + VIX) and selectivity sweep (win rate by confidence slice). |
| `tests/` | 48 gate tests: features leak-free, labels, scaler discipline, paper scoring, cohort parity, next-open fills, earnings windows, engine behaviours. CI runs them on every push. |

### Strategy & live operation
| File | Does |
|---|---|
| `strategy.py` | THE cohort picker (sector-capped top-15, inverse-NATR weights, regime exposure) — one code path shared by backtest and live. + Monte-Carlo risk study (research). |
| `backtest.py` | Cohort simulator: honest next-open entries, ±3% exits, costs, bear = cash; equity/trades CSVs + SPY comparison. |
| `backtest_trailing.py` | Event-driven engine for overlapping/variable holds (trailing stops), same honesty rules. |
| `predict_live.py` | Daily picks: train on full history, rank latest day, earnings blackout, cohort every 10 sessions; logs `pending` → fills at next open → scores via ±3% exits into `paper_trades`. |
| `earnings.py` | U1: earnings-date collector (yfinance) + blackout logic (skip names reporting within 5 sessions). |
| `insider.py` | SEC EDGAR Form 4 collector → `insider_flow` (tested: no lift at 10d; kept for research). |
| `broker_alpaca.py` | The ONLY broker-facing file. Paper-locked by construction (PK-key + `APCA_PAPER` + endpoint hard-fails), HALT kill switch, notional caps, idempotent order ids, retry backoff. |
| `paper_trader.py` | Nightly broker pass: reconcile real fills into the record (fills = truth, slippage logged), mirror scored exits, keep the U9 SPY core on target, queue tonight's picks for the open. |
| `run_daily.py` | The automated daily cycle (Task Scheduler, weekdays 22:00): data → reconcile+fill pendings → score → new cohort if due → broker pass. Logs to `logs/`. One-click: `run_paper.bat`. |
| `dashboard.py` | Streamlit control room (localhost-only): overview, picks, backtest, model, logs + buttons to run everything. |

### Experiments (each = one FINDINGS entry; kill-listed if dead)
| File | Verdict |
|---|---|
| `run_exit_sweep.py` | Exit geometry on honest entries → **±3% adopted** |
| `run_sym_label.py` | U3: sym-label retrain → higher AUC, worse book → rejected |
| `run_trail_label.py` | U2: trailing-label retrain → AUC 0.49, uncertified → rejected |
| `run_debeta.py` | U8: de-beta'd labels → edge dies with the beta → rejected |
| `run_lstm.py` / `run_ensemble.py` | LSTM OOF + ensemble evaluation → blend rejected for live |
| `run_insider_lift.py` / `run_insider_backfill.py` | Insider features → −0.002 AUC (horizon mismatch) |
| `run_crosssec_lift.py` | Cross-sectional ranks → +0.001 (noise) |
| `run_horizon_experiment.py` | 20/40/60d horizons → AUC decays → 10d confirmed |
| `run_backtest.py` / `run_strategy_compare.py` | Early Monte-Carlo book study / risk-control comparison (superseded by `backtest.py`) |

### Docs, config, generated
| File | Does |
|---|---|
| `README.md` | Setup + how to run. `FINDINGS.md` — the research journal (the crown jewel). `SECURITY.md` — threat model; dashboard is local-only. `PROJECT_PLAN.md` — private roadmap (gitignored). |
| `requirements.txt` | Pinned deps; torch installs separately (cu128). `.github/workflows/ci.yml` — pytest + pip-audit on push. `.streamlit/config.toml` — theme, localhost bind. `tickers.csv` — S&P 500 universe input. |
| Generated (gitignored) | `trading.db` (~220MB), `walk_forward_oof*.csv` (OOF predictions), `backtest_*.csv` (equity/trades), `logs/` (daily runs). |
| `faceoff/` (gitignored, personal) | Old-FYP-vs-new arena: `train_fyp.py` (V4 resurrected in PyTorch), `run_faceoff.py`, `money_table.py` (£200 table), `build_arena_cache.py`, `app.py` (versus UI + time machine). |

---

## Concepts (what we use and why)

**Triple-barrier label.** Each stock-day asks: does price hit +3% before −1% within 10 days? First barrier touched decides. Asymmetric on purpose — U3 proved the asymmetry *is* the selection edge.

**Purged walk-forward.** 12 folds, each trains only on data ending 14+ days before its test year — the model is always scored on a future it never saw. Its predictions on test years = **OOF (out-of-fold)** — the honest raw material for every result.

**AUC.** Ranking skill: 0.50 = coin flip. Ours ≈ 0.56 — small and real. Hard-won lesson (U2+U3): **pooled AUC is not the objective; the top-15 book is.**

**Null certification.** Shuffle labels 2,000×, ask if the real result beats every fake. Ours does (~124σ). Any new label must pass or die.

**Survivorship bias.** The universe is today's survivors — no delistings — so absolute backtest levels are inflated (uncapped exits most of all). Comparisons stay meaningful; the forward paper record is the only clean verdict.

**Next-open entries (F1).** A signal computed on the close cannot be bought at that close. All entries fill at the next session's open — this halved the fantasy returns and is non-negotiable.

**Regimes.** SPY vs its 200-day MA + VIX ⇒ bull / sideways / bear. Exposure 100% / 60% / **0%** — the model provably cannot rank in bears (fold-9 AUC 0.49), so bear = cash.

**Cohort.** One 15-name book per 10 trading sessions (matching the hold window), max 3 per sector, weights ∝ 1/NATR (volatility-targeted: risk ≈ equal per name). 10 bps round-trip cost per position.

**Exits.** Adopted: symmetric ±3% (or 10-day timeout). Trailing/no-stop exits return more but fail the max-drawdown gate and are the most survivorship-flattered — shelved, not forgotten.

**Win rate vs positive rate.** Barrier win = hit +3% before −1% (label metric, breakeven 25%). Positive rate = trade closed green under the real exits (~54%, breakeven ~50%+costs). Both tracked; don't mix them.

**Key stats.** Expectancy = mean net return/trade. Profit factor = gross wins/gross losses. Sharpe = return per unit of volatility (annualised). Max drawdown = worst peak-to-trough. CAGR = compound annual growth.

**The beta tilt.** The label mechanically rewards high-beta names (top-15 book β≈1.6). U8 proved you can't label it away — the edge substantially *is* the beta/vol payoff — so beta is managed at the portfolio layer (hedging U7, core–satellite U9).

**Earnings blackout (U1).** Skip any candidate reporting within 5 sessions — earnings gaps through stops are unrankable tail risk.

**Forward paper record.** Every cohort logged (`pending` → filled at real next open → scored by the real exits) — survivorship-free, the final judge. Gates: ~500 closed picks for Gate A, kill criteria pre-registered in the handoff.

**The data ceiling.** Sentiment, insider flow, longer horizons, cross-sectional ranks — all tested, all ≈ +0.00 AUC. Liquid large-cap price data supports a thin edge, and ours already extracts it.

---

## Current adopted configuration (post U1-U3, U8)

| Piece | Setting |
|---|---|
| Model | XGBoost on ~30 wide features (LSTM/blend: research only) |
| Label | 3:1 triple barrier (+3%/−1%/10d) — certified 0.561 AUC |
| Exits | Symmetric ±3%, 10-day timeout |
| Book | Top-15, sector cap 3, 1/NATR weights, bear = cash, cohort per 10 sessions |
| Filters | Earnings blackout, 5 sessions |
| Honest baseline (2014-26) | +72% total, Sharpe 0.52, −20.3% DD vs SPY +322% / 0.86 / −24.9% |
| Status | Real, small, certified, index-lagging long-only; forward paper trial running nightly |

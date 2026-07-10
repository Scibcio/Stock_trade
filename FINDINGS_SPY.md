# FINDINGS_SPY — SPY anomaly + execution-layer audit

Read-only investigation per `CLAUDE_CODE_PROMPT_SPY_AUDIT.md` (Stages 1–2).
**No code was changed.** Every claim carries file:line, DB rows, or live Alpaca
API responses, pulled 2026-07-10 from paper account PA3J9LJP5ME3.

Note on the audit's context block: the live system is XGBoost-only (the
CNN-LSTM/ensemble is a research artifact, see FINDINGS.md/F6) and there is no
two-model agreement gate. This does not change any hypothesis below.

---

## STAGE 1 — Why is it buying SPY?

### H1 — Universe leakage → **REFUTED**

- The tradeable universe is `SELECT ticker FROM stocks WHERE ml_ready = 1`
  ([pipeline.py:34-39](pipeline.py)), populated from `tickers.csv` by
  `database.py`. The six feature instruments live in a **different table**,
  `market_baselines` (reads e.g. [threshold_analysis.py:43](threshold_analysis.py),
  [backtest.py:211](backtest.py)). No concat/union/extend joins them anywhere.
- DB evidence: `SELECT COUNT(*) FROM paper_trades WHERE ticker IN
  ('SPY','RSP','HYG','UUP','^VIX','^TNX')` → **0 rows**. No feature instrument
  has ever been a pick.
- Order-log evidence: every SPY order on the account carries the client id
  `core|SPY|<YYYYMMDD>|buy` — minted ONLY by `paper_trader._rebalance_spy`
  ([paper_trader.py:116-134](paper_trader.py)). Model picks carry
  `<cohort_id>|<ticker>` ids. RSP/HYG/UUP: **zero orders ever**. `^VIX`/`^TNX`:
  **no submission was ever attempted** (so no rejection exists to find — the
  absence is consistent with H1 being false, not with silent failures).

**SPY is not leaking into the universe. The SPY buys are the U9 core–satellite
SPY core** (75% of the $10k cap), built deliberately in ≤$2,000 chunks per
night (`MAX_ORDER_NOTIONAL` cap) until the $7,500 target is reached.

### H2 — Cadence → **REFUTED for picks; SPY nightly cadence is by design**

- The cohort gate exists and gates **pick generation**:
  `predict_live.cohort_due` ([predict_live.py:102-113](predict_live.py)),
  called at the top of `main()`; it last evaluated true on 2026-07-02 and every
  nightly log since prints `Cohort not due yet (one book per 10 trading
  sessions)` (e.g. `logs/daily_2026-07-09.log`).
- The SPY core rebalance is **intentionally not cohort-gated** — it runs
  nightly until the core is within the 2% drift band
  ([paper_trader.py:116-134](paper_trader.py)). Three-plus consecutive SPY
  buys = the core building up in capped chunks: fills of $2,000 on 07-06,
  07-08, 07-09 plus a queued $1,458.71 — converging on $7,500, then stopping.

### H3 — Idempotency → **REFUTED (no over-buying)**

- Same-day resubmission is a broker-side no-op via deterministic per-day ids
  ([paper_trader.py:127](paper_trader.py); duplicate → `DUPLICATE` sentinel,
  [broker_alpaca.py](broker_alpaca.py) `submit_notional`).
- Fills vs signals, from the API: SPY fills = 3 × $2,000 (07-06/08/09), avg
  entries $749.31/$743.28/$745.86 → position 8.0413 shares, market value
  **$6,036.30 vs the $7,500 target** — under-deployed, not 3× oversized. Each
  chunk respects the per-order cap; total core ≤ (1−w)·cap by construction.

### H4 — Not a bug (determinism / beta) → **MOOT for SPY, already documented**

SPY never passes through the model, so feature-vector autocorrelation cannot
explain its orders (H1–H3 evidence above fully accounts for them). The beta
tilt itself (top-15 book β≈1.6) is real, previously measured, and documented in
FINDINGS.md ("The hidden beta tilt", U8) — per the audit brief, not re-derived.

---

## STAGE 2 — Execution layer audit (live evidence)

### 1. Bracket legs → **CONFIRMED: entries carry NO legs — a documented design decision, with a real caveat**

- API evidence: parent orders return no child legs (plain notional market DAY
  orders).
- Why: Alpaca allows fractional/notional **only** as simple DAY orders (bracket
  + notional is rejected). Exits are therefore **loop-managed**: the nightly
  run scores ±3%/timeout on closes ([predict_live.py:222](predict_live.py))
  and `_mirror_exits` closes the broker position
  ([paper_trader.py:94-113](paper_trader.py)). This matches the *verified
  backtest semantics* (close-based barriers), which bracket legs would not.
- **Honest caveat the audit is right about:** between nightly runs there is no
  live stop. An intraday gap through −3% exits at the *next open after the
  crossing close*, not at −3%. The backtest books the crossing close; the gap
  between the two is measured slippage (Gate B input), not hidden.

### 2. HALT file → **CONFIRMED (live test, this audit)**

Created `HALT`, ran the full broker pass, counted orders via the API:
**before=25, after=25 — zero new orders**; the pass printed
`HALT file present - SPY entry blocked` (it visibly blocked a due SPY top-up),
then `HALT` was deleted. Design note: HALT blocks **new risk (buys)** only;
exits remain allowed deliberately — the kill switch must never trap a position.

### 3. Paper-endpoint hard failure → **CONFIRMED (live test, this audit)**

There is no base-URL config to point at live — `TradingClient(..., paper=True)`
is hardcoded and the endpoint enum is verified
([broker_alpaca.py](broker_alpaca.py) `_paper_endpoint_ok`). Equivalent guard
flips, run live: `APCA_PAPER=false` → `BrokerSafetyError` raised;
live-style `AK…` key → `BrokerSafetyError` raised. Both are raises, not logs.

### 4. HOLD_DAYS time-exit → **CONFIRMED in code; first live exercise ~2026-07-16**

`score_paper_trades` times out every position at the HOLD_DAYS window's last
close ([predict_live.py:222](predict_live.py)); scored rows are closed at the
broker by `_mirror_exits`; unfillable picks are retired by
`expire_stale_pending` ([predict_live.py:205](predict_live.py)). The 07-02
cohort matures around 07-16 — the first live round-trip proof is pending.

### 5. Slippage → **CONFIRMED implemented / INCONCLUSIVE live (zero datapoints — see F-A)**

`reconcile` writes fill-vs-signal-close slippage per trade into
`alpaca_fills.slippage_bps` ([paper_trader.py:65-92](paper_trader.py); gate
test `test_reconcile_applies_fill_price`). Live table: **empty** — because the
entry batch it would have measured never filled (F-A below). Gate B remains
unmeasurable until cohort 2 fills.

---

## NEW FINDINGS (outside the audit's list — the real problems)

### F-A — The 2026-07-04 order batch was CANCELED and never resubmitted → broker book is missing cohort 1 · **CONFIRMED, HIGH**

API evidence: all **15 orders submitted Sat 2026-07-04 show status CANCELED,
fill $0.00** (14 stock entries + that day's SPY chunk). Orders submitted on
actual trading evenings (07-06 onward) filled normally — DAY orders queued
across the holiday weekend were canceled by the broker, not filled at Monday's
open. Because `_submit_cohort` marks `submitted_at` on submission and
`reconcile` only matches **fills**, a *submitted-then-canceled* order is
invisible: the record holds 15 open positions, the broker holds none of them.
Consequence: the Alpaca mirror is record-only for cohort 1; slippage (Gate B)
has no data. The paper RECORD itself (scored on DB closes) is unaffected —
Gate A is intact.

### F-B — Foreign/manual trades on the account · **CONFIRMED — needs the owner's attention**

The account shows qty-based orders our system cannot produce (it submits
notional-only, buy-only, with deterministic ids): RCL buy 07-08 → sold 07-09;
CIFR (not in the S&P universe) buy+sell 07-09; and — important — **A (Agilent)
SOLD SHORT 100 shares, current position −$13,359**, with a covering A BUY
still ACCEPTED. The system correctly leaves foreign symbols untouched
(`_mirror_exits` only touches tickers ever present in `paper_trades`,
[paper_trader.py:94-113](paper_trader.py)), but manual trades contaminate
account equity and any account-level performance read.

---

## Proposed remediation (ordered commits — **no code written, awaiting approval**)

1. `fix:` reconcile detects CANCELED/EXPIRED/REJECTED entry orders and clears
   `submitted_at` so the next nightly pass resubmits (heals F-A class
   permanently; idempotent ids make retries safe).
2. `fix:` clear `submitted_at` for the 15 orphaned 07-02 rows **only if** the
   cohort is still young at merge time; otherwise let cohort 1 finish as
   record-only and let cohort 2 (~07-16) be the first fully-mirrored book —
   recommended, cleaner for Gate B.
3. `feat:` divergence check in the nightly pass: count record-open positions
   missing at the broker, print + store in `alpaca_state`; warn in the
   dashboard Live tab.
4. `docs:` FINDINGS.md entry for F-A/F-B; operating rule: **no manual trading
   on this account** (open a second paper account for hand-trading — Alpaca
   allows several).
5. *(optional)* `feat:` pre-submission calendar check (`client.get_clock()`)
   to defer submissions when the next session is >1 day away — belt-and-braces
   on top of (1).

— End of audit. Stage 3 untouched, gates unmoved, n=3 is still n=3.

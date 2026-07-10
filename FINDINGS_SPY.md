# FINDINGS_SPY — SPY anomaly + execution-layer audit

Read-only investigation per `CLAUDE_CODE_PROMPT_SPY_AUDIT.md` (Stages 1–2).
**No code was changed.** Every claim carries file:line, DB rows, or live Alpaca
API responses, pulled 2026-07-10 (morning) from paper account PA3J9LJP5ME3.

> **Independently re-verified** (2026-07-10) by two adversarial read-only
> reviewers. Every Stage-1 and Stage-2 verdict below HELD under re-derivation.
> Their four corrections are folded in inline and marked **[rev]**: the empty
> slippage table is structural not just cancellation-caused; F-A's cause is
> unproven (could be a manual Cancel-All); exit-side slippage is *not* measured;
> and remediation-fix-1 must use fresh order ids. The universe lens (H1) could
> not be re-run (session limit), but its direct DB/id evidence is unambiguous.

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
- **Honest caveat:** between nightly runs there is no live stop. An intraday gap
  through −3% exits at the *next open after the crossing close*, not at −3%. The
  backtest books the crossing close; the difference is real execution slippage.
  **[rev]** This exit-side gap is currently **NOT measured** — `reconcile` only
  matches BUY fills ([paper_trader.py:80](paper_trader.py)) and exits go out via
  `close_position` with broker-generated ids it can't match, so `alpaca_fills`
  can only ever hold entries. Measurable from Alpaca order history; no code does
  it yet (remediation 5).

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

### 4. HOLD_DAYS time-exit → **CONFIRMED in code; first live exercise ~2026-07-20 [rev]**

`score_paper_trades` times out every position at the HOLD_DAYS window's last
close ([predict_live.py:222](predict_live.py)); scored rows are closed at the
broker by `_mirror_exits`; unfillable picks are retired by
`expire_stale_pending` ([predict_live.py:205](predict_live.py)). **[rev]** The
07-02 cohort's guaranteed time barrier lands ~**07-20** (11 forward sessions,
minus the 07-03 holiday), not 07-16 — earlier only if a ±3% barrier fires first.

### 5. Slippage → **CONFIRMED implemented (entries only) / no live data — structurally, not just F-A [rev]**

`reconcile` writes entry fill-vs-signal-close slippage into
`alpaca_fills.slippage_bps` ([paper_trader.py:65-92](paper_trader.py); gate test
`test_reconcile_applies_fill_price`). Live table: **empty**. **[rev]** Two
independent reasons, not one: (a) it only writes into `status='pending'` rows,
but cohort 1 was inserted as legacy `open`/`entry_open=NULL` rows (see F-A), so
they were never reconcilable *regardless of the cancellation*; (b) even on a
clean cohort it records **entry** slippage only. Gate B slippage remains
unmeasured until cohort 2 fills AND exit slippage is added (remediation 5).

---

## NEW FINDINGS (outside the audit's list — the real problems)

### F-A — Cohort 1 never reached the broker book · **CONFIRMED, HIGH**

API evidence: all **15 orders from Sat 2026-07-04 show status CANCELED, fill
$0.00**, bulk-canceled within 2 seconds on 2026-07-05 19:04 UTC. Orders from
actual trading evenings (07-06+) filled normally. Because `_submit_cohort` marks
`submitted_at` on submission and `reconcile` only matches **fills**, a
submitted-then-canceled order is invisible — the record holds 15 open positions
the broker never held.

**[rev] Corrections from the independent review:**
- **Cause is UNPROVEN.** I asserted "canceled by the broker over the weekend."
  The API does not record *who* canceled; a manual **Cancel-All** by the owner
  (who hand-trades this account — see F-B) fits the 2-second bulk cancel equally.
  Downgrade the cause to *unknown*; the remediation works either way.
- **Cohort 1 is a LEGACY pre-Phase-3 book** (rows inserted `open`/`entry_open
  NULL` on 07-02, proven by the old 3-field STATUS log). So it scores on
  signal-close semantics and its broker mirror was *never* going to feed Gate B
  slippage — the empty `alpaca_fills` is structural, not purely the cancel.
- **HON blind spot:** the HON row produced **no broker order at all**
  (non-fractionable, slot $54 < 1 share ≈$224 → skipped, then marked submitted).
  An order-log scan can never heal it; only a position-vs-record **divergence
  check** catches it. So a perfect cohort-1 resubmit mirrors at most 14/15 names.

Consequence: the Alpaca mirror is record-only for cohort 1. The paper RECORD
(scored on DB closes) is unaffected — **Gate A is intact.**

### F-B — Foreign/manual trades on the account · **CONFIRMED — needs the owner's attention**

The account shows qty-based orders our system cannot produce (it submits
notional-only, buy-only, with deterministic ids): RCL buy 07-08 → sold 07-09;
CIFR (not in the S&P universe) buy+sell 07-09; and — important — **A (Agilent)
SOLD SHORT 100 shares, current position −$13,359**, with a covering A BUY
still ACCEPTED. The system correctly leaves foreign symbols untouched
(`_mirror_exits` only touches tickers ever present in `paper_trades`,
[paper_trader.py:94-113](paper_trader.py)), but manual trades contaminate
account equity and any account-level performance read.

**[rev] Update (2026-07-10 ~13:35 UTC):** the covering A BUY (100 sh @ $134.48)
has FILLED — **the Agilent short is closed**; live positions now show only SPY
($7,459, core converged inside the 2% drift band exactly as Stage-1 predicted).
The −$13,359 figure was a morning snapshot. The finding stands: foreign trades
happened. **Operating rule: do not hand-trade this account** (use a second paper
account for manual play — Alpaca permits several).

---

## Proposed remediation (ordered commits — **no code written, awaiting approval**)

**[rev] Re-ordered per the review** (divergence check is the only fix that
catches the HON class; fix-1's id reuse would silently no-op):

1. `feat:` **broker↔record divergence check** (PRIMARY) — each nightly pass
   counts record-`open` tickers missing at the broker (and vice-versa), prints
   + stores in `alpaca_state`, surfaces in the dashboard Live tab. This is the
   only mechanism that catches *both* canceled orders **and** the HON
   never-ordered case.
2. `fix:` resubmit healing — on CANCELED/EXPIRED/REJECTED with `filled_qty==0`,
   re-queue with a **fresh attempt-scoped `client_order_id`** (mirroring the
   date-scoped SPY-core ids), NOT the original id (reusing it → 422 duplicate →
   silent no-op loop). Validate with one live paper order before merge.
3. `decision:` leave **cohort 1 record-only** (recommended). It's a legacy
   close-entry book that can never feed Gate B slippage anyway; let cohort 2
   (~07-20) be the first cleanly-mirrored book. (So fix 2 heals *future*
   cancels, not the 07-02 rows.)
4. `feat:` record **exit-side slippage** — match `close_position` fills (broker
   ids) back to scored rows so Gate B measures round-trip, not entry-only, cost.
5. `feat:` pre-submission calendar guard (`client.get_clock()`) — defer entries
   when the next session is >1 day out (belt-and-braces vs weekend cancels).
6. `docs:` fold F-A/F-B into FINDINGS.md; encode the **no-manual-trading** rule.

Not fixing (correct as-is): the SPY nightly cadence, the buy-only HALT, the
no-bracket-legs design, bear=cash. None are bugs.

— End of audit. Stage 3 untouched, gates unmoved, n=3 is still n=3.

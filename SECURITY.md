# Security — dashboard & public deployment

Threat review for `dashboard.py` and a **hard checklist** before it ever faces the public
internet. A publicly-reachable *financial* app carries real liability and data-exposure
risk, so this is not optional.

---

## Posture built into the code

`dashboard.py` is hardened by construction:

- **Read-only database.** Opened with `mode=ro` — the app *cannot* write, alter, or drop
  anything, even if a bug tried to.
- **No SQL injection surface.** Every query is parameterized; there is **no free-text user
  input that reaches SQL** (only fixed queries and bounded `LIMIT`s). Filters, if added,
  must stay as validated `selectbox`/`slider` widgets — never raw text into a query.
- **Bounded + cached.** All tables are capped (`MAX_ROWS`) and loaders are cached, so a
  refresh-spam can't hammer the DB.
- **No information leakage.** Data access is wrapped in `try/except` returning empty frames —
  users see a friendly message, never a stack trace, file path, or SQL error.
- **No secrets.** The app reads only local files; there are no API keys, tokens, or
  credentials in the code or repo (and `.gitignore` excludes `.env`, the DB, and OOF data).
- **No dangerous capabilities** *(read-only build)*. No file uploads, no `eval`/`exec`, no
  network calls at request time, and — critically — **no trade execution of any kind.**
- **Disclaimer on every page.** Educational/research framing, not financial advice.

---

## ⚠️ The Control Panel makes this build LOCAL-ONLY

`dashboard.py` now ships a **Control Panel** (sidebar buttons) that runs the pipeline
scripts via `subprocess` — `database.py`, `predict_live.py`, `run_daily.py`, `pipeline.py`,
`baseline_null.py`. This is deliberate: it lets *you* operate the whole system without a
terminal.

**But it means the app executes code.** That is safe on your own machine (localhost) and
**must never be exposed to the public internet** — a reachable "run script" button is a
remote-code-execution surface. Before any public deploy you MUST **delete the entire
`CONTROL PANEL` section** of `dashboard.py` (and the `run_script` helper) to return to the
read-only posture below. Also bind Streamlit to localhost only (`--server.address 127.0.0.1`)
and never port-forward it.

---

## Pre-launch checklist (MUST do before public deploy)

1. **Do NOT ship `trading.db`.** It's ~220 MB and holds the full dataset. For a public app,
   export a small **sanitized snapshot** (latest picks + paper-trade summary + OOF metrics)
   to a few small CSV/JSON files and point the dashboard at those. Never serve the raw DB,
   and never expose a file-download of it.
2. **Disable Streamlit error details.** In `.streamlit/config.toml`:
   `[client]\nshowErrorDetails = false` — so no traceback ever reaches a visitor.
3. **HTTPS only.** Deploy behind TLS (Streamlit Community Cloud does this; if self-hosting,
   put it behind a TLS reverse proxy). No plain HTTP.
4. **Rate limiting / resource caps.** Enforce at the host / reverse proxy (requests-per-IP,
   memory/CPU limits). Queries are already bounded and cached, but add a network-layer cap.
5. **Authentication (if the data isn't meant to be fully public).** Add a login/gate; do
   not rely on "nobody knows the URL."
6. **Pin dependencies + patch them.** Commit a `requirements.txt` with versions; watch for
   Streamlit / pandas advisories and update. Run `pip-audit` periodically.
7. **Legal.** Keep the disclaimer prominent, add a Terms/Privacy page, and get a **legal
   review for publishing financial content** in your jurisdiction. Publishing stock "picks"
   publicly can carry regulatory exposure.
8. **No PII / analytics without disclosure.** Don't collect user data. If you add analytics,
   say so and comply with GDPR/CCPA as applicable.
9. **Keep it read-only forever.** Never add write endpoints, user uploads, or (above all)
   any order-placement / brokerage integration to a public-facing app.

---

## The paper broker (Phase 3) — paper-only by construction

`broker_alpaca.py` is the ONLY file that talks to a broker, and it cannot reach a
live account: it hard-fails unless `APCA_PAPER=true`, unless the key has the paper
`PK` prefix (live keys are `AK`), and unless the SDK client resolves to
`paper-api.alpaca.markets`. Keys live only in `.env` (gitignored, never logged,
never shown in the dashboard). Safety rails: a `HALT` file blocks all NEW orders
(never exits — the kill switch must not trap positions), a hard per-order notional
cap, Alpaca's $1 minimum respected, deterministic client-order ids make
resubmission idempotent, and 3-try backoff on transient API errors. The dashboard
reads broker state from the database only — it makes no API calls and has no order
buttons. **There is no live-money code path; adding one is out of scope, forever.**

---

## What this app deliberately will NOT do

No writes · no code execution · no file uploads · no external requests per view · no secrets ·
**no trading or money movement.** If a future change would add any of these to a public
deployment, treat it as a new security review, not a feature.

---

## Reporting

Found an issue? Open a private security advisory on the GitHub repo rather than a public
issue.

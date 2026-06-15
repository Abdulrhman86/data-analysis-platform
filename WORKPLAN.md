# Workplan — from "tested code" to "actually working & used"

This plan takes the project from its current state (a 45-test-green app that has
**never been run end-to-end or deployed**) to something real people can open and
use. It is written for a **Claude Code** agent to execute, with an explicit split
of what the agent can do vs. what only the human operator can do.

> **Decide the game first (from the council).** "Working" means different things:
> - **Grade / defense:** critical path = Phase A → C → E3 (defense pack). E1 (real users) optional.
> - **Portfolio / demo:** critical path = Phase A → B → C → D.
> - **Product / traction:** all of A–E, and E1 (real-user validation) is the real test.
>
> Phases **A → D are the universal critical path** (run it, make it safe, deploy it,
> frame it). Everything else is goal-dependent.

## Legend
- 🤖 **Claude Code can do this** (write/run code, drive the browser, fix bugs, prep deploy, build assets).
- 🧑 **Only you can do this** (accounts, clicking "deploy", recruiting/watching users, the defense).
- 🤝 **Both** (Claude prepares; you trigger/confirm).

Environment facts: repo root = `data_mining/`; app entry = `app/home.py`; test venv at
`data_mining/.venv` (full stack installed); run via `cd app && streamlit run home.py`.

---

## Phase A — Make it actually RUN (the #1 gap) 🤖
*The council was unanimous: tests verify functions, not the live Streamlit app. Until it's
driven end-to-end in a browser, "working" is unproven. Claude can do this now via Playwright.*

| # | Task | Who | Deliverable | Effort |
|---|------|-----|-------------|--------|
| A1 | Create a realistic **messy sample dataset** (missing values, mixed types, a date column, categoricals, an outlier, a high-cardinality column) | 🤖 | `app/static/sample_sales.csv` | S |
| A2 | Launch the app locally (`streamlit run home.py`, background) | 🤖 | running server on :8501 | S |
| A3 | **Drive the full workflow in a real browser:** upload → data-quality → preprocess (impute + encode + scale) → 3 charts → save a dashboard → train 2 models → tune one → predict on a new file. Capture screenshots + console/network errors at each step | 🤖 | screenshot trail + a list of every runtime error pytest missed | M |
| A4 | **Fix every runtime bug found** (session-state, widget keys, upload edge cases, dashboard persistence, predict-tab schema handling); re-run A3 until clean | 🤖 | green end-to-end run | M–L |
| A5 | Add a scripted Playwright walkthrough as a smoke-test regression guard | 🤖 | `app/tests/test_smoke_e2e.py` (or a `scripts/` walkthrough) | M |

**Exit criteria:** the entire workflow completes in a browser on the sample data with zero unhandled errors, captured in screenshots.

---

## Phase B — Make it SAFE to put online 🤖
*Before a public URL: enforce limits, warn about data, and fit Community Cloud's constraints.*

| # | Task | Who | Deliverable | Effort |
|---|------|-----|-------------|--------|
| B1 | Wire the unused `Config.MAX_FILE_SIZE_MB` and add `.streamlit/config.toml` (`[server] maxUploadSize=50`) — Cloud caps ~1 GB RAM, so cap uploads | 🤖 | enforced upload limit | S |
| B2 | Add a **privacy notice** on the upload page ("Public demo — don't upload confidential or personal data") | 🤖 | notice in `app/pages/1_upload_data.py` | S |
| B3 | Add `runtime.txt` (`python-3.11`) so the deploy matches local | 🤖 | `app/runtime.txt` or repo-root | S |
| B4 | Handle Cloud's **ephemeral disk**: `.app_data/dashboards.json` is wiped on restart. Make dashboard persistence degrade gracefully (keep session + JSON export; note the limitation in-app) | 🤖 | graceful behavior + note | S–M |
| B5 | **Memory guard** the data-quality engine + upload for big files (sample rows like the viz page does) so Cloud doesn't OOM | 🤖 | row/size guard | M |

**Exit criteria:** app stays within Cloud limits and never persists or invites sensitive data.

---

## Phase C — DEPLOY to a live URL 🤝
*Claude prepares everything; the account + "deploy" click are yours.*

| # | Task | Who | Deliverable | Effort |
|---|------|-----|-------------|--------|
| C1 | Verify repo is Cloud-ready: requirements at repo root, main file path `app/home.py`, no absolute paths | 🤖 | deploy-ready repo | S |
| C2 | Push to GitHub (Claude can run `git`/`gh` **if a remote + auth exist**; otherwise you create the repo and push) | 🤝 | repo on GitHub | S |
| C3 | share.streamlit.io → connect repo → main file `app/home.py` → Deploy | 🧑 | **live public URL** | S |
| C4 | Drive the **deployed** URL end-to-end (Claude via Playwright on the public URL; you confirm on your phone) — cold start, no local paths, real memory | 🤝 | verified live app | M |

**Exit criteria:** a stranger could open the URL and complete the workflow.

---

## Phase D — Make it UNDERSTANDABLE (framing) 🤖
*The Outsider couldn't tell what it is or whether it's for them. Sell the outcome, not the machinery.*

| # | Task | Who | Deliverable | Effort |
|---|------|-----|-------------|--------|
| D1 | Rewrite the landing headline/subtitle to **benefit-first** ("Upload a spreadsheet, understand your data and build prediction models — no code") in `app/home.py` | 🤖 | new landing copy | S |
| D2 | Add a **"Try it with sample data"** button (loads the A1 dataset) so users see the payoff before uploading their own | 🤖 | sample button on upload page | S–M |
| D3 | Produce a **60-second demo**: a script + a silent screen-capture **GIF** (Claude via Playwright); you re-record with voiceover if desired | 🤝 | `demo.gif` + script | M |
| D4 | A crisp "What is this / who is it for" block in the README + landing page | 🤖 | updated README + page | S |

**Exit criteria:** a first-time visitor knows what it does and can try it in one click.

---

## Phase E — VALIDATE + DEFENSE PACK 🤝
*The council: claims must be shown, not asserted; the educator is your most plausible first user.*

| # | Task | Who | Deliverable | Effort |
|---|------|-----|-------------|--------|
| E1 | Hand the URL to **2–3 real people** (ideally a professor/TA), watch silently, note friction | 🧑 | friction notes | S |
| E2 | Turn friction notes into a prioritized fix list; implement the quick wins | 🤖 | fixes | M |
| E3 | **Defense pack:** "Why not just use KNIME/Orange/Excel?" slide content; a one-page methodology summary (leakage-free pipeline, before/after); and a runnable script that **demonstrates the no-leakage claim live** (fit-on-train proof) | 🤖 | `docs/DEFENSE.md` + `scripts/show_no_leakage.py` | M |
| E4 | Positioning one-pager: "rigorous ML you can't get wrong," educator angle | 🤖 | `docs/POSITIONING.md` | S |

**Exit criteria:** real people have used it; you can *demonstrate* (not just claim) correctness at a defense.

---

## Phase F — Optional hardening (deferred items) 🤖
*Lower priority than A–E. Only after it runs, deploys, and is validated.*

- Fold the 8 viz-page `*_viz.py` modules onto the shared `charts.py` core (finishes the unification).
- `class_weight='balanced'` option for imbalanced classification; `PreprocessingStep` fitted-state.
- Per-chart PNG export (needs a non-kaleido path) if static images are wanted.

---

## Recommended sequence & "start here"

```
A (run it) ──► B (make safe) ──► C (deploy) ──► D (frame) ──► E (validate + defense)
   ▲ start here                                                   F = optional, last
```

**Start here, today: Phase A.** It's the council's #1 priority, it's the prerequisite for
everything else, and Claude Code can execute A1–A5 immediately (Playwright + the installed
venv). The first real end-to-end browser run is where the unknown bugs live — find them
before a public URL or a defense does.

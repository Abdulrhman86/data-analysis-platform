# Backend Completion Workplan — to a 100% working product

Goal: take the backend from "works on the bundled sample + 47 green tests" to
"works on a real user's messy file, end-to-end, without silent failures or data
loss." Grounded in a 3-agent code audit (ML core · data/preprocessing/viz ·
tests/cross-cutting) — every item below cites real `file:line` findings, not
guesses.

## Baseline (what's already solid — not redone)
- Leakage-free sklearn `Pipeline`+`ColumnTransformer` (fit on train only; CV/tuning
  rebuild per fold), stratified split, `class_weight='balanced'`, configurable scaler.
- Security: XSS/JS-injection sanitized, pipeline-replay allowlist, JSON-safe numpy export.
- Data-quality engine robust (empty/all-NaN/50k guards, cached). `charts.py` pure + tested.
- Paths anchored to `__file__`; no bare `except:`; 47 tests green; Docker + deploy ready.

## Phases (priority order; each = implement → tests → browser-verify if UI → commit)

| # | Phase | Closes (audit findings) | Effort | Risk |
|---|-------|------------------------|--------|------|
| **1** | **Real-world data ingestion & type robustness** | `numeric_as_string` type leak (cols vanish from viz/preprocess); no numeric coercion at viz boundary (crash); CSV has no encoding/delimiter fallback; no empty/single-col/duplicate-header guards; Excel-date heuristic clobbers numeric cols; uncached `detect_column_types` | M | M |
| **2** | **Self-contained model (Predict & Export on raw data)** | saved/exported model isn't self-contained — page-3 feature engineering not bundled, so raw uploads to Predict fail/mispredict; single-model export omits preprocessing steps | M | M |
| **3** | **ML edge-case hardening** | multiclass ROC-AUC crashes when test split misses a class; CV folds not clamped to min-class/n-samples (tuning/CV crash on small data); regression can't use categorical features; zero-variance + high-cardinality columns degrade silently; `learning_curve` crashes on tiny data; feature-importance length-mismatch | M | M |
| **4** | **Preprocessing replay correctness** | log-transform silent `-inf`/NaN on replay; strategy-based impute freezes a stale mean/mode into params (should recompute); datetime re-detected/parsed every rerun; dead ECDF branch; hour-availability check; `.str.len()` on non-string | M | M |
| **5** | **Persistence hardening + observability** | dashboard persistence is all-or-nothing — one bad record wipes ALL dashboards (silent, `print`-only); ZERO logging in the codebase; ephemeral-disk caveat undocumented | S–M | Low |
| **6** | **Test coverage expansion (47 → ~85)** | untested: `model_export` round-trip, persistence robustness, multi-step replay, `feature_engineering`, page-level viz builders, messy-CSV parse, `Config` helpers | M–L | Low |
| **7** | **Chart-engine unification** | viz-page `*_viz.py` only partially folded onto `charts.py`: two divergent KDE impls (seaborn vs gaussian_kde), duplicate bar/pie/scatter/histogram/correlation; fold the safe Plotly ones, leave seaborn/pairplot | M | M |
| **8** | **Polish & cleanup** | dead `FeatureFlags` (wire or remove); debug `print`s → logger; magic numbers → `Config`; `pytest` warning filter; misc P2 | S | Low |

### Phase detail

**Phase 1 — ingestion robustness.** Reclassify `numeric_as_string` (coerce or treat as categorical) everywhere the numeric/categorical/datetime lists are built (`preprocessing_utils.detect_column_types` + all 3 consumers). Add `pd.to_numeric(errors="coerce")` + all-NaN warning at the viz/stat boundary (`single_variable_viz` et al.). `read_file`: retry `latin-1` + delimiter sniff (`sep=None, engine="python"`). De-dupe duplicate column names on upload; short-circuit empty/0-col with a clear message. Gate the Excel-serial-date heuristic on name keywords + full-column ratio. `@st.cache_data` on `detect_column_types`. **Tests:** messy-CSV parse suite.

**Phase 2 — self-contained model.** Bundle the page-3 `PreprocessingPipeline` (it already serializes via `to_dict` and replays via `.apply`) into the saved/exported artifact; in the Predict tab re-apply it to raw uploads before `model.predict`. Include `preprocessing_steps.json` in the single-model export path. **Tests:** train → export → reload → predict on RAW unseen rows matches.

**Phase 3 — ML edge cases.** ROC-AUC over present classes; clamp effective CV folds to `min(cv, min_class_count, n_samples)` with a warning; allow categorical features for regression (let `ColumnTransformer` encode; only target must be numeric); drop/flag zero-variance columns; cap one-hot cardinality (`max_categories`); guard `learning_curve` sample count + `n_jobs=1`; fix feature-importance name length on the encoded path. **Tests:** single-class, tiny-data, constant-feature, regression-with-categorical.

**Phase 4 — replay correctness.** Guard non-positive input in `apply_log_transform` (log1p/clip). Record imputation *strategy* (recompute mean/mode on replay) instead of freezing the value. Reuse cached `detect_column_types` instead of re-parsing datetimes each rerun. Fix the dead ECDF option string, the always-true `.dt.hour` check, and `.str.len()` on non-string object cols. **Tests:** multi-step replay determinism + strategy-recompute.

**Phase 5 — persistence + logging.** `save_dashboards`: atomic write (`tmp` + `os.replace`). `load_dashboards`: per-record try/except (skip one bad entry, keep the rest) + quarantine a corrupt file to `.corrupt` rather than overwrite. New `utils/logging_setup.py` (level from `LOG_LEVEL`); replace the 5 `print`s with `logger.exception`; log lifecycle (upload+shape, model trained, dashboard saved/loaded, replay failures). Document the ephemeral-disk caveat (caption + DEPLOY/DOCKER); optional `DASHBOARD_DIR` env override. **Tests:** truncated/malformed/partial-write persistence.

**Phase 6 — coverage.** Fill the remaining gaps from the audit's test plan not already covered by phases 1–5: `feature_engineering`, page-level viz builders (smoke-build a figure each), `Config.stepper/metric_card/empty_state`. Target ~85 tests.

**Phase 7 — chart unification (lower priority).** In order, each a small Plotly→Plotly PR + test: (1) KDE → `charts.build_kde`; (2) single-var bar/pie + relationship scatter → `charts.build_*`; (3) add `build_histogram_overlay`; (4) add `build_correlation`. Leave seaborn `pairplot`/countplot and bespoke time-series subplots as the deliberate "risky, not folded" set.

**Phase 8 — polish (lower priority).** Delete `ENABLE_RECENT_FILES`; wire or remove the other two flags. Debug `print`s → `logger.debug`. Lift `MAX_DASHBOARD_ROWS` + Excel-date bounds into `Config`. Add `filterwarnings` for the matplotlib pyparsing noise.

## Notes
- Phases 1–6 are the "100% working" core; 7–8 are polish/de-dup (optional, can stop after 6).
- Every phase is Claude-doable and verified (pytest + browser where UI-affecting). No operator action needed.
- Cadence: ask permission before each phase; on finishing a phase, ask before the next.

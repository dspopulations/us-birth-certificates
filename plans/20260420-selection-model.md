# Implementation plan: Bayesian selection model for DS livebirth ascertainment

**Target repo:** `dspopulations/us-birth-certificates`
**Branch (in progress):** `selection-model-phase-1`
**Status:**

| Phase | State | Commit(s) |
|---|---|---|
| 1. Foundation & data pipeline | ✅ complete | `f96ecac` |
| 2. Diagnostics module + rendering CLI | ✅ complete | `9ce27a1`, `c6a7c4b` (region removal) |
| 3. Parameter-recovery validation | ✅ complete | `679ce54` |
| 4. Real-data fits — scaffolding | ✅ complete | `2630985`, `2a9da20` (render-loop cleanup) |
| 4. Real-data fits — variant sweep | pending (overnight) | — |
| 5. Analysis & Quarto report — scaffolding | ✅ complete | `b8e04c9` |
| 5. Dev full-spec validation fit | ✅ complete | `94ad2e2` (findings note) |
| 6. Reproducibility polish (README) | ✅ complete | `b51d567` |
| idata size trim + dev profile bump | ✅ complete | this commit |

**Estimated remaining effort:** ~4–6 hours of focused work (cross-variant analysis, Quarto re-render review, write-up) plus a single overnight reporting-profile sweep (≈ 4–12 h wall-clock for 4 variants via `scripts/run_all_selection_variants.py`).

---

## 0. Before you start

Read in order:

1. `plans/docs/bayesian_selection_model.md` — full model specification and justification (design doc; still authoritative on the maths).
2. `plans/docs/dag.svg` — visual of the three-stage structure.
3. `src/dspopulations_us_birth_certificates/selection/` — docstrings on `priors.py`, `model.py`, `data.py`, `diagnostics.py`.
4. This plan — integration conventions and open phases.

If a design question is not answered by these, **stop and ask**. Identifiability properties are sensitive to exactly how priors, covariates, and stages interact; a plausible-looking deviation can silently break things. Section 10 lists invariants that must not move without agreement.

---

## 1. Context

### What existed before this plan

- A DuckDB-backed NCHS natality file (`data/us_births.db`) covering 1989–2024 with derived columns (`mage_c`, `mracehisp_c`, `down_ind`, etc.) built by `scripts/duckdb_prepare.py`.
- A LightGBM classifier producing `p_ds_lb_pred_01` per birth, driven by `scripts/fit_model.py`.
- A per-month `ds_pred_missing` flag (top-⌈1.5 × recorded⌉ non-recorded births).
- A Bayesian subpackage at `src/dspopulations_us_birth_certificates/bayes/` with one registered model (`m1-year-age`) driven by `scripts/fit_bayes_model.py`. This package is the **template for integration conventions** (config presets, script shape, Quarto template copying, plot-style reuse) — the selection-model subpackage mirrors its structure.

### What this plan adds

A three-stage Bayesian selection model that decomposes observed recording into:

```
P(R=1 | X) = θ_LB(age) · η(X) · s(X) + (1 − θ_LB·η) · f
```

- `θ_LB` — baseline DS livebirth rate in absence of screening (Morris 2002 / de Graaf 2015).
- `η = 1 − η_detect · η_term` — screening/termination pass-through.
- `s` — birth-certificate sensitivity (Boulet 2011 / Salemi 2017).
- `f` — false-positive rate (Ohio/NY study), fixed at 7.8e-5.

The LightGBM approach cannot separate `s` from `η_term`. The Bayesian model does so via external priors plus the Dobbs 2022 shock.

---

## 2. Deviations from the original plan

The plan in this file has been edited against reality. Material deviations to be aware of:

### 2.1 Package layout: integrated, not standalone

The original plan placed the code under a sibling package `src/ds_model/pymc/`. To stay consistent with the repo's actual layout, it lives inside the existing distribution package as **`dspopulations_us_birth_certificates.selection`**, parallel to the existing `dspopulations_us_birth_certificates.bayes`.

### 2.2 No state / region dimension

`data/us_births.db` has no state-level column (only `mbstate_rec` = maternal nativity). The model dropped `region` entirely in commit `c6a7c4b`:

- `build_model` no longer takes `n_region`.
- `eta_term_ry[region, year]` became `eta_term_year[year]` with the same pre/post-Dobbs heterogeneous sigma.
- Cells no longer carry `region_idx`.
- Variant D still works but identifies termination only via the national pre-vs-post-2022 year shift — a weaker test than the original plan's region × year contrast.

If restricted-use NCHS files or a state-linkage source land later, restoring the region dimension means reintroducing the old `eta_term_ry` block in `model.py`, the `region_idx` column in `data.py`, and the per-region Dobbs forest in `diagnostics.py` — all clean additions rather than refactors.

### 2.3 Recorded rate is ~5.3e-4, not ~1e-3

Plan §2.6 predicted `R_total` of 30–50 k and a rate near 1e-3. The actual DB produces ~17.8 k recorded DS over ~33.5 M births (rate ≈ 5.3e-4) across 2016–2024. This matches the existing `m1-year-age` model's `alpha_mu = logit(5e-4)` intercept and is **supportive** evidence for the selection model's premise (roughly half of true livebirths go unrecorded). Downstream prior-predictive checks in Phase 4/5 must be pinned to ~5e-4, not 1e-3.

### 2.4 `diagnostics.py` was written from scratch

The plan described `diagnostics.py` as "⚠️ PARTIAL"; no such file was delivered in `plans/src/`. The module under `selection/diagnostics.py` is a from-scratch implementation with six figure-returning functions plus tidy-DataFrame companions (see §5). `dobbs_forest_plot` became `dobbs_year_trajectory_plot` as a consequence of §2.2.

---

## 3. Local conventions (mirror these in new code)

### 3.1 Environment

- Python 3.14 via conda env **`dspop-us-birth-certificates`** (see `environment.yml`).
- Distribution name `dspopulations-us-birth-certificates`, import name `dspopulations_us_birth_certificates`. Installed editable (`pip install -e .`) as part of environment creation.
- `pyproject.toml` already lists every dependency the selection model needs: `pymc`, `arviz`, `duckdb`, `numpy`, `pandas`, `matplotlib`, `nutpie`, `numpyro`, `jaxlib`, `scipy`. **Do not add new dependencies without a clear justification**; the repo deliberately uses only what's in environment.yml.
- Shared sibling package `dse_research_utils` (imported from `../../dseinternational/research/src/python`) provides:
  - `environment.setup.init_script()` — matplotlib style, call at top of every script's `main()`.
  - `metadata.packages.report_package_versions(PACKAGE_LIST)` — environment reproducibility banner.
  - `plot.styles.FIGSIZE_*`, `COLOUR_*`, `DPI_*`, `TEXT_COLOUR` — style constants; **prefer these over hardcoded literals**.
- `PACKAGE_LIST` is re-exported from `dspopulations_us_birth_certificates.__init__`.

### 3.2 Quality gates

Before every PR, run both of:

```bash
ruff check src tests scripts
npm run spellcheck
```

- Real lint errors: fix them.
- False-positive cspell "unknown word" flags (author names, journal abbreviations, etc.): add to `config/spellcheck/allow-en.txt` — in alphabetical order in the matching section. Do **not** reword the prose.
- British English (en-GB) except for proper nouns / NVSS codings.

CI (`.github/workflows/ci.yml`) runs `ruff check src scripts tests`, `npm run spellcheck`, and `pytest`. The CI test job installs the package `--no-deps` and then pins only the light runtime deps; **heavy Bayesian deps (pymc/arviz/nutpie/jaxlib/numpyro) are not installed**. Consequently:

- Modules that import `pymc`/`pytensor` at the module top level will break CI import of the `selection` package.
- `selection/bayes/models.py` demonstrates the convention: `TYPE_CHECKING` import for type hints, runtime `import pymc as pm` inside the function body.
- **Follow-up (Phase 3 Task 3.0):** move the top-level `pymc` / `pytensor.tensor` imports in `selection/model.py` into `build_model` to match. (Phase 1 landed them at module top; tests pass locally because pymc is installed there, but CI will fail on the import.)

### 3.3 Script/CLI shape

Model `scripts/fit_selection_model.py` (to be written in Phase 4) on **`scripts/fit_bayes_model.py`**:

- Top-of-`main()` calls `setup.init_script()` and `package_metadata.report_package_versions(PACKAGE_LIST)`.
- Arg parsing returns a dataclass of CLI settings.
- Run-config preset dataclass (`BayesRunConfig` in `bayes.config`) with `dev` / `reporting` profiles; per-flag overrides via `dataclasses.replace`. **Reuse `BayesRunConfig`** (by importing it) or add a parallel `SelectionRunConfig` if the defaults need to diverge — the presets are the same shape either way.
- Status output via `cli_output` (`banner`, `section`, `info`, `success`, `warning`, `kv_table`) rather than print statements.
- Artefact layout under a run directory (`idata.nc`, `cells.parquet`, `config.json`, `run_config.json`, `summary.csv`, `plots/`, `tables/`).
- Optional `--render` flag calls `render_quarto()` on the template copied next to the run.

Default output directory: `output/selection/<variant>/<spec>/<timestamp>/`. The existing `bayes` pipeline writes to `output/bayes/<model-id>/<outcome>/<timestamp>/` — keep the parallel structure.

### 3.4 Quarto templates

One template per model variant at `docs/models/<model-id>/index.qmd`, pattern-matched on the `m1-year-age` template. It is **copied** into each run directory by `bayes.io.copy_docs_template` (or the selection equivalent) and rendered there, so each fit's HTML sits next to its own `idata.nc`/`cells.parquet`. Templates read artefacts via relative paths; they do not traverse the repo.

For the selection model, use a single template at `docs/models/selection/index.qmd` that loads the fit's variant from `config.json`, rather than one template per variant.

### 3.5 Notebooks

- Jupytext pairs: `.ipynb` gitignored; only `.py:percent` committed. Jupytext config is in `pyproject.toml`.
- Notebooks use `init_workbook()`, not `init_script()`.
- The user is moving away from notebooks — prefer package code + CLI scripts.

### 3.6 Data handling

- Raw NCHS microdata is under the NCHS Data Use Agreement — never commit records, summaries below publication-ready aggregation, or anything that could identify a birth.
- `data/` is gitignored.
- The SQL pipeline is in `scripts/duckdb_prepare.py`; `src/dspopulations_us_birth_certificates/variables.py` documents every column and code meaning. Derived columns used by the selection model: `year`, `mage_c`, `mracehisp_c`, `meduc`, `pay_rec`, `gestrec10`, `ca_cchd`, `ab_nicu`, `ab_aven1`, `down_ind`.

---

## 4. Repository layout (actual)

```
dspopulations/us-birth-certificates/
├── data/                                   # gitignored (NCHS DUA)
│   └── us_births.db
├── plans/
│   ├── 20260420-selection-model.md         # this file
│   ├── docs/bayesian_selection_model.md    # design doc
│   ├── docs/dag.svg                        # DAG figure
│   └── src/                                # original delivery (kept for history)
├── docs/
│   ├── analysis/predicted.qmd              # existing classifier analysis
│   └── models/
│       ├── m1-year-age/index.qmd           # bayes pipeline template
│       ├── usbc10/index.qmd                # lightgbm pipeline template
│       └── selection/index.qmd             # NEW (Phase 5 Task 5.1)
├── src/dspopulations_us_birth_certificates/
│   ├── bayes/                              # existing — HSGP cell-model pipeline
│   ├── selection/                          # NEW — this plan
│   │   ├── __init__.py                     ✅ (Phase 1)
│   │   ├── priors.py                       ✅ (Phase 1)
│   │   ├── model.py                        ✅ (Phase 1)
│   │   ├── simulate.py                     ✅ (Phase 1)
│   │   ├── data.py                         ✅ (Phase 1)
│   │   ├── diagnostics.py                  ✅ (Phase 2)
│   │   └── config.py                       pending (Phase 4 Task 4.0)
│   └── ...
├── scripts/
│   ├── fit_bayes_model.py                  existing — template for selection CLI
│   ├── render_selection_diagnostics.py     ✅ (Phase 2)
│   ├── fit_selection_model.py              pending (Phase 4 Task 4.1)
│   └── run_all_selection_variants.py       pending (Phase 4 Task 4.2)
├── tests/
│   ├── test_selection_priors.py            ✅ 10 tests
│   ├── test_selection_simulate.py          ✅ 10 tests
│   ├── test_selection_model_compile.py     ✅  7 tests
│   ├── test_selection_data.py              ✅  9 tests
│   ├── test_selection_diagnostics.py       ✅  9 tests
│   ├── test_render_selection_diagnostics.py ✅ 3 tests
│   └── test_selection_parameter_recovery.py  pending (Phase 3 Task 3.1)
├── output/
│   └── selection/<variant>/<spec>/<ts>/    NEW — fit artefacts (gitignored)
└── config/spellcheck/allow-en.txt          add author/journal terms here
```

---

## 5. Phase 1 — Foundation & data pipeline (COMPLETE)

Commit `f96ecac`. 36 tests landed across four files (priors / simulate / model compile / data).

Delivered:

- `selection/priors.py` — Morris/Natoli/Kuppermann/Boulet values; four variants A/B/C/D; `ModelPriors` dataclass (read §10 before editing values).
- `selection/model.py` — `build_model(cells, priors, *, spec, n_year, post_dobbs_year_start)` returning a `pm.Model` across four specs (`theta_only`, `theta_s`, `single_eta`, `full`); helpers `extract_true_counts`, `posterior_subgroup_rate`.
- `selection/simulate.py` — `TrueParams.from_priors(...)` and `simulate_cells(...)` for parameter-recovery validation.
- `selection/data.py` — `prepare_cells(con, *, year_range, post_dobbs_year, table, columns)` returning a cell frame with integer-index columns + `N_cell` / `R_cell` totals. Schema-drift handled by an explicit `columns` override.
- `tests/test_selection_*.py` — priors lock-down, simulate shape/determinism, model-compile × 4 specs, data SQL / in-memory DuckDB fixture.

Real-data self-test: 60 057 cells, `N_total = 33 498 266`, `R_total = 17 776`, rate ≈ 5.3e-4. `cells.attrs = {n_year: 9, post_dobbs_year_start: 6, year_range: (2016, 2024), N_total, R_total}`.

---

## 6. Phase 2 — Diagnostics module + rendering CLI (COMPLETE)

Commits `9ce27a1` (initial) + `c6a7c4b` (region removal). 12 additional tests (9 diagnostics, 3 CLI).

Delivered in `selection/diagnostics.py`:

1. `identifiability_pairplot(idata)` + `identifiability_table(idata)` — per-race scatter of `eta_term_race` vs `s_race` draws, with `|r|>0.7` flagged as prior-driven.
2. `dobbs_year_trajectory_plot(idata, *, post_dobbs_year_start)` + `dobbs_year_trajectory_table(idata, ...)` — year-level trajectory of `eta_term_year` with pre/post-Dobbs colouring; headline mean(post)−mean(pre) shift and CI.
3. `cchd_consistency_check(idata, cells, *, published_cchd_prevalence=0.225)` + `cchd_consistency_summary(...)` — posterior CCHD prevalence among true DS livebirths vs EUROCAT target.
4. `posterior_predictive_by_stratum(idata, cells, *, stratum_col)` — observed vs predicted recorded counts, aggregated by year/race/age.
5. `decomposition_by_race(idata, cells)` — stacked recorded/missed + implied prenatal terminations; tidy frame attached to `fig._selection_data`.
6. `age_curve_check(idata, cells)` + `age_curve_table(idata)` — posterior `θ_LB` per 1,000 livebirths vs Morris/de Graaf anchors.

Parity helpers `summary_table` and `convergence_health` mirror `bayes.diagnostics`.

Delivered in `scripts/render_selection_diagnostics.py`:

- Arg parsing with either `--fit-dir` (auto-discovers `idata.nc` and `cells.parquet`) or explicit `--idata / --cells / --out-dir`.
- Reads `post_dobbs_year_start` from `cells.attrs` unless overridden via CLI.
- Saves PNG + SVG per figure under `<out-dir>/plots/` and tidy CSVs under `<out-dir>/tables/` — mirrors `bayes.plots._save`.
- Logs through `cli_output`; each diagnostic is guarded so one failure does not kill the run.
- Prefers a cached `summary.csv` on re-runs; computes fresh via `az.summary` only when the cache is absent (commit `2a9da20`).

---

## 7. Phase 3 — Parameter recovery validation (COMPLETE)

Commit `679ce54`. 9 slow-marked tests covering both array-parameter 95% CI coverage (≥ 70% on `theta_lb_age`, `eta_term_race`, `eta_term_year`, `s_race`) and scalar tolerance (±0.5 logit on `eta_term_int`, `s_int`, `s_preterm`, `s_cchd`).

### Task 3.0 — Clean up pymc import convention *(done)*

Moved `import pymc as pm` / `import pytensor.tensor as pt` inside `build_model` under a `TYPE_CHECKING` guard. The `selection` subpackage now imports cleanly in pymc-free environments (the CI test job), matching the `bayes.models` convention.

### Task 3.1 — Parameter-recovery test *(done)*

**File:** `tests/test_selection_parameter_recovery.py`

Verify the Bayesian fit recovers known parameters from simulated data. If this fails, nothing downstream is trustworthy.

Mark with `@pytest.mark.slow` and exclude from default `pytest -q` via `addopts = "-q -m 'not slow'"` in `pyproject.toml`. Document `pytest -m slow` to opt in.

Landed as `tests/test_selection_parameter_recovery.py`, 9 cases, runs in ~20 s on nutpie (vs. hours on the pymc default). 70% coverage threshold on small arrays accommodates the discrete-coverage variance on N∈{6,7,9}. Hard R̂ < 1.05 gate so broken fits don't slip past as bad recovery.

---

## 8. Phase 4 — Real-data fits

### Task 4.0 — `selection.config` + run-config preset *(done, `2630985`)*

`src/dspopulations_us_birth_certificates/selection/config.py` adds `SelectionModelConfig` (JSON-serialisable snapshot with variant/spec/year_range/post_dobbs_year/priors/notes) and `selection_run_config(name)` factory returning `BayesRunConfig` with selection-tuned presets.

Profiles (updated on `2025-04-20`):

| profile | draws | tune | chains | target_accept | sampler |
|---|---:|---:|---:|---:|---|
| `dev` | 1000 | 1000 | 2 | 0.9 | nutpie |
| `reporting` | 1500 | 1500 | 4 | 0.95 | nutpie |

The `dev` preset was bumped from 400×2 after the dev-validation fit (`94ad2e2`) landed max R̂ 1.02 / min ESS 100 on the full spec — under the reporting gates of R̂ < 1.01 / ESS ≥ 400. `1000×2` should clear ESS on the named RVs without the ~1 h wall-clock of reporting.

### Task 4.1 — `fit_selection_model.py` CLI *(done, `2630985`)*

`scripts/fit_selection_model.py` mirrors `scripts/fit_bayes_model.py`:

```
--variant {A,B,C,D}                 # sensitivity variant
--spec {theta_only,theta_s,single_eta,full}   # default: full
--profile {dev,reporting}           # default: dev
--years YYYY-YYYY                   # default 2016-2024
--post-dobbs-year YYYY              # default 2022
--duckdb-path data/us_births.db
--output-dir output/selection/<variant>/<spec>/<ts>    # auto
--prior-only                        # skip NUTS, run prior-predictive only
--draws / --tune / --chains / --target-accept  # profile overrides
--render                            # quarto render after fit
```

Pipeline: `selection.prepare_cells` → `selection.build_model` → `bayes.sampling.sample` (prior predictive + NUTS + posterior predictive) → `bayes.io.save_artefacts` → `bayes.io.copy_docs_template("selection")` → `selection.render.render_all` → optional `bayes.io.render_quarto`.

The rendering loop was factored into `selection/render.py` so the post-hoc `render_selection_diagnostics.py` and the fit CLI share a single code path (commit `2630985`). A subsequent cleanup (`2a9da20`) removed a duplicate `az.summary` call that was doubling wall-clock time on per-cell deterministics.

### Task 4.1c — `docs/models/selection/index.qmd` *(done, `2630985` + `b8e04c9`)*

Single Quarto template branching on `variant` and `spec` read from `config.json`. Embeds the six diagnostic figures, a convergence summary, a side-by-side classifier comparison section (added in Task 5.2), and explicit caveats on the no-region model and the dev-vs-reporting distinction.

### Task 4.2 — Baseline `theta_only` run on real data *(done)*

`output/selection/C/theta_only/baseline/` — R̂ ≤ 1.01 on all 7 `theta_lb_age` parameters, min ESS 928. Posterior rates sit 21–43 % of Morris at every age band (monotonicity preserved). This is the expected signature of selection in `theta_only`: with η = s = 1 imposed, the selection pressure absorbs into `θ_LB`. The plan's original criterion ("within 0.2 logit") was over-optimistic given 60 k cells × 33.5 M births; the revised acceptance is monotone-with-age plus systematic ~0.4× shift below Morris.

### Task 4.2a — Per-cell Deterministic trim *(done, this commit)*

`selection/model.py` now declares only `p_ds_lb` and `p_recorded` as per-cell `pm.Deterministic`. The previous `theta_lb`, `eta_detect`, `eta_term`, `eta`, and `s` are inline tensor expressions, shrinking the saved `idata.nc` by ~5/7. `decomposition_by_race` reconstructs per-cell `theta_lb` from `theta_lb_age[age_idx]` and computes `terminated = theta_lb − p_ds_lb` (the algebraic identity, no `eta_detect`/`eta_term` needed).

Measured size: theta_only prior-only goes from 298 MB → 46 MB. Full-spec reporting extrapolates from the dev-validation 4.2 GB to ~1 GB per variant (still large; watch the `_run_logs/` disk budget during overnight sweeps).

### Task 4.3 — Fit all four variants (full spec, reporting profile)

**Driver:** `scripts/run_all_selection_variants.py` *(done, this commit)*

```bash
# Overnight run. All four variants, full spec, reporting profile.
python scripts/run_all_selection_variants.py --profile reporting --render

# Resumable after interruption.
python scripts/run_all_selection_variants.py --profile reporting --render --skip-existing

# Extra flags passthrough (e.g. tighter target_accept).
python scripts/run_all_selection_variants.py --profile reporting \
    --extra-args "--target-accept 0.98"
```

Runs the variants sequentially as subprocesses so a failure in one doesn't sink the batch. Per-variant log goes to `output/selection/_run_logs/<batch_ts>_<variant>.log`.

**Wall-time estimate** (nutpie, 60 k cells, `reporting` profile after the Deterministic trim): 1–3 h per variant for a total of ~4–12 h. Post-trim idata.nc should be ~1 GB per variant (~4 GB total). The dev profile at 1000×2 is reasonable for inner-loop iteration at ~30 min per fit.

**Acceptance per variant:**

- Max R̂ < 1.01 (hard fail)
- Min ESS bulk > 400 on the named RVs
- Divergences < 10 (re-run at `--target-accept 0.98` if not)
- `summary.csv`, `plots/`, `tables/` all present under the run dir

### Task 4.4 — Identifiability review

For each variant's `tables/identifiability.csv`:

- |r| > 0.7 on most race panels → decomposition is **prior-driven**. Report prominently; do not over-interpret individual race effects on η_term vs s.
- |r| < 0.7 AND Variant C race effects agree with Variant D (Dobbs-only) → **genuine identification**.
- In between → **partial** — report with caveats.

**Dev-validation preview** (variant C, full spec, `output/selection/C/full/dev_validation/`): all six race panels have |r| ≤ 0.15, well below the 0.7 threshold (see `notes/202604201706-selection-full-spec-dev-validation.md`). The clinical-marker → `s` channel plus the year-level Dobbs signal identify the decomposition even without state-level contrast. **Positive result for the no-region model.** Reporting-profile runs should confirm.

**Caveat on the Dobbs year effect** (also in the validation note): dev-profile posterior mean(post) − mean(pre) = +0.69 on logit — *opposite* sign to what Dobbs would predict. The no-region model cannot separate year-level termination from year-level detection drifts. The Dobbs trajectory is a diagnostic, not a causal estimate, until state-level data restores `eta_term_ry[region, year]`.

---

## 9. Phase 5 — Analysis & Quarto report

### Task 5.1 — Quarto template *(done, `2630985`)*

`docs/models/selection/index.qmd` matches the `m1-year-age` pattern. Copy-into-run via `bayes.io.copy_docs_template("selection", out_dir)` at the end of `fit_selection_model.py`. Sections:

1. Run metadata callout (reads `config.json` + `run_config.json`).
2. Cell aggregation summary.
3. Convergence section (max R̂ + min ESS from `summary.csv`).
4. Age curve — Stage 1 sanity check against Morris anchors.
5. Identifiability pair-plot + table.
6. Dobbs year trajectory + effect-size summary row.
7. CCHD consistency.
8. Decomposition by race.
9. Posterior predictive checks (PPC by year / race / age).
10. Classifier comparison (Task 5.2).
11. Caveats.

### Task 5.2 — Classifier comparison *(done, `b8e04c9`)*

Section 10 of the template computes headline Bayesian numbers (total DS livebirths, NH Black share of missed, CCHD co-occurrence in missed) directly from `idata.nc` + `cells.parquet` and puts them alongside placeholder classifier values drawn from `docs/analysis/predicted.qmd`. Final polish (pulling the classifier values automatically from the upstream pipeline) can happen once the reporting-profile variant runs land.

### Task 5.3 — Cross-variant aggregation script *(done, `b8e04c9`)*

`scripts/compare_selection_variants.py` auto-discovers the latest `output/selection/{A,B,C,D}/full/<ts>/` directories (or accepts explicit `--fit-dirs`), aggregates posterior means + 95% CIs for `total_true`, per-race `eta_term_race` / `s_race`, identifiability |r|, and the Dobbs year-effect summary into a long-format `comparison.csv`, plus a forest-plot figure of the two race-effect families across variants. 4 tests against synthetic fit dirs.

### Task 5.x — Dev full-spec validation *(done, `94ad2e2`)*

`output/selection/C/full/dev_validation/` + `notes/202604201706-selection-full-spec-dev-validation.md`. Key findings:

- All six identifiability panels data-informed (|r| ≤ 0.15).
- Race effects qualitatively prior-consistent, magnitudes larger than priors (data is contributing).
- Dobbs year effect has **wrong sign** — flagged as a no-region identification weakness. The year trajectory is a diagnostic, not a causal estimate, without state-level data.
- idata.nc size drove the per-cell Deterministic trim (Task 4.2a above).
- Dev profile at 400×2 undershoots convergence gates; bumped to 1000×2.

### Phase 5 still to do (after Task 4.3 reporting sweep)

- Re-render each variant's Quarto against the reporting-profile fit.
- Run `scripts/compare_selection_variants.py --profile reporting` across the four variants and append a cross-variant analysis section to the master report.
- Write-up of findings (separate note under `notes/`).

---

## 10. Phase 6 — Reproducibility

### Task 6.1 — Documentation *(done, `b51d567`)*

- Top-level `README.md` has an **Analyses** section pointing at both Bayesian pipelines.
- `src/dspopulations_us_birth_certificates/selection/README.md` summarises the public API, fit-→-diagnostics flow, output layout, profile presets, variants, and invariants.
- `CLAUDE.md` / `AGENTS.md` / `.github/copilot-instructions.md` updated in lockstep (commit `679ce54`) with the `pytest -m slow` marker convention.

### Task 6.2 — CI *(no changes needed)*

The existing workflow (`.github/workflows/ci.yml`) runs `ruff check src scripts tests`, `npm run spellcheck`, and `pytest`. Tests marked `@pytest.mark.slow` are deselected by the default `-m 'not slow'` in `pyproject.toml`, so CI skips the pymc-dependent and long-running cases without needing the Bayesian stack in CI's pinned install. The `selection` subpackage imports cleanly in the pymc-free CI env (verified via `sys.meta_path` blocking).

### Task 6.3 — No Makefile *(decided, no work)*

The repo uses `scripts/` entry points instead. The entry points now include `scripts/fit_selection_model.py`, `scripts/render_selection_diagnostics.py`, `scripts/compare_selection_variants.py`, and `scripts/run_all_selection_variants.py`.

---

## 11. Acceptance criteria (overall)

The implementation is complete when:

- [x] Phase 1 — package structure, priors, simulator, data aggregator.
- [x] Phase 2 — diagnostics module and rendering CLI.
- [ ] Task 3.0 — pymc import convention in `selection/model.py`.
- [ ] Task 3.1 — parameter-recovery test passes.
- [ ] `output/selection/C/full/<ts>/` exists with max R̂ < 1.01, min ESS > 400, 0 divergences.
- [ ] The identifiability pair-plot for Variant C is documented in the Quarto report.
- [ ] Variants A, B, C, D all fitted and compared in the Quarto report.
- [ ] CCHD-consistency check for Variant C has a 95% CI containing the EUROCAT target (≈22.5%); otherwise documented as a limitation.
- [ ] `docs/models/selection/index.qmd` renders without errors.
- [ ] Repo `README.md` references the new analysis.
- [ ] `ruff check src tests scripts` and `npm run spellcheck` both clean.

---

## 12. Design invariants (do not change without approval)

These are not suggestions; they have specific identifiability consequences.

1. **Morris priors stay tight** (`MORRIS_SIGMA = 0.10` on logit). Loosening lets the data drag `θ_LB` around, absorbing variation that belongs to η or s.
2. **Clinical features (CCHD, NICU, Aven, Preterm) enter only `s`**, never `η`. They are observed after the pregnancy filters and cannot causally influence detection or termination. Adding them to η is a causal-structure violation.
3. **False-positive rate `f` is fixed at 7.8e-5**, not estimated. Ohio/NY pins it.
4. **Reference levels:** Race = NH White (idx 0); Education = Some college (idx 2); Payer = Private (idx 1). Changing references changes every other coefficient's interpretation.
5. **Year coding is `year − year_start`** (0-based within the window). `post_dobbs_year_start = 2022 − year_start`. Do not decouple.
6. **Stage 1 is θ_LB (baseline livebirth rate), not θ (conception rate).** That is what makes Morris directly usable and removes the need for an η_loss stage.
7. **Region is intentionally absent** (§2.2). Do not fake a region (e.g. `mbstate_rec` is not a region). Restore only if a genuine state-level column appears.

If you find a reason to change any of these, stop and discuss before implementing.

---

## 13. When in doubt

- **Data schema questions.** Inspect `data/us_births.db` with `DESCRIBE us_births` before editing `selection/data.py`. `src/.../variables.py` is authoritative on code meanings.
- **Convergence failures.** If R̂ > 1.01 after `--target-accept 0.98`, stop — something is structurally wrong. Don't crank up draws as a workaround.
- **Prior modifications.** Values in `priors.py` come from specific publications (docstrings cite them). Don't tune them to fit the data better.
- **Identifiability diagnostic.** If \|r\| > 0.7 for race effects on η_term vs s, that is a **finding** about the model, not a bug. Report it; don't paper over it by tightening a prior.
- **Pre-commit checks.** `ruff check` and `npm run spellcheck` are required before merging. Add false-positive words to the allowlist rather than rewording.

---

## 14. Estimated remaining effort

| Task | Human effort | Machine effort |
|---|---|---|
| Task 4.3 — overnight variant sweep | ~0.5 h kick off + review | **4–12 h MCMC** overnight |
| Task 4.4 — identifiability review across 4 variants | ~1 h | negligible |
| Task 5-post — re-render Quarto, run `compare_selection_variants.py`, append cross-variant analysis | ~2–3 h | negligible |
| Task 5-writeup — findings note under `notes/` | ~1–2 h | — |
| **Total human-attended** | **~4–6 h** | — |

Everything else (Phases 1–3, Phase 4 scaffolding, Phase 5 scaffolding + dev validation, Phase 6 docs) is done on the `selection-model-phase-1` branch. The overnight sweep is the last technical gate; the remaining human effort is interpretation.

---

*End of plan.*

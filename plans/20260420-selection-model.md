# Implementation plan: Bayesian selection model for DS livebirth ascertainment

**Target repo:** `dspopulations/us-birth-certificates`
**Branch (in progress):** `selection-model-phase-1`
**Status:**

| Phase | State | Commit(s) |
|---|---|---|
| 1. Foundation & data pipeline | ✅ complete | `f96ecac` |
| 2. Diagnostics module + rendering CLI | ✅ complete | `9ce27a1`, `c6a7c4b` (region removal) |
| 3. Parameter-recovery validation | pending | — |
| 4. Real-data fits (4 variants) | pending | — |
| 5. Analysis & Quarto report | pending | — |
| 6. Reproducibility polish | pending | — |

**Estimated remaining effort:** ~10–15 hours of focused work plus ≥20 hours of MCMC wall-clock in Phase 4.

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
- Writes a `convergence_summary.csv` with max Rhat / min ESS.

---

## 7. Phase 3 — Parameter recovery validation

### Task 3.0 — Clean up pymc import convention *(small, do first)*

**File:** `src/dspopulations_us_birth_certificates/selection/model.py`

Currently imports `pymc as pm` and `pytensor.tensor as pt` at module top. CI installs the package without pymc/pytensor, so `from dspopulations_us_birth_certificates import selection` will fail at collection time. Fix by matching the `bayes.models` pattern:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pymc as pm
    import pytensor.tensor as pt


def build_model(...) -> "pm.Model":
    import pymc as pm
    import pytensor.tensor as pt
    ...
```

**Acceptance:** `ruff check` stays clean; tests still pass; `python -c "from dspopulations_us_birth_certificates.selection import build_model"` succeeds in an environment without pymc installed (the selection package imports, `build_model` raises ImportError only if called).

### Task 3.1 — Parameter-recovery test

**File:** `tests/test_selection_parameter_recovery.py`

Verify the Bayesian fit recovers known parameters from simulated data. If this fails, nothing downstream is trustworthy.

Mark with `@pytest.mark.slow` and exclude from default `pytest -q` via `addopts = "-q -m 'not slow'"` in `pyproject.toml`. Document `pytest -m slow` to opt in.

```python
@pytest.mark.slow
def test_parameter_recovery_full_spec() -> None:
    truth = TrueParams.from_priors(variant_C_default(), seed=42)
    cells = simulate_cells(
        truth, n_cells_per_month=60,
        n_year=9, post_dobbs_year_start=6, seed=42,
    )
    model = build_model(
        cells, variant_C_default(), spec="full",
        n_year=9, post_dobbs_year_start=6,
    )
    with model:
        idata = pm.sample(
            500, tune=500, chains=2, target_accept=0.9,
            random_seed=42, progressbar=False,
        )

    # 95% CI coverage over at least 80% of each parameter family.
    params = {
        "theta_lb_age": truth.theta_lb_age_logit,
        "eta_term_race": truth.eta_term_race,
        "eta_term_year": truth.eta_term_year,
        "s_race": truth.s_race,
    }
    for name, true_vals in params.items():
        post = idata.posterior[name]
        lo = post.quantile(0.025, dim=("chain", "draw")).values
        hi = post.quantile(0.975, dim=("chain", "draw")).values
        covered = ((true_vals >= lo) & (true_vals <= hi)).mean()
        assert covered >= 0.8, f"{name}: only {covered:.0%} covered"
```

Note the plan's original `eta_term_ry` entry is now `eta_term_year`.

**Acceptance:** test passes. If it fails, the model is mis-specified or the sampler is too short — investigate before moving on to Phase 4.

---

## 8. Phase 4 — Real-data fits

### Task 4.0 — `selection.config` + run-config preset

**File:** `src/dspopulations_us_birth_certificates/selection/config.py` (NEW)

Either:

- Re-export `BayesRunConfig` from `bayes.config` (simplest — the shape is identical), **or**
- Add a `SelectionRunConfig` with selection-specific `dev` / `reporting` presets. Suggested `reporting` defaults: `draws=1000`, `tune=1000`, `chains=4`, `target_accept=0.95`, `nuts_sampler="nutpie"`. (Selection needs higher `target_accept` than the bayes M1 model.)

Add a `SelectionModelConfig` dataclass (analogous to `BayesModelConfig`) snapshotting `variant`, `spec`, `year_range`, `post_dobbs_year`, `priors` digest, and `notes`.

### Task 4.1 — `fit_selection_model.py` CLI

**File:** `scripts/fit_selection_model.py` (NEW)

Mirror `scripts/fit_bayes_model.py`. Flags:

```
--variant {A,B,C,D}                 # sensitivity variant
--spec {theta_only,theta_s,single_eta,full}   # default: full
--profile {dev,reporting}           # default: dev
--years YYYY-YYYY                   # default 2016-2024
--db-path data/us_births.db
--output-dir output/selection/<variant>/<spec>/<ts>    # auto
--prior-only                        # skip NUTS, run prior-predictive only
--draws / --tune / --chains / --target-accept  # profile overrides
--render                            # quarto render after fit
```

Pipeline steps (all logged via `cli_output.section`):

1. Load cells (`selection.prepare_cells`).
2. Build model (`selection.build_model(spec, variant_*, ...)`).
3. Prior-predictive (always).
4. Sample (unless `--prior-only`).
5. Posterior predictive.
6. Write artefacts: `idata.nc`, `cells.parquet`, `config.json`, `run_config.json`, `summary.csv`.
7. Copy `docs/models/selection/index.qmd` into the run dir.
8. Render diagnostics via `selection.diagnostics.*` through the shared `render_selection_diagnostics.py` codepath (extract the save-loop into a reusable function).
9. Optionally `quarto render` if `--render`.

### Task 4.2 — Baseline `theta_only` run on real data

```bash
python scripts/fit_selection_model.py \
    --variant C --spec theta_only \
    --profile reporting
```

**Verification:**

- Max R̂ < 1.01
- Posterior `theta_lb_age` means tight around Morris prior (within 0.2 on logit scale)
- Observed recorded DS age distribution roughly matches `θ_LB(age) · η · s` with η, s at their prior-mean scalar anchors (~0.5 × 0.4 ≈ 0.2, so ratio ≈ 0.2 vs Morris rates).

If this fails, **stop** — the data aggregation or age coding is wrong. Don't try the full model yet.

### Task 4.3 — Fit all four variants (full spec)

**File:** `scripts/run_all_selection_variants.py` (NEW — a Python runner, not a shell script, to match repo conventions)

```python
# Iterates variants A, B, C, D; shells out to fit_selection_model.py
# or calls its main() in-process for shared setup; records each variant's
# output path into a summary manifest.
```

**Wall-time estimate** (without region dimension, ~60k cells, `reporting` profile, 4 chains × 1000 draws, target_accept=0.95): 1–3 hours per variant on a modern workstation with `nutpie`; longer with the pymc default sampler.

**Acceptance per variant:**

- Max R̂ < 1.01 (hard fail if violated)
- Min ESS bulk > 400
- Divergences < 10 (re-run at `--target-accept 0.98` if not)
- `summary.csv`, `plots/`, `tables/` all present under the run dir

### Task 4.4 — Identifiability review

For each variant's identifiability pair-plot (`plots/identifiability.png` + `tables/identifiability.csv`):

- \|r\| > 0.7 on most race panels → decomposition is **prior-driven**. Report prominently; do not over-interpret individual race effects on η_term vs s.
- \|r\| < 0.7 and Variant C race effects agree with Variant D (Dobbs-only) → **genuine identification**.
- In between → **partial** — report with caveats.

Under the no-region model, Variant D's Dobbs signal is a national pre-vs-post-2022 year shift rather than a treated-vs-untreated-state contrast, so identification is weaker than the original plan anticipated. Expect some prior-driven decomposition on at least a subset of race panels.

---

## 9. Phase 5 — Analysis & Quarto report

### Task 5.1 — Quarto template at `docs/models/selection/index.qmd`

Write a single template that:

- Reads `config.json` to discover variant, spec, year range.
- Loads `summary.csv`, `tables/*.csv`, and the figures from `plots/`.
- Sections mirror `docs/models/m1-year-age/index.qmd` where applicable:
  1. Run metadata callout.
  2. Headline numbers — posterior total DS livebirths 2016–2024 with 95% CI vs recorded count.
  3. Age-specific rates — posterior θ_LB · η vs observed recorded rates.
  4. Race-specific decomposition — the `decomposition_by_race` plot plus an identification commentary driven by the `identifiability.csv` \|r\| column.
  5. Dobbs year-trajectory analysis — the trajectory plot plus the summary effect size.
  6. Sensitivity variants — a side-by-side table across A/B/C/D for key demographic effects (requires reading from sibling fit dirs; see Task 5.3).
  7. CCHD consistency — `cchd_consistency` figure + summary.

Copy-into-run pattern (see `bayes.io.copy_docs_template`) keeps template in version control while each run's HTML sits next to its artefacts.

### Task 5.2 — Classifier comparison section

Append a comparison section reading from both pipelines' output:

| quantity | classifier | Bayesian | note |
|---|---|---|---|
| Total predicted DS livebirths 2016–2024 | from `docs/analysis/predicted.qmd` | posterior CI | classifier fixed at 1.5× quota |
| NH Black share among missing | classifier | posterior CI | classifier ~flat; model decomposes |
| CCHD co-occurrence in missing | classifier 25.6% | posterior CI | true ~22.5% |

### Task 5.3 — Cross-variant aggregation script

**File:** `scripts/compare_selection_variants.py` (NEW) — model on the existing `scripts/compare_variants.py`. Reads the four variant runs' `summary.csv` + `tables/`, builds the side-by-side table for Task 5.1 §6, and writes a combined figure under a parent run dir (e.g. `output/selection/_compare_<ts>/`).

---

## 10. Phase 6 — Reproducibility

### Task 6.1 — Documentation

- Add a "Selection model" section to the repo `README.md` pointing at `plans/docs/bayesian_selection_model.md` and `docs/models/selection/index.qmd`.
- Add a short module `README.md` at `src/dspopulations_us_birth_certificates/selection/README.md` summarising the public API and the fit-→-diagnostics-→-render pipeline.
- **Do not** create a top-level `USAGE.md` — the repo has no such convention; docstrings + the Quarto template cover it.

### Task 6.2 — CI

The existing workflow (`.github/workflows/ci.yml`) runs `ruff check src scripts tests`, `npm run spellcheck`, and `pytest`. The test job installs the package without pymc — see Task 3.0. No workflow changes are needed if Task 3.0 lands. Do not add a slow-tests job in CI; `@pytest.mark.slow` tests need pymc and take minutes — run locally before release.

### Task 6.3 — No Makefile

The original plan proposed `Makefile` targets. The repo uses `scripts/` entry points instead. Do not add a Makefile; the entry points already in `scripts/` plus the new `scripts/fit_selection_model.py` and `scripts/run_all_selection_variants.py` are the "Makefile".

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

| Phase | Tasks | Human effort | Machine effort |
|---|---|---|---|
| 3. Recovery test + pymc-import cleanup | 3.0, 3.1 | ~1.5 h | ~5 min runtime |
| 4. Real data fits (4 variants) | 4.0–4.4 | ~4–6 h code | ~4–12 h MCMC |
| 5. Analysis & Quarto | 5.1–5.3 | ~4–6 h | negligible |
| 6. Reproducibility | 6.1–6.3 | ~1 h | — |
| **Total human-attended** | | **~10–15 h** | |

Phase 4 is dominated by wall-clock MCMC time on `output/selection/<variant>/full/`, not implementation effort.

---

*End of plan.*

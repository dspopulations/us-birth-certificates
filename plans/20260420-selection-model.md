# Implementation plan: Bayesian selection model for DS livebirth ascertainment

**Target agent:** Claude Code CLI
**Target repo:** `dspopulations/us-birth-certificates`
**Estimated scope:** 15–25 hours of focused work, 6 phases, ~15 discrete tasks

## 0. Before starting

Read these in order — they contain the design rationale and will clarify many "why are we doing it this way" questions:

1. `docs/bayesian_selection_model.md` — full model specification and justification
2. `docs/dag.svg` — visual of the three-stage structure
3. `src/README.md` — implementation overview
4. This plan

If you hit a design question not answered by these, **stop and ask** rather than inventing an answer. The model has specific identifiability properties that depend on exactly how priors, covariates, and stages interact; plausible-looking deviations can silently break things.

---

## 1. Context

### What exists already

The repo currently has:
- A DuckDB-backed NCHS natality file (`data/us_births.db`)
- A LightGBM classifier producing `p_ds_lb_pred_01` per birth
- A Quarto report at `docs/analysis/predicted.qmd` that compares recorded vs predicted-missing DS births
- A flag `ds_pred_missing` derived from a per-month quota of ⌈1.5 × recorded⌉ top non-recorded births

### What we're adding

A three-stage Bayesian selection model that decomposes observed recording into:

```
P(R=1 | X) = θ_LB(age) · η(X) · s(X) + (1 − θ_LB·η) · f
```

where

- `θ_LB` is the baseline DS livebirth rate in the absence of screening (Morris 2002 / de Graaf 2015)
- `η = 1 − η_detect · η_term` is the screening/termination pass-through
- `s` is BC sensitivity (Boulet 2011)
- `f` is the small false-positive rate (Ohio/NY study)

The classifier approach is structurally unable to separate `s` from `η_term`; the Bayesian model uses external priors + the Dobbs 2022 natural experiment to decompose them.

### Deliverables already produced

You should find these in the repo at:

```
docs/
├── bayesian_selection_model.md          # full model report
└── dag.svg                              # published-quality DAG

src/ds_model/pymc/
├── README.md                            # module overview
├── priors.py                            # ✅ complete — encodes Morris/Natoli/Boulet values
├── model.py                             # ✅ complete — PyMC model builder, 4 staged specs
├── simulate.py                          # ✅ complete — synthetic-data generator
├── data.py                              # ✅ complete — DuckDB → cells
├── fit.py                               # ✅ complete — end-to-end CLI runner
└── diagnostics.py                       # ⚠️ PARTIAL — see Task 2.1
```

Everything marked ✅ has been sanity-tested and runs without errors. `diagnostics.py` was cut off mid-implementation and needs completion.

---

## 2. Repository layout (target)

```
dspopulations/us-birth-certificates/
├── data/
│   └── us_births.db                     # existing
├── docs/
│   ├── bayesian_selection_model.md      # existing
│   ├── dag.svg                          # existing
│   └── analysis/
│       ├── predicted.qmd                # existing classifier analysis
│       └── bayesian.qmd                 # NEW — posterior analysis report
├── src/
│   └── ds_model/
│       ├── __init__.py                  # NEW
│       └── pymc/
│           ├── __init__.py              # NEW
│           ├── priors.py                # existing
│           ├── model.py                 # existing
│           ├── simulate.py              # existing
│           ├── data.py                  # existing
│           ├── fit.py                   # existing
│           ├── diagnostics.py           # COMPLETE (Task 2.1)
│           └── README.md                # existing
├── tests/                                # NEW directory
│   ├── __init__.py
│   ├── test_priors.py                   # NEW (Task 1.2)
│   ├── test_simulate.py                 # NEW (Task 1.3)
│   ├── test_model_compile.py            # NEW (Task 1.4)
│   ├── test_data.py                     # NEW (Task 1.5)
│   └── test_parameter_recovery.py       # NEW (Task 3.1) — slow test
├── fits/                                # NEW directory (git-ignore)
│   ├── variantA.nc
│   ├── variantB.nc
│   ├── variantC.nc                     # main specification
│   └── variantD.nc
├── scripts/
│   └── run_all_variants.sh              # NEW (Task 4.2)
├── pyproject.toml                       # update (Task 1.1)
└── IMPLEMENTATION_PLAN.md               # this file
```

---

## 3. Phase 1: Foundation & integration

### Task 1.1 — Package setup

**Goal:** Establish the Python package structure and dependencies.

**Steps:**
1. Create `src/ds_model/__init__.py` (can be empty) and `src/ds_model/pymc/__init__.py` that re-exports `build_model`, `ModelPriors`, and `prepare_cells`.
2. Update `pyproject.toml` (or create one) to include:
   - `pymc >= 5.10`
   - `arviz >= 0.17`
   - `duckdb >= 0.10`
   - `pandas`, `numpy`
   - `matplotlib` (for diagnostics)
   - Dev deps: `pytest`, `pytest-xdist`
3. Verify `from ds_model.pymc import build_model, variant_C_default, prepare_cells` works in a fresh Python session.

**Acceptance:**
- `pip install -e .[dev]` succeeds
- `python -c "from ds_model.pymc import build_model; print('OK')"` prints OK
- `pytest --collect-only` lists tests (even if the test files are empty)

---

### Task 1.2 — Unit tests for priors

**File:** `tests/test_priors.py`

**Goal:** Lock down the published-literature values so later refactors can't silently change the model's behaviour.

**Tests to write:**

```python
def test_morris_age_rates_monotone():
    """Morris rates should increase with maternal age (except the terminal flattening)."""
    # ... assert MORRIS_THETA_LB[:-1] is monotone non-decreasing

def test_morris_rates_match_de_graaf_2015():
    """Values must match the de Graaf 2015 corrigendum to EJHG."""
    expected = [0.66, 0.70, 0.84, 1.48, 4.72, 15.22, 30.71]
    # ... assert np.allclose(MORRIS_THETA_LB_PER_1000, expected)

def test_factor_level_lengths_match_arrays():
    """Prior arrays must be the same length as their factor-level vocabularies."""
    # ... for race, education, payer: check ETA_DETECT_*, ETA_TERM_*, S_*

def test_sensitivity_variants_differ_as_expected():
    """Variant A tightens s, widens eta_term; Variant B the reverse."""
    # ... assert variant_A.s_race_sigma < variant_C.s_race_sigma
    # ... assert variant_B.eta_term_race_sigma < variant_C.eta_term_race_sigma
    # ... assert variant_D.eta_term_race_sigma > variant_C.eta_term_race_sigma

def test_logit_round_trip():
    """logit/inv_logit should be inverse."""
    # ... for p in [0.1, 0.5, 0.9]: assert np.isclose(inv_logit(logit(p)), p)
```

**Acceptance:** All tests pass.

---

### Task 1.3 — Unit tests for simulate.py

**File:** `tests/test_simulate.py`

**Tests to write:**

```python
def test_simulate_produces_valid_cells():
    """Output DataFrame has all required columns with correct dtypes."""
    # ... required cols: year_idx, age_idx, race_idx, edu_idx, payer_idx,
    #     region_idx, preterm, cchd, nicu, aven, N_cell, R_cell
    # ... R_cell <= N_cell everywhere

def test_recorded_rate_in_expected_range():
    """Total recorded rate should be plausible (5e-4 to 2e-3)."""
    # Real-world range is ~9e-4 to 1.2e-3 per livebirth

def test_true_probabilities_stored():
    """Columns true_theta_lb, true_eta, true_s present for recovery tests."""

def test_rng_determinism():
    """Same seed -> same cells DataFrame."""
    # ... assert simulate_cells(..., seed=0).equals(simulate_cells(..., seed=0))

def test_recorded_increases_with_age():
    """Older-mother cells should have higher recorded DS rates (in expectation)."""
    # ... group by age_idx, check monotone trend in R_cell / N_cell
```

**Acceptance:** All tests pass. This is CPU-fast (no PyMC).

---

### Task 1.4 — Model compile test

**File:** `tests/test_model_compile.py`

**Goal:** Verify each of the four specs builds and survives prior predictive sampling.

**Tests to write:**

```python
@pytest.mark.parametrize("spec", ["theta_only", "theta_s", "single_eta", "full"])
def test_build_model_and_prior_predict(spec, tiny_cells):
    """All four spec levels compile and draw from prior predictive."""
    model = build_model(tiny_cells, variant_C_default(), spec=spec,
                       n_year=9, n_region=4, post_dobbs_year_start=6)
    with model:
        prior = pm.sample_prior_predictive(draws=10, random_seed=0)
    assert "R_obs" in prior.prior_predictive
    # R_obs must be nonnegative integer-valued and <= N_cell
    # ...
```

Use a pytest fixture `tiny_cells` to build ~200-row synthetic cells for all tests. Mark this test module with `@pytest.mark.slow` if it takes >30s.

**Acceptance:** All four specs compile; prior predictive draws are valid.

---

### Task 1.5 — Unit tests for data.py

**File:** `tests/test_data.py`

**Goal:** Verify the DuckDB → cells pipeline. Because we don't want to depend on the real database being present, build a small in-memory DuckDB with known contents.

**Tests to write:**

```python
def test_prepare_cells_maps_raw_to_indices():
    """Raw NCHS codes map correctly to factor indices."""
    # Build a DuckDB with 100 rows of known coded data
    # Call prepare_cells(con)
    # Assert age_idx, race_idx, edu_idx, payer_idx all in valid ranges

def test_prepare_cells_filters_year_range():
    """Year filter (2016-2024) is respected."""

def test_prepare_cells_drops_unmapped():
    """Rows with unmapped race/edu codes are dropped."""

def test_prepare_cells_aggregates_correctly():
    """N_cell and R_cell sum correctly across cell grouping."""
    # Build a DF with known (age, race, ...) combos and known down_ind values
    # After aggregation, N_cell and R_cell should match hand-computed totals

def test_preterm_derivation():
    """gestrec10 1-5 -> preterm=1, 6-10 -> preterm=0, 99 -> dropped."""

def test_dobbs_classification():
    """TX, AL, AR -> 1; CA, NY, MA -> 0."""
```

**Acceptance:** All tests pass; no dependency on the real `us_births.db`.

---

### Task 1.6 — Adapt data.py to your actual schema

**Goal:** The `data.py` SQL uses column names that match the current 2003-revised birth certificate (`mracehisp`, `ab_nicu`, etc.) but **you must verify these match your DuckDB schema exactly**.

**Steps:**
1. Open `data/us_births.db` and run `DESCRIBE births` (or the equivalent).
2. Compare column names to what `data.py` uses.
3. If names differ, update `data.py` — either by changing the SQL or by adding a column-alias mapping dict at the top of the file.
4. If encoding differs (e.g., race coded 0–9 rather than 1–9, or as strings), update the `*_MAP` dicts.
5. Run `python -m ds_model.pymc.data` (the self-test at the bottom of the file) — it should print cell counts without errors.
6. Run `python -c "import duckdb; from ds_model.pymc.data import prepare_cells; con = duckdb.connect('data/us_births.db', read_only=True); c = prepare_cells(con); print(c.shape, c.attrs)"` — should print a cells DataFrame summary from real data.

**Acceptance:**
- `prepare_cells` on real data returns a non-empty DataFrame
- `cells.attrs` contains `n_year`, `n_region`, `post_dobbs_year_start` with sensible values
- Total `N_cell` sum is in the expected range (~30M livebirths for 2016–2024)
- Total `R_cell` sum is in the expected range (~30k–50k recorded DS)
- Ratio is approximately 1e-3 (1 per 1,000 livebirths)

**If real data fails these bounds:** stop and report the discrepancy. Don't silently adjust the model to match wrong data.

---

## 4. Phase 2: Diagnostics module

### Task 2.1 — Complete diagnostics.py

**File:** `src/ds_model/pymc/diagnostics.py` — currently partial. The file ends mid-function `decomposition_by_race`.

**Goal:** Complete the five diagnostic functions.

**Functions to finish:**

1. **`identifiability_pairplot(idata)`** — ✅ already complete
2. **`dobbs_forest_plot(idata, post_dobbs_year_start)`** — ✅ already complete
3. **`cchd_consistency_check(idata, cells, published_cchd_prevalence=0.225)`** — ✅ already complete
4. **`posterior_predictive_by_stratum(idata, cells, stratum_col)`** — ✅ already complete
5. **`decomposition_by_race(idata, cells)`** — ⚠️ incomplete — finish the implementation and add the title/labels at the end

Then add one additional function:

6. **`age_curve_check(idata, cells)`** — posterior mean θ_LB by age band vs Morris/de Graaf published values. A sanity check that the Stage 1 prior is being respected and not being pulled around by data fitting.

**Acceptance:**
- `python -c "from ds_model.pymc.diagnostics import *"` imports successfully
- All six functions return matplotlib Figures
- A smoke test in `tests/test_diagnostics.py` runs each function on a small InferenceData (produced from fitting `spec='full'` on tiny synthetic cells) and verifies the figure contains at least one Axes object

---

### Task 2.2 — Diagnostics-rendering CLI

**File:** `src/ds_model/pymc/render_diagnostics.py` — NEW

**Goal:** A script that takes a fitted InferenceData and writes all diagnostic figures to disk.

```bash
python -m ds_model.pymc.render_diagnostics \
    --idata fits/variantC.nc \
    --cells fits/cells_variantC.parquet \
    --out-dir docs/figures/variantC/
```

Writes: `identifiability.png`, `dobbs_forest.png`, `cchd_consistency.png`, `ppc_by_year.png`, `ppc_by_race.png`, `ppc_by_age.png`, `decomposition.png`, `age_curve.png`.

**Acceptance:** Running the script on a fitted InferenceData produces all 8 PNGs at ≥150 DPI.

---

## 5. Phase 3: Parameter recovery validation

### Task 3.1 — Parameter-recovery test

**File:** `tests/test_parameter_recovery.py`

**Goal:** Verify the Bayesian fit can recover known parameters from simulated data. If this doesn't pass, nothing downstream is trustworthy.

**Mark this test `@pytest.mark.slow`** and exclude it from the default `pytest` run — add `-m "not slow"` to default and document how to run slow tests.

**Test:**

```python
@pytest.mark.slow
def test_parameter_recovery_full_spec():
    """
    Simulate data from a known truth, fit the full model, verify
    posterior means are within 95% CI of the true values for the main
    parameters.
    """
    truth = TrueParams.from_priors(variant_C_default(), seed=42)
    cells = simulate_cells(
        truth, n_cells_per_month=60,
        n_year=9, n_region=4, post_dobbs_year_start=6, seed=42,
    )
    model = build_model(cells, variant_C_default(), spec="full",
                       n_year=9, n_region=4, post_dobbs_year_start=6)
    with model:
        idata = pm.sample(500, tune=500, chains=2, target_accept=0.9,
                         random_seed=42, progressbar=False)

    # For each parameter family, check that the posterior 95% CI
    # contains the true value for at least 80% of the parameters
    # (95% CI coverage is the gold standard, but we allow slack for
    # finite-chain noise).
    params = {
        "theta_lb_age": truth.theta_lb_age_logit,
        "eta_term_race": truth.eta_term_race,
        "s_race": truth.s_race,
    }
    for name, true_vals in params.items():
        post = idata.posterior[name]
        lo = post.quantile(0.025, dim=("chain", "draw")).values
        hi = post.quantile(0.975, dim=("chain", "draw")).values
        covered = ((true_vals >= lo) & (true_vals <= hi)).mean()
        assert covered >= 0.8, f"{name}: only {covered:.0%} of true values in 95% CI"
```

**Acceptance:** test passes. If it fails, the model is mis-specified or the sampler is too short — investigate before moving on.

---

## 6. Phase 4: Real-data fits

### Task 4.1 — Run theta-only baseline on real data

**Goal:** Before the full model, fit the simplest spec to confirm that the Morris age curve reproduces observed age patterns.

```bash
python -m ds_model.pymc.fit \
    --db data/us_births.db \
    --spec theta_only \
    --variant C \
    --draws 500 --tune 500 --chains 4 \
    --output fits/theta_only.nc
```

**Verification:**
- Max R̂ < 1.01
- Posterior `theta_lb_age` means are tight around the Morris prior (within 0.2 on logit scale)
- The implied DS livebirth rate per age group, multiplied by a constant ~0.4, matches the observed recorded-DS age pattern

If this doesn't work, **stop** — something is wrong with the age coding or the data aggregation. Don't try to fit the full model.

---

### Task 4.2 — Run all four sensitivity variants on full spec

**File:** `scripts/run_all_variants.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
for V in A B C D; do
    python -m ds_model.pymc.fit \
        --db data/us_births.db \
        --spec full \
        --variant "$V" \
        --draws 1000 --tune 1000 --chains 4 \
        --target-accept 0.95 \
        --posterior-predictive \
        --output "fits/variant${V}.nc"
done
```

**Expected wall time:** ~4–8 hours per variant on a workstation, longer if using state-level region grouping. Run overnight.

**Acceptance per variant:**
- Max R̂ < 1.01 (hard fail if violated)
- Min ESS bulk > 400
- Divergences < 10 (ideally 0); if >10, increase `target-accept` to 0.98 and re-run
- Sidecar `.meta.json` written with correct convergence stats

---

### Task 4.3 — Diagnostic reports for each variant

After Task 4.2 finishes, for each variant:

```bash
for V in A B C D; do
    python -m ds_model.pymc.render_diagnostics \
        --idata "fits/variant${V}.nc" \
        --cells "fits/cells.parquet" \
        --out-dir "docs/figures/variant${V}/"
done
```

**Critical check:** open `docs/figures/variantC/identifiability.png`. For each race panel:
- If posterior correlation `|r| > 0.7` for most race panels: **the decomposition is prior-driven**. Report this prominently; do not over-interpret posterior race effects on η_term and s individually.
- If correlations are <0.7 and Variant C race-effect estimates agree with Variant D (Dobbs-only identification): **genuine identification**. The decomposition can be interpreted.
- Something in between: **partial identification**; report with appropriate caveats.

---

## 7. Phase 5: Analysis & comparison to existing work

### Task 5.1 — Bayesian analysis Quarto report

**File:** `docs/analysis/bayesian.qmd`

Mirror the structure of `docs/analysis/predicted.qmd` but use the Bayesian fit instead of the classifier.

Sections to include:
1. **Headline numbers** — posterior total DS livebirth estimate 2016–2024, with 95% CI, compared to the recorded count
2. **Age-specific rates** — posterior θ_LB · η against observed recorded rates, by age band
3. **Race-specific decomposition** — the decomposition_by_race plot, with explicit identification commentary based on the pair-plot diagnostic
4. **Dobbs analysis** — the forest plot, with state-by-state effect sizes
5. **Sensitivity variants** — side-by-side posterior estimates across variants A/B/C/D for the key demographic effects
6. **CCHD consistency** — the cchd_consistency_check output

Each figure should reference the corresponding section of `bayesian_selection_model.md` for methodology.

---

### Task 5.2 — Comparison to classifier approach

Add a final section to `docs/analysis/bayesian.qmd` comparing:

| quantity | classifier estimate | Bayesian posterior | note |
|---|---|---|---|
| Total predicted DS livebirths | ~99,035 | ... [CI] | classifier uses fixed 1.5× quota |
| NH Black share among missing | ... | ... [CI] | classifier shows ~flat; model decomposes to η_term and s |
| CCHD co-occurrence in missing | 25.6% | ... [CI] | true value ~22.5% |

Flag disagreements explicitly.

---

## 8. Phase 6: Reproducibility

### Task 6.1 — Documentation updates

- Update repo `README.md` with a section "Bayesian analysis" pointing to `docs/bayesian_selection_model.md` and `docs/analysis/bayesian.qmd`
- Add a `USAGE.md` showing the full pipeline: DuckDB → prepare_cells → fit → diagnostics → Quarto report
- Ensure all module docstrings have runnable examples in their top-level docstrings

### Task 6.2 — CI

Add a `.github/workflows/ci.yml` that runs:
- `pytest -m "not slow"` on push/PR
- `pytest -m slow` nightly only (it takes ~5 minutes)

Do not run `scripts/run_all_variants.sh` in CI — it's too slow.

### Task 6.3 — Makefile

Add a `Makefile` with targets:
- `make test` — run fast tests
- `make test-slow` — run slow tests including parameter recovery
- `make fit-theta` — run Task 4.1
- `make fit-all-variants` — run Task 4.2
- `make diagnostics` — render all diagnostic figures for all variants
- `make report` — render the Quarto `bayesian.qmd`

---

## 9. Acceptance criteria (overall)

The implementation is complete when:

- [ ] All tests in `tests/` pass, including the slow parameter-recovery test
- [ ] `fits/variantC.nc` has max R̂ < 1.01, min ESS > 400, 0 divergences
- [ ] The identifiability pair-plot for Variant C has been inspected and documented in the Quarto report
- [ ] Variants A, B, C, D all fitted and their posteriors compared in the Quarto report
- [ ] The CCHD-consistency check for Variant C produces a 95% CI containing the EUROCAT target (≈22.5%); if not, documented as a limitation
- [ ] The Quarto report `docs/analysis/bayesian.qmd` renders without errors and produces a publication-grade HTML
- [ ] Repo README references the new analysis

---

## 10. Design decisions that must NOT be changed without explicit approval

These are not suggestions; they're requirements with specific identifiability consequences:

1. **Morris rates stay tight (σ=0.10 on logit).** Loosening lets the data drag θ_LB around, absorbing variation that belongs to η or s.
2. **Clinical features (CCHD, NICU, Aven, Preterm) enter only `s`**, never `η`. They are observed after the pregnancy filters and cannot causally influence detection or termination. If you're tempted to add them to η "for better fit", don't — it's a causal-structure violation.
3. **False-positive rate is fixed, not estimated.** The Ohio/NY study pins it; estimating it would add a poorly-identified parameter.
4. **Reference levels for categorical effects.** Race reference = NH White (index 0). Education reference = Some college (index 2). Payer reference = Private (index 1). Changing these changes the interpretation of every other coefficient.
5. **Year coding is year − year_start (0-based within window).** The Dobbs classification depends on `post_dobbs_year_start = 2022 - year_start`; don't decouple these.
6. **Stage 1 is θ_LB (baseline livebirth rate), not θ (conception rate).** This is what makes the η_loss stage unnecessary and what makes Morris directly usable. Converting to conception rates would require adding η_loss back, which is not identifiable.

If you find a reason to change any of these, stop and discuss before implementing.

---

## 11. When in doubt

- **Ask about data schema questions.** The SQL in `data.py` is a reasonable default but may not exactly match the production DuckDB schema. Don't guess — inspect and ask.
- **Ask about convergence failures.** If R̂ > 1.01 after `target_accept=0.98`, something is structurally wrong — don't just keep cranking up draws.
- **Ask before modifying priors.** The values in `priors.py` come from specific publications and were chosen deliberately. Don't tune them to fit the data better.
- **Ask if identifiability diagnostic fails.** If |r| > 0.7 for race effects on η_term vs s, that's a finding about the model, not a bug to fix. Report it; don't paper over it by tightening a prior.

## 12. Glossary of files produced

After completion:

```
Code:
  src/ds_model/pymc/*.py                  ~1,500 lines
  tests/test_*.py                         ~500 lines
  scripts/run_all_variants.sh, Makefile

Reports:
  docs/bayesian_selection_model.md        ~700 lines (existing)
  docs/analysis/bayesian.qmd              new, ~400 lines

Data:
  fits/variantA.nc, B.nc, C.nc, D.nc      4 × ~100–500 MB InferenceData

Figures:
  docs/figures/variantC/*.png             ~8 files × ~150 KB
  docs/figures/variantA/*.png             similar
  ...
  docs/dag.svg                            ~10 KB (existing)
```

## 13. Estimated total effort

| Phase | Tasks | Est. effort |
|---|---|---|
| 1. Foundation | 1.1–1.6 | 3–5 hours |
| 2. Diagnostics | 2.1–2.2 | 2–3 hours |
| 3. Recovery test | 3.1 | 1 hour + ~5 min runtime |
| 4. Real data fits | 4.1–4.3 | 1–2 hours code + 20+ hours runtime (overnight) |
| 5. Analysis | 5.1–5.2 | 3–5 hours |
| 6. Reproducibility | 6.1–6.3 | 1–2 hours |
| **Total human-attended** | | **~15–20 hours** |

Phase 4 is dominated by wall-clock MCMC time, not implementation effort.

---

*End of plan.*

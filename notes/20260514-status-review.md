# Project status review — US birth certificates, DS ascertainment

> [!WARNING]
> This note was drafted by an AI coding assistant (Claude, Opus 4.7) on
> 2026-05-14 after the bug-sweep PR (#38) was opened on
> `dev/frank/project-review`. Inventory, timeline, and pipeline structure
> are pulled directly from the repo and PR history; the interpretation
> of findings and the prioritised next steps are AI-synthesised from the
> existing `notes/` corpus and should be confirmed by a human reviewer
> before any external citation or planning use.

## 1. Project aims

From `plans/readme.md`:

1. **Document** DS births 1989–2024 and compare against surveillance-based
   estimates.
2. **Identify** predictors of *recorded* DS cases 2016–2024; train, prune,
   evaluate.
3. **Use** the predictive model to identify *likely missed* cases
   2016–2024 and estimate the true total.
4. **Document** the predicted-missing cohort 2016–2024 against
   surveillance and against prior co-occurring-condition estimates.
5. **Develop** a statistical model of factors influencing DS live births
   and how those factors evolve.

The reason this is non-trivial: NCHS birth-certificate sensitivity for DS
is well below 100%; national recording sits at roughly 35–40%, so naïve
counts under-state births by about 60–65%. The
project is trying to close that gap with ML *and* a structural Bayesian
model.

> **Correction (2026-08-04).** This passage originally attributed the ~40%
> recording rate to Boulet (2011). Boulet reports **18%** sensitivity for Down
> syndrome, measured in metropolitan Atlanta; 40% appears nowhere in the paper.
> The national figure survives — transporting Boulet's estimate to national
> recording level gives 0.374, and Salemi's gives 0.319 — but the attribution
> was wrong. See
> [the study-area transport note](20260804-salemi-boulet-study-area-transport.md).

## 2. Approach

Three layers, all wired through DuckDB.

**Data layer.**  `scripts/download_data.py` → `import_parquet.py` →
`combine_parquet.py` → `prepare_parquet.py` → `duckdb_create.py` →
`duckdb_prepare.py`. The pipeline lands a single `data/us_births.db`
with harmonised columns (race/ethnicity recodes that bridge the 1989,
2003/4, and 2014 NCHS revisions; combined Down-syndrome target
`ca_down_c`; `mage_c`, `mrace_c`, `mhisp_c`, `mracehisp_c`).

**ML layer.**  LightGBM, declarative model definitions, 9 variants in
`MODELS`:

| Family | C+P label | C-only label |
|---|---|---|
| Full features (`usbc10_*`) | M0, M1, M2 | M0_CN, M1_CN |
| Clinical + maternal-age only (`usbc11_*`) | M0, M1 | M0_CN, M1_CN |

Each variant has tuned hyper-params committed to its class. Predicted
probabilities and a year×month `ceil(1.5 × recorded)` "predicted-
missing" flag are written back to DuckDB (one column-pair per variant)
by `scripts/fit_model.py --write-predictions`.

**Bayesian layer.**  `selection/` implements a three-stage decomposition

```
P(R=1 | X) = theta_LB(age) · eta(X) · s(X) + (1 − theta_LB·eta) · f
```

where θ_LB is the Morris/de Graaf baseline, η = 1 − η_detect · η_term
covers screening-pass-through, *s* is birth-certificate sensitivity
(Boulet/Salemi), *f* is the small false-positive rate. Four staged
builds (`theta_only`, `theta_s`, `single_eta`, `full`) and three
prior-sensitivity variants (A: tight *s* / loose η_term; B: tight
η_term / loose *s*; C: default). NUTS via nutpie.

Separate `output/bayes/m1-year-age/` Bayesian *outcomes* regression also
exists (continuous *t* = year + month/12, HSGP on age) — used for
descriptive trend analysis, not the structural decomposition.

**Engineering scaffolding.**  Reproducibility manifest (git SHA + env +
package versions + schema hash), per-run `RunConfig` presets
(`dev / test / reporting`), Quarto templates per model family, CI with
ruff + pytest + cspell, 122 fast tests + slow parameter-recovery test
(currently runs locally only — pymc not in CI).

## 3. Current status

### What works

- **Engineering quality is high.** The April 17–25 sprint delivered 29
  PRs that turned a notebook prototype into a registered, tested,
  reproducible pipeline. Today's bug sweep removed 22 latent issues
  without breaking a test. `ruff check src tests scripts` is clean.
- **The data pipeline is correct after fixes.** Two notable bugs landed
  today: a duplicate `ca_disor` column in `load_predictors_data`
  (DuckDB returning two same-name columns), and the `mrace_c` year
  guard that was nulling out 2014–2019 entirely. A pre-existing rebuilt
  DB will still have those NULLs until `duckdb_prepare.py` is re-run.
- **The ML side appears sound.** After the apgar5/apgar10 filter bug
  (#26) was found and all four model families re-tuned and re-pruned
  (#29), the predicted-missing cohort under C-only labels behaves as
  physicality predicts: shifts toward advanced maternal age, more
  severe clinical outcomes, recovers the Boulet race-ethnicity signal
  once sociodemographic features are pruned away.
- **A reportable comparison exists.** `analyse_predicted` +
  `compare_predicted` produce Quarto reports across 30 variables × 3
  populations × 4 model variants.

### What doesn't work

**The three-stage Bayesian selection model gives implausible posteriors
at production scale.** From
`notes/202604202200-selection-reporting-sweep-findings.md`: across all
three variants (A/B/C),

- θ_LB sits 10–15σ above Morris.
- *s_int* sits 6–10σ below the external sensitivity evidence.
- η_detect_int sits ~6σ below prior.
- Implied total true DS livebirths 2016–2024 ≈ 231k vs surveillance
  ≈ 378k (~42k/year).
- Implied BC sensitivity 4–7% vs ~37% nationally (transported from
  Boulet's 18% metro-Atlanta measurement; the ≈ 40% originally cited
  here as Boulet's own figure is withdrawn — see the correction above).

Cause: with 33.5M rows the likelihood overwhelms even σ = 0.10 priors.
The Morris prior is *not* tight enough at this data scale. The
identifiability pair-plot diagnostic the model ships with looks at
race-level residuals only and missed the intercept-level collinearity.
Design invariant #1 ("Morris priors stay tight") is empirically broken.

The plumbing is good; the model is not yet usable for inference.

**Aim 1 (1989–2024 longitudinal) is partially blocked.** The race
recoding bridges 1989-cert / 2003-cert / 2014-cert codings *for the
mother* (`mrace_c` / `mhisp_c` / `mracehisp_c`), but there's no
paternal equivalent yet and the `mracehisp` artefact note
(`202604201112-mracehisp-error.md`) documents how easy it is to mix
coding regimes silently. The current modelling sticks to 2016+ for
exactly this reason.

### Open decisions (per the notes)

1. **C-only vs C+P as the primary label.**
   `notes/20260422-compare-confirmed-only.md §1` shows that under C+P
   the model rides a `ca_disor = Pending` tautology (29% of predicted-
   missing vs 4.5% baseline). Under C-only the tautology collapses.
   C-only is the cleaner label, but downstream analyses (selection
   model, the Bayesian outcomes model, the predicted-analyses report)
   all still use C+P. Recommendation in the note: switch the default.
2. **`apgar5` post-fix retention.** Newly retained after the filter
   fix but not deeply analysed. Importance was 2.65e-3 under C-only,
   the largest gain in that re-derivation — may want a closer look.
3. **Prior magnitudes after today's race-label fix.** PR #38 corrected
   `RACE_LEVELS` positions 2 vs 3 (data has narrow NH AIAN at idx 2
   and broad NH Asian/PI/Other at idx 3; labels said the opposite).
   The *values* of `ETA_TERM_RACE` / `S_RACE` / `ETA_DETECT_RACE` at
   those positions weren't moved — they may now be applying to the
   wrong demographic and need re-derivation against Boulet/Natoli/
   Salemi.

### Inventory

- **Code:** 73 Python files in `src/scripts/tests` (excluding
  notebooks); 122 fast tests; 1 slow parameter-recovery test.
- **Models:** 9 LightGBM variants registered + 3 Bayesian selection
  variants × 4 staged builds + 2 Bayesian outcome variants
  (`recorded` / `recorded+predicted`).
- **Outputs committed:** 4 Bayesian-outcome run directories under
  `output/bayes/m1-year-age/` (gitignored except for index.qmd
  templates). Per-variant LightGBM runs and the selection-model A/B/C
  reporting sweep are reproducible but their artefacts aren't in git
  (NCHS DUA).
- **Cadence:** intense April 17–25 (29 substantive PRs); quiet April
  26 → May 13 (only dependabot bumps); today's bug sweep is the first
  substantive commit in 19 days.

## 4. Suggested next steps, prioritised

### Immediate (this week-ish)

1. **Land PR #38** (bug sweep). Re-run `duckdb_prepare.py` against an
   existing DB to refresh `mrace_c` for 2014–2019. Re-run the LightGBM
   `--write-predictions` step to refresh `ds_pred_missing_*` columns
   now that the `ca_disor` duplicate column is gone.
2. **Re-evaluate the corrected `RACE_LEVELS` against the prior
   magnitudes** — confirm the offsets at positions 2 and 3 are still
   defensible against Boulet 2011 / Natoli 2012 for the newly-corrected
   demographics, or update them.

### Short term (1–2 sprints)

3. **Pick the canonical label.** Decide whether downstream is C-only,
   C+P, or both-as-comparison. The `ca_disor` tautology argument is
   strong; the note recommends C-only as primary.
4. **Selection model v2.** The current model needs a redesign before
   its outputs can be cited. Options from
   `notes/202604202200-selection-reporting-sweep-findings.md`:
   - Replace σ = 0.10 logit priors with σ ≈ 0.01 or a pinned-
     deterministic-plus-residual structure that genuinely forbids
     drift at 33.5M-row scale.
   - Reconsider invariant #2 (clinical features enter only *s*):
     epidemiologically CCHD is *caused by* DS in ~50% of cases, so the
     model's "CCHD independent of DS-status" structure is the wrong
     sign.
   - Add an intercept-level identifiability scatter
     (θ_LB × *s_int*) so the failure mode that bit this sweep would
     be visible diagnostically, not only via post-hoc σ counts.
   - Calibrate against surveillance (~42k DS births/year) as a
     structural constraint, not just a post-hoc check.

### Medium term

5. **Aim 1 (longitudinal 1989–2024).** Needs paternal `fracehisp_c`
   recode + an audit of any other regime-breaking raw NCHS columns.
   Sketch already in `notes/202604201112-mracehisp-error.md`.
6. **Bayesian outcomes model M2/M3.** Add ethnicity (M2) and education
   (M3) dims now that the monthly-resolution M1 baseline is
   established. The `smooth_coords` refactor in PR #15 was written for
   this.

### Quality of life

7. The pre-PR-#38 `output/` directory has stale runs from before
   today's fixes (notably anything that reads `ca_disor` or `mrace_c`
   for 2014–2019). After the data pipeline is re-run, the M0 fit →
   permutation prune → M1 retune → final fit cycle from PR #29 needs
   to run again across all four families to refresh everything
   cleanly.

## 5. Headline

Engineering is in good shape; the ML side produces credible predicted-
missing cohorts (especially under C-only); the structural Bayesian
model is the load-bearing scientific blocker. The current pause looks
like a deliberate one — the April 25 retune-and-write-predictions PR
was a natural breakpoint, and the next move is a non-trivial scientific
decision (selection-model v2) rather than more engineering.

# PyMC implementation: Bayesian selection model for DS livebirth ascertainment

This directory contains a runnable PyMC implementation of the three-stage
Bayesian selection model described in `../bayesian_selection_model.md`.

## Files

| file | purpose |
|---|---|
| `priors.py` | Published-literature values encoded as informative priors (Morris, Natoli, Kuppermann, Boulet, Salemi); four sensitivity-analysis variants |
| `model.py` | PyMC model builder with four staged specs (`theta_only`, `theta_s`, `single_eta`, `full`) |
| `simulate.py` | Forward simulator generating synthetic cell-level data from a known ground truth; used for parameter-recovery validation |

## Quick start

```bash
pip install pymc arviz pandas numpy
```

Run the self-tests:

```bash
python priors.py     # prints the Morris age curve and baseline rates
python simulate.py   # generates synthetic data and prints summary statistics
```

End-to-end fit on synthetic data (parameter-recovery test):

```python
from priors import variant_C_default
from simulate import TrueParams, simulate_cells
from model import build_model
import pymc as pm

truth = TrueParams.from_priors(variant_C_default(), seed=42)
cells = simulate_cells(
    truth, n_cells_per_month=30,
    n_year=9, n_region=4, post_dobbs_year_start=6, seed=42,
)

model = build_model(
    cells, variant_C_default(), spec="full",
    n_year=9, n_region=4, post_dobbs_year_start=6,
)
with model:
    idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9)
```

## Expected workflow on real data

1. **Aggregate NCHS natality files to cells.** Using DuckDB, group 2016–2024
   US birth records by the factor combination in `priors.AGE_LEVELS` ×
   `RACE_LEVELS` × `EDU_LEVELS` × `PAYER_LEVELS` × year × region × preterm ×
   CCHD × NICU × assisted-ventilation. Count total livebirths (`N_cell`) and
   recorded DS (`R_cell`) per cell. You'll get on the order of 100,000–500,000
   non-empty cells depending on how you bin region (state vs 4-region).

2. **Verify the model specification on simulated data first.** Run
   `simulate.py` and then fit `build_model(..., spec="full")`. Confirm that
   posterior means for the key parameters (θ_LB age curve, race effects on
   η_term and s) are within credible-interval distance of the true values.
   If recovery fails, the model is mis-specified or the priors are too
   restrictive.

3. **Build up staged models.** Start with `spec="theta_only"` on the real
   data to verify the age-coding matches Morris, then progress through
   `theta_s`, `single_eta`, `full`. Each stage should fit in progressively
   more time and produce progressively more structure in the posterior.

4. **Fit the full model.** On full 2016–2024 cell-level data expect several
   hours of NUTS sampling. Use `chains=4, draws=1000, tune=1000` as a
   starting point. If sampling is slow or chains mix poorly, reparameterise
   hierarchical effects non-centrally (replace `pm.Normal("x", mu, sigma)`
   with `x = mu + sigma * pm.Normal("x_raw", 0, 1)`).

5. **Run sensitivity variants.** Repeat the full fit with
   `variant_A_tight_s`, `variant_B_tight_eta_term`, and `variant_D_dobbs_only`.
   Compare posterior race and education effects on η_term and s across
   variants; the spread across variants is the effective uncertainty on
   the decomposition.

## Key design choices

**Morris, not Hook.** θ_LB is the DS livebirth rate *in the absence of
screening*, which Morris, Mutton & Alberman (2002) measured directly using
the UK NDSCR. This supersedes Hook's 1981 rates (which are cited in older
sources but were based on smaller samples and assumed an exponential curve
above age 45 that Morris shows is actually sigmoidal). Using Morris as the
Stage-1 anchor makes the model simpler than a conception-rate framing,
because natural fetal loss is already baked into Morris's numbers — there's
no separate η_loss stage to identify.

**Clinical features enter only s.** CCHD, NICU, assisted ventilation, and
preterm status are observed after the pregnancy is delivered and therefore
cannot causally influence η_detect or η_term. They enter only the Stage-3
sensitivity term, where they pick up the "workup not complete at
certificate submission" mechanism that Boulet identified. This modelling
discipline is what lets the clinical-marker signals identify s separately
from η.

**Post-Dobbs region × year interaction is the key identifying variation.**
The Dobbs decision (June 2022) was a near-experimental shock that affected
η_term without plausibly affecting s (birth-certificate recording practices
did not change in mid-2022). The `eta_term_ry` parameter has heterogeneous
sigma — tight pre-2022, wide post-2022 — so that the data can speak about
post-Dobbs effects without being constrained by the prior. If you compare
Variant C (informative race priors on η_term) with Variant D (priors shrunk
to zero, only Dobbs variation identifies η_term), agreement is evidence of
genuine identification rather than prior-driven decomposition.

**False-positive rate fixed, not estimated.** The Ohio/NY false-positive
study pins `f ≈ 7.8e-5`. Estimating it would add a poorly-identified
parameter with little substantive value. If you want to check sensitivity
to `f`, just change `priors.false_positive_rate` and re-fit.

## Diagnostics to run after fitting

The single most important diagnostic is a **posterior pair-plot of race
effects in η_term vs s**. If these have a posterior correlation below about
−0.7, the decomposition is genuinely prior-driven and should be reported
with explicit sensitivity bounds. If the correlation is weaker, the data
are providing meaningful identification.

Other essential checks:

- **Posterior predictive on recorded counts** by year, by race, by age.
  Fit should reproduce observed aggregate rates.
- **Pre/post-Dobbs forest plot for state-level `eta_term_ry` changes.**
  Treated states should show positive `eta_term_ry` shifts after mid-2022
  (fewer terminations → more livebirths).
- **CCHD co-occurrence check.** Posterior true-DS-livebirth cells with
  CCHD=1 should match published prevalence (~20–25% for cyanotic lesions,
  per EUROCAT and Heinke et al. 2021). If they don't, the clinical-marker
  effects on s need re-examining.

## Sampling performance

Approximate wall-clock times on a modern workstation:

| cells | chains × draws | spec | wall time |
|---:|---:|:---:|---:|
| 3k (synthetic) | 2 × 300 | full | ~90s |
| 50k | 4 × 1000 | full | ~1–2h |
| 500k | 4 × 1000 | full | ~8–12h |

Variational inference (`pm.fit(method='advi')`) gives usable approximate
posteriors in minutes for the larger datasets, but the correlation
structure — especially the race × η_term vs race × s decomposition that
the sensitivity analysis depends on — is not reliably recovered by ADVI.
Use ADVI for development iteration only; final results should come from
NUTS.

## References

See `../bayesian_selection_model.md` for the full reference list with
citations. The most directly implementation-relevant references:

- Morris, J.K., Mutton, D.E. & Alberman, E. (2002). *J Med Screen* 9:2–6.
- de Graaf, G., Buckley, F. & Skotko, B.G. (2015). *EJHG* 23:1140 (corrigendum).
- Natoli, J.L., et al. (2012). *Prenat Diagn* 32:142–153.
- Kuppermann, M., et al. (2006). *Obstet Gynecol* 107:1087–1097.
- Boulet, S.L., et al. (2011). *Public Health Rep* 126:186–194.
- Salemi, J.L., et al. (2017). *Paediatr Perinat Epidemiol* 31:67–75.
- Chaiken, S.R., et al. (2023). *JAMA Netw Open* 6:e233684.

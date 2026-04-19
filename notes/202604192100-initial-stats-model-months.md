# Initial statistical model — monthly time resolution (M1)

> [!WARNING]
> This note was drafted by an AI coding assistant (Claude) from a
> working session on 2026-04-19. Figures, numbers, and claims reflect
> the specific fit runs recorded below; a human reviewer should verify
> the interpretation before citing.

## Context

Session goal: take the initial Bayesian model (`m1-year-age`) from
annual to **monthly** temporal resolution, while keeping the shape of
the model otherwise unchanged, and fit it at `reporting` quality on
both outcome constructions.

The motivating question was twofold:

1. Does a finer time axis reveal structure that the annual-scale
   smooth was averaging through?
2. Does the `recorded_plus_predicted` outcome carry month-to-month
   artefacts from the year×month quota used by `scripts/fit_model.py`
   when it flags predicted-missing cases?

Seasonality was explicitly not in scope — no cyclic term, no month
dim, just a finer trend.

## Model change

Before (`m1-year-age`, v1):

- cells grouped by `(year, mage_c)` → ~315 cells.
- `logit(p) = alpha + f_year(year) + f_age(age)`.
- HSGP basis count `m = 12` for both smooths.

After (PR #15):

- cells grouped by `(year, dob_mm, mage_c)` → 4,212 cells.
- A derived coord `t = year + (dob_mm - 1) / 12` is attached to the
  cell frame via a new `prepare_cells` hook on `BayesModelDefinition`.
- `logit(p) = alpha + f_t(t) + f_age(age)`.
- `t_m = 24` (resolves ~108 monthly points across 2016–2024);
  `age_m = 12` unchanged.
- Priors otherwise identical: `alpha ~ Normal(logit(5e-4), 1)`,
  ExpQuad HSGP with `c = 1.5`, `ls ~ InverseGamma(2, 1)`,
  `eta ~ HalfNormal(1)`.

Supporting changes:

- `BayesModelDefinition` gains `smooth_coords` (separates SQL grouping
  dims from the coord axes the model smooths over) and
  `prepare_cells` (default identity; M1 overrides to derive `t` and
  an `age` alias for `mage_c`).
- CLI iterates `smooth_coords` for prior-draws, trend plots, and
  `key_vars` instead of hardcoding year/age names.
- `diagnostics.prior_predictive_summary` renamed internal row labels
  (`time_trend_rate_ratio`, `ls_t_coord_units`) and param names
  (`time_coord`, `age_coord`).
- Smoke test, Quarto report template, and `DEFAULT_SUMMARY_VAR_NAMES`
  updated to the new coord naming.

`ruff check src tests scripts` and `npm run spellcheck` clean.
`pytest tests/test_bayes_model_smoke.py` passes in ~7s.

## Runs

Both at the `reporting` profile: 4 chains × (2000 tune + 2000 draws),
nutpie sampler, `target_accept = 0.9`, `random_seed = 47`.

| outcome | output dir | n_cells | n_total | y_total | overall rate |
|---|---|---:|---:|---:|---:|
| `recorded` | `output/bayes/m1-year-age/recorded/20260419-204041/` | 4,212 | 33,527,704 | 17,809 | 5.31 / 10,000 |
| `recorded_plus_predicted` | `output/bayes/m1-year-age/recorded_plus_predicted/20260419-204653/` | 4,212 | 33,527,704 | 44,551 | 13.29 / 10,000 |

Case-count ratio 44,551 / 17,809 ≈ 2.50 — exactly the
`1 + 1.5 × recorded` quota the upstream LightGBM top-up applies.

## Convergence

Both runs sit at the edge of the default `r_hat < 1.01` threshold:

| outcome | max Rhat | min ESS_bulk |
|---|---:|---:|
| `recorded` | 1.0100 | 430 (`alpha`) |
| `recorded_plus_predicted` | 1.0100 | 592 (`alpha`) |

The offenders are `alpha`, `eta_age`, and the first couple of
`f_age_hsgp_coeffs` — the classic soft-identifiability between a
global intercept and a zero-mean GP with non-trivial amplitude. On
the rate scale (where cells are identified) the PPC is clean in both
runs, so the joint is well-identified; only the partition between
`alpha` and the smooth intercept mixes slowly.

For publication, bump `target_accept` to 0.95 or re-run with more
draws. Neither was deemed worth doing for this preliminary session.

## Findings

### Maternal age dominates both outcomes

The age smooth carries logit-scale amplitude `eta_age` ≈ 1.0–1.2
with tight posteriors, compared to `eta_t` ≈ 0.24–0.26 for the time
smooth. Age posterior curves follow a textbook hockey-stick:

- `recorded`: plateau ≈ 0.3 / 1,000 from ages 17–32; peaks at
  **6.1 / 1,000 at age 45**; drops sharply beyond 47.
- `recorded_plus_predicted`: plateau ≈ 0.9 / 1,000 from ages 17–32;
  peaks at **11.1 / 1,000 at age 45**; same sharp drop-off.

### The top-up is not a uniform 2.5× scaling

Comparing the two age curves at specific points:

| age | recorded rate | r+p rate | ratio |
|---:|---:|---:|---:|
| 20 | 0.29 / 1,000 | 0.94 / 1,000 | **3.2×** |
| 25 | 0.25 / 1,000 | 0.88 / 1,000 | **3.5×** |
| 35 | 0.62 / 1,000 | 1.54 / 1,000 | 2.5× |
| 40 | 2.37 / 1,000 | 4.52 / 1,000 | 1.9× |
| 45 | 6.12 / 1,000 | 11.15 / 1,000 | 1.8× |

The LightGBM-predicted-missing cases disproportionately land in the
*middle-age* range, where recorded rates are lowest. At the peak age,
the top-up is smaller than the headline 2.5× because the recorded
rate is already high there — diminishing returns on flagging likely
cases. This is the most consequential qualitative difference between
the two outcome constructions and is worth keeping in mind when
interpreting posteriors from downstream models that use
`recorded_plus_predicted`.

### Time is a small, slow story — and consistent across outcomes

Both fits give a `t`-scale correlation length of ≈ 4 years (on the
original year scale, not standardised) and `eta_t` whose 94 % HDI
barely excludes zero.

`recorded`, posterior-mean rate by `t`:

- 2016.0: 5.33 / 10,000
- 2017.8: 5.55 / 10,000 (peak)
- 2020.8: 5.18 / 10,000 (trough)
- 2022.0: 5.23 / 10,000
- 2024.9: 5.30 / 10,000

`recorded_plus_predicted`, posterior-mean rate by `t`:

- 2016.0: 13.40 / 10,000
- 2020.5: 13.35 / 10,000
- 2024.9: 13.15 / 10,000

The shape of the smooth is essentially the same under both outcomes —
a shallow bump in 2017–2018 followed by a very slow decline through
2024. What differs is only the level. That consistency is the main
positive result for the quota-based top-up: it isn't introducing
fake trends, and it isn't smoothing real ones away.

### No smoking-gun quota artefacts at the monthly scale

The answer to the second motivating question:

> Does `recorded_plus_predicted` carry month-to-month artefacts from
> the year×month quota?

Not visibly, at the smoothness the ExpQuad HSGP admits. The posterior
trend under `recorded_plus_predicted` shows no stepwise jumps at
month boundaries. Observed monthly rates scatter widely around the
band (±5 % range), but that scatter is consistent with Poisson noise
at ~150–400 cases/month, not with systematic quota steps.

Caveat: a ~4-year correlation length is itself an averaging choice.
A stronger test would be to plot observed monthly rate vs the smooth
directly (the `trend_t.csv` companion has exactly this — see
`observed_rate` and `posterior_mean` columns) and look for
autocorrelation in the residuals at month-scale. That was not done
this session.

### Prior predictive sanity

Prior-implied quantities for both runs (same priors):

- `baseline_rate` median 5.2e-4, 94 % HDI [1.6e-5, 2.5e-3] — covers
  the population rate with ~5× slack either side.
- `time_trend_rate_ratio` 94 % HDI up to ~230× — wider than needed
  but the posterior narrows far below this.
- `age_gradient_rate_ratio` 94 % HDI up to ~115× — observed gradient
  is ~20× between age 20 and age 45, well inside the prior.
- `ls_t_coord_units` median 1.5 years, HDI [0.29, 5.9] — plausible.

All checks pass; no prior needs tightening.

## What's next

Not done in this session; candidates for follow-up:

1. Harden convergence (`target_accept = 0.95`, non-centred GP
   re-parameterisation) if `alpha` soft-identifiability becomes a
   blocker downstream.
2. M2 / M3: add ethnicity (M2) and education (M3) as further
   stratification dims. The `smooth_coords` refactor in PR #15 was
   written with those in mind.
3. Direct residual plot (observed monthly rate minus posterior-mean
   band) to stress-test the "no quota artefacts" conclusion at the
   month scale.
4. Publication-grade run with 4,000+ draws for final estimates.

## Artefacts

- PR: <https://github.com/dspopulations/us-birth-certificates/pull/15>
- Run outputs: `output/bayes/m1-year-age/{recorded,recorded_plus_predicted}/2026041*/`
- Logs: `output/bayes/logs/m1-reporting-{recorded,rpp}.log`

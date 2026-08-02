> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# DSP004 and DSP005 exact-age ablations

**Date:** 2026-08-02
**Status:** Implemented and fitted. `DSP004` is the preferred simple-resolution
baseline; the other models remain named sensitivity or diagnostic models.

## Question

`DSP001` and `DSP002` evaluate the Morris maternal-age curve at seven broad age
bands. `DSP003` simultaneously changed that resolution and added age-specific
combined reduction. Its improved age fit therefore could not show how much came
from resolving the Morris curve more accurately and how much came from assigning
an age pattern to reduction.

These two ablations change only the maternal-age resolution:

| Model | Maternal-age resolution | Recording | Combined reduction | Role |
| --- | --- | --- | --- | --- |
| `DSP001` | Seven bands | Constant `s` | One value per year | Band-resolution reference |
| `DSP004` | NCHS single-age codes | Constant `s` | One value per year | Exact-age ablation of `DSP001` |
| `DSP002` | Seven bands | Partially pooled `s_year` | One value per year | Year-varying-recording reference |
| `DSP005` | NCHS single-age codes | Partially pooled `s_year` | One value per year | Exact-age ablation of `DSP002` |
| `DSP003` | NCHS single-age codes | Constant `s` | Smooth age pattern within year | Age-allocation diagnostic |

The term *exact age* is shorthand. NCHS code 12 pools ages 10-12 and code 50
pools ages 50 and over; only codes 13-49 are literal single years. The Morris
curve is evaluated at representative ages 12 and 50 for the two pooled
endpoints. Both tails therefore use a pooled-cell approximation; for the 50+
cell, age 50 is specifically a lower-endpoint approximation.

## Reporting fits

Both fits use confirmed-or-pending certificate flags for 33,527,704 births from
2016-2024. Intervals below are 89% equal-tailed intervals (ETIs).

| Model | True DS livebirths, mean (89% ETI) | Recording sensitivity centre, mean (89% ETI) | Age PPC | Age-year PPC | Seven-band PPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `DSP004` | 44,280 (41,934-46,562) | 0.340 (0.322-0.360) | 18/39 | 286/351 | 1/7 |
| `DSP005` | 45,059 (41,839-48,169) | 0.337 (0.314-0.363) | 17/39 | 281/351 | 1/7 |

Relative to their otherwise matched band-resolution models, the posterior mean
total changes modestly:

- `DSP001` to `DSP004`: +452 births;
- `DSP002` to `DSP005`: +524 births.

The overlapping intervals do not by themselves prove equivalence, but these
small shifts indicate that replacing the band approximation does not overturn
the aggregate headline under the same reduction and recording assumptions.
It does materially change the cell-level predictive check.

## Common-grid comparison

Raw likelihood or information-criterion values cannot be compared fairly when
one model is fitted to seven bands and the other to single-age cells. The
comparison script therefore reconstructs both models on the same 351
age-by-year cells. This is an in-sample posterior-predictive audit, not held-out
predictive evidence.

| Comparison | 89% interval coverage on common grid | Mean absolute standardised residual |
| --- | ---: | ---: |
| `DSP001` to `DSP004` | 221/351 to 282/351 | 1.690 to 1.019 |
| `DSP002` to `DSP005` | 221/351 to 284/351 | 1.687 to 1.018 |

The mean absolute standardised residual is the average absolute difference
between observed and posterior-predictive mean counts after dividing by the
posterior-predictive standard deviation; lower values indicate closer
in-sample calibration on this grid.

The common-grid counts differ slightly from each model's native age-year PPC
counts above because they are reconstructed afresh from posterior draws with a
fixed random seed and common aggregation procedure. They answer a fair
cross-resolution comparison question; the native counts describe each saved
fit on its own reporting grid.

The common-grid improvement shows that broad-band evaluation of a steep,
non-linear age-risk curve was a material discretisation artefact. It does not
solve the remaining age pattern: both exact-age simple models still cover only
one of seven age bands after aggregation.

## DSP004 compared with DSP003

`DSP004` and `DSP003` use the same age resolution and constant recording
sensitivity. Their main difference is whether combined reduction is common
across age within a year or receives a smooth age pattern.

| Metric | `DSP004` | `DSP003` |
| --- | ---: | ---: |
| True DS livebirths, posterior mean | 44,280 | 41,834 |
| Common-grid coverage | 282/351 | 320/351 |
| Mean absolute standardised residual | 1.019 | 0.770 |
| Seven-band coverage on the common grid | 1/7 | 7/7 |

`DSP003` has a posterior mean total lower by 2,447 and substantially better
in-sample age calibration. That improvement is not evidence that age-specific
prenatal reduction is the mechanism. The model assigns the residual age shape
to reduction because recording is constrained to be constant by age. After
fixing the false-positive rate, the certificate excess rate above that rate
principally identifies
`theta_age * (1-rho_year_age) * (s-f)`. Another allocation between reduction
and recording can therefore fit the same age pattern under different external
assumptions.

## Decision

1. Promote `DSP004` as the preferred **simple-resolution baseline**. It removes
   an avoidable discretisation approximation while retaining the transparent
   `rho_year + constant s` accounting model.
2. Retain `DSP001` as a discretisation-sensitivity model. It records how much
   the original seven-band approximation changes the headline and PPCs.
3. Retain `DSP005` as the year-varying-recording sensitivity model. Its extra
   flexibility does little to the remaining age residual and should not be
   treated as automatically preferable to `DSP004`.
4. Retain `DSP003` as an age-structure diagnostic. Its better in-sample fit
   demonstrates unresolved age structure, but it does not establish a prenatal
   mechanism or replace the simple baseline headline.

“Preferred” here is deliberately narrow: `DSP004` is the cleanest simple
accounting baseline, not a claim that its age fit is adequate or that its
latent components are identified by certificates alone.

## Assumptions that remain visible

- The false-positive rate is fixed at `f=7.8e-5` per non-DS birth. It is not
  learned from these data, so the total and recording estimates are conditional
  on that transported measurement assumption. A subsequent
  [DSP004 sensitivity grid](20260802-dsp004-false-positive-surveillance-sensitivity.md)
  finds that the working `f` range barely changes the true-total estimate under
  the current reduction priors but materially changes recording sensitivity;
  widening the annual reduction priors has a larger effect on the total and its
  interval width.
- After fixing `f`, the birth-certificate excess rate mainly identifies
  `theta_age * (1-rho_year) * (s-f)`. External Morris and surveillance
  information separate the terms; the data do not do so unaided.
- Posterior predictive checks reported here are in-sample. A more flexible
  model is expected to fit the observed cells better; held-out checks are still
  needed before preferring flexibility for prediction.
- These are population-level ascertainment corrections. They cannot identify
  which individual unflagged births were missed DS cases.

## Reproducible artefacts

The saved reporting fits are under:

```text
output/selection_core_reduction/DSP004/20260802-base-reporting
output/selection_core_reduction/DSP005/20260802-base-reporting
```

The matched common-grid comparisons are under:

```text
output/selection_core_reduction/comparisons/DSP001-vs-DSP004/20260802-base-reporting
output/selection_core_reduction/comparisons/DSP002-vs-DSP005/20260802-base-reporting
output/selection_core_reduction/comparisons/DSP004-vs-DSP003/20260802-base-reporting
output/selection_core_reduction/comparisons/DSP004-vs-DSP005/20260802-base-reporting
```

The comparison metadata explicitly records that common-grid PPCs are
reconstructed in-sample evidence and that raw information-criterion scores are
not compared across different cell aggregations.

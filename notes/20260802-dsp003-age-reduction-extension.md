> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# DSP003 age-specific combined-reduction extension

**Date:** 2026-08-02
**Status:** Implemented and fitted as a diagnostic extension; retained as an
age-structure diagnostic because the headline is sensitive to the false-positive
and smoothing assumptions. The subsequent exact-age ablations promote `DSP004`,
not `DSP003`, as the preferred simple-resolution baseline.

## Question

`DSP001` and `DSP002` reproduce the recorded total by year, but both miss the
maternal-age distribution. `DSP003` asks whether a smooth age pattern in the
combined pre-livebirth reduction can explain that residual while retaining the
simple `DSP001` assumption of one certificate-recording sensitivity.

This remains a population accounting model. It does not identify individual
unrecorded births, and it does not identify age-specific termination separately
from age-specific certificate recording.

## Model change

`DSP003` branches directly from `DSP001`:

- exact maternal-age cells replace the seven broad age bands;
- the Morris probability is evaluated at each represented age;
- `s` remains constant across year and age;
- a centred RW1 supplies a common smooth age effect on the logit of combined
  reduction;
- a differentiable intercept calibration preserves the sampled national
  reduction margin in every year:

```text
sum_age N[year, age] * theta[age] * rho[year, age]
------------------------------------------------- = rho_year_anchor
       sum_age N[year, age] * theta[age]
```

The saved `rho_year` is this natural-DS-weighted margin. The saved
`rho_year_anchor` is the surveillance-informed target before calibration, and
`rho_year_margin_error` checks the numerical identity.

The [2024 NCHS user guide](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2024.pdf)
shows that the endpoint codes require care. `MAGER=12` denotes ages 10-12 and
`MAGER=50` denotes ages 50 and over; ages 13-49 are literal single years.
`DSP003` labels the endpoints `10-12` and `50+`, but evaluates Morris at the
representative values 12 and 50. Those endpoints, and ages 48-49, are sparse and
should not be described as independently data-rich.

## Measurement and anchor audit

The new read-only audit reports the certificate-code composition, exact-age
Morris expectation, the seven-band approximation, the false-positive assumption
on both probability scales, and the product identified from observed counts.

For the 2016-2024 core cohort:

- 33,527,704 births enter the model; another 61,524 age-known births have
  unknown `down_ind` and are excluded;
- 17,809 births have a confirmed-or-pending DS flag: 7,984 confirmed and 9,825
  pending;
- exact-age Morris expectation is 73,396.6, compared with 72,511.5 under the
  seven-band approximation; the banding difference is -885.1 (-1.21%);
- applying the inherited `f=7.8e-5` to all births gives about 2,615 false flags
  as an upper approximation; fitted non-DS exposure gives about 2,612, or 14.7%
  of all C/P flags in this cohort;
- the implied share is highly age-dependent because `f` is fixed per non-DS
  birth: about 27-29% in the youngest broad bands and 1.7% at age 45+;
- forcing a 7.8% false-coded share in this cohort corresponds to approximately
  `f=4.15e-5`. This is a cohort-calibrated sensitivity scenario, not a transported
  validation estimate.

The inherited conversion came from combining a false-coded share reported by
[Johnson et al. (1985)](https://doi.org/10.1002/gepi.1370020203) in an older
Ohio/New York validation study with an assumed recorded rate of 10 per 10,000.
A false-coded share among observed flags is not the same quantity as `f`, which
is a probability per non-DS birth. [Ammar et al.
(2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11506645/) supply a useful
Tennessee contrast but used a selected suspected-DS cohort, so that study does
not provide a national replacement estimate for `f`.

## Fit results

All DSP003 runs below used four chains, 1,500 or more retained draws per chain,
`target_accept=0.95`, and the PyMC sampler. Every run had maximum Rhat 1.000 and
minimum effective sample size above 900. The 3,000-draw base run had minimum ESS
2,823. The independently recomputed reduction margin differed from the saved
margin by at most `3.33e-16`.

| Run | `f` | RW1 step sigma | Recorded definition | True DS livebirths, mean (89% ETI) | Recording sensitivity, mean | Age PPC | Age-year PPC |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `DSP001` reporting baseline | `7.8e-5` | n/a | C/P | 43,828 (41,444-46,130) | 0.344 | 1/7 broad bands | n/a |
| `DSP003` base | `7.8e-5` | 0.10 | C/P | 41,834 (39,585-44,007) | 0.364 | 35/39 | 319/351 |
| lower smoothing | `7.8e-5` | 0.05 | C/P | 40,845 (38,505-43,194) | 0.372 | 31/39 | 310/351 |
| higher smoothing | `7.8e-5` | 0.20 | C/P | 43,548 (41,315-45,744) | 0.349 | 38/39 | 329/351 |
| still higher smoothing | `7.8e-5` | 0.30 | C/P | 44,511 (42,313-46,670) | 0.342 | 39/39 | 331/351 |
| cohort-calibrated `f` | `4.15e-5` | 0.10 | C/P | 41,111 (38,997-43,155) | 0.400 | 36/39 | 321/351 |
| no false positives | `0` | 0.10 | C/P | 39,601 (37,652-41,568) | 0.450 | 36/39 | 326/351 |
| joint lower `f`, higher smoothing | `4.15e-5` | 0.20 | C/P | 43,043 (40,916-45,143) | 0.382 | 38/39 | 332/351 |
| confirmed-only sensitivity | `0` | 0.10 | confirmed only | 42,971 (40,659-45,204) | 0.186 | 35/39 | 324/351 |

The confirmed-only recording parameter is confirmation sensitivity, not the C/P
recording sensitivity, so its value should not be compared directly with the
other `s` estimates.

## Interpretation

The extension answers one question positively: a shared smooth age structure
can remove most of the severe maternal-age misfit without introducing
year-specific recording sensitivity. The reporting base covers all nine year
margins, 35 of 39 age margins, and 319 of 351 joint age-year cells at the 89%
posterior-predictive interval.

It does not pass the robustness gate for a preferred model:

1. The total changes by about 4,900 births across the fitted `f` and smoothing
   scenarios (roughly 39,600-44,500).
2. At fixed `f`, changing the RW1 step scale from 0.05 to 0.30 changes the mean
   total by about 3,700 births. The sparse upper tail is influential: with step
   sigma 0.10, ages 48, 49, and 50+ are overpredicted; sigma 0.30 absorbs that
   tail and covers every age margin.
3. Lowering `f` moves the inferred recording sensitivity and total materially,
   especially because the inherited `f` accounts for a large share of observed
   flags at young ages.
4. Better in-sample PPC coverage does not prove that the age pattern belongs to
   prenatal reduction. `DSP003` assigns it there by construction because `s` is
   held constant by age.

The defensible conclusion is therefore conditional: the certificates contain a
strong age pattern in `(1-rho_age) * (s_age-f)`, but they do not determine which
factor owns it.

## Decision and next sequence

1. Keep `DSP004` as the preferred simple-resolution baseline, `DSP001` as its
   age-discretisation sensitivity, and `DSP003` as an age-structure diagnostic.
   Do not promote the `DSP003` base total as a new headline.
2. Replace the arbitrary fixed RW1 scale with a pre-specified sensitivity set or
   a carefully regularised learned scale, then assess it with held-out years.
   In-sample PPC alone will favour flexibility.
3. Treat `f` as an externally anchored scenario axis. Do not estimate it from
   the certificate counts alone and do not translate a false-coded share without
   its source-population denominator.
4. A mirrored age-on-recording sibling is now warranted as an assumption-bound
   diagnostic: keep reduction common across age and put the smooth age effect on
   `s_age`. Compare it with `DSP003` under identical `f`, smoothing, endpoint,
   and holdout rules.
5. Do not allow flexible `rho_age` and `s_age` in the same model without new
   age-specific external information; that would expose a non-identifiable
   ridge rather than resolve it.
6. Stop before race, education, or payer extensions until the measurement,
   smoothness, and age-allocation checks are resolved.

## Reproducible commands and artefacts

Base reporting fit:

```bash
python scripts/fit_core_reduction_model.py DSP003 \
  --profile reporting \
  --draws 3000 --tune 3000 --chains 4 \
  --target-accept 0.95 --prior-predictive-samples 1000 \
  --nuts-sampler pymc --render \
  --output-dir output/selection_core_reduction/DSP003/20260802-base-reporting
```

Measurement audit:

```bash
python scripts/audit_core_reduction_assumptions.py \
  --duckdb-path data/us_births.db \
  --fit-dir output/selection_core_reduction/DSP003/20260802-base-reporting \
  --target-fp-share 0.078 \
  --output-dir \
    output/selection_core_reduction/DSP003/20260802-base-reporting/assumption_audit
```

Direct comparison:

```bash
python scripts/compare_core_reduction_models.py \
  output/selection_core_reduction/DSP001/20260802-152403 \
  output/selection_core_reduction/DSP003/20260802-base-reporting \
  --output-dir \
    output/selection_core_reduction/comparisons/DSP001-vs-DSP003-base-reporting
```

The base run contains the rendered HTML report, exact-age and age-year PPC
tables, the residual heatmap, age-specific reduction table, weighted-margin
audit, NetCDF posterior, and configuration. Output data remain local and
gitignored.

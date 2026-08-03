> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# Review of the DSP001-DSP005 core accounting model family

**Date:** 2026-08-03

**Status:** Independent methodological review. No model was refitted and no
tracked input was changed. Every figure attributed to an existing note is quoted
from that note; every figure marked **[new]** was computed for this review from
`data/us_births.db` and the saved `DSP004` reporting artefacts, and should be
reproduced in a release-conformant environment before it is relied on.

## Scope

This reviews the `DSP001`-`DSP005` family as it stands after
[#86](https://github.com/dseinternational/dspopulations-us-birth-certificates/pull/86):
the model code in `src/dspopulations_us_birth_certificates/selection/`
(`core_reduction.py`, `core_models.py`, `core_reporting.py`), the fit and
comparison scripts, the seven design notes from 2026-08-02 and 2026-08-03, the
model inventory in `docs/models/README.md`, and the saved run artefacts under
`output/selection_core_reduction/`.

The frozen reference used throughout is the independent, unshifted reporting fit
at `output/selection_core_reduction/DSP004/20260803-calibration-base-reporting`,
the same artefact the race-surveillance audit froze.

## Verdict

The approach is sound and the documentation is unusually candid about its own
limits. Reducing the five-stage selection model to a two-parameter accounting
identity was the right decision, and the notes correctly and repeatedly state
that the certificate data see only the product
`theta_age * (1 - rho_year) * (s - f)`.

Two things should change before any number is published. First, the headline
89% interval is roughly three times narrower than the family's own evidence
supports, because the reported baseline treats annual surveillance errors as
independent. Second, the one clear model-adequacy failure — the maternal-age fit
— has a specific, testable cause that none of the notes considers, and
correcting it moves the total.

A third item is now the largest open risk: the reduction series that determines
the headline total has no derivation recorded anywhere in this repository.

## What the family gets right

These are genuine strengths and should be preserved through any revision.

- The accounting identity is the correct spine, and stating it before layering
  explanatory structure was the right call. The earlier five-stage model asked
  the certificate data to separate detection, termination and recording
  simultaneously; this family does not pretend to.
- The non-identifiability is documented in every note rather than buried, and
  the coherent-calibration note quantifies it directly: the posterior
  correlation between the standardised reduction-trajectory error index and `s`
  is `0.929`-`0.979`.
- The `DSP001` to `DSP004` exact-age ablation is a real improvement with a real
  diagnostic payoff: common-grid mean absolute standardised residual falls from
  `1.690` to `1.019` while the posterior mean total moves only `+452` births.
- The equicorrelation implementation in `core_reduction.py:174-187` is correct.
  `(1 - lambda) I + lambda J` scaled by the outer product of the sigmas is the
  equicorrelated covariance, and it does preserve every marginal variance as
  claimed.
- Refusing to pool or model-average scenario draws, and stating explicitly that
  the scenario envelope carries no coverage probability, is the right call.
- The race-surveillance audit is exemplary. Recording `calibration_eligible=false`
  on a single incomplete window, rather than calibrating a race layer from it,
  is exactly the discipline this study needs.
- Numerical hygiene is good: `round_to="none"` in `summary_table` so display
  rounding cannot move a value across a convergence threshold; reconstruction
  identities checked to `1e-11`; the race audit verifying that it performs no
  DuckDB writes.

## Finding 1 — the total's level is a prior, and its interval is too narrow

The notes say the data see only the product. The stronger statement, which the
notes do not quite make, is that this determines what the posterior for the
total can and cannot be.

Since `T = sum(N * theta * eta)` with `theta` fixed and `N` observed, the total
is a deterministic function of `eta`. The cell likelihood identifies
`eta_year * s` for each of nine years: nine constraints for ten unknowns under
constant `s`. The eight relative year contrasts are therefore data-identified,
and the common level is not. Along that one remaining direction only the prior
speaks, and the total's level is proportional to it.

**[new]** This is arithmetically checkable against the frozen fit. Because
`f` applies to a denominator that is almost the whole cohort,

```text
(17,809 - 33,527,704 * 7.8e-5) / 0.340182 = 44,665
```

against a posterior mean total of `44,255` — agreement to under 1%. The
recorded counts fix `T * s`; the reduction prior fixes `s`; the total follows.

Two consequences.

**`lambda = 0` is the least defensible point on the correlation axis, and it is
the reported baseline.** The reduction series is one derived series with a
common methodology, a common Morris denominator and a common extrapolation.
Errors in its *level* are close to perfectly correlated across years. Treating
them as independent lets nine annual errors partly cancel when summed, which
manufactures precision in precisely the direction that is otherwise
unidentified. The family's own numbers show the size of the effect from two
directions:

| Evidence | Width on the total |
| --- | ---: |
| Prior simulation, `lambda=0` | `6,539` |
| Prior simulation, `lambda=0.9` | `17,562` |
| Baseline posterior 89% ETI | `4,631` |
| Span of primary scenario means (`37,610`-`50,190`) | `12,580` |
| Span including both joint corners (`34,806`-`50,190`) | `15,384` |

**[new]** The scenario-mean span is `2.7` times the baseline interval width, and
`3.3` times it once the corners are included — the same factor the prior
simulation gives independently (`17,562 / 6,539 = 2.7`).

**The scenario grid should become a parameter.** A common logit-scale level
error on the reduction trajectory, the level uncertainty in the Morris curve,
and any systematic transport error from the surveillance source population to
the NCHS cohort all act on `T` in exactly the same way. They collapse into a
single scale nuisance parameter. Giving that one parameter an explicit prior and
integrating over it would produce an honest interval — plausibly around `±14%`
rather than `±5%` — and would retire the seven-scenario grid, the two corners,
the envelope-is-not-an-interval caveat, and the borderline-MCSE adjudication in
one step.

It would also finally let Morris uncertainty propagate. At present `theta` is a
fixed `pm.Data` node (`core_reduction.py:519`) with no uncertainty at all, and
in the older model `MORRIS_SIGMA = 0.001` is explicitly a pin rather than an
uncertainty estimate. Pinning `theta` for *identification* was correct; treating
it as known exactly for *uncertainty* is not, because `T` is directly
proportional to its level. No standard errors or covariance for the Morris
double-logistic parameters `(7.33, 4.211, 0.2815, 37.23)` are recorded anywhere
in the repository.

## Finding 2 — the reduction series has no recorded provenance

`data/us-births-reduction-rates-1989-2024.csv` supplies the prior mean for
`rho_year` and therefore determines the headline total. No script, notebook or
documented calculation in this repository produces it. `git log --follow` gives
three commits, all import or move operations; the file arrived already-final in
the port from `previous/`, and the originating workbook was not ported.

The repository already records the gap.
`notes/20260627-data-prep-adjustments-validation.md` says "Math correct;
**provenance missing**", and `docs/data-preparation.md` says the in-repo
provenance of these series "is currently thin and should be recorded (source,
vintage, method)".

**[new]** Reconstructed numerically, the series is
`1 - surveillance_prevalence / Morris_expected_prevalence`, where the Morris
expectation is births-weighted. It is *not* derived from
`data/us-births-degraaf-prevalence-recording-2000-2024.csv` and not from
`scripts/derive_recording_rates.py`. That is a genuinely separate path, and the
`DSP00x` models never import `recording_anchor`, so the circularity that
`notes/20260707-s-anchor-and-identifiability-diagnostic.md` documents for the
older model does not apply here.

The question that matters cannot be answered from the repository at all:
**whether the surveillance numerator is raw surveillance-programme prevalence or
an ascertainment-adjusted modelled estimate.** If it is the adjusted estimate,
the anchor already contains a recording-rate correction and the independence the
family relies on is weaker than assumed. There is no citation, no vintage and no
method note. A precision discontinuity at 2014/2015 — six significant figures
before, nine after — is consistent with two spliced source vintages, and
`docs/data-preparation.md` already flags it.

Everything downstream is conditional on this file, including the whole
coherent-calibration analysis. It should be the first thing resolved.

Relatedly, the `0.20` and `0.45` logit standard deviations have no documented
derivation either. They are asserted in the `from_reduction_csv` signature and
restated in the notes as fixed inputs. Since the headline interval is
essentially a deterministic image of those two numbers, they need a stated
basis wherever the total is published.

## Finding 3 — the extrapolated tail is not internally coherent

Reduction is a ratio whose denominator is the births-weighted Morris
expectation, and that denominator is not static: maternal ageing raised it
substantially over the period. The tail was built by extrapolating the *ratio*
linearly, which implicitly holds the denominator near its 2018 level.

**[new]** The 2020-2024 values are exactly an ordinary-least-squares line fitted
through 2008-2019, slope `+0.005964956` per year. Recomputing the ratio against
each year's actual Morris expectation, holding the surveillance numerator at its
last grounded value, gives a materially different trajectory:

| Year | CSV prior | Coherent recomputation | `DSP004` posterior mean |
| ---: | ---: | ---: | ---: |
| 2019 | `0.3724` | `0.3838` | `0.3702` |
| 2020 | `0.3783` | `0.3936` | `0.3839` |
| 2021 | `0.3843` | `0.4058` | `0.4213` |
| 2022 | `0.3903` | `0.4221` | `0.4537` |
| 2023 | `0.3962` | `0.4309` | `0.4532` |
| 2024 | `0.4022` | `0.4392` | `0.4556` |

The Morris expectation per birth rose from `2.110542e-03` in 2018 to
`2.362013e-03` in 2024, `+11.9%`. The prior-implied 2016-2024 total falls from
`45,698` under the CSV series to `44,432` under the coherent recomputation, a
change of `-2.8%`, landing almost exactly on the published posterior mean of
`44,255`.

This changes the interpretation of the tail. Roughly half of the prior-data
conflict in 2021-2024 — which a constant-`s` reading would attribute to
recording holding steady while reduction rose — is instead the extrapolation
failing to track a denominator that is known and moving. Any statement that
combined reduction rose sharply after 2020 is currently confounded with the
extrapolation method.

**2019 is extrapolated but receives the tight observed prior.** **[new]** The
Morris denominator implied by the 2019 entry is `2.110442e-03`, which equals the
*2018* expectation of `2.110542e-03` to five parts in a hundred million, not the
actual 2019 value of `2.149455e-03`. The 2019 value also lies on the
extrapolation line to within `3e-10`. So 2019 was filled from the line and the
last surveillance-grounded year is 2018. But
`DEFAULT_EXTRAPOLATED_REDUCTION_START = 2020` (`core_reduction.py:51`) gives
2019 the observed sigma of `0.20` rather than the extrapolated `0.45`. Six
extrapolated years are treated as five, with the mislabelled year given less
than half its intended prior width.

Separately, the extrapolation carries a pre-2019 behavioural trend straight
through the COVID-19 disruption to prenatal care and the June 2022 *Dobbs*
decision, with no structural break, in exactly the years of greatest policy
interest. Because termination decisions precede birth by several months, a
mid-2022 break maps mostly onto 2023 births.

## Finding 4 — assisted reproductive technology explains most of the age misfit

The broad-age posterior-predictive coverage of `1/7` is treated across three
notes as an unresolved allocation between age-varying reduction and age-varying
recording, with the mirrored age-on-recording diagnostic named as the next gate.
A third explanation is available, it is testable in data already extracted, and
it accounts for most of the failure.

`rf_artec` (assisted reproductive technology) is already in the pipeline via
`data_utils.py`. **[new]** Comparing observed recorded counts against the
`DSP004` posterior-mean prediction, by ART status:

| Maternal age | ART births | ART observed/predicted | Non-ART observed/predicted |
| --- | ---: | ---: | ---: |
| under 35 | `194,597` | `0.97` | `1.05` |
| 35-39 | `176,425` | `0.43` | `0.98` |
| 40-44 | `82,329` | `0.25` | `1.09` |
| 45+ | `25,643` | `0.06` | `0.98` |

The coding is unambiguous. At 45 and over, `rf_artec='X'` with `rf_inftr='N'`
covers `59,545` births with `391` recorded cases (`6.57` per 1,000), while
`rf_artec='Y'` with `rf_inftr='Y'` covers `25,643` births with `11` recorded
cases (`0.43` per 1,000) — a fifteen-fold difference.

**[new]** At 45 and over, ART is `28.9%` of births but `2.7%` of recorded cases,
and accounts for `162` of the band's `+175` over-prediction, or `93%`. Once ART
births are set aside the band fits almost exactly, `0.968` against `0.697` with
them included. At ages 48-50, where roughly half of all births are ART, there
were **zero** recorded Down syndrome cases among `9,779` ART births.

Three mechanisms push the same way and two of them are `theta` effects rather
than reduction effects: donor oocytes carry the donor's age-related risk rather
than the recipient's; preimplantation genetic testing for aneuploidy removes
trisomy-21 embryos before transfer; and ART pregnancies are screened far more
intensively. Applying the Morris curve at face value to ART births therefore
overstates the natural expectation where `theta` is largest.

**[new]** ART births are `1.4%` of the cohort but contribute `3,403` of the
`73,397` Morris-expected cases, `4.6%`. Correcting for it would lower the
headline total by roughly `3`-`4%` and raise `s` correspondingly — close to the
family's own `5%` materiality gate.

This also explains a puzzle in the `DSP003` note, which records that ages 48, 49
and 50+ are over-predicted at `sigma=0.10`, that the sparse tail is influential,
and that the total swings `39,601`-`44,511` across smoothing choices. Those are
the ART-dominated ages. `DSP003`'s smooth age effect is being asked to absorb an
ART composition artefact at the top of the range and a genuine screening-uptake
gradient at the bottom, which is why it is so sensitive to `sigma`.

**The residual is real and points one way.** **[new]** After excluding ART, a
monotone gradient survives at younger ages: observed/predicted of `1.332` at
under-20, `1.194` at 20-24, `1.098` at 25-29, against `0.955` at 30-34. That is
consistent with age-varying reduction. Here the notes are over-cautious in
treating the two allocations as symmetric: `priors.py` already encodes a steep
externally-anchored age gradient on detection (`ETA_DETECT_AGE`, `-1.5` to
`+1.9` on the logit, described as well anchored to advanced-maternal-age
uptake), while there is no external evidence for a steep opposite-signed age
gradient in certificate recording. If anything, prenatally diagnosed cases —
concentrated among older mothers — should be better documented, which would make
the misfit worse rather than better. Treating the two as equally plausible
discards evidence the project has already assembled.

## Finding 5 — the time-trend allocation is set by a prior-width ratio

**[new]** The clearest empirical signal in the data is that the quantity the
likelihood actually identifies, `(R/N - f) / (sum(N * theta) / N)`, falls
monotonically:

| Year | Identified `eta * s` | Standard error |
| ---: | ---: | ---: |
| 2016 | `0.2336` | `0.0058` |
| 2018 | `0.2262` | `0.0057` |
| 2020 | `0.2118` | `0.0056` |
| 2022 | `0.1873` | `0.0051` |
| 2024 | `0.1870` | `0.0051` |

That is a `-19.9%` decline over nine years. No single year-to-year step exceeds
`1.8` standard errors, but the cumulative trend is unambiguous. Every
substantive conclusion about change over time is a choice about how to divide
that decline between rising reduction and falling recording, and the certificate
data cannot make the division.

`DSP004` assigns all of it to reduction. The project's own de Graaf-derived
recording anchor assigns much of it to recording:
`notes/figures/recording_rates_anchor.csv` shows `s` for NH White falling from
`0.4615` in 2016 to `0.3831` in 2024, `-17%`. These two artefacts in the same
repository resolve the same discrepancy in opposite directions, and the `DSP00x`
notes never mention the second. The two rest on different tail assumptions — the
anchor holds the survival ratio flat from 2018, the reduction CSV extrapolates
it — but between them they bracket the answer, and only one bracket is currently
reported.

`DSP005` does not settle it. Its split between `rho_year` and `s_year` is
determined by the ratio of prior widths — `0.20`/`0.45` on reduction against
`0.35` on the centred recording offsets — and that ratio has no evidential
basis. `DSP005` reads as a test of year-varying recording but functions as an
arbitrary apportionment. Its reported `s_year` decline of `0.363` to `0.331`,
`-8.8%`, is roughly half the anchor's and should not be read as evidence about
the true split.

**[new]** Rising ART use also contributes a pure composition artefact that the
model currently books as rising termination. ART's share of Morris-expected
cases grows from `3.40%` in 2016 to `5.91%` in 2024, and the identified
`eta * s` declines `-17.5%` among non-ART births against `-19.9%` overall — so
about `12%` of the observed trend is composition, not behaviour.

## Finding 6 — two assumptions carrying more weight than their evidence

**The false-positive rate has a documented units error.** `f = 7.8e-5` implies
that `14.7%` of all recorded flags are false. The `DSP003` note records the
derivation: a `7.8%` false-coded *share* from Johnson et al. (1985) multiplied
by an *assumed* recorded rate of `1e-3`. `audit_core_reduction_assumptions.py`
carries a warning function stating the same and that the audit "does not treat
that conversion as transportable". A share among recorded flags, times an
assumed rate per all births, is being applied as a probability per non-DS birth.
The project knows this and the value still ships as the reference default. To
the notes' credit `T` is nearly invariant to `f`; `s` is not, moving `0.401` to
`0.313` across the grid.

**The numerator definition interacts with the only external check on `s`.** Of
the `17,809` flags, `9,825` are pending and `7,984` confirmed — `55%` pending.
Confirmed-only gives `s = 0.186`; confirmed-or-pending at `f=0` gives `0.401`.
Boulet's record-linkage sensitivity of approximately `40%` is genuinely
independent evidence about `s`, and it is the only external check available on
the one non-identified direction. It is not currently reported as such. Whether
Boulet's denominator corresponds to confirmed-or-pending or to confirmed alone
determines whether that evidence corroborates the model or implies a total
roughly `80%` higher. That question is more consequential than anything
currently on the roadmap.

## Finding 7 — verification gaps

The infrastructure is well built, but the checks that would catch this family's
specific failure modes are the ones missing.

1. **No parameter recovery or simulation-based calibration for any core model.**
   `tests/test_selection_parameter_recovery.py` exercises `selection/model.py`;
   its recovered parameters do not exist in the core models. Nothing probes the
   `rho`/`s` ridge that every note names as the central weakness. Every test in
   `tests/test_core_reduction_model.py` uses prior-predictive draws or graph
   inspection; none calls `pm.sample`. There is no core-cell simulator.
2. **The fit script never fails.** `scripts/fit_core_reduction_model.py:374-384`
   prints a warning and returns `0` regardless, and divergences are not read at
   fit time at all. `output/selection_core_reduction/DSP001/20260802-152232`
   records max Rhat `1.1400` and minimum ESS `12` and carries a complete set of
   tables, plots and `year_summary.csv`, indistinguishable from a healthy run.
3. **The convergence gate misses free random variables.** `summary_vars` omits
   `rho_logit_year`, which is free in the default uncorrelated configuration,
   and `rho_age_step`, which is `DSP003`'s 38 actual free parameters. Only the
   cumulated, centred transform `rho_age_offset` is checked.
4. **Coverage is reported without a reference level.** The `DSP004` `f=0` run
   shows `55.3%` cell coverage against a nominal `89%` and no code path flags
   it. The only coverage-versus-threshold comparison in the repository is in the
   old model's recovery test.
5. **`rho_year_margin_error` is computed, saved to the NetCDF, and read by
   nothing** — no tolerance in any test, the fit script or the report template.
   It converges to machine epsilon on the current grid, but a wider
   `reduction_age_step_sigma`, a large calibration shift or an extended age
   range could break the fixed 12-iteration solve silently.
6. **No MCSE, energy/BFMI or tree-depth diagnostics** anywhere in the core
   pipeline, despite the coherent-calibration note's decision rule depending on
   combined MCSEs.
7. **Prior-predictive draws are sampled and saved but never consumed.** There is
   no prior-predictive check on `R_obs` for any core model. What the report
   calls prior-versus-posterior is an analytic logit-normal quantile
   calculation.
8. **No manifest for core fits.** `manifest.py` records the git SHA and package
   versions but is used only by `scripts/fit_model.py`. Core-model artefacts
   carry no provenance and no recorded health status.
9. **The five safety rejections in `_exact_grid_for_comparison` are untested** —
   mismatched recorded definition, year range, false-positive rate, cell grid
   and cohort totals. These are the checks that stop a scientifically invalid
   comparison.
10. **Artefact-contract drift.** All nine `DSP003` runs predate
    `core_ppc_by_age_band.csv`, which `compare_core_reduction_sensitivities.py`
    lists in `REQUIRED_TABLES`, so the `DSP003` sensitivity grid cannot be run
    through the sensitivity comparison without refitting.

Model comparison is deliberately descriptive — no LOO or WAIC, self-documented
and test-pinned. That is defensible across differing cell aggregations, but it
means `DSP003`'s better in-sample fit carries no penalty for its 38 extra
parameters, and there is no held-out or temporal cross-validation anywhere.

## Finding 8 — documentation and internal consistency

- **Two `DSP004` baselines are in circulation.** `44,280` with an interval of
  `41,934`-`46,562` in the ablation and false-positive notes, and `44,255` with
  `41,934`-`46,565` in the coherent-calibration note, which anchors all of its
  materiality percentages and which the race audit then freezes. The runs differ
  in sampler (pymc against nutpie) and in retained draws (1,500 against 3,000).
- **The "matched settings" claim does not hold.** The ablation note presents
  `DSP004`/`DSP005` as changing only the age resolution relative to
  `DSP001`/`DSP002`, but the false-positive note records 1,500 retained draws
  against 3,000. The headline `+452` and `+524` shifts compare runs at different
  simulation precision, as does the `DSP004`-versus-`DSP003` gap of `2,447`.
- **The most consequential result is missing from the spine note.**
  `notes/20260802-core-reduction-recording-model.md` was revised at
  [#86](https://github.com/dseinternational/dspopulations-us-birth-certificates/pull/86),
  after the coherent-calibration work landed at
  [#82](https://github.com/dseinternational/dspopulations-us-birth-certificates/pull/82),
  yet it links every other note except that one and still reports only the
  approximately `3%` effect of widening independent priors. A reader of the
  primary note comes away believing headline uncertainty is around `±5%` when
  the corpus establishes `±13`-`15%`.
- **The `DSP002` note is stale.** It still names `DSP001` as the primary simple
  model and carries no supersession marker, unlike the `DSP003` note.
- **A retracted claim was never revised.** The false-positive note's
  "effectively indistinguishable" conclusion rests on an MCSE argument that the
  coherent-calibration note explicitly disowns because seed 47 was reused. The
  claim is repeated in the spine note's follow-up.
- **A label disagrees across documents.** `docs/models/README.md` calls
  `DSP002` a band-resolution sensitivity; the notes treat it as the
  year-varying-recording reference.
- **A test cites a non-existent section.** `tests/test_selection_priors.py:82`
  cites "plan §10 #3"; `plans/readme.md` has no numbered sections.
- **Rounding cluster.** Several headline deltas cannot be re-derived from the
  published tables because they were taken on unrounded posteriors without
  saying so. Individually trivial, collectively it means no table in the corpus
  is self-checking.

## Recommended next steps

Ordered by the ratio of consequence to effort. Items 1-4 should be settled
before any figure leaves the repository.

### 1. Record the provenance of the reduction series

Establish and document, in `docs/data-preparation.md`, the source, vintage and
method behind `data/us-births-reduction-rates-1989-2024.csv`. The specific
question to answer is whether the surveillance numerator is raw
surveillance-programme prevalence or an ascertainment-adjusted modelled
estimate; the independence of the whole family turns on it. Resolve the
2014/2015 vintage splice at the same time. If the numerator cannot be
established, that limitation belongs in the abstract, not in a note.

### 2. Rebuild the extrapolated tail coherently and relabel 2019

Recompute 2019-2024 against each year's actual births-weighted Morris
expectation rather than extrapolating the ratio, so the series stops conflating
behavioural change with maternal-age composition change. Set
`DEFAULT_EXTRAPOLATED_REDUCTION_START = 2019` in `core_reduction.py:51` so the
first extrapolated year receives the extrapolated prior width. Expect the
prior-implied total to fall by roughly `2.8%` and the tail prior-data conflict
to roughly halve. Record whether a *Dobbs* structural break is being assumed
absent, and say so where the 2022-2024 nowcasts are reported.

### 3. Replace the scenario grid with one estimated scale parameter

Add a single common logit-scale level parameter on the reduction trajectory with
an explicit prior combining reduction-level uncertainty, Morris level
uncertainty and transport error, and integrate over it. Report that interval as
the headline. This retires `--reduction-calibration-shift-logit`, the seven
scenarios, the two corners, the envelope caveat and the MCSE adjudication, and
it is the only way the published interval can honestly represent shared
surveillance uncertainty. Keep `lambda` as a sensitivity if useful, but stop
reporting `lambda=0` as the reference.

### 4. Add ART to the cell definition, before the age-on-recording gate

Stratify cells by `rf_artec`, or at minimum run the ART-stratified
posterior-predictive check, and re-examine broad-age coverage. The 45+ failure
appears fully explained by it. Do this before spending further effort on the
mirrored age-on-recording diagnostic, which is currently being asked to explain
a misfit that is largely a `theta` misspecification. Expect the total to fall by
roughly `3`-`4%`. Re-run the `DSP003` smoothing sensitivity afterwards; its
`sigma` sensitivity should shrink.

### 5. Resolve the numerator definition against the external evidence on `s`

Determine whether Boulet's approximately `40%` sensitivity corresponds to
confirmed-or-pending or to confirmed-only flags, and report the comparison
against the posterior `s` either way. This is the only quasi-independent check
on the non-identified direction and it currently goes unreported. Re-derive `f`
on the correct scale, or retire the `7.8e-5` default in favour of the
cohort-calibrated `4.15e-5` with the units error stated.

### 6. Make the trend allocation explicit rather than implicit

Either anchor `s_year` externally, or state plainly that the division of the
`-19.9%` decline in the identified quantity between reduction and recording is
set by the ratio of prior widths and report both bracketing extremes. Reconcile
`notes/figures/recording_rates_anchor.csv` with the `DSP00x` constant-`s`
assumption, or explain in the notes why the anchor's declining `s` is not
evidence against it.

### 7. Close the verification gaps, in this order

1. Make `scripts/fit_core_reduction_model.py` exit non-zero on a failed gate,
   read `sample_stats.diverging`, and extend `summary_vars` to the actual free
   random variables (`rho_logit_year`, `rho_age_step`).
2. Quarantine or delete
   `output/selection_core_reduction/DSP001/20260802-152232`.
3. Write a core-cell simulator and a recovery test aimed at the `rho`/`s` ridge,
   plus simulation-based calibration if the budget allows.
4. Assert `rho_year_margin_error` against a tolerance, and add a `DSP003`
   Newton-solve test at realistic scale (39 ages, large step sigma, non-zero
   calibration shift).
5. Compare reported coverage against its nominal `89%` with a binomial
   tolerance band, and fail or flag when it falls outside.
6. Write a manifest for core fits, and record gate status in the artefacts.
7. Test the five rejections in `_exact_grid_for_comparison`.

### 8. Repair the documentation record

Reconcile the two `DSP004` baselines and state which is the reference. Refit the
1,500-draw runs at 3,000 draws, or drop the "matched" claim. Add the
coherent-calibration result and link to the spine note. Add a supersession
marker to the `DSP002` note. Revise the false-positive note's disowned MCSE
claim. Fix the `DSP002` label in `docs/models/README.md` and the plan citation
in `tests/test_selection_priors.py`.

## Reproducing the new figures

Every **[new]** figure comes from read-only aggregate queries against
`data/us_births.db` for 2016-2024 with `down_ind IS NOT NULL` and
`mage_c IS NOT NULL`, combined with the posterior means in
`output/selection_core_reduction/DSP004/20260803-calibration-base-reporting/year_summary.csv`
and `chance.get_ds_lb_nt_probability_array`. Predicted counts use
`N * (theta * eta_year * s + (1 - theta * eta_year) * f)` at posterior means,
which is an approximation to the full posterior-predictive distribution and
adequate for the residual decomposition here but not for interval statements.
These figures were produced in a local environment that is not
release-conformant and must be regenerated before they are cited.

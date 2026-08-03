> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# Identifying group effects on reduction separately from recording

**Date:** 2026-08-03

**Status:** Diagnostic analysis and design proposal. Nothing was refitted. All
figures were computed from `data/us_births.db` and the frozen `DSP004` reporting
run at `output/selection_core_reduction/DSP004/20260803-calibration-base-reporting`
in a local environment that is **not release-conformant**; they must be
regenerated before citation. Companion to the
[model-family review](20260803-dsp-core-model-family-review.md) and the
[false-positive channel note](20260803-false-positive-channel-identification.md).

## Question

Aims 4 and 5 require race and socioeconomic effects on the combined pre-livebirth
reduction and on certificate recording. Both act on the same observed quantity,
and the literature says both are real: recording varies by race and severity
(Boulet 2011; Salemi 2017), and termination after prenatal diagnosis varies with
access and socioeconomic status.

This note establishes that stratification alone can never separate them, records
what *is* identified without any allocation assumption, and proposes a design that
does separate them using variation already present in the microdata.

## Stratifying cannot separate them

For stratum `g` the likelihood identifies the product `eta(g) * s(g)`. Each new
stratum contributes one new identified product and two new unknowns, so the
accounting never improves. No amount of demographic detail helps; adding
covariates multiplies the number of unidentified directions rather than reducing
it. This is the same structure as the year-level problem described in the review,
and it is why the race audit was right to fail closed.

Something must break the symmetry. Record linkage of the project's own data is not
available, so the routes below rely on external published estimates, on exclusion
restrictions, or on partial identification.

## What is identified with no assumption at all

The ratio of recorded flags to Morris-expected cases requires nothing beyond the
Morris curve, and it equals `eta * s` plus a false-positive term.

**[new]** By race/Hispanic-origin group, 2016-2024:

| Group | Births | Mean maternal age | Flags | Raw ratio | False-positive share | Adjusted ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NH AIAN | `246,101` | `27.3` | `184` | `0.4444` | `10.4%` | `0.3981` |
| Hispanic | `8,256,529` | `28.3` | `4,969` | `0.2925` | `13.0%` | `0.2546` |
| NH Multi-race | `771,018` | `27.9` | `383` | `0.2654` | `15.7%` | `0.2237` |
| NH White | `17,044,320` | `29.7` | `9,419` | `0.2535` | `14.1%` | `0.2177` |
| NH Black | `4,737,259` | `28.2` | `1,998` | `0.2045` | `18.5%` | `0.1667` |
| NH Asian/PI | `2,165,760` | `32.0` | `669` | `0.1012` | `25.3%` | `0.0756` |

A `4.39`-fold spread raw, `5.26`-fold adjusted. These are real, data-identified
findings. Reporting them as the primary group-level estimand — with bounds on each
factor from `eta <= 1` and `s <= 1`, and with sign restrictions where the recording
literature supplies a direction — would let the study publish substantive group
differences without choosing an allocation. Given how much of this work is shaped
by non-identification, that partial-identification framing is worth adopting
regardless of what else is built.

Two contaminants must be removed before these numbers carry weight. The
false-positive share varies by a factor of `2.4` across groups (see the companion
note). And ART composition varies too: **[new]** from `0.24%` of births and `0.97%`
of Morris-expected cases for NH AIAN to `3.06%` and `7.81%` for NH Asian/Pacific
Islander. Excluding ART births narrows the between-group spread from `4.39` to
`4.17`-fold — a smaller effect than the false-positive adjustment, but in the
opposite direction, and concentrated in the same group.

## The time-based screening design is underpowered

The natural exclusion restriction is screening technology: cfDNA adoption has a
datable time profile and certificate recording has no reason to track it, which is
the logic `selection/priors.py` already applies to the national year dimension.

Tested directly, the time dimension will not carry it. **[new]** The 2016-2024
decline in the identified ratio by age band, ART excluded, with Poisson standard
errors:

| Age band | 2016 | 2024 | Change | SE | Flags 2016/2024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| under 25 | `0.4261` | `0.3824` | `-10.3%` | `9.1%` | `300`/`199` |
| 25-29 | `0.3506` | `0.3207` | `-8.5%` | `8.2%` | `337`/`266` |
| 30-34 | `0.2622` | `0.2266` | `-13.6%` | `7.1%` | `430`/`374` |
| 35-39 | `0.2461` | `0.1981` | `-19.5%` | `5.8%` | `623`/`565` |
| 40+ | `0.2408` | `0.2071` | `-14.0%` | `6.6%` | `447`/`458` |

The prediction that screening expansion to average-risk pregnancies should steepen
the decline at younger ages is not supported, but neither is it refuted: the
youngest-versus-35-39 difference is `9.2` percentage points against a combined
standard error of `10.8`. Nine years and roughly two thousand flags per band is
simply not enough resolution. Treat the time dimension as a corroborating check,
not as the identifying variation.

## The payer-by-age interaction is a usable exclusion restriction

The cross-sectional contrast has far more power and a much sharper structure.
**[new]** Pooled 2016-2024, ART excluded, false-positive-adjusted:

| Age band | All payers | Medicaid | Private | Medicaid ÷ Private | Flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| under 20 | `0.315` | `0.317` | `0.318` | `0.99` | `433` |
| 20-24 | `0.269` | `0.275` | `0.254` | `1.08` | `1,642` |
| 25-29 | `0.236` | `0.239` | `0.227` | `1.05` | `2,610` |
| 30-34 | `0.194` | `0.238` | `0.165` | `1.45` | `3,594` |
| 35-39 | `0.201` | `0.267` | `0.163` | `1.63` | `5,252` |
| 40-44 | `0.223` | `0.287` | `0.176` | `1.63` | `3,650` |
| 45+ | `0.198` | `0.293` | `0.155` | `1.89` | `392` |

Pooled across ages the unadjusted Medicaid-to-private ratio is `1.50` with a
log-ratio standard error of `0.016`, `z = 25.2`. Payer mix is stable across the
window — Medicaid `42.6%` to `40.4%` of births, private `48.6%` to `50.6%`,
unknown under `1%` — so this is not composition drift.

**Below age 30 the two payers are indistinguishable; above 30 the gap opens to
between `45%` and `89%`.** Separately, the private-insured profile falls from
`0.318` to `0.155` across the age range while the Medicaid profile barely moves,
`0.317` to `0.293`.

### Why the interaction bounds the reduction contrast

Prenatal screening access and termination produce this interaction naturally,
because screening is age-triggered: an access disparity can only bite at ages where
screening happens and must vanish where it does not.

The restriction is **not** that recording is free of such an interaction. It is not.
The obvious mechanism is that income rises with maternal age among the privately
insured while Medicaid eligibility caps it, so if higher-income pregnancies are
better recorded then `s_P` improves with age relative to `s_M`. What matters is that
this mechanism has a **determinable sign**, and that the sign works against the
observed pattern rather than producing it.

Write the observed ratio as `r(g,a) = eta(g,a) * s(g,a)` and let
`psi(a) = s_M(a) / s_P(a)`. Then

```text
log r_M(a) - log r_P(a) = [log eta_M(a) - log eta_P(a)] + log psi(a)
```

The observed left-hand side rises by `0.647` in logs across the age range, from
`log 0.99` to `log 1.89`. The socioeconomic-recording mechanism makes `psi`
non-increasing, so `log psi` contributes nothing positive and the reduction contrast
must account for at least the whole rise:

```text
eta_M / eta_P grows by at least exp(0.647) = 1.91x from under-20 to 45+
```

The observed interaction is therefore a **lower bound** on the reduction contrast,
not an estimate of it. Deflating the finding would require `psi` to be *increasing*
— Medicaid recording improving relative to private as mothers get older — for which
there is no plausible mechanism and, as the next subsection shows, no empirical
support.

A second mechanism points the same way. Among privately insured older mothers the
DS births that occur despite high detection are disproportionately cases diagnosed
prenatally whose parents continued, which are the best-documented cases available:
a known diagnosis at delivery. Among Medicaid older mothers more DS births are
surprises at delivery, plausibly recorded worse. Both effects raise `s_P` relative
to `s_M` with age.

Three further properties make the bound robust.

**The Morris curve cancels.** Within an age band the Medicaid-to-private ratio is
the ratio of two false-positive-adjusted recorded rates; `theta(a)` appears in both
denominators and drops out. The interaction is therefore immune to Morris
misspecification, including the ART problem, which is excluded here anyway.

**The false-positive rate does not cancel, but it only strengthens the
interaction.** **[new]** Recomputing the ratio across the plausible range of `f`:

| Age band | `f = 0` | `f = 7.8e-05` | `f = 1.007e-04` | `f = 1.457e-04` |
| --- | ---: | ---: | ---: | ---: |
| under 20 | `1.00` | `0.99` | `0.99` | `0.99` |
| 25-29 | `1.04` | `1.05` | `1.06` | `1.07` |
| 35-39 | `1.57` | `1.63` | `1.65` | `1.70` |
| 40+ | `1.64` | `1.66` | `1.66` | `1.67` |

The pattern — indistinguishable below 30, strongly divergent above 35 — holds at
every value including `f = 0`, and a larger `f` widens rather than narrows it. So
the interaction does not depend on resolving the false-positive question first,
even though the *levels* do.

**Recording level is separately readable at the young end, but only conditional on
`f`.** If prenatal selection is negligible below age 20, the under-20 ratio *is* the
recording sensitivity — and it is indistinguishable between Medicaid and private,
which contradicts the assumption that socioeconomic status drives recording. Because
any residual selection at those ages implies a higher `s`, that ratio is a lower
bound on `s`, and `T = (R - N f) / s` therefore gives an upper bound on the total.

**[new]** Both the ratio and the bound depend strongly on the assumed
false-positive rate, and they move in opposite directions, so the bound must be
reported across the range rather than at one value:

| `f` | Under-20 ratio | Implies `s >=` | `T <=` | `T <=` if `eta <= 0.85` |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `0.432` | `0.432` | `41,215` | `35,033` |
| `7.8e-05` | `0.315` | `0.315` | `48,177` | `40,950` |
| `1.007e-04` | `0.281` | `0.281` | `51,288` | `43,595` |

At the value the [companion note](20260803-false-positive-channel-identification.md)
argues for, `1.007e-04`, the unconditional bound of `51,288` is no longer
informative, and the bound conditional on at most `15%` reduction below age 20 moves
from `40,950` to `43,595` — still under the `DSP004` posterior mean of `44,255`, but
only just. So this line of argument points the same way as the Boulet comparison,
that the fitted `s` is at the low end and the total at the high end, but it is not
strong enough to carry a claim on its own until `f` is settled. An earlier draft
quoted `T <= 48,235` and `T <= 41,000` at a single `f` without that dependence.

### The education test fixes the sign empirically

Maternal education is a second socioeconomic axis, and the two explanations make
**opposite** predictions for it. If socioeconomic status acts through recording, then
better-off mothers are better recorded and their ratio should be *higher*. If it
acts through selection, they screen more and their ratio should be *lower*, and only
at ages where screening happens.

**[new]** At ages 40 and over, ART excluded, false-positive-adjusted:

| Education | Births | Flags | Ratio | SE |
| --- | ---: | ---: | ---: | ---: |
| At most high school | `321,744` | `1,638` | `0.305` | `0.008` |
| Some college | `245,814` | `1,002` | `0.249` | `0.008` |
| Bachelor | `271,691` | `813` | `0.181` | `0.006` |
| Graduate or professional | `238,608` | `492` | `0.123` | `0.006` |

Monotone decreasing, a factor of `2.5` end to end, with every gap many standard
errors wide. Higher education gives a **lower** ratio — the opposite sign to the
recording explanation. The same ordering holds within private insurance alone
(`0.253` at high school or less against `0.114` at graduate level), so it is not a
payer artefact.

**[new]** Under age 25, where screening is minimal, the same groups converge:

| Education | Births | Flags | Ratio | SE |
| --- | ---: | ---: | ---: | ---: |
| At most high school | `5,181,396` | `1,405` | `0.280` | `0.007` |
| Some college | `2,032,468` | `547` | `0.273` | `0.012` |
| Bachelor | `340,109` | `99` | `0.299` | `0.030` |

All within noise of one another. The graduate cell is omitted because it rests on
four flags. That convergence is itself evidence against the premise of the recording
explanation: where selection is not operating there is no detectable socioeconomic
gradient in the ratio, which is difficult to reconcile with recording varying
strongly with income.

One incidental check falls out of the same table. The lowest-education profile is
flat to slightly rising with age, `0.280` to `0.305`. A systematically too-steep
Morris curve would drag *every* group's profile downward with age, so at least one
flat group is mild evidence that the curve is about right for non-ART births — which
supports reading the steep high-socioeconomic-status decline as real selection rather
than as curve error.

### The finding this already yields

Without resolving the level confounding at all: **the maternal-age gradient in
prenatal selection is concentrated among privately insured and higher-educated
mothers, and is close to flat among Medicaid and lowest-education mothers.** That is
an Aim 5 result, it is bounded from below by the data, and it requires no external
calibration.

State it as a bound rather than a point estimate. The `1.89` ratio at 45 and over is
not an estimate of the reduction contrast; the defensible claim is that the contrast
grows by at least `1.91`-fold across the age range under the sign restriction above.

## Proposed design

### Route A — group offsets on recording, group-by-year on reduction

Mostly a refactor of existing machinery. Hierarchical sum-to-zero group offsets on
`s`, held time-invariant; group-by-year deviations on `rho`; the national annual
margin preserved by the same weighted-intercept calibration `DSP003` already uses
across ages. On its own this identifies *changes* by group but not levels. Build it
first because the other routes hang on it.

### Route B — screening access via payer and age

```text
logit rho(g, a, y) = rho_national(y) + beta(g) * screening_exposure(a) + ...
logit s(g)         = sigma_0 + sigma(g)     # no age dependence: this is the restriction
```

`screening_exposure(a)` is a monotone age profile representing age-triggered
screening intensity; `beta(g)` scales it per group. Identification is that `s(g)`
shifts a group's whole age profile while `beta(g)` tilts it. The tilt is estimable;
the shift remains confounded. Retain the composition-preserving calibration from
Route A.

External cfDNA uptake series by payer would still add value — dating the mechanism,
and separating access expansion from improved test sensitivity — but they are no
longer load-bearing, which matters because the staggered state-Medicaid-coverage
design that would be strongest needs geography this extract does not have (see
below).

### Route C — severity as a recording-side covariate

`selection/priors.py` drops the clinical flags because "they correlate with true DS
prevalence, not recording". That is a sound warning against a reduced-form model,
where the flags would act as DS predictors. It does not apply to a latent-status
model, where they can be confined to the recording channel:

```text
P(D = 1 | age, group, year) = theta(age) * eta(group, year)   # flags absent here
P(marker m | D)                                               # two distributions
P(flag | D = 1, m, group)   = s(group) * h(m)                 # severity enters only here
P(flag | D = 0, m)          = f(m)
```

The observable becomes the joint flag-by-marker table per cell rather than the flag
count. `P(m | D = 0)` is pinned almost exactly by the non-DS births,
`P(m | D = 1)` is Aim 4's co-occurrence target and can carry an external prior, and
`h(m)` is the severity gradient in recording that Salemi measured. Severity supplies
variation in `s` that does not covary with `theta`, which is what pins the level
Route B leaves free.

Honest caveat: severe anomalies are more detectable prenatally, so a path runs from
severity through detection to termination, and severity is therefore not a clean
instrument. It is nonetheless the only recording-side signal present in the
microdata, and it is currently discarded.

## Sequencing and cautions

Build in the order A, B, C — skeleton, then age identification, then level
identification. Two prerequisites come first, because both differentially distort
group comparisons and would otherwise be absorbed by whichever new parameter is
most flexible: the ART correction from the review's Finding 4, and the
confirmed/pending channel split from the companion note.

- **Interpretation.** Payer bundles race, education, age, parity and geography, and
  Medicaid eligibility is partly determined by pregnancy itself. The exclusion
  restriction survives that — it needs only that recording is free of a
  payer-by-age interaction — but the effect must be reported as a socioeconomic
  bundle, not as insurance per se. Use education as a corroborating stratifier.
- **Geography is absent.** The only geographic field in the extract is
  `mbstate_rec`, mother's nativity; public-use natality files suppress state of
  residence and occurrence from 2005. A post-2022 state-level design — abortion
  access as an exogenous shock to termination that cannot affect recording — would
  be the strongest available identification, and would also carry the *Dobbs*
  structural break the reduction extrapolation currently ignores. It requires the
  restricted-use file under a NAPHSIS or NCHS agreement. That is a data-access
  decision rather than a modelling one, and it may be the highest-value item on the
  roadmap.
- **These are diagnostics, not results.** Every figure here is an aggregate ratio
  at posterior means with conditional standard errors. The interaction needs
  estimating inside the model, with uncertainty, before it is a finding.
- **The identification is a sign restriction, not an exclusion restriction.** An
  earlier draft of this note claimed recording could not produce a payer-by-age
  interaction. That was too strong. Recording mechanisms exist; the argument is that
  their sign is determinable and adverse, so the observed interaction bounds the
  reduction contrast from below. Route B should be specified and reported on that
  basis, and the bound should be stated wherever the interaction is.
- **Do not use the de Graaf-derived recording anchor to arbitrate.** It is computed
  from the same recorded counts and the same Morris curve, as
  [the anchor diagnostic](20260707-s-anchor-and-identifiability-diagnostic.md)
  records. Published record-linkage estimates are independent; that anchor is not.

## Reproducing

Read-only aggregate queries over `data/us_births.db` for 2016-2024 with
`down_ind IS NOT NULL` and `mage_c IS NOT NULL`, grouping on `mracehisp_c` and
`pay_rec` (`1` Medicaid, `2` private, `3` self-pay, `4` other, `9` unknown), with
`rf_artec` as the ART stratifier and `chance.get_ds_lb_nt_probability_array` for
`theta`. Ratios use `f = 7.8e-5` where adjusted. The education tables group `meduc`
as `1-3` at most high school, `4-5` some college, `6` bachelor and `7-8` graduate or
professional, excluding `9` (not stated). Standard errors are Poisson on the flag
count and are conditional on `eta` at `DSP004` posterior means, so they understate
total uncertainty.

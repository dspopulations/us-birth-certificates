> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# Estimating the false-positive rate from the age gradient and the confirmed/pending split

**Date:** 2026-08-03

**Status:** Diagnostic analysis and modelling proposal. Nothing was refitted. All
figures were computed from `data/us_births.db` and the frozen `DSP004` reporting
run at `output/selection_core_reduction/DSP004/20260803-calibration-base-reporting`
in a local environment that is **not release-conformant**; they must be
regenerated before citation. Companion to the
[model-family review](20260803-dsp-core-model-family-review.md) and the
[group-identification note](20260803-group-reduction-recording-identification.md).

## Question

`f` is fixed at `7.8e-5` in every core model, which implies that `2,612` of the
`17,809` recorded flags — `14.7%` — are false. The
[false-positive sensitivity note](20260802-dsp004-false-positive-surveillance-sensitivity.md)
treats `f` as an externally supplied scenario axis on the grounds that the
evidence cannot support estimating it.

This note asks whether that is actually true. It is not: the maternal-age
gradient identifies `f` in principle, and the confirmed/pending split supplies a
second, distinctive signal that the existing models discard by summing the two.

## Why the current value should not be a fixed default

The derivation is recorded in the
[DSP003 note](20260802-dsp003-age-reduction-extension.md) and warned about in
`scripts/audit_core_reduction_assumptions.py`: a `7.8%` false-coded *share* among
observed flags from Johnson et al. (1985), multiplied by an *assumed* recorded
rate of `1e-3`. A share among recorded flags times an assumed rate per all births
is then applied as a probability per non-DS birth. The repository already flags
the units mismatch and states that the conversion is not transportable.

Two responses are available and neither is the current one. Setting `f = 0` is a
stronger claim than the evidence supports — the sensitivity note is right to
refuse it. Fixing a mis-derived point value with no uncertainty is the other
extreme. The correct response to weak evidence is a prior, and the correct
response to *available* evidence is an estimate.

## The maternal-age gradient identifies `f`

Per year, the recorded rate is affine in the Morris probability:

```text
p_recorded(y, a) = f + theta(a) * eta(y) * (s - f)
```

so the intercept is `f` and the slope is `eta * s`. Across the 39 single-year age
codes `theta` spans `6.6e-04` at code 12 to `3.8e-02` at code 50, a factor of
about `58`. That is ample leverage for the intercept.

**[new]** Fitting `R_cell = s * (N * theta * eta) + f * N` across the 351 cells by
weighted least squares, with `eta` at `DSP004` posterior means and Poisson
weights:

| Cohort | Channel | `s` (SE) | `f` (SE) |
| --- | --- | ---: | ---: |
| All births | confirmed | `0.1388` (`0.0031`) | `3.414e-05` (`3.24e-06`) |
| All births | pending | `0.1407` (`0.0033`) | `8.642e-05` (`3.80e-06`) |
| All births | confirmed or pending | `0.2537` (`0.0063`) | `1.457e-04` (`7.19e-06`) |
| Excluding ART | confirmed | `0.1605` (`0.0033`) | `2.061e-05` (`3.24e-06`) |
| Excluding ART | pending | `0.1657` (`0.0033`) | `7.039e-05` (`3.60e-06`) |
| Excluding ART | confirmed or pending | `0.3249` (`0.0054`) | `1.007e-04` (`5.61e-06`) |

Two observations. The ART-excluded combined slope of `0.3249` nearly reproduces
the `DSP004` posterior mean `s` of `0.340`, which is a useful cross-check on both.
And every specification puts `f` **above** the `7.8e-5` default — `1.007e-04`
excluding ART, `+29%`, and `1.457e-04` including it.

**The essential caveat.** The intercept absorbs any age misspecification. If
reduction rises with maternal age, a single-slope fit compensates with a
spuriously positive intercept, so these are upper bounds rather than estimates.
The two ART rows demonstrate the mechanism directly: leaving the ART births in —
whose Morris expectation is overstated, per Finding 4 of the review — moves `f`
from `1.007e-04` to `1.457e-04` and drags `s` down from `0.3249` to `0.2537`.
This is exactly the effect the sensitivity note identified when it declined to
select high `f` on posterior-predictive grounds, and it is a reason to estimate
`f` only in a model that also carries flexible age structure, so the competition
is visible rather than displaced.

## The confirmed/pending split is the unused signal

Of the `17,809` flags, `9,825` are pending and `7,984` confirmed. A pending
notation means the anomaly was unconfirmed at filing, so false positives should
concentrate there. That yields a specific prediction: because false positives are
flat per birth while true positives scale with `theta`, the pending share of
flags should fall as maternal age rises.

**[new]** It does, monotonically:

| Age band | Confirmed | Pending | Pending share | Model-implied false-positive share |
| --- | ---: | ---: | ---: | ---: |
| under 20 | `155` | `278` | `64.2%` | `27.0%` |
| 20-24 | `669` | `975` | `59.3%` | `29.3%` |
| 25-29 | `1,085` | `1,531` | `58.5%` | `28.3%` |
| 30-34 | `1,555` | `2,101` | `57.5%` | `21.1%` |
| 35-39 | `2,532` | `2,806` | `52.6%` | `7.6%` |
| 40-44 | `1,777` | `1,942` | `52.2%` | `2.4%` |
| 45+ | `211` | `192` | `47.6%` | `1.7%` |
| All | `7,984` | `9,825` | `55.2%` | `14.7%` |

The pending-share gradient is much shallower than the implied false-positive
gradient, so pending flags are plainly not *only* false positives — most pending
flags at older ages are presumably real cases awaiting confirmation. But the
differential is real, it runs in the predicted direction, and the channel-specific
regressions quantify it: excluding ART, the false-positive rate is `7.039e-05` in
the pending channel against `2.061e-05` in the confirmed channel, a factor of
`3.42`, while the true-case sensitivities are nearly identical at `0.1657` and
`0.1605`.

## Why `f` matters more for group work than for the total

The sensitivity note correctly establishes that the national total is nearly
invariant to `f`, because `T = sum(N * theta * eta)` and `eta` is prior-pinned;
`f` only reallocates observed flags between false positives and incomplete
recording. What the note does not record is that `f` materially distorts *group*
comparisons, because its share of a group's flags is `f / (R/N)` and therefore
depends on that group's recorded rate.

**[new]** Across the seven race/Hispanic-origin groups the implied false-positive
share of flags ranges from `10.4%` (NH AIAN) to `25.3%` (NH Asian/Pacific
Islander), a factor of `2.4`. Subtracting the false-positive term widens the
between-group spread in the identified ratio from `4.39` to `5.26`-fold. So `f` is
close to irrelevant for Aim 3 and central to Aim 4.

This has a direct bearing on the
[race-surveillance audit](20260803-dsp004-race-surveillance-audit.md), which
recorded a material 2018 composition discrepancy with Asian/Pacific Islander and
Hispanic among the qualifying contrasts. Asian/Pacific Islander carries both the
highest false-positive share and the highest ART share of Morris-expected cases,
so part of that discrepancy may be artefactual. That strengthens rather than
weakens the audit's fail-closed decision.

## Proposal

Model confirmed and pending as two binomial channels rather than summing them:

```text
confirmed[y,a] ~ Binomial(N, p_ds_lb * s_C + (1 - p_ds_lb) * f_C)
pending[y,a]   ~ Binomial(N, p_ds_lb * s_P + (1 - p_ds_lb) * f_P)
```

This gives 702 cell constraints instead of 351 for the cost of two extra
parameters, and it gives `f` an internal identifying signature — enrichment in the
pending channel — rather than an imported constant. Give `f_C` and `f_P` priors
wide enough to reflect the genuine state of the evidence rather than fixed values.

Notes on scope and sequencing:

- Do **not** interpret the resulting `s_C` and `s_P` as comparable to the existing
  `confirmed_only` sensitivity fit, whose `s = 0.186` is a different estimand.
- Estimate `f` only alongside flexible age structure, and with ART handled, or the
  intercept will absorb the age misfit. The regression above shows the size of that
  displacement.
- The channel split identifies the false-positive process. It does **not** help
  separate reduction from recording, because both channels scale with the same
  true-case total. That separation needs the design in the
  [group-identification note](20260803-group-reduction-recording-identification.md).
- The `f=7.8e-5` and `f=4.15e-5` scenarios should be retired in favour of an
  estimated `f` once the channel model is fitted, with the Johnson et al. figure
  retained only as a historical note on provenance.

## Reproducing

Read-only aggregate queries over `data/us_births.db` for 2016-2024 with
`down_ind IS NOT NULL` and `mage_c IS NOT NULL`, using `ca_down_c` for the
confirmed/pending split, `rf_artec` for the ART stratifier, and
`chance.get_ds_lb_nt_probability_array` for `theta`. The regressions use `eta` at
`DSP004` posterior means, which fixes rather than propagates that uncertainty, so
the reported standard errors are conditional and understate total uncertainty.

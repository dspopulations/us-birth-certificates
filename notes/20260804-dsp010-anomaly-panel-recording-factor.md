> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Bringing evidence to the post-window allocation (`DSP010`)

**Date:** 2026-08-04

**Status:** Implemented and fitted. `DSP010` is registered, tested and documented
in `docs/models/README.md`. The reported fits are extended runs over 2004-2024 —
4 chains × 4,000 draws — produced in a local environment that is **not
release-conformant**, so regenerate before citation. Direct successor to the
[`DSP009` note](20260804-dsp009-post-anchor-recording-drift.md), whose
recommendation 3 asked for exactly this, and companion to the
[surveillance workbook extraction note](20260803-degraaf-surveillance-workbook-extraction.md)
and the [trend-pin note](20260804-dsp010-control-prevalence-trend-pins.md), which
supplies the headline specification's pinned control trends.

**The headline is a negative result, and it is the useful part.** `DSP010` builds
the anomaly panel the `DSP009` note proposed: control conditions sharing the Down
syndrome certificate item that have no prenatal reduction channel, whose common
movement should read as the item's recording sensitivity in exactly the years no
surveillance window reaches. It works, and it says the item's recording
sensitivity fell about `6.3%` over 2016-2024 — which contradicts `DSP008`'s
constant-`s` reading of the same years from an independent channel. But the two
things that would turn that into an identification of the Down syndrome split both
fail: the controls **disagree with each other** at `I² = 91%`, and the five years
where the panel and the anchor overlap carry **almost no information** about
whether Down syndrome recording tracked the item at all. The panel narrows the
range that `DSP009`'s prior alone would give. It does not divide the decline.

Three of the four controls' own prevalence trends are now measured from active
surveillance rather than assumed flat, and the headline fit uses them. That
measurement moved the factor from the `7.5%` an earlier version of this note
reported, and it removed — rather than reduced — one of the design's two
prior-driven assumptions. It did nothing at all to the other one, the loading.

## Question

`DSP009` established that the post-2020 split between falling prevalence and
falling recording is set by prior width, not by data, and made that explicit with
two reportable corners. Its recommendation 3 named the one route that could do
better:

> Build the anomaly-panel design as `DSP010`. It is the only route identified so
> far that can actually *divide* the post-window decline rather than parameterise
> the division, and it works in exactly the years the anchor does not reach.

The idea rests on an **exclusion restriction**. The 2003 certificate revision
records Down syndrome as one checkbox in a single congenital-anomaly item. Several
other checkboxes on that item describe conditions with no prenatal
detection-and-termination channel worth speaking of, so their recorded rate is not
a mixture of prevalence and recording — it is close to a direct reading of how
well the item is being completed. If the item's completion changed, those
conditions should show it, and Down syndrome should have inherited it.

## Design

### Curating the controls is most of the work

`data/us-births-anomaly-panel-conditions.csv` names every checkbox on the item
with a role and a reason. Four qualify as controls: **hypospadias** (male-only
and not prenatally diagnosable, so no reduction channel exists at all — the
strongest control available), **cleft palate alone** (poorly detectable on
routine ultrasound), **cleft lip with or without palate** (detectable but
essentially never terminated in the US), and **limb reduction defect**
(termination rare and confined to severe multi-limb cases).

Two exclusions carry more information than the inclusions.

**Gastroschisis** has no material reduction channel and would otherwise qualify
on the stated criterion. Its composition-adjusted recorded rate falls `25.9%`
across 2016-2018 to 2022-2024, which is a real decline in US birth prevalence
after a long rise. Reading that as recording would be simply wrong, and it shows
that "no reduction channel" is not sufficient: the control's *own prevalence* must
also be stable.

**Cyanotic congenital heart disease** *rose* `14.6%` over the same window while
every control fell, because universal newborn pulse-oximetry screening was phased
in across the states over 2011-2018. So a single item-wide recording factor is
already refuted for at least one checkbox on the item. That is why the Down
syndrome loading is a parameter rather than an assumption.

Both are recorded in the curation table with their reasons rather than dropped
silently, because "which conditions did you leave out" is the first question this
design should be asked.

### The model

For each panel year and control condition, on the logit scale — and these rates
are of order `5e-4`, so a logit-scale offset is a multiplicative change in the
rate to within the rate itself:

```
logit p[y,c] = level[c]
             + composition[y,c]        fixed: maternal-age mix, centred on 2016
             + known_trend[y,c]        fixed: any published prevalence trend
             + common[y]               DATA: what every control did together
             + trend[c] * (y - 2016)   each control's own trend
             + noise[y,c]
```

and Down syndrome is tied to it by

```
logit s[y] = logit recording_s + loading * ( common[y] - gamma * (y - 2016) )
```

where `gamma` is a true-prevalence trend shared by every control. The bracket is
the *recording* part of the common movement — what is left after removing the part
attributed to real prevalence.

Three choices in that structure are load-bearing.

**`gamma` is subtracted, not fitted against the walk.** A prevalence trend common
to hypospadias, both clefts and limb reduction is perfectly confounded with a
common recording trend; nothing inside the panel can distinguish them. Rather
than let the two compete for the same signal — which would put a ridge in the
posterior — the data-driven total (`common[y]`) and the prior-driven
reinterpretation (`gamma`) are separate parameters, and the recording factor is
their difference. The geometry stays clean and the provenance stays legible.

**The panel starts in 2016.** Revised-certificate coverage reaches exactly 100%
that year; before it, the revised subset is a changing and non-random set of
states, and a panel reaching further back would read state composition as
recording behaviour. `prepare_anomaly_panel` refuses earlier years rather than
letting a fit quietly include them.

**`recording_s` keeps its meaning.** The factor is exactly zero before 2016 and at
the 2016 reference year itself, so `recording_s` remains the reference-year
revised sensitivity and stays directly comparable with `DSP006`, `DSP008` and
`DSP009`. `recording_s_panel_ratio` reports the final year against that level.

Note that `recording_s_panel_ratio` is **not** comparable with
`recording_s_drift_ratio` despite the parallel names: each divides by its own
model's reference level, and those levels sit in different years — 2016 here,
2020 for the drift. See the caution under the results table.

## The controls disagree, and that changed the specification twice

The four controls do not agree about the common recording change.
Composition-adjusted, 2016-2018 against 2022-2024:

| Control | Change | Poisson SE | Flags/year |
| --- | ---: | ---: | ---: |
| Hypospadias | `-15.5%` | `1.8%` | `2,071` |
| Limb reduction | `-12.9%` | `3.8%` | `459` |
| Cleft palate alone | `-7.7%` | `2.8%` | `857` |
| Cleft lip ± palate | `-2.2%` | `1.9%` | `1,924` |

Hypospadias and cleft lip differ by `13.2` points against a combined SE of `2.6`
— about five standard errors. Formally: `Q = 33.4` on 3 degrees of freedom,
`p < 1e-6`, `I² = 91%`, between-condition SD `7.4%`. A fixed-effect mean of these
four is `-9.3% ± 1.1%`. The honest random-effects mean, whose SE widens with the
disagreement instead of ignoring it, is **`-9.6% ± 4.0%`**.

This is not a footnote. It forced two specification decisions, and in both cases
the naive choice produced a number I would have reported and should not have.

**Per-condition trends must not be centred to sum to zero.** Hard centring asserts
that the controls' own trends average exactly to the item-wide factor — that these
four hand-picked conditions are interchangeable measurements of one thing. Given
`I² = 91%` that is the fixed-effect fallacy. Measured: centring returned a
common-change SD of `1.7%`, against `4.0%` from a random-effects treatment of the
same data. An uncentred deviation with an estimated scale reproduces the honest
width. This is the same error the project already fought over with
`anchor_obs_sigma` — a narrow interval asserting precision nothing supports — and
it arrived by a completely different route.

**The common walk's innovation scale must be fixed, not estimated.** With it
estimated, the common change carries its own adaptive shrinkage towards zero, and
that shrinkage competes with the deviations' shrinkage for the same confounded
signal. The posterior then splits the controls' movement wherever the two priors
happen to balance rather than where the data put it: the fitted common change moved
from `-10.0%` to `-4.6%` while the total across factor and deviations stayed put.
The common change is the estimand, so it must not be shrunk; only the deviations
may be.

Because the panel's whole credibility rests on this, `panel_heterogeneity`
computes it at load time, it travels in every `config.json`, and the fit prints it
and warns above `I² = 50%`. A run cannot quietly present a shared factor the
panel itself contradicts.

## Results

All rows are 2004-2024, 4 chains × 4,000 draws, surveillance observation SD fixed
at `0.05`, **zero divergences throughout**, every run CLEAN on the per-chain audit.

**The headline `DSP010` is the pinned, exclusion-exact fit.** Three of the four
controls now carry a true-prevalence trend measured from active surveillance rather
than assumed to be zero, and with that measurement in place the shared prevalence
trend is set to zero instead of carried as an invented prior — the reading the
measurement supports. The pinning, its one refusal and the fits behind the three
`DSP010` variants are in
[the trend-pin note](20260804-dsp010-control-prevalence-trend-pins.md).

| Fit | 2016-2024 total | 89% ETI | Width | vs. corner | `s₂₀₂₄` vs. own reference | Prevalence 2024 vs 2018 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `DSP008` — all prevalence | `44,536` | `43,290`–`45,795` | `5.63%` | — | `1.0000` (constant) | `-6.22%` |
| `DSP009` — drift prior `0.06` | `45,370` | `43,596`–`47,186` | `7.91%` | `+1.87%` | `0.9623` (vs `2020`) | `-1.60%` |
| `DSP009` — flat + `0.20`, all recording | `45,780` | `44,321`–`47,253` | `6.40%` | `+2.79%` | `0.9444` (vs `2020`) | `+0.18%` |
| **`DSP010` — panel, pinned, γ = 0** | **`45,666`** | **`43,879`–`47,665`** | **`8.29%`** | **`+2.54%`** | **`0.9593` (vs `2016`)** | **`-3.01%`** |
| `DSP010` — pinned, γ free at `0.004` | `45,653` | `43,799`–`47,782` | `8.72%` | `+2.51%` | `0.9595` (vs `2016`) | `-3.06%` |
| `DSP010` — unpinned curation, γ free | `45,828` | `43,887`–`48,097` | `9.19%` | `+2.90%` | `0.9516` (vs `2016`) | `-2.37%` |
| `DSP010` — pinned, γ = 0, loading `1` | `45,742` | `44,047`–`47,483` | `7.51%` | `+2.71%` | `0.9561` (vs `2016`) | `-2.86%` |

`recording_s` is `0.3363`–`0.3371` across every row, so the anchored level is
untouched to three decimals and every difference sits in the post-2016 factor.

**Two provenance warnings about reading down that table.** The four `DSP010` rows
were fitted at one seed and differ only in the stated setting, so they compare to
each other cleanly. The `DSP008` and `DSP009` rows are as recorded when they were
run, at other seeds, and their artefacts are no longer on disk. Totals survive that
— the Monte Carlo standard error on a total is around `10` births against a
`1,300`-birth spread down the column — but **widths move by about `0.2` percentage
points between seeds**, so small width differences across the `DSP008`/`DSP009`
boundary should not be read as specification effects.

**The `s` ratios are not comparable across models, and the denominators are shown
because of it.** `DSP009`'s drift is zero until the last window closes, so its ratio
spans 2020-2024; `DSP010`'s factor starts at the 2016 panel reference, so its ratio
spans 2016-2024. Per year that reverses the naive reading of the column:
`DSP009`'s prior-driven drift is `-0.94%` a year against the headline's
panel-measured `-0.52%`. **The panel implies a shallower annual recording decline
than `DSP009`'s default prior, applied over twice as many years.** Only the total
and its interval are directly comparable between rows.

### What the panel measures

| Quantity | Posterior | Prior |
| --- | --- | --- |
| Common log-rate change, 2024 vs 2016 | `-6.34%` [`-13.07%`, `+1.48%`] | — (data) |
| Item recording factor, 2024 vs 2016 | `-6.34%` [`-13.07%`, `+1.48%`] | — |
| Down syndrome `s`, 2024 vs 2016 | `-4.07%` [`-10.52%`, `+0.92%`] | — |
| Down syndrome loading | `0.941 ± 0.456` | `1 ± 0.5` |
| Common prevalence trend, log/year | fixed at `0` | measured at `-0.00262` externally |
| Between-condition trend SD, log/year | `0.0147` [`0.0071`, `0.0267`] | `HalfNormal(0.02)` |

The first two rows are identical by construction: with the shared prevalence trend
set to zero, the common movement of the controls *is* the item's recording factor.
That collapse is the point of the headline specification, and it is the reason the
factor's interval is `2.5` percentage points narrower here than with γ free.

Three of those rows are the whole story.

**The panel does measure something.** The item's recording sensitivity fell about
`6.3%` over 2016-2024. `DSP008`'s constant-`s` reading of the same period is
refuted by an independent channel: booking the entire recorded decline to
prevalence requires the controls to have held flat, and they did not — not on the
certificate, and not in active surveillance either. It should be said plainly that
the factor's 89% interval still includes zero. This is a point estimate with a
direction and an argument behind it, not an interval that excludes no change.

**The loading is not identified, and the total is sensitive to it anyway.** Its
posterior SD is `0.456` against a prior of `0.500` — the five overlap years
reduced the uncertainty about whether Down syndrome recording tracked the item by
under **10%**. The falsification test the `DSP009` note asked for now exists, and it
cannot discriminate. Pinning the trends did not help: the loading is identified by
the anchor/panel overlap, and the pins add nothing there.

That does *not* make the choice harmless. Pinning the loading at `1` moves the
2016-2024 total by `+76.4` births, `+0.17%` — `4.7` Monte Carlo standard errors of
the difference, so a real gap between the two posteriors rather than sampling noise.
The sensitivity is about half what it was before the trends were pinned (`+141.6`
births, `10` standard errors, on the unpinned pair), which is the one place pinning
bought robustness rather than just moving a number. It is still a measurable move
driven by an assumption the data cannot pin.

**The shared prevalence trend is no longer a parameter.** With three controls' own
trends measured, their mean implies a shared trend of `-0.00262` log per year —
comfortably inside the `0 ± 0.004` prior the earlier specification carried, which is
external support for a choice that had been pure assertion. The headline therefore
sets it to zero rather than re-estimating a quantity the panel provably cannot see:
with γ free the posterior returned the prior untouched in every fit. Switching it off
moves the total by `+13` births, `0.8` standard errors — nothing — and removes `2.5`
percentage points of factor width that was pure prior.

The corner where that reading is weakest is hypospadias, which is the one control
with no pin. Its own trend sits in neither the offsets nor γ, so γ = 0 asserts
slightly more than the measurement delivers. The γ-free row is kept in the envelope
for exactly that reason.

### The per-chain audit is clean

`scripts/audit_anchored_chain_health.py --strict` reports every row of the table
above CLEAN. Chains agree on `recording_s` to better than `0.09%`, and the worst
single-chain `eta` maximum across the vintage is `0.80` — against `50.9` in the
escaped chain the `DSP009` note documents. The audit now also prints
`recording_panel`, because without it a `DSP010` run is indistinguishable from
`DSP008` in the audit table, and what varies `s` after the last window is exactly
what that note's factorial turns on.

A second likelihood channel was a plausible way to reopen the anchor-off mode —
the panel gives the fit another way to satisfy the certificate counts. It did not:
the widest-spread `DSP010` row sits at `0.0872%`, two orders of magnitude below the
`1%` suspect threshold, and no `DSP010` fit has ever entered the anchor-off mode.

**One `DSP010` configuration does need more tuning than the default, and it is the
headline's.** The first attempt at the pinned γ = 0 fit lost a chain during tuning —
step size collapsed to zero, `panel_condition_trend_scale` stuck at the boundary of
its own HalfNormal, R-hat `1.53`, min ESS `4`, and *zero divergences* to warn you.
The other three chains agreed with the γ-free fit to four decimals. Removing γ
tightens the geometry around a hierarchical scale that can then be pulled harder
towards zero, so this corner sits closest to the funnel. It converges cleanly at
6,000 tuning iterations and `target_accept` `0.995`; the audit independently marks
the failed run SUSPECT. The lesson is procedural: that run wrote a complete,
ordinary-looking `idata.nc`, and only the R-hat verdict distinguished it.

### One internal check that passes

The model's fitted between-condition trend SD is `0.0147` log per year
[`0.0071`, `0.0267`]. The load-time `panel_heterogeneity` diagnostic, computed by a
completely different route — DerSimonian-Laird on composition-adjusted span means,
no PyMC involved — gives `0.0124` per year on the observed rates. The model recovers
the disagreement that the independent statistic measures, which is some evidence the
hierarchical layer is doing what it is supposed to rather than absorbing something
else.

Pinning tightened that agreement rather than loosening it. `panel_heterogeneity`
reads observed rates *before* the model subtracts the pinned trends, so it cannot
see them; netting them out by hand raises the between-condition SD to `0.0137` per
year, which is what the fit actually has to explain. Against the fitted `0.0147`
that is a `7%` gap where the unpinned comparison left `17%`. The diagnostic's
blindness to the pins is a real gap in the code and is listed as such below.

## What this buys, and what it does not

**Evidence widened the interval, then more evidence narrowed it back — and both
moves were honest.** The panel on its own gives `9.19%` against `DSP009`'s `7.91%`
and `DSP008`'s `5.63%`: bringing a second channel to the allocation *increased* the
reported uncertainty, because it exposed a disagreement between the controls that
`DSP009`'s single prior had no way to express. Measuring the controls' own
prevalence then brings it back to `8.29%`, recovering about half of that widening.

The two directions are the same principle applied twice. `DSP009`'s `7.91%` is the
width of an invented prior. The panel's `9.19%` is what the evidence supports once
the controls' disagreement and the loading's freedom are carried through. The
headline's `8.29%` is that same width after two of the inputs it rested on —
the controls' prevalence trends, and the shared trend they imply — stopped being
assumptions. What never happened at any stage was a narrowing bought by asserting
more.

**What genuinely improves is the standing of the central estimate.** `DSP009` sits
at `+1.87%` above the `DSP008` corner because its drift SD was set to bracket a
range. The headline sits at `+2.54%` because four conditions on the same certificate
item fell, three of them by more than active surveillance says their true prevalence
moved, while `DSP008` needs the item flat. Those are different kinds of claim, even
though both are numbers with intervals around them.

**The envelope narrows at one end.** Across this vintage the seven rows span
`44,536` to `45,828`, and the panel argues against the low end specifically:
`DSP008`'s `-6.22%` post-2018 prevalence decline requires an item-wide recording
change of zero, which the controls contradict. The `DSP010` rows put the
prevalence decline at `-2.4%` to `-3.1%`. So the useful output is not a point but
a **shifted floor** — and the pinned rows put that floor slightly deeper than the
unpinned ones did, because part of what looked like recording turned out to be
real prevalence.

**It does not divide the decline.** The two things that would make this an
identification both fail, and pinning fixed neither. The controls disagree at
`I² = 91%`, so the shared factor is a weighted compromise between conditions that
are telling different stories — and their measured prevalence trends do not explain
that disagreement, so it is not resolved by knowing them. And the loading — the
parameter carrying "did Down syndrome inherit this?" — is still `91%` prior.
`DSP010` is a *partial identification* result: it moves the allocation from
prior-only towards evidence-informed without arriving at evidence-determined.

**The most interesting single number is `0.456` against `0.500`.** That is the
measured value of five overlap years, and it says where effort should go. Not more
controls — the panel already has the four defensible ones and adding weaker
conditions would raise `I²`, not lower it. Not tighter priors, which would be
assertion. **More overlap.** Extending the surveillance anchor by two mid-years
would add four years in which both channels speak and roughly double the
information about the loading. That reframes extending the anchor from a
data-availability wish into the binding constraint on this whole line of work.

## Caveats

- **One control's prevalence trend is still assumed.** Three of four are pinned
  from active surveillance — see
  [the trend-pin note](20260804-dsp010-control-prevalence-trend-pins.md) — but
  hypospadias was refused a pin, because its series is a rising line, a `-16%` level
  break at 2019 and a rising line again, which no single slope describes. It is the
  control carrying the largest certificate decline, so this is the largest remaining
  hole in the headline. What is known cuts one way: both segments rise, so no reading
  of that series supports a sustained prevalence *decline*, and a segment-informed
  pin would have made the recording finding stronger rather than weaker.
- **The shared prevalence trend is unfalsifiable from inside the panel, and the
  headline now asserts it.** With γ free the posterior returns the prior essentially
  unchanged — the panel provably cannot see this quantity. The headline sets it to
  zero on external grounds instead: the pinned controls imply `-0.00262` log per
  year, well inside the old `0 ± 0.004` prior. That is evidence, but it is evidence
  from three controls out of four and from one state, so the γ-free row stays in the
  envelope as the corner where this assumption is relaxed.
- **The pins carry no uncertainty into the model.** `true_trend_log_per_year` enters
  the likelihood as a fixed offset, so the headline asserts three point estimates
  whose standard errors (`0.005`–`0.007` log per year) are not small relative to the
  slopes themselves. Carrying them needs a per-condition trend prior, which does not
  exist yet.
- **Two of the four controls are not independent, and the model assumes they
  are.** Each condition gets its own Binomial on the same birth denominator, which
  treats the four as independent outcomes. They are not quite: `ca_clpal` ("cleft
  palate alone") and `ca_cleft` ("cleft lip with or without cleft palate") are
  definitionally mutually exclusive on the certificate, yet co-occur on `1,254`
  records over 2016-2024 — `16.3%` of all cleft-palate-alone flags. That is a
  certificate coding inconsistency, not a clinical fact, and it means the panel
  has fewer than four effectively independent controls, so the estimated
  between-condition SD is slightly too precise. The other five pairs overlap on
  under `1%` each, and `3.4%` of flagged records carry two or more control flags
  overall.

  **Checked, and it changes nothing material.** Refitting with the overlapping
  control dropped (`--panel-conditions ca_hypo,ca_cleft,ca_limb`, same profile and
  a distinct seed) moves the 2016-2024 total from `45,821` to `45,805` — `16`
  births, inside a Monte Carlo standard error — and *widens* the interval from
  `8.99%` to `9.26%`, which is the expected direction with one fewer control. The
  item recording factor moves `-7.51%` to `-6.92%` and the loading's prior-SD
  ratio stays at `0.89`. Every conclusion in this note survives. That check was run
  on the unpinned curation and has not been repeated against the headline; nothing
  about the overlap interacts with the pins, but the figures quoted here are from the
  older specification.
- **Four controls is a small sample of an unobservable population.** The estimand
  is "how the item's completion changed", and the panel estimates it from four
  conditions selected by judgement. `panel_condition_trend_mean` is reported so
  the confounding between the common factor and the mean of the deviations stays
  visible, but no amount of reporting turns four into many.
- **The exclusion restriction is about termination, not detection.** Cleft lip and
  limb reduction are prenatally detectable; the claim is only that detection does
  not lead to termination at a rate that matters in the US. That is a clinical
  judgement carried in the curation table, and it is the assumption most likely to
  be challenged.
- **The panel and the anchor could disagree for reasons other than the loading.**
  A mismatch over 2016-2020 would be read as a loading below one, but it could
  equally be surveillance inaccuracy, a Down syndrome-specific recording change,
  or the false-positive rate. The overlap test is not clean.
- **Hypospadias is male-only.** Its denominator is nominally wrong. The male share
  of US births is stable to within `0.08%` across 2016-2024 against double-digit
  changes in the rates, so the mis-scaling is absorbed by the condition's own
  level and contributes no trend — but it is the most precise control and so the
  most influential, which is worth knowing.
- **All figures come from a non-release-conformant environment** and must be
  regenerated before external citation.

## Reproducing

```bash
# HEADLINE: pinned controls, shared prevalence trend asserted zero.
# Needs the longer tuning -- at the default this configuration can lose a chain
# to the funnel on panel_condition_trend_scale. Check the R-hat verdict.
python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 \
    --panel-conditions-csv data/us-births-anomaly-panel-conditions-pinned.csv \
    --panel-prevalence-trend-sigma 0 --random-seed 20260804 \
    --profile reporting --draws 4000 --tune 6000 --chains 4 --target-accept 0.995

# Corner: relax the shared prevalence trend back to a prior
python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 \
    --panel-conditions-csv data/us-births-anomaly-panel-conditions-pinned.csv \
    --profile reporting --draws 4000 --tune 4000 --chains 4 --target-accept 0.99

# Corner: the pre-pinning specification, controls assumed flat
python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 \
    --profile reporting --draws 4000 --tune 4000 --chains 4 --target-accept 0.99

# Corner: the strict shared-factor restriction the panel argues against
python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 \
    --panel-conditions-csv data/us-births-anomaly-panel-conditions-pinned.csv \
    --panel-prevalence-trend-sigma 0 --panel-loading-fixed 1.0 --random-seed 20260804 \
    --profile reporting --draws 4000 --tune 6000 --chains 4 --target-accept 0.995

# Robustness: drop the control that overlaps another (see caveats)
python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 \
    --panel-conditions ca_hypo,ca_cleft,ca_limb --profile reporting \
    --draws 4000 --tune 4000 --chains 4 --target-accept 0.99

# Sensitivity: add an excluded condition back deliberately, with its pinned trend
python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 \
    --panel-conditions-csv data/us-births-anomaly-panel-conditions-pinned.csv \
    --panel-conditions ca_hypo,ca_clpal,ca_cleft,ca_limb,ca_gast

# Always audit an anchored fit per chain
python scripts/audit_anchored_chain_health.py --strict
```

## Recommended next steps

1. ~~**Pin the controls' prevalence trends against published surveillance.**~~
   **Done, with one refusal** —
   [the trend-pin note](20260804-dsp010-control-prevalence-trend-pins.md). The
   source is the Texas Birth Defects Registry rather than the NBDPN pooled reports:
   the national estimates are five-year cohorts that cannot give an annual slope,
   and hypospadias is absent from them entirely. Cleft palate alone, cleft lip ±
   palate and limb reduction are pinned; gastroschisis is pinned at `-0.038` log
   per year, confirming the exclusion and making it available as the fifth-control
   check. Hypospadias is refused — its 2019 level shift is a discontinuity, not a
   trend. The refit is done and the pinned, γ = 0 fit is now the headline above.
   Two follow-ons remain: **build the channel that carries the pins' standard
   errors**, since a fixed offset asserts a point estimate and the errors are not
   small relative to the slopes; and **resolve the hypospadias break**, by
   correspondence with the registry or by finding a second active-surveillance annual
   series, since it is the one control still unpinned and the one carrying the largest
   certificate decline.
2. **Make `panel_heterogeneity` net out the pinned trends.** It reads observed rates
   before the model subtracts `known_log_trend`, so a pinned run prints the same
   `I² = 91%` as an unpinned one and the `I² > 50%` warning fires identically. Netting
   them out by hand gives `92.5%` and a between-condition SD of `8.23%` against
   `7.45%` — the pins make the controls disagree *more*, not less, because their
   surveillance trends are uncorrelated with their certificate declines. The
   diagnostic should report what the fit actually has to explain.
3. **Widen the overlap rather than the panel.** The loading is weakly identified
   because only five years carry both channels. A surveillance anchor extended by
   even two mid-years would add four overlap years and roughly double the
   information available about the loading — a far better return than any change
   to the panel itself. That makes extending the anchor a modelling priority, not
   just a data-availability wish.
4. **Consider the hybrid the validation deliberately rejects.** `DSP010` refuses
   to combine the panel with `DSP009`'s post-anchor walk, because the two are not
   separately identified over the unanchored years. A version confining the walk
   to the panel's *residual* — what recording did beyond what the controls
   explain — would be identified and would let the panel inform rather than
   replace the drift. That is a different model, and worth one.
5. **Decide how to handle the cleft overlap properly.** The two cleft checkboxes
   are definitionally exclusive but co-flagged on `16.3%` of cleft-palate-alone
   records, so the panel's four Binomials are not four independent observations.
   The robustness check above says it does not matter for any conclusion here, but
   the clean fixes are to merge them into one "any cleft" control or to model the
   panel as a multinomial over the item's checkboxes, and the second would handle
   every pair at once.
6. **Read divergences at fit time.** Still open from the `DSP009` note, and now
   applies to a model with a second likelihood channel where a mis-specified
   panel would plausibly show up there first.
7. **Do not report a single post-2020 total.** Across `DSP008`, `DSP009` and
   `DSP010` the envelope is what is defensible. `DSP010` narrows it and explains
   part of it; it does not collapse it to a point.

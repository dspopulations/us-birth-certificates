> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Bringing evidence to the post-window allocation (`DSP010`)

**Date:** 2026-08-04

**Status:** Implemented and fitted. `DSP010` is registered, tested and documented
in `docs/models/README.md`. The reported fits are extended runs over 2004-2024 —
4 chains × 4,000 tune + 4,000 draws at `target_accept` `0.99` — produced in a
local environment that is **not release-conformant**, so regenerate before
citation. Direct successor to the
[`DSP009` note](20260804-dsp009-post-anchor-recording-drift.md), whose
recommendation 3 asked for exactly this, and companion to the
[surveillance workbook extraction note](20260803-degraaf-surveillance-workbook-extraction.md).

**The headline is a negative result, and it is the useful part.** `DSP010` builds
the anomaly panel the `DSP009` note proposed: control conditions sharing the Down
syndrome certificate item that have no prenatal reduction channel, whose common
movement should read as the item's recording sensitivity in exactly the years no
surveillance window reaches. It works, and it says the item's recording
sensitivity fell about `7.5%` over 2016-2024 — which contradicts `DSP008`'s
constant-`s` reading of the same years from an independent channel. But the two
things that
would turn that into an identification of the Down syndrome split both fail: the
controls **disagree with each other** at `I² = 91%`, and the five years where the
panel and the anchor overlap carry **almost no information** about whether Down
syndrome recording tracked the item at all. The panel narrows the range that
`DSP009`'s prior alone would give. It does not divide the decline.

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
on the stated criterion. Its composition-adjusted recorded rate falls `26.0%`
across 2016-2018 to 2022-2024, which is a real decline in US birth prevalence
after a long rise. Reading that as recording would be simply wrong, and it shows
that "no reduction channel" is not sufficient: the control's *own prevalence* must
also be stable.

**Cyanotic congenital heart disease** *rose* `14.5%` over the same window while
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
`DSP009`. `recording_s_panel_ratio` reports the final year against that level —
the same headline as `recording_s_drift_ratio`, on the same scale.

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

All six rows below are one vintage: 2004-2024, 4 chains × 4,000 tune + 4,000
draws at `target_accept` `0.99`, surveillance observation SD fixed at `0.05`,
distinct seeds, **zero divergences throughout**, max R-hat between `1.0008` and
`1.0036`.

| Fit | 2016-2024 total | 89% ETI | Width | vs. corner | `s₂₀₂₄`/reference | Prevalence 2024 vs 2018 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `DSP008` — all prevalence | `44,536` | `43,290`–`45,795` | `5.63%` | — | `1.0000` | `-6.22%` |
| `DSP009` — drift prior `0.06` | `45,370` | `43,596`–`47,186` | `7.91%` | `+1.87%` | `0.9623` | `-1.60%` |
| `DSP009` — flat + `0.20`, all recording | `45,780` | `44,321`–`47,253` | `6.40%` | `+2.79%` | `0.9444` | `+0.18%` |
| **`DSP010` — panel** | **`45,821`** | **`43,896`–`48,013`** | **`8.99%`** | **`+2.89%`** | **`0.9521`** | **`-2.48%`** |
| `DSP010` — exclusion asserted exactly | `45,813` | `43,969`–`47,852` | `8.48%` | `+2.87%` | `0.9520` | `-2.44%` |
| `DSP010` — loading pinned at `1` | `45,963` | `44,111`–`47,867` | `8.17%` | `+3.20%` | `0.9462` | `-2.03%` |

`recording_s` is `0.3363`–`0.3371` across all six, so the anchored level is
untouched to three decimals and every difference sits in the post-2016 factor.

### What the panel measures

| Quantity | Posterior | Prior |
| --- | --- | --- |
| Common log-rate change, 2024 vs 2016 | `-7.45%` [`-14.19%`, `+0.46%`] | — (data) |
| Item recording factor, 2024 vs 2016 | `-7.51%` [`-15.43%`, `+1.45%`] | — |
| Down syndrome `s`, 2024 vs 2016 | `-4.79%` [`-12.04%`, `+0.86%`] | — |
| Down syndrome loading | `0.936 ± 0.450` | `1 ± 0.5` |
| Common prevalence trend, log/year | `0.0001 ± 0.0039` | `0 ± 0.004` |
| Between-condition trend SD, log/year | `0.0145` [`0.0068`, `0.0266`] | `HalfNormal(0.02)` |

Three of those rows are the whole story.

**The panel does measure something.** The item's recording sensitivity fell about
`7.5%` over 2016-2024. `DSP008`'s constant-`s` reading of the same period is
refuted by an independent channel: booking the entire recorded decline to
prevalence requires the controls to have held flat, and they did not.

**The loading is not identified.** Its posterior SD is `0.450` against a prior of
`0.500` — the five overlap years reduced the uncertainty about whether Down
syndrome recording tracked the item by **10%**. The falsification test the
`DSP009` note asked for now exists, and it cannot discriminate. Pinning the
loading at `1` moves the total by `+0.31%`, well inside a Monte Carlo standard
error of the difference, so nothing in the data prefers either.

**The common prevalence trend is untouched.** Posterior `0.0001 ± 0.0039` against
a prior of `0 ± 0.004`: the panel returns the prior essentially unchanged, exactly
as the confounding predicts. Asserting it to be zero moves the total by `-0.02%`.
That row is a prior, wearing a posterior's clothes, and it is reported so no one
mistakes it for evidence.

### The per-chain audit is clean

`scripts/audit_anchored_chain_health.py --strict` reports **all six runs CLEAN**.
Chains agree on `recording_s` to better than `0.09%`, and the worst single-chain
`eta` maximum across the vintage is `0.80` — against `50.9` in the escaped chain
the `DSP009` note documents. The audit now also prints `recording_panel`, because
without it a `DSP010` run is indistinguishable from `DSP008` in the audit table,
and what varies `s` after the last window is exactly what that note's factorial
turns on.

A second likelihood channel was a plausible way to reopen the anchor-off mode —
the panel gives the fit another way to satisfy the certificate counts. It did not:
`DSP010` is the widest-spread row at `0.0872%`, which is two orders of magnitude
below the `1%` suspect threshold.

### One internal check that passes

The model's fitted between-condition trend SD is `0.0145` log per year
[`0.0068`, `0.0266`]. The load-time `panel_heterogeneity` diagnostic, computed by a
completely different route — DerSimonian-Laird on composition-adjusted span means,
no PyMC involved — gives `0.0124`. The model recovers the disagreement that the
independent statistic measures, which is some evidence the hierarchical layer is
doing what it is supposed to rather than absorbing something else.

## What this buys, and what it does not

**The interval gets wider, not narrower, and that is the honest direction.**
`DSP010`'s width is `8.99%` against `DSP009`'s `7.91%` and `DSP008`'s `5.63%` —
the widest of the six. Bringing evidence to the allocation *increased* the
reported uncertainty. That is not a failure of the design. `DSP009`'s `7.91%` is
the width of an invented prior; `DSP010`'s `8.99%` is the width the evidence
actually supports once the controls' disagreement and the loading's freedom are
both carried through. A narrower number here would have meant asserting more, not
knowing more.

**What genuinely improves is the standing of the central estimate.** `DSP009` sits
at `+1.87%` above the `DSP008` corner because its drift SD was set to bracket a
range. `DSP010` sits at `+2.89%` because four conditions on the same certificate
item fell while `DSP008` needs them flat. Those are different kinds of claim, even
though both are numbers with intervals around them.

**The envelope narrows at one end.** Across this vintage the six rows span
`44,536` to `45,963`, and the panel argues against the low end specifically:
`DSP008`'s `-6.22%` post-2018 prevalence decline requires an item-wide recording
change of zero, which the controls contradict. The `DSP010` rows put the
prevalence decline at `-2.0%` to `-2.5%`. So the useful output is not a point but
a **shifted floor**.

**It does not divide the decline.** The two things that would make this an
identification both fail. The controls disagree at `I² = 91%`, so the shared
factor is a weighted compromise between conditions that are telling different
stories. And the loading — the parameter carrying "did Down syndrome inherit
this?" — is 90% prior. `DSP010` is a *partial identification* result: it moves the
allocation from prior-only towards evidence-informed without arriving at
evidence-determined.

**The most interesting single number is `0.450` against `0.500`.** That is the
measured value of five overlap years, and it says where effort should go. Not more
controls — the panel already has the four defensible ones and adding weaker
conditions would raise `I²`, not lower it. Not tighter priors, which would be
assertion. **More overlap.** Extending the surveillance anchor by two mid-years
would add four years in which both channels speak and roughly double the
information about the loading. That reframes extending the anchor from a
data-availability wish into the binding constraint on this whole line of work.

## Caveats

- **The controls' own prevalence trends are assumed, not pinned.** Every
  `true_trend_log_per_year` in the curation table is `0.0`, meaning "believed
  stable, not verified". Pinning them against published NBDPN surveillance is the
  single most valuable open input to this design, and gastroschisis shows the
  assumption is not free: one condition that looked admissible on the reduction
  criterion had a `26%` prevalence-driven decline.
- **The common prevalence trend is unfalsifiable from inside the panel.** It is
  carried as a prior at `0.004` log per year and the posterior returns the prior
  essentially unchanged, which is the correct behaviour and also a statement that
  this assumption is doing real work without any evidence behind it.
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
# The panel-driven allocation
python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 \
    --profile reporting --draws 4000 --tune 4000 --chains 4 --target-accept 0.99

# Corner: the strict shared-factor restriction the panel argues against
python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 \
    --panel-loading-fixed 1.0 --profile reporting --draws 4000 --tune 4000 \
    --chains 4 --target-accept 0.99

# Corner: the exclusion restriction asserted exactly
python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 \
    --panel-prevalence-trend-sigma 0 --profile reporting --draws 4000 \
    --tune 4000 --chains 4 --target-accept 0.99

# Sensitivity: add a condition back deliberately
python scripts/fit_core_reduction_model.py DSP010 --years 2004-2024 \
    --panel-conditions ca_hypo,ca_clpal,ca_cleft,ca_limb,ca_gast

# Always audit an anchored fit per chain
python scripts/audit_anchored_chain_health.py --strict
```

## Recommended next steps

1. **Pin the controls' prevalence trends against published surveillance.** This
   is the highest-value open input and the one that would most change what the
   panel can claim. NBDPN annual reports and the state-based birth-defects
   surveillance literature give birth prevalence for all four controls; entering
   them in `true_trend_log_per_year` converts an assumption into an offset. Do
   gastroschisis too, as a check that the machinery recovers a known decline.
2. **Widen the overlap rather than the panel.** The loading is weakly identified
   because only five years carry both channels. A surveillance anchor extended by
   even two mid-years would add four overlap years and roughly double the
   information available about the loading — a far better return than any change
   to the panel itself. That makes extending the anchor a modelling priority, not
   just a data-availability wish.
3. **Consider the hybrid the validation deliberately rejects.** `DSP010` refuses
   to combine the panel with `DSP009`'s post-anchor walk, because the two are not
   separately identified over the unanchored years. A version confining the walk
   to the panel's *residual* — what recording did beyond what the controls
   explain — would be identified and would let the panel inform rather than
   replace the drift. That is a different model, and worth one.
4. **Read divergences at fit time.** Still open from the `DSP009` note, and now
   applies to a model with a second likelihood channel where a mis-specified
   panel would plausibly show up there first.
5. **Do not report a single post-2020 total.** Across `DSP008`, `DSP009` and
   `DSP010` the envelope is what is defensible. `DSP010` narrows it and explains
   part of it; it does not collapse it to a point.

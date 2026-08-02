> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# DSP004 false-positive and surveillance-precision sensitivity

**Date:** 2026-08-02
**Status:** Implemented and fitted. The false-positive term is retained, but its
transported default is now treated as a scenario rather than a known constant.

## Question

`DSP004` fixes the false-positive probability at `f=7.8e-5` per non-DS
birth and uses surveillance-informed annual combined-reduction priors. This
audit asks two separate questions:

1. Does uncertainty about `f` materially change the true-livebirth estimate,
   recording sensitivity, or age fit?
2. Does the stated precision of the surveillance-derived reduction priors
   materially constrain the result?

Setting `f=0` is not assumption-free: it assumes perfect specificity for the
confirmed-or-pending certificate endpoint. Conversely, estimating `f` freely
from these cells would use the same maternal-age residual that `DSP004` already
fits inadequately. The first analysis therefore treats `f` as a fixed scenario
axis rather than selecting it from in-sample fit.

## Scenarios

All runs use the exact same 2016-2024 DSP004 cohort, Morris curve, constant
recording sensitivity and model structure. The current reduction-prior logit
SDs are `0.20` before 2020 and `0.45` from 2020. The deliberately wide stress
doubles them to `0.40` and `0.90`.

| Scenario | `f` | Reduction-prior logit SDs | Role |
| --- | ---: | ---: | --- |
| Perfect-specificity stress | `0` | `0.20 / 0.45` | Lower false-positive extreme; not an assumption-free baseline |
| Cohort-scaled scenario | `4.15e-5` | `0.20 / 0.45` | Makes expected false flags approximately 7.8% of this cohort's observed flags; not externally validated |
| Transported default | `7.8e-5` | `0.20 / 0.45` | Existing DSP004 operational baseline |
| High diagnostic stress | `1.20e-4` | `0.20 / 0.45` | Deliberately high scenario; not an estimated upper bound |
| Perfect specificity, independent-wide | `0` | `0.40 / 0.90` | False-positive endpoint crossed with wider independent annual priors |
| Transported default, independent-wide | `7.8e-5` | `0.40 / 0.90` | Direct reduction-prior-width contrast |

The wide scenarios test independent annual imprecision. They do not represent a
shared calibration error that moves the full surveillance trajectory together.

## Fit quality

Each fit used four chains and the reporting profile. The current-prior fits used
1,500 retained draws per chain. The wider-prior fits exposed a stronger
recording-reduction ridge, so they were rerun with 3,000 retained draws per
chain. All final runs had zero divergences, maximum Rhat below `1.01`, and
minimum effective sample size above 600.

## Results

| Scenario | True DS livebirths, mean (89% ETI) | ETI width | Recording `s`, mean | Age-year PPC | Seven-band PPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Perfect-specificity stress | 44,507 (42,172-46,862) | 4,690 | 0.401 | 194/351 | 1/7 |
| Cohort-scaled scenario | 44,326 (41,934-46,708) | 4,775 | 0.367 | 249/351 | 1/7 |
| Transported default | 44,280 (41,934-46,562) | 4,627 | 0.340 | 286/351 | 1/7 |
| High diagnostic stress | 44,228 (42,020-46,507) | 4,487 | 0.313 | 293/351 | 2/7 |
| Perfect specificity, independent-wide | 45,685 (41,170-50,119) | 8,950 | 0.391 | 195/351 | 1/7 |
| Transported default, independent-wide | 45,802 (40,897-50,161) | 9,264 | 0.330 | 283/351 | 1/7 |

At the current reduction-prior widths:

- moving from the cohort-scaled `f=4.15e-5` scenario to the transported
  `f=7.8e-5` default gives separately fitted posterior means only 46 births
  apart (0.10%). That difference is smaller than their combined Monte Carlo
  standard error, so the totals are effectively indistinguishable at this
  simulation precision;
- the same change lowers recording sensitivity by 0.027, from 0.367 to 0.340;
- even the full `f=0` to `f=1.20e-4` stress changes the mean total by only
  -279 births (-0.63%), but changes `s` by -0.087;
- age-year PPC coverage rises with `f`, but broad-age coverage remains only
  one or two of seven bands. A larger additive false-positive floor is partly
  absorbing the known age residual and must not be selected for that reason.

The expected false-positive accounting makes the recording result easier to
interpret:

| `f` | Expected false flags | Share of 17,809 observed flags |
| ---: | ---: | ---: |
| `0` | 0 | 0.0% |
| `4.15e-5` | approximately 1,390 | 7.8% |
| `7.8e-5` | approximately 2,612 | 14.7% |
| `1.20e-4` | approximately 4,018 | 22.6% |

Thus `f` is not costless. Under the current surveillance prior it mostly changes
how observed flags are divided between false positives and incomplete recording,
rather than changing the estimated number of true livebirths.

## Surveillance-prior precision

Widening the independent annual reduction priors has the larger headline effect:

- at the transported default `f`, the mean total rises by 1,521 births (3.44%)
  and the 89% ETI width increases from 4,627 to 9,264 births, approximately
  doubling;
- at `f=0`, the mean rises by 1,179 births (2.65%) and the ETI width increases
  by 91%;
- the full six-scenario union spans posterior means of 44,228-45,802 and ETI
  bounds of 40,897-50,161. This is a sensitivity envelope, not a posterior
  interval or a probability statement.

Within this six-scenario grid, the deliberately doubled independent-width stress
affects the apparent precision of the base total more than the working `f`
range. This is not a general ranking of uncertainty sources. The wider runs may
still understate surveillance uncertainty: independent yearly errors can average
down, whereas errors in a shared surveillance source may move every annual
anchor in the same direction.

## Decision

1. Retain the false-positive term in the observation model. Do not describe
   `f=0` as dropping an assumption.
2. Retain `DSP004` as the preferred simple-resolution baseline, but describe
   its total and `s` as conditional on both `f` and the surveillance-prior
   specification.
3. Treat `4.15e-5` and `7.8e-5` as a working scenario comparison, not as a
   validated interval for `f`. The true-total result is stable across those two
   values; the recording-sensitivity result is not.
4. Do not select the high-`f` scenario using in-sample PPC. Its improved cell
   coverage is consistent with an additive intercept absorbing unresolved age
   structure.
5. Before using a single publication headline interval, add a coherent
   surveillance-level sensitivity: shift the complete reduction trajectory on
   the logit scale or introduce a shared calibration-bias term. The independent
   `0.40 / 0.90` stress does not answer that question.
6. If suitable validation counts become available, model confirmed and pending
   flags separately and construct a prior for `f` from the relevant non-DS
   denominator. Until then, scenario reporting is more defensible than a
   seemingly data-estimated `f`.

## Reproducible artefacts

The fitted runs are under:

```text
output/selection_core_reduction/DSP004/20260802-f-zero-reporting
output/selection_core_reduction/DSP004/20260802-f-cohort-calibrated-reporting
output/selection_core_reduction/DSP004/20260802-base-reporting
output/selection_core_reduction/DSP004/20260802-f-high-stress-reporting
output/selection_core_reduction/DSP004/20260802-f-zero-wide-reduction-reporting
output/selection_core_reduction/DSP004/20260802-f-legacy-wide-reduction-reporting
```

The combined tables and figures are generated under:

```text
output/selection_core_reduction/comparisons/
  DSP004-f-reduction-sensitivity/20260802-reporting
```

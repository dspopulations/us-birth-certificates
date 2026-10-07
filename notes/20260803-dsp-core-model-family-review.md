> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# August review of the core model family

**Original date:** 3 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

This review compared the evidence and assumptions behind the core model family. Later work added surveillance-anchored models, explicit post-anchor recording drift and an anomaly panel. The [September fixes](20260905-dsp-code-review-fixes.md) addressed several implementation findings. The old completed task lists are removed here.

## Which evidence controls the estimate

| Evidence | Scale | Width on the total |
| --- | --- | ---: |
| Prior simulation, `lambda=0` | prior | `6,539` |
| Prior simulation, `lambda=0.9` | prior | `17,562` |
| Baseline 89% ETI | posterior | `4,631` |
| Span of primary scenario means (`37,610`-`50,190`) | scenario | `12,580` |
| Span including both joint corners (`34,806`-`50,190`) | scenario | `15,384` |

The recorded-count likelihood constrains a combination of true prevalence, recording and false positives. In unanchored models, the reduction priors largely determine the scale of true prevalence. A large birth denominator does not independently identify each mechanism.

The reported uncertainty differs by evidence source and by what is held fixed. Compare assumptions on a common scale before treating one interval as more informative than another. A posterior interval conditional on a fixed benchmark excludes error in that benchmark.

## Reduction-input consistency

| Year | CSV prior | Coherent recomputation | `DSP004` posterior mean |
| ---: | ---: | ---: | ---: |
| 2019 | `0.3724` | `0.3837` | `0.3702` |
| 2020 | `0.3783` | `0.3936` | `0.3839` |
| 2021 | `0.3843` | `0.4058` | `0.4213` |
| 2022 | `0.3903` | `0.4220` | `0.4537` |
| 2023 | `0.3962` | `0.4307` | `0.4532` |
| 2024 | `0.4022` | `0.4392` | `0.4556` |

The annual reduction input must use a compatible counterfactual denominator. Its tail includes extrapolated values. A reconstructed ratio is not the same as a directly observed annual termination probability.

## Assisted reproduction and maternal age

| Maternal age | ART births | ART observed/predicted | Non-ART observed/predicted |
| --- | ---: | ---: | ---: |
| under 35 | `194,597` | `0.97` | `1.05` |
| 35-39 | `176,425` | `0.43` | `0.98` |
| 40-44 | `82,329` | `0.25` | `1.09` |
| 45+ | `25,643` | `0.06` | `0.97` |

The discrepancy in ART births is a check of the age benchmark and selected population. It does not identify the cause. Maternal age need not equal oocyte age, and donor-egg use can make a maternal-age benchmark unsuitable. Excluding ART changes the population and is a sensitivity analysis, not an automatic correction.

## Products and subgroup contrasts

| Year | Identified `eta * s` | Standard error |
| ---: | ---: | ---: |
| 2016 | `0.2336` | `0.0058` |
| 2018 | `0.2262` | `0.0057` |
| 2020 | `0.2118` | `0.0056` |
| 2022 | `0.1873` | `0.0051` |
| 2024 | `0.1870` | `0.0051` |

Ratios relative to the Morris benchmark remain conditional on that benchmark, age aggregation and `f`. They can constrain products. They cannot alone establish a subgroup's termination or recording rate. The [group note](20260803-group-reduction-recording-identification.md) records these limits.

## Confirmed-only definition

| run | `f` | `s` (89% ETI) | true DS livebirths |
| --- | ---: | ---: | ---: |
| control, reproduces the table row | `0` | `0.1861` (`0.1760`-`0.1971`) | 42,971 (40,659-45,204) |
| refit | `1.1e-5` | `0.1757` (`0.1658`-`0.1861`) | 43,403 (41,056-45,784) |

Confirmed-only excludes pending karyotypes. It changes the ascertainment channel, rather than isolating all true cases. Comparisons need channel-specific sensitivity and false-positive assumptions. The [study-area review](20260804-salemi-boulet-study-area-transport.md) corrects the old Boulet attribution.

## Findings addressed later

The [workbook audit](20260803-degraaf-surveillance-workbook-extraction.md) resolved five-year window timing and prompted `DSP006` to `DSP008`. The [drift note](20260804-dsp009-post-anchor-recording-drift.md) made the post-window allocation explicit. The [panel note](20260804-dsp010-anomaly-panel-recording-factor.md) introduced control conditions and their transport assumptions.

The September update replaced the age-margin solver, modelled overlapping anchor errors jointly, removed a reused observation from the starting prior, respected feasibility barriers in prior sampling and expanded fit validation. Old numerical results require refitting under those changes.

Remaining scientific concerns are source uncertainty, transport to the national population, the post-anchor trend, subgroup denominators and the lack of linked validation records. More sampler effort cannot resolve these evidence limits.

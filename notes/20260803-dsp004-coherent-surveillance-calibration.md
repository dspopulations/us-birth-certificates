> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Coherent surveillance scenarios for DSP004

**Original date:** 3 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

This sensitivity analysis changed the surveillance calibration coherently rather than varying a reduction multiplier alone. `lambda` scales the external prevalence level and `delta` changes its time trajectory. The scenarios are alternatives, not draws from a validated uncertainty distribution.

| Scenario | $\lambda$ | $\delta$ | Purpose |
| --- | ---: | ---: | --- |
| Baseline | `0` | `0` | Existing independent, unshifted DSP004 prior |
| Moderate correlation | `0.5` | `0` | Errors share half their standardised variance |
| Strong correlation | `0.9` | `0` | Errors strongly co-move while retaining marginal variances |
| Negative shift, larger | `0` | `-0.4` | Directional common-level stress |
| Negative shift, smaller | `0` | `-0.2` | Directional common-level stress |
| Positive shift, smaller | `0` | `+0.2` | Directional common-level stress |
| Positive shift, larger | `0` | `+0.4` | Directional common-level stress |

| Joint corner | $\lambda$ | $\delta$ |
| --- | ---: | ---: |
| Strong correlation, larger negative shift | `0.9` | `-0.4` |
| Strong correlation, larger positive shift | `0.9` | `+0.4` |

| $\lambda$ | Prior mean total | Prior 89% interval | Interval width |
| ---: | ---: | ---: | ---: |
| `0` | 45,361 | 42,032-48,570 | 6,539 |
| `0.5` | 45,357 | 38,282-52,045 | 13,763 |
| `0.9` | 45,349 | 36,228-53,790 | 17,562 |

## Original fitted results

| Scenario | $\lambda$ | $\delta$ | Model-implied true DS livebirths, mean (89% ETI) | Mean change | ETI-width change | Aggregate rule |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | `0` | `0` | 44,255 (41,934-46,565) | reference | reference | No |
| Moderate correlation | `0.5` | `0` | 42,469 (39,211-45,808) | -4.04% | +42.43% | **Yes: width** |
| Strong correlation | `0.9` | `0` | 40,883 (38,859-42,921) | -7.62% | -12.28% | **Yes: mean** |
| Negative shift, larger | `0` | `-0.4` | 50,190 (48,178-52,177) | +13.41% | -13.67% | **Yes: mean** |
| Negative shift, smaller | `0` | `-0.2` | 47,390 (45,108-49,512) | +7.09% | -4.92% | **Yes: mean** |
| Positive shift, smaller | `0` | `+0.2` | 41,056 (38,661-43,480) | -7.23% | +4.05% | **Yes: mean** |
| Positive shift, larger | `0` | `+0.4` | 37,610 (35,023-40,262) | -15.02% | +13.13% | **Yes: mean** |

| Scenario | Mean $s$ | Change in mean $s$ | Model-implied missed true cases, mean (89% ETI) | Missed-count change | Decomposition rule |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline | 0.340 | reference | 29,216 (26,885-31,519) | reference | No |
| Moderate correlation | 0.355 | +0.015 | 27,433 (24,153-30,771) | -6.10% | No |
| Strong correlation | 0.368 | +0.028 | 25,844 (23,797-27,871) | -11.54% | **Yes: missed count** |
| Negative shift, larger | 0.300 | -0.040 | 35,151 (33,122-37,146) | +20.31% | **Yes: missed count** |
| Negative shift, smaller | 0.318 | -0.023 | 32,352 (30,066-34,482) | +10.73% | **Yes: missed count** |
| Positive shift, smaller | 0.367 | +0.027 | 26,019 (23,605-28,454) | -10.94% | **Yes: missed count** |
| Positive shift, larger | 0.401 | +0.060 | 22,572 (19,997-25,233) | -22.74% | **Yes: $s$ and missed count** |

| Joint corner | True total, mean (89% ETI) | Mean change | Mean $s$ | Missed true cases, mean (89% ETI) |
| --- | ---: | ---: | ---: | ---: |
| $\lambda=0.9,\delta=-0.4$ refined | 46,336.589 (44,331.066-48,337.931) | +4.704% | 0.325 | 31,298 (29,276-33,312) |
| $\lambda=0.9,\delta=+0.4$ | 34,806 (32,727-36,924) | -21.35% | 0.433 | 19,769 (17,674-21,876) |

The total and its allocation into true recorded and missed cases respond to different restrictions. A small response under one scenario grid supports a claim only within that grid. It does not establish insensitivity to unknown source error or all plausible trends.

The table's expected true and missed counts are functions of parameter draws. They are not realised latent case counts. The uncertainty in fixed Morris rates, a fixed `f` and omitted source transport is not supplied by these intervals.

The [workbook audit](20260803-degraaf-surveillance-workbook-extraction.md) later corrected the time-window interpretation. Use that definition and the September covariance changes for a new calibration comparison. The [race audit](20260803-dsp004-race-surveillance-audit.md) addresses the limits of extending pooled conclusions to subgroups.

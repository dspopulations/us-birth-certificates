> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# False-positive and reduction-prior sensitivity in DSP004

**Original date:** 2 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

This varied the fixed false-positive probability and uncertainty in the annual reduction priors. It tests the consequences of supplied assumptions, rather than estimating the true false-positive rate.

| Scenario | `f` | Reduction-prior logit SDs | Role |
| --- | ---: | ---: | --- |
| Perfect-specificity stress | `0` | `0.20 / 0.45` | Lower false-positive extreme; not an assumption-free baseline |
| Cohort-scaled scenario | `4.15e-5` | `0.20 / 0.45` | Makes expected false flags approximately 7.8% of this cohort's observed flags; not externally validated |
| Transported default | `7.8e-5` | `0.20 / 0.45` | Existing DSP004 operational baseline |
| High diagnostic stress | `1.20e-4` | `0.20 / 0.45` | Deliberately high scenario; not an estimated upper bound |
| Perfect specificity, independent-wide | `0` | `0.40 / 0.90` | False-positive endpoint crossed with wider independent annual priors |
| Transported default, independent-wide | `7.8e-5` | `0.40 / 0.90` | Direct reduction-prior-width contrast |

| Scenario | True DS livebirths, mean (89% ETI) | ETI width | Recording `s`, mean | Age-year PPC | Seven-band PPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Perfect-specificity stress | 44,507 (42,172-46,862) | 4,690 | 0.401 | 194/351 | 1/7 |
| Cohort-scaled scenario | 44,326 (41,934-46,708) | 4,775 | 0.367 | 249/351 | 1/7 |
| Transported default | 44,280 (41,934-46,562) | 4,627 | 0.340 | 286/351 | 1/7 |
| High diagnostic stress | 44,228 (42,020-46,507) | 4,487 | 0.313 | 293/351 | 2/7 |
| Perfect specificity, independent-wide | 45,685 (41,170-50,119) | 8,950 | 0.391 | 195/351 | 1/7 |
| Transported default, independent-wide | 45,802 (40,897-50,161) | 9,264 | 0.330 | 283/351 | 1/7 |

| `f` | Expected false flags | Share of 17,809 observed flags |
| ---: | ---: | ---: |
| `0` | 0 | 0.0% |
| `4.15e-5` | approximately 1,390 | 7.8% |
| `7.8e-5` | approximately 2,612 | 14.7% |
| `1.20e-4` | approximately 4,018 | 22.6% |

When the reduction prior largely fixes true prevalence, changing `f` mainly changes how the model divides recorded flags between true and false positives and how it estimates recording sensitivity. A stable total in that setting is conditional on the reduction prior.

The expected false flags are `sum(N * (1 - p_true) * f)`. The approximation `N * f` is close only because true Down syndrome prevalence is small. The percentage among recorded flags is a derived quantity, not `f` itself.

A better age predictive fit under a larger `f` does not establish that value as correct. The intercept can also compensate for misspecified age patterns or the natural-rate benchmark. See the [channel proposal](20260803-false-positive-channel-identification.md).

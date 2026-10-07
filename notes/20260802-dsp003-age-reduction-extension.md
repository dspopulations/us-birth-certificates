> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Age-dependent reduction in DSP003

**Original date:** 2 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

`DSP003` adds an age term to combined prenatal reduction. It centres that term so the birth-weighted annual reduction matches the intended annual margin. This tests the age pattern that the constant-within-year reduction model cannot express.

## Original scenarios

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

The table varies the fixed false-positive probability, reduction smoothness and certificate case definition. Its predictive checks concern recorded counts. Improved fit does not determine whether the age pattern is prenatal reduction, recording, the natural-rate benchmark or another omitted factor.

The default `f = 7.8e-5` came from a false-coded share among flags multiplied by an assumed recorded prevalence. A share among observed flags and a probability among births without Down syndrome have different denominators. The conversion requires assumptions and is not a direct study estimate of `f` for this population.

The September implementation replaced the fixed Newton iterations with a bounded solve and added an explicit annual-margin check. Refit before interpreting this table's totals or margins. See the [false-positive sensitivity note](20260802-dsp004-false-positive-surveillance-sensitivity.md) for the conditional allocation effect.

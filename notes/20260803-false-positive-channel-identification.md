> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# False positives and the confirmed/pending channels

**Original date:** 3 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

This diagnostic asked whether age patterns and separate confirmed/pending counts could inform `f`. It proposed a model extension but did not refit one. The regressions held reduction at historical `DSP004` posterior means, so they were not independent validation estimates.

## Conditional age regressions

If true prevalence is exactly `theta[a] * eta[y]` with age-constant `eta` and `s`, then the recorded rate is affine in `theta`:

```text
p_recorded[y,a] = f + theta[a] * eta[y] * (s - f)
```

Under those restrictions, age variation can inform an intercept `f`. Age-dependent reduction, recording, ART use or benchmark error can also affect the intercept. It is not identified without such restrictions. Misspecification does not make the fitted intercept a guaranteed upper bound.

| Cohort | Channel | `s` (SE) | `f` (SE) |
| --- | --- | ---: | ---: |
| All births | confirmed | `0.1388` (`0.0031`) | `3.414e-05` (`3.24e-06`) |
| All births | pending | `0.1407` (`0.0033`) | `8.642e-05` (`3.80e-06`) |
| All births | confirmed or pending | `0.2537` (`0.0063`) | `1.457e-04` (`7.19e-06`) |
| Excluding ART | confirmed | `0.1605` (`0.0033`) | `2.061e-05` (`3.24e-06`) |
| Excluding ART | pending | `0.1657` (`0.0033`) | `7.039e-05` (`3.60e-06`) |
| Excluding ART | confirmed or pending | `0.3249` (`0.0054`) | `1.007e-04` (`5.61e-06`) |

The separate regressions used count-dependent weights, so their coefficients do not add to the combined coefficients. A joint model should preserve the mutually exclusive outcomes and propagate uncertainty in prevalence.

## Confirmed and pending counts

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

Pending flags make up a lower share at older maternal ages in this extract. That pattern could reflect false positives, confirmation timing, diagnosis or other age-dependent recording. It does not by itself estimate the true-case proportion in either channel.

The "model-implied false-positive share" column depends on the supplied `f` and the old fit. It is not an observed share of adjudicated errors. The implied share matters more in groups with low recorded rates, even if the national total is fixed largely by an external prevalence restriction.

## Possible joint model

For mutually exclusive confirmed, pending and unflagged outcomes, use one multinomial per birth cell, with probabilities

```text
p_C = p_true * s_C + (1 - p_true) * f_C
p_P = p_true * s_P + (1 - p_true) * f_P
p_U = 1 - p_C - p_P
```

Constrain `s_C + s_P <= 1` and `f_C + f_P <= 1`. Two independent binomials with the same denominator would ignore the outcome dependence.

The split adds information about recorded channels. It also adds unknown sensitivities and false-positive rates. Identification still requires justified shared structure or external constraints. Test that prospect with simulation and source validation before presenting it as an estimate of `f`.

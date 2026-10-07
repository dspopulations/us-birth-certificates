> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!WARNING]
> This note was drafted by an AI coding assistant (Claude) from an
> autonomous session on 2026-04-20. Figures and claims below reflect a
> single dev-profile fit; a human reviewer should verify before citing.
> A reporting-profile replication is required before any substantive
> interpretation.

> [!NOTE]
> This note was written against an earlier model specification that
> carried a heteroscedastic pre/post-2022 sigma on `eta_term_year`.
> That asymmetric year treatment has since been removed in favour of a
> single homoscedastic year sigma. The year-effect findings below
> reflect the old specification and should not be read as current
> results.

# Early development check of the full selection model

**Original date:** 20 April 2026. Historical research note, revised for clarity in October 2026.

This was a development fit of the early selection model. Its priors and recording parameterisation were later changed. It is useful as a check of that implementation, rather than a current estimate.

## Original checks and estimates

| quantity | value | target |
|---|---:|---|
| max R̂ across **79 named RVs** | 1.02 | < 1.01 |
| min ESS bulk across RVs | 100 | ≥ 400 |
| min ESS tail across RVs | 188 | ≥ 400 |

| race | \|r\| | interpretation |
|---|---:|---|
| NH White | 0.02 | low marginal correlation |
| NH Black | 0.06 | low marginal correlation |
| NH AIAN/NHOPI/Other | 0.04 | low marginal correlation |
| NH Asian | 0.13 | low marginal correlation |
| Hispanic | 0.08 | low marginal correlation |
| Unknown | 0.15 | low marginal correlation |

| race | mean | 94% HDI |
|---|---:|---|
| NH White | −0.26 | [−0.61, +0.06] |
| NH Black | −0.81 | [−1.18, −0.46] |
| NH AIAN/NHOPI/Other | −0.33 | [−0.69, +0.05] |
| NH Asian | +0.00 | [−0.36, +0.42] |
| Hispanic | −0.47 | [−0.84, −0.14] |
| Unknown | −0.02 | [−0.36, +0.34] |

| race | mean | 94% HDI |
|---|---:|---|
| NH White | −0.22 | [−0.41, −0.05] |
| NH Black | −0.98 | [−1.18, −0.79] |
| NH AIAN/NHOPI/Other | +0.13 | [−0.17, +0.39] |
| NH Asian | −1.10 | [−1.33, −0.90] |
| Hispanic | −0.22 | [−0.43, −0.05] |
| Unknown | −0.41 | [−0.66, −0.19] |

The original table used a threshold on posterior correlation to classify parameters as data-informed or prior-driven. Those interpretations were too strong. High correlation can show a trade-off. Low correlation cannot establish that the data identify each parameter, especially under tight priors.

The natural-rate, screening, termination and recording stages appear together in the recorded-count probability. A fit can reproduce the recorded counts without learning their separate contributions. The [July diagnostic review](20260707-s-anchor-and-identifiability-diagnostic.md) sets out the required sensitivity checks.

Use the [selection guide](../src/dspopulations_us_birth_certificates/selection/README.md) for current profiles. A profile sets sampler effort, not a guarantee of adequate sampling.

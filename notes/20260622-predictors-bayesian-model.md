> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!WARNING]
> This note was drafted by an AI coding assistant.

# What classifier scores and selection models can estimate

**Original date:** 22 June 2026. Historical research note, revised for clarity in October 2026.

This note explains the distinction that guided later work. Its original fitted examples came from June selection-model runs and are now historical.

## The observed and unobserved quantities

For cell `j`, the selection model uses a recorded count `R_j` among `N_j` births. It models the recorded probability as

```text
p_true = theta_lb * eta
eta = 1 - detection * termination
p_recorded = p_true * s + (1 - p_true) * f
R ~ Binomial(N, p_recorded)
```

`theta_lb` is the expected Down syndrome live-birth rate without selective termination. `eta` is the remaining fraction relative to that counterfactual. `s` is the recording probability among true cases. `f` is the false-positive probability among births without Down syndrome.

The recorded counts constrain `p_recorded`. They cannot separate all of its factors without restrictions or other evidence. Priors and surveillance observations supply those restrictions. A posterior interval describes uncertainty conditional on them; it does not include every possible error in the model or source data.

## Historical fitted examples

| quantity                      | estimate (2016–2024)                                                                | interpretation                                                   |
| ----------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **True DS livebirths**        | **~40,000** (pin recording ≈0.40) **to ~48,000** (let the data set recording ≈0.32) | recorded + missed; the range _is_ the recording-assumption bound |
| Recorded (data)               | 17,776 (~15,200 after removing estimated false positives)                           | what the certificates caught                                     |
| **Implied missed**            | **~25,000–33,000 (≈62–68%)**                                                        | true babies the certificates didn't record                       |
| Recording rate `s`            | 0.38 (pinned) → 0.32 (data-preferred, variant B)                                    | fraction of true DS recorded                                     |
| Termination reduction `1 − η` | ~34–45% overall; age-graded ~5% (under 20) → ~64% (45+)                             | share of DS pregnancies electively terminated                    |

| variant                | recording demographics | termination demographics | gaps load onto | total |
| ---------------------- | ---------------------- | ------------------------ | -------------- | ----- |
| **C** (main spec)      | moderately pinned      | moderately free          | balanced       | ~40k  |
| **A** (tight `s`)      | pinned harder          | freer                    | termination    | ~40k  |
| **B** (tight `η_term`) | freer                  | pinned harder            | recording      | ~48k  |

These totals depend on the June priors and parameterisation. Similar estimates across variants can arise from shared assumptions. The [anchor review](20260707-s-anchor-and-identifiability-diagnostic.md) explains why low posterior correlation is also insufficient evidence of identification.

## Classifier scores

A classifier trained against the certificate label estimates `P(recorded DS | X)`. The recording sensitivity is `P(recorded DS | true DS, X)`. They are different probabilities.

With no false positives, `P(recorded DS | X) = P(true DS | X) * s(X)`. Recovering true prevalence from this product requires information about `s(X)`. False positives add another term. A calibrated score against the certificate label therefore remains calibrated to that label, rather than to true Down syndrome.

The [positive-unlabelled learning survey](https://arxiv.org/abs/1811.04820) discusses this kind of problem and the assumptions used to estimate prevalence or classify cases. The usual setup assumes labelled positives are true positives; the certificate analysis must also consider false labels. Feature removal alone does not verify the assumptions needed for correction.

## Co-occurring conditions

Let `q_obs` be the proportion of recorded DS cases with a recorded co-occurring condition. If DS recording is `r` times as likely among cases with that condition as among cases without it, then, under accurate condition labels and no DS false positives,

```text
q_true = q_obs / (r * (1 - q_obs) + q_obs)
```

At `r = 1`, this equals the recorded proportion. That is an assumption of equal recording, not a result established by a lack of evidence for unequal recording. Validation samples with many cardiac conditions do not estimate the within-DS recording-rate ratio by themselves. Condition under-recording, diagnosis timing and DS false positives require further checks.

The certificate's cyanotic congenital heart disease item is narrower than all congenital heart defects. Do not compare those rates as if the definitions matched.

The original claims that recorded co-occurring rates were the supported population answer, and that classifier-selected cohorts proved the opposite, have been removed. Both claims required evidence the analysis did not supply.

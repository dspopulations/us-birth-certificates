> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!WARNING]
> This note was drafted by an AI coding assistant (Claude) from a
> working session on 2026-04-19. Figures, numbers, and claims reflect
> the specific fit runs recorded below; a human reviewer should verify
> the interpretation before citing.

# Initial monthly outcome models

**Original date:** 19 April 2026. Historical research note, revised for clarity in October 2026.

This note records early models of monthly recorded Down syndrome births and a cohort selected by a classifier. Those outcome models are historical. The current model families are in the [model guide](../docs/models/README.md).

## Recorded results

| outcome | output dir | n_cells | n_total | y_total | overall rate |
|---|---|---:|---:|---:|---:|
| `recorded` | `output/bayes/m1-year-age/recorded/20260419-204041/` | 4,212 | 33,527,704 | 17,809 | 5.31 / 10,000 |
| `recorded_plus_predicted` | `output/bayes/m1-year-age/recorded_plus_predicted/20260419-204653/` | 4,212 | 33,527,704 | 44,551 | 13.29 / 10,000 |

| outcome | max Rhat | min ESS_bulk |
|---|---:|---:|
| `recorded` | 1.0100 | 430 (`alpha`) |
| `recorded_plus_predicted` | 1.0100 | 592 (`alpha`) |

| age | recorded rate | r+p rate | ratio |
|---:|---:|---:|---:|
| 20 | 0.29 / 1,000 | 0.94 / 1,000 | **3.2×** |
| 25 | 0.25 / 1,000 | 0.88 / 1,000 | **3.5×** |
| 35 | 0.62 / 1,000 | 1.54 / 1,000 | 2.5× |
| 40 | 2.37 / 1,000 | 4.52 / 1,000 | 1.9× |
| 45 | 6.12 / 1,000 | 11.15 / 1,000 | 1.8× |

The output paths identify the original runs. A classifier-selected cohort consists of high scores subject to a quota. It does not provide verified missed cases. A seasonal pattern in this cohort may come from the recording process, the classifier, or the quota.

Maternal age needs explicit adjustment because the expected Down syndrome rate changes with age. The monthly model did not separate changes in birth prevalence from changes in recording. Its convergence checks describe sampling under its assumptions; they do not validate the outcome definition.

The later [predictor note](20260622-predictors-bayesian-model.md) explains this limit. Use the [current workflow](../docs/modelling-workflow.md) for commands rather than the original run paths.

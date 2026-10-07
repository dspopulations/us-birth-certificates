> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Annual recording sensitivity in DSP002

**Original date:** 2 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

`DSP002` replaces the constant recording sensitivity in `DSP001` with a smooth year-specific term. Both use seven maternal-age bands and the same externally constrained reduction trajectory.

## Original comparison

| Model | Max Rhat | Min ESS |
| --- | ---: | ---: |
| `DSP001` | 1.0000 | 1519 |
| `DSP002` | 1.0000 | 7055 |

| Metric | `DSP001` | `DSP002` | Difference |
| --- | ---: | ---: | ---: |
| True DS livebirths, posterior mean | 43,828 | 44,535 | +706 |
| True DS livebirths, 89% ETI | 41,444-46,130 | 41,376-47,617 |  |
| Aggregate reduction, posterior mean | 0.396 | 0.386 | -0.010 |
| Global recording sensitivity, posterior mean | 0.344 | 0.341 | -0.002 |

| Year | Posterior mean | 89% ETI |
| --- | ---: | ---: |
| 2016 | 0.363 | 0.325-0.409 |
| 2017 | 0.341 | 0.305-0.383 |
| 2018 | 0.363 | 0.323-0.410 |
| 2019 | 0.348 | 0.309-0.394 |
| 2020 | 0.346 | 0.280-0.435 |
| 2021 | 0.335 | 0.267-0.425 |
| 2022 | 0.326 | 0.257-0.414 |
| 2023 | 0.328 | 0.260-0.418 |
| 2024 | 0.331 | 0.259-0.424 |

The fit can shift a recorded trend between reduction and recording only within the restrictions supplied by the priors. A year-specific recording term does not independently identify recording change.

Similar totals in these two fits reflect their shared reduction inputs. They do not show that the total is insensitive to all plausible recording assumptions. Later exact-age and surveillance-anchored models test other restrictions. The original recommendation to use `DSP001` as the main model is superseded by the [current inventory](../docs/models/README.md).

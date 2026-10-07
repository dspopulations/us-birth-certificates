> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Post-anchor recording drift in DSP009

**Original date:** 4 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

`DSP009` allows recording sensitivity to drift after the last year contributing to a surveillance window. With the latest window ending in 2020, drift begins in 2021. Its fixed innovation scale is a sensitivity choice, rather than a quantity identified from post-window certificate counts alone.

## Allocation scenarios and chain audit

| Configuration | Post-window decline attributed to |
| --- | --- |
| `--recording-s-drift-sigma 0` | Prevalence (identical to `DSP008`) |
| default drift `0.06` | Both, in the ratio of the drift SD to the anchor's state variances |
| `--anchor-forecast-flat --recording-s-drift-sigma 0.20` | Recording |

| Quantity | Chains 0, 2, 3 | Chain 1 |
| --- | ---: | ---: |
| `anchor_obs_sigma` | `0.0125` | `0.8417` |
| `anchor_level_sigma` | `0.0119` | `0.1310` |
| `recording_s` | `0.3350` | `0.0060` |
| `prevalence_year` 2004, per 10,000 | `12.69` | `720.9` |
| `eta_year` 2004 | `0.702` | `39.88` |

| Component | Chains 0, 2, 3 | Chain 1 | Delta |
| --- | ---: | ---: | ---: |
| Cell Binomial log-likelihood | `-3,596.8` | **`-3,513.1`** | `+83.7` |
| Surveillance observation log-likelihood | `+41.5` | `-157.2` | `-198.7` |
| `anchor_obs_sigma` prior | `+2.74` | `-138.92` | `-141.66` |
| `anchor_level_sigma` prior | `+3.51` | `-17.77` | `-21.27` |
| `recording_s_logit` prior | `-1.15` | `-13.99` | `-12.84` |
| **Total** | | | **`≈ -291`** |

| Model | Drift | Observation SD | Prevalence tail | Verdict |
| --- | --- | --- | --- | --- |
| `DSP007` | none | free | forecast | CLEAN |
| `DSP008` | none | free | forecast | CLEAN |
| `DSP009` | `post_anchor` | free | forecast | **DEGENERATE** |
| `DSP009` | `post_anchor` | free | flat | CLEAN |
| `DSP009` | `post_anchor` | fixed `0.05` | forecast | CLEAN |
| `DSP009` | `post_anchor` | fixed `0.10` | forecast | CLEAN |

One original chain combined inflated surveillance uncertainty with a different prevalence/recording allocation. Aggregate summaries concealed the difference. Fixing surveillance observation uncertainty and adding a feasibility barrier improved that particular fit. Those results support checking chains separately, not assuming that every anchored fit has the same problem.

The original log-density gap of about 194 corresponds to a very small density ratio, about `5e-85`. It is not zero at double precision. A lower-density region can still trap a sampler, so density comparisons do not replace sampling diagnostics.

## Original sensitivity results

| Fit | 2016-2024 total | 89% ETI | Width | vs. corner | MCSE |
| --- | ---: | --- | ---: | ---: | ---: |
| `DSP008`, all prevalence | `44,544` | `43,295`-`45,819` | `5.66%` |, | `5.6` |
| `DSP009`, drift `0.06` | `45,370` | `43,612`-`47,201` | `7.91%` | `+1.85%` | `7.6` |
| `DSP009`, flat + `0.20`, all recording | `45,795` | `44,316`-`47,303` | `6.52%` | `+2.81%` | `6.2` |

| Fit | `s` revised | `s₂₀₂₄` / anchored | Prevalence 2024 vs 2018 | 2021-2024 subtotal |
| --- | ---: | ---: | ---: | ---: |
| `DSP008` | `0.3363` | `1.0000` | `-6.25%` | `18,858` |
| `DSP009` drift `0.06` | `0.3364` | `0.9629` | `-1.68%` | `19,616` |
| `DSP009` flat corner | `0.3364` | `0.9437` | `+0.18%` | `19,981` |

| Fit | 2016-2024 total | 89% ETI | Width | vs. corner | Prevalence 2024 vs 2018 |
| --- | ---: | --- | ---: | ---: | ---: |
| `DSP008`, all prevalence | `44,405` | `42,318`-`46,582` | `9.60%` |, | `-6.34%` |
| `DSP009`, drift `0.06` | `45,184` | `42,744`-`47,763` | `11.11%` | `+1.75%` | `-1.95%` |

Allowing drift moved the historical mean total by about 1.8%. This is a change under an alternative assumption, not a measured bias of the constant-recording model. The counts after the final anchor constrain a prevalence/recording product. They cannot decide its allocation without additional information.

The current default fixes anchor observation log-SD at 0.05. Use `--anchor-obs-sigma-estimated` explicitly to test an estimated scale; omitting the fixed-scale flag does not free it. A larger number of draws cannot guarantee adequate sampling.

The September code now includes feasibility-aware prior sampling, diagnostics for the active parameterisation and a fit-time failure status for invalid runs. The older pending tasks to add those checks are complete. See [the fix record](20260905-dsp-code-review-fixes.md) and [workflow](../docs/modelling-workflow.md).

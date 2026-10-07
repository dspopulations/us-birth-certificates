> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Exact-age comparisons in DSP004 and DSP005

**Original date:** 2 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

`DSP004` uses 39 maternal-age codes instead of seven bands. `DSP005` uses that resolution with annual recording sensitivity. These comparisons test how age aggregation affects the recorded-count fit.

| Model | Maternal-age resolution | Recording | Combined reduction | Role |
| --- | --- | --- | --- | --- |
| `DSP001` | Seven bands | Constant `s` | One value per year | Band-resolution reference |
| `DSP004` | NCHS single-age codes | Constant `s` | One value per year | Exact-age ablation of `DSP001` |
| `DSP002` | Seven bands | Partially pooled `s_year` | One value per year | Year-varying-recording reference |
| `DSP005` | NCHS single-age codes | Partially pooled `s_year` | One value per year | Exact-age ablation of `DSP002` |
| `DSP003` | NCHS single-age codes | Constant `s` | Smooth age pattern within year | Age-allocation diagnostic |

| Model | True DS livebirths, mean (89% ETI) | Recording sensitivity centre, mean (89% ETI) | Age PPC | Age-year PPC | Seven-band PPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `DSP004` | 44,280 (41,934-46,562) | 0.340 (0.322-0.360) | 18/39 | 286/351 | 1/7 |
| `DSP005` | 45,059 (41,839-48,169) | 0.337 (0.314-0.363) | 17/39 | 281/351 | 1/7 |

| Comparison | 89% interval coverage on common grid | Mean absolute standardised residual |
| --- | ---: | ---: |
| `DSP001` to `DSP004` | 221/351 to 282/351 | 1.690 to 1.019 |
| `DSP002` to `DSP005` | 221/351 to 284/351 | 1.687 to 1.018 |

| Metric | `DSP004` | `DSP003` |
| --- | ---: | ---: |
| True DS livebirths, posterior mean | 44,280 | 41,834 |
| Common-grid coverage | 282/351 | 320/351 |
| Mean absolute standardised residual | 1.019 | 0.770 |
| Seven-band coverage on the common grid | 1/7 | 7/7 |

Comparisons use a common age grid where possible. A predictive interval covering more recorded cells is evidence about that observed-data fit. It does not validate the number of true cases or identify the split between prenatal reduction and recording.

The seven-band Morris probabilities average a nonlinear risk curve. The exact-age versions use the available age codes, including their endpoint recodes; they do not recover each mother's unrecoded exact age.

Use `DSP004` as the exact-age unanchored reference in a model comparison, as described in the [inventory](../docs/models/README.md). The September changes require new fits before publication.

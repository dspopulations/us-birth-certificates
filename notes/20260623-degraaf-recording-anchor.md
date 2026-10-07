> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!WARNING]
> Work in progress. All data and models are preliminary. Totals quoted here are from the
> **dev** sampler preset (2 chains, 1000+1000); re-run at the `reporting` preset before citing.

# First surveillance-derived recording anchor

**Original date:** 23 June 2026. Historical research note, revised for clarity in October 2026.

This records the first recording anchor for the selection model. Its prevalence extraction was corrected in [June](20260628-degraaf-corrected-prevalence-extraction.md). The [August workbook audit](20260803-degraaf-surveillance-workbook-extraction.md) later distinguished annual labels from five-year surveillance windows.

The generator reconstructed a true count from surveillance prevalence and the birth denominator, then formed `s = recorded / reconstructed_true`. It standardised by the observed maternal-age distribution and extrapolated a retained-fraction trajectory where surveillance inputs were absent. A pre-2015 holdout favoured a constant tail over a linear tail; that does not establish a constant tail after 2020.

## Original development results

| race | de-Graaf `s` | old hand-set prior |
|------|------|------|
| NH White | 0.46 | 0.40 |
| NH Black | 0.33 | 0.31 |
| NH AIAN | 0.71 (noisy) | 0.33 |
| NH Asian/PI | 0.37 | 0.38 |
| Hispanic | 0.38 | 0.35 |

| | s-only anchor | full-margin | de Graaf target |
|---|---|---|---|
| Total true DS 2016–2024 | 36,930 (95% 31.6–42.4k) | **40,718 (95% 35.5–45.9k)** | 45,928 |
| Overall `s` | 0.481 | 0.437 | ~0.38 |

The tables describe the original extraction and sampler settings. They do not support choosing a current total or tightening the surveillance uncertainty to reach a preferred estimate.

The anchor uses the same recorded counts later fitted by the model. It is therefore not fully independent of the recorded-count likelihood. Combining a derived recording prior with a prevalence observation can reuse the same evidence. Convergence improvement does not settle this dependence.

The current generator has seven race rows. Unknown and multi-race have weak fallback recording priors and no prevalence-margin target. See the [selection guide](../src/dspopulations_us_birth_certificates/selection/README.md).

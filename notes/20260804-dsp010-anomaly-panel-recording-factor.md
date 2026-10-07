> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# An anomaly panel for recording change in DSP010

**Original date:** 4 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

`DSP010` uses other certificate anomaly items as controls for shared recording change. It combines the DS likelihood and surveillance anchor with condition-specific levels and trends, a common recording factor, and a DS loading on that factor.

## Original panel patterns

| Control | Change | Poisson SE | Flags/year |
| --- | ---: | ---: | ---: |
| Hypospadias | `-15.5%` | `1.8%` | `2,071` |
| Limb reduction | `-12.9%` | `3.8%` | `459` |
| Cleft palate alone | `-7.7%` | `2.8%` | `857` |
| Cleft lip ± palate | `-2.2%` | `1.9%` | `1,924` |

Age standardisation removes one source of composition change. It does not prove that a remaining common trend is recording. The interpretation assumes suitable controls and adequate restrictions on their true prevalence trends.

## Historical fits

| Fit | 2016-2024 total | 89% ETI | Width | vs. corner | `s₂₀₂₄` vs. own reference | Prevalence 2024 vs 2018 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `DSP008`, all prevalence | `44,536` | `43,290`–`45,795` | `5.63%` |, | `1.0000` (constant) | `-6.22%` |
| `DSP009`, drift prior `0.06` | `45,370` | `43,596`–`47,186` | `7.91%` | `+1.87%` | `0.9623` (vs `2020`) | `-1.60%` |
| `DSP009`, flat + `0.20`, all recording | `45,780` | `44,321`–`47,253` | `6.40%` | `+2.79%` | `0.9444` (vs `2020`) | `+0.18%` |
| **`DSP010`, panel, pinned, γ = 0** | **`45,666`** | **`43,879`–`47,665`** | **`8.29%`** | **`+2.54%`** | **`0.9593` (vs `2016`)** | **`-3.01%`** |
| `DSP010`, pinned, γ free at `0.004` | `45,653` | `43,799`–`47,782` | `8.72%` | `+2.51%` | `0.9595` (vs `2016`) | `-3.06%` |
| `DSP010`, unpinned curation, γ free | `45,828` | `43,887`–`48,097` | `9.19%` | `+2.90%` | `0.9516` (vs `2016`) | `-2.37%` |
| `DSP010`, pinned, γ = 0, loading `1` | `45,742` | `44,047`–`47,483` | `7.51%` | `+2.71%` | `0.9561` (vs `2016`) | `-2.86%` |

| Quantity | Posterior | Prior |
| --- | --- | --- |
| Common log-rate change, 2024 vs 2016 | `-6.34%` [`-13.07%`, `+1.48%`] |, (data) |
| Item recording factor, 2024 vs 2016 | `-6.34%` [`-13.07%`, `+1.48%`] |, |
| Down syndrome `s`, 2024 vs 2016 | `-4.07%` [`-10.52%`, `+0.92%`] |, |
| Down syndrome loading | `0.941 ± 0.456` | `1 ± 0.5` |
| Common prevalence trend, log/year | fixed at `0` | measured at `-0.00262` externally |
| Between-condition trend SD, log/year | `0.0147` [`0.0071`, `0.0267`] | `HalfNormal(0.02)` |

The loading maps a panel factor onto the DS recording logit. A change on that scale is not the same percentage change in every item's recorded rate. The posterior loading SD relative to its prior SD describes remaining dispersion; it is not a percentage of the estimate supplied by the prior.

Fixed control trends, a shared prevalence-trend restriction and the prior on the loading all affect the inferred DS drift. If an interval includes zero, the corresponding contrast does not exclude no change at that interval level.

## Control-trend inputs

The [Texas trend note](20260804-dsp010-control-prevalence-trend-pins.md) supplied point estimates for three controls and retained zero for hypospadias after a level-shift diagnostic. Those estimates come from one state and another period. Their sampling and transport uncertainty does not enter the current model as per-condition uncertainty.

## Dependence and source limits

The panel uses condition-specific binomials on the same births. Conditions can co-occur, and the two cleft fields can be coded inconsistently. The August audit found 1,254 co-flagged cleft records, or 16.3% of cleft-palate-alone flags. Dropping one control barely changed an unpinned fit, but that check did not validate every specification or remove dependence.

Hypospadias has a male-only clinical denominator, while the model uses the common birth denominator and a condition level. A stable male share can absorb the level difference, but a changing share could affect its trend.

The panel's causal interpretation also depends on diagnosis timing, clinical coding, termination and surveillance coverage. A shared pattern among selected controls cannot independently rule out shared prevalence or ascertainment changes.

`DSP010` does not combine the panel with `DSP009`'s unrestricted drift. A proposed residual-drift extension would still need identification checks; defining a residual does not automatically supply them. Current commands and diagnostics are in the [workflow guide](../docs/modelling-workflow.md).

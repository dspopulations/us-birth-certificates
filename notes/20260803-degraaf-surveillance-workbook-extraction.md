> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Audit of the surveillance workbook and overlapping windows

**Original date:** 3 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

The audit reconstructed the de Graaf workbook, checked race denominators and used its pooled surveillance series to build anchored models. It separates the workbook's observed surveillance prevalence from fitted correction factors and projections.

## Reconstruction evidence

| Check | Max relative error |
| --- | ---: |
| `U == Q/R` | `0.000e+00` |
| `G == U` where surveillance observed (75 cells) | `0.000e+00` |
| `G ==` fitted line where surveillance missing (50 cells) | `0.000e+00` |
| `O ==` centred five-year sum of `C` | exact |
| `R == L` | `0.000e+00` |

| Race | n | Slope refit | Rel. error | Intercept refit | Rel. error | R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nhw | 15 | `0.001692095` | `2.2e-13` | `0.421962462` | `1.3e-15` | `0.704` |
| nhb | 15 | `0.005150633` | `6.5e-14` | `0.220699387` | `8.8e-16` | `0.906` |
| his | 15 | `0.001980572` | `1.7e-10` | `0.316338543` | `5.0e-13` | `0.235` |
| as/pi | 15 | `0.001836718` | `2.2e-14` | `0.320411014` | `1.7e-16` | `0.137` |
| ai/an | 15 | `0.009189841` | `2.7e-14` | `0.335633908` | `3.3e-16` | `0.318` |
| pooled | 15 | `0.001714238` | `2.3e-13` | `0.366315772` | `1.5e-16` | `0.762` |

| Years | Reporting fraction `G` |
| --- | --- |
| 2000–2001 | fitted line, extrapolated **backwards** |
| 2002–2014 | observed |
| 2015 | fitted line, interpolated |
| 2016 | observed |
| 2017 | fitted line, interpolated |
| 2018 | observed |
| 2019–2024 | fitted line, extrapolated **forwards** |

The source surveillance values represent centred five-year windows. Row labels mark their mid-year, not an independent annual prevalence observation. The pooled workbook correction is constructed from race rows; it is not an additional independent surveillance measurement.

Multi-race handling and Pacific Islander grouping differed across source denominator series. The pooled total was more stable than the race-specific allocations. Do not use matched pooled totals to validate each race denominator.

## Levels and time coverage

| Mid-year | Window | Births (5yr) | Prev./10⁴ | Expected/yr | Recorded/yr | Reported | Residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 1998–2002 | `20,031,387` | `11.6821` | `4,680.2` | `1,791.2` | `0.3827` | `1.0%` |
| 2002 | 2000–2004 | `20,337,729` | `12.5183` | `5,091.9` | `1,869.0` | `0.3671` | `0.8%` |
| 2004 | 2002–2006 | `20,661,214` | `12.4901` | `5,161.2` | `1,956.2` | `0.3790` | `0.7%` |
| 2006 | 2004–2008 | `21,116,910` | `12.5924` | `5,318.2` | `2,022.2` | `0.3802` | `0.7%` |
| 2008 | 2006–2010 | `20,997,330` | `12.5272` | `5,260.8` | `2,016.0` | `0.3832` | `0.7%` |
| 2010 | 2008–2012 | `20,322,113` | `12.5292` | `5,092.4` | `1,949.0` | `0.3827` | `0.7%` |
| 2012 | 2010–2014 | `19,868,060` | `12.7820` | `5,079.1` | `1,941.6` | `0.3823` | `1.1%` |
| 2014 | 2012–2016 | `19,844,580` | `13.3721` | `5,307.3` | `2,026.0` | `0.3817` | `2.0%` |
| 2016 | 2014–2018 | `19,609,281` | `13.1618` | `5,161.9` | `2,072.4` | `0.4015` | `2.9%` |
| 2018 | 2016–2020 | `18,999,754` | `13.4780` | `5,121.6` | `2,060.8` | `0.4024` | `3.1%` |

| Anchor basis | Prevalence/10⁴ | Expected total | vs `DSP004` |
| --- | ---: | ---: | ---: |
| 2016 window held flat | `13.1618` | `44,128` | `-0.3%` |
| 2018 window held flat | `13.4780` | `45,189` | `+2.1%` |
| Log-linear trend extrapolated | `13.539` | `45,393` | `+2.6%` |

| Year | Implied true prevalence /10⁴ (col H) | Mean `G` |
| --- | ---: | ---: |
| 2016 | `13.664` | `0.4259` |
| 2018 | `13.782` | `0.3976` |
| 2020 | `13.442` | `0.4024` |
| 2022 | `12.579` | `0.4103` |
| 2024 | `12.665` | `0.4183` |

The workbook's reporting fraction and the model's sensitivity differ when certificate flags include false positives. The ratio of recorded to true counts includes those flags; `s` conditions on true cases only.

A five-year window ending in 2020 constrains years through 2020 even when its label is 2018. The post-window tail begins after the latest contributing year. Earlier annual-label interpretations are superseded.

## Certificate-era audit

| Era | Pooled reporting fraction |
| --- | --- |
| 15 windows, mid-2000 to 2014 | **`0.3793`**, sd `0.0047` |
| 2 windows, 2016 and 2018 | **`0.4019`** |

| Model of the pooled reporting fraction | Coefficient | R² |
| --- | ---: | ---: |
| Linear trend in year | `+0.001219` | `0.560` |
| Revised-certificate coverage | `+0.015561` | `0.439` |
| **Step at 2015** | `+0.022668` | **`0.746`** |
| Linear trend, 2000–2014 only | `+0.000563` |, |

| Start | Anomaly status unknown | Race unknown | Source field | Confirmed/pending |
| --- | ---: | ---: | --- | --- |
| 1989 | **`17.57%`** | `5.12%` | `downs` | none |
| 1993 | `6.52%` | `1.26%` | `downs` | none |
| 1996 | `2.12%` | `1.46%` | `downs` | none |
| 2003 | `1.34%` | `0.70%` | `uca_downs` | none |
| **2004** | **`1.17%`** | `0.79%` | `uca_downs` + `ca_down(s)` | within revised only |
| 2016 | `0.17%` | `0.92%` | `ca_down(s)` | complete |

The certificate revision was adopted over several years. Early unrecorded or unknown flags, race coding and the confirmed/pending split are not directly comparable to 2016 onward. An era offset can model an observed difference; it does not establish that the revision caused it.

## Original model comparisons

| Fit | 2016–2024 total | ETI width | `s` |
| --- | ---: | ---: | --- |
| `DSP004`, 2016–2024 (frozen) | `44,255` [`41,934`–`46,565`] | `10.46%` | `0.3402` |
| `DSP004`, 2004–2024 | `45,866` [`44,566`–`47,155`] | `5.64%` | `0.3271` |
| **`DSP006`, 2004–2024** | **`44,505`** [`43,219`–`45,814`] | **`5.83%`** | `0.3379` revised / `0.2954` unrevised |

| Fit | Level from | `s` | 2016–2024 total | ETI width |
| --- | --- | --- | ---: | ---: |
| `DSP004`, 2016–2024 (frozen) | reduction CSV | `0.3402` | `44,255` [`41,934`–`46,565`] | `10.46%` |
| `DSP004` | reduction CSV | `0.3271` | `45,866` [`44,566`–`47,155`] | `5.64%` |
| `DSP006` | reduction CSV | `0.3379` / `0.2954` | `44,505` [`43,219`–`45,814`] | `5.83%` |
| `DSP007` | **surveillance** | `0.3254` | `45,487` [`44,804`–`46,240`] | `3.16%` |
| **`DSP008`** | **surveillance** | `0.3347` / `0.2970` | **`44,589`** [`43,952`–`45,231`] | **`2.87%`** |

| Surveillance observation SD | 2016–2024 total | ETI width |
| --- | ---: | ---: |
| estimated (`0.012`) | `44,580` | `2.87%` |
| fixed `0.05` | `44,522` | `5.86%` |
| fixed `0.10` | `44,441` | `9.69%` |
| fixed `0.20` | `44,088` | `16.19%` |

These fits preceded the September changes. Small intervals here depend strongly on the assigned surveillance uncertainty and other restrictions. A narrower interval is not evidence that an uncertainty choice is correct.

## Shared-window errors

Matching each observed window to the mean of its latent annual values is necessary, but it does not make shared sampling errors independent. Windows containing the same births share observation error as well as latent years.

The original note claimed that averaging latent years handled this dependence. The September model now uses a joint covariance based on window overlap for log-prevalence observations. See the [fix record](20260905-dsp-code-review-fixes.md) for its assumptions and implementation. That covariance still does not cover every shared systematic error in the surveillance source.

Remaining source questions concern the case definition, surveillance programme coverage, uncertainty in observed prevalence, and comparability of race denominators. Current commands and model options are in the [workflow guide](../docs/modelling-workflow.md).

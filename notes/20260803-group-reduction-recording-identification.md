> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# What subgroup certificate ratios constrain

**Original date:** 3 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

This diagnostic compared recorded rates with the Morris age benchmark across race, payer, year and education. It did not fit a subgroup model. The numerical ratios below use the August data and false-positive scenarios.

## Original diagnostic tables

| Group | Births | Mean maternal age | Flags | Raw ratio | False-positive share | Adjusted ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NH AIAN | `246,101` | `27.3` | `184` | `0.4444` | `10.4%` | `0.3981` |
| Hispanic | `8,256,529` | `28.3` | `4,969` | `0.2925` | `13.0%` | `0.2546` |
| NH Multi-race | `771,018` | `27.9` | `383` | `0.2654` | `15.7%` | `0.2237` |
| NH White | `17,044,320` | `29.7` | `9,419` | `0.2535` | `14.1%` | `0.2177` |
| NH Black | `4,737,259` | `28.2` | `1,998` | `0.2045` | `18.5%` | `0.1667` |
| NH Asian/PI | `2,165,760` | `32.0` | `669` | `0.1012` | `25.3%` | `0.0756` |

| Age band | 2016 | 2024 | Change | SE | Flags 2016/2024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| under 25 | `0.4261` | `0.3824` | `-10.3%` | `9.1%` | `300`/`199` |
| 25-29 | `0.3506` | `0.3207` | `-8.5%` | `8.2%` | `337`/`266` |
| 30-34 | `0.2622` | `0.2266` | `-13.6%` | `7.1%` | `430`/`374` |
| 35-39 | `0.2461` | `0.1981` | `-19.5%` | `5.8%` | `623`/`565` |
| 40+ | `0.2408` | `0.2071` | `-14.0%` | `6.6%` | `447`/`458` |

| Age band | All payers | Medicaid | Private | Medicaid ÷ Private | Flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| under 20 | `0.315` | `0.317` | `0.318` | `0.99` | `433` |
| 20-24 | `0.269` | `0.275` | `0.254` | `1.08` | `1,642` |
| 25-29 | `0.236` | `0.239` | `0.227` | `1.05` | `2,610` |
| 30-34 | `0.194` | `0.238` | `0.165` | `1.45` | `3,594` |
| 35-39 | `0.201` | `0.267` | `0.163` | `1.63` | `5,252` |
| 40-44 | `0.223` | `0.287` | `0.176` | `1.63` | `3,650` |
| 45+ | `0.198` | `0.293` | `0.155` | `1.89` | `392` |

| Age band | `f = 0` | `f = 7.8e-05` | `f = 1.007e-04` | `f = 1.457e-04` |
| --- | ---: | ---: | ---: | ---: |
| under 20 | `1.00` | `0.99` | `0.99` | `0.99` |
| 25-29 | `1.04` | `1.05` | `1.06` | `1.07` |
| 35-39 | `1.57` | `1.63` | `1.65` | `1.70` |
| 40+ | `1.64` | `1.66` | `1.66` | `1.67` |

| `f` | Under-20 ratio | Implies `s >=` | `T <=` | `T <=` if `eta <= 0.85` |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `0.432` | `0.432` | `41,215` | `35,033` |
| `7.8e-05` | `0.315` | `0.315` | `48,177` | `40,950` |
| `1.007e-04` | `0.281` | `0.281` | `51,288` | `43,595` |

| Education | Births | Flags | Ratio | SE |
| --- | ---: | ---: | ---: | ---: |
| At most high school | `321,744` | `1,638` | `0.305` | `0.008` |
| Some college | `245,814` | `1,002` | `0.249` | `0.008` |
| Bachelor | `271,691` | `813` | `0.181` | `0.006` |
| Graduate or professional | `238,608` | `492` | `0.123` | `0.006` |

| Education | Births | Flags | Ratio | SE |
| --- | ---: | ---: | ---: | ---: |
| At most high school | `5,181,396` | `1,405` | `0.280` | `0.007` |
| Some college | `2,032,468` | `547` | `0.273` | `0.012` |
| Bachelor | `340,109` | `99` | `0.299` | `0.030` |

The "adjusted ratio" removes a supplied false-positive term and uses the natural-rate benchmark. It estimates a product involving prenatal retention and recording only if that benchmark and the false-positive assumption are suitable. It is not free of assumptions.

Within a broad age band, groups can still have different single-year age distributions. A band-level comparison does not exactly cancel age risk. Standardisation needs the available age resolution and a stated target population.

## Conditional restrictions

If `eta <= 1` and the adjusted ratio equals `eta * (s - f)`, the ratio can imply a lower bound on `s`. That bound is conditional on the benchmark, correctly specified `f` and the probability model. ART, age miscoding or source mismatch can invalidate its interpretation.

A payer or education contrast cannot be allocated to recording or prenatal reduction without further restrictions. A proposed inequality such as higher screening access implying lower retention needs external evidence and compatible populations. Even then, it may constrain a range rather than identify each group's rate.

Lack of a detectable education contrast is not evidence of equal recording. Similar products can hide opposing changes in their factors. A model with more groups does not create the missing evidence.

## Connection to later work

The [false-positive proposal](20260803-false-positive-channel-identification.md) discusses the intercept and channel assumptions. The [study-area note](20260804-salemi-boulet-study-area-transport.md) gives measured local sensitivities and a conditional national rescaling. The [race audit](20260803-dsp004-race-surveillance-audit.md) explains why source window and denominator alignment must precede a subgroup fit.

The original claim that the table established separate group reduction and recording rates has been removed. The useful result is a set of observed patterns and conditional restrictions to test in a later model.

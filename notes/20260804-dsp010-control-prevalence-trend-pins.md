> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Texas surveillance trends used by the anomaly panel

**Original date:** 4 August 2026. Historical research note, revised for clarity in October 2026.

Fit results below predate the [September model fixes](20260905-dsp-code-review-fixes.md). They have not been regenerated for this review. Use them to trace decisions, and refit before citing estimates or intervals.

This fitted annual Texas Birth Defects Registry counts to supply control-condition trends for `DSP010`. The registry uses medical-record surveillance rather than the live-birth certificate anomaly item as its main diagnostic source. That reduces one source of circularity, but does not make a Texas trend an exact national trend.

## Method and original estimates

The analysis fitted quasi-Poisson log-linear trends with a birth-denominator offset and dispersion-scaled errors. The primary window was 2010 to 2022; a 2016 to 2022 sensitivity used fewer annual points. Denominators reconstructed from rounded published rates were checked against Texas resident births. Hypospadias used male births in this source analysis.

| Condition | Slope, log/yr | SE | Dispersion | Largest level shift | Pinned |
| --- | ---: | ---: | ---: | --- | :---: |
| `ca_hypo` hypospadias | `+0.00524` | `0.00453` | `5.01` | **`-16.3%` at 2019, z = `-6.0`** | **no** |
| `ca_clpal` cleft palate alone | `-0.00118` | `0.00657` | `1.78` | `+12.0%` at 2017, z = `+1.2` | yes |
| `ca_cleft` cleft lip ± palate | `-0.00281` | `0.00495` | `1.81` | `+10.0%` at 2018, z = `+1.4` | yes |
| `ca_limb` limb reduction | `-0.00648` | `0.00537` | `1.24` | `+17.2%` at 2019, z = `+2.4` | yes |
| `ca_gast` gastroschisis | `-0.03798` | `0.00533` | `0.73` | `-12.5%` at 2016, z = `-1.7` | yes |
| `reference_ds` Down syndrome | `+0.00130` | `0.00392` | `1.56` | `+13.1%` at 2018, z = `+2.7` | never |

| Fit | Slope, log/yr | Dispersion |
| --- | ---: | ---: |
| 2010-2018 | `+0.02347 ± 0.00426` | `1.47`, a line fits |
| 2019-2022 | `+0.01611 ± 0.01234` | `0.65`, a line fits |
| 2010-2022 | `+0.00524 ± 0.00453` | **`5.01`, a line does not fit** |

| Condition | National, log/yr | Texas, log/yr | |
| --- | ---: | ---: | :--- |
| `ca_clpal` | `+0.01062` | `+0.00231` | agree |
| `ca_cleft` | `-0.00525` | `-0.00298` | agree |
| `ca_limb` | `-0.01035` | `-0.00929` | agree |
| `ca_gast` | `-0.03703` | `-0.04514` | agree |

The level-shift scan and dispersion threshold were diagnostic rules chosen for this analysis. They are not model-free tests or universal criteria for a valid trend. A scan over multiple candidate breakpoints also needs care when interpreting a selected shift's nominal standard error.

The whole-window hypospadias series did not fit a simple linear trend well. It retained a zero trend in the panel. That default is a scenario, not a measured flat national prevalence trend or an established conservative choice.

## Effect of the supplied trends

| Condition | Certificate change | Pin removes | Residual |
| --- | ---: | ---: | ---: |
| `ca_hypo` | `-16.78%` | `+0.00%` | `-16.78%` |
| `ca_clpal` | `-8.02%` | `-0.71%` | `-7.32%` |
| `ca_cleft` | `-2.25%` | `-1.69%` | `-0.56%` |
| `ca_limb` | `-13.79%` | `-3.89%` | `-9.91%` |

| | Unpinned | Pinned | Pinned, γ = 0 |
| --- | ---: | ---: | ---: |
| 2016-2024 total | `45,828` | `45,653` | `45,666` |
| 89% ETI | `43,887`-`48,097` | `43,799`-`47,782` | `43,879`-`47,665` |
| Width | `9.19%` | `8.72%` | `8.29%` |
| Item recording factor | `-7.63%` | `-6.38%` | `-6.34%` |
| its 89% ETI | `-15.71%` to `+1.31%` | `-14.39%` to `+2.67%` | `-13.07%` to `+1.48%` |
| its width | `17.02` pp | `17.06` pp | `14.55` pp |
| `s` 2024 vs 2016 | `0.9516` | `0.9595` | `0.9593` |
| Prevalence 2024 vs 2018 | `-2.37%` | `-3.06%` | `-3.01%` |
| `recording_s` | `0.3370` | `0.3369` | `0.3368` |
| Loading | `0.935 ± 0.454` | `0.936 ± 0.453` | `0.941 ± 0.456` |
| Shared trend γ | `+0.00009 ± 0.00397` | `+0.00019 ± 0.00387` | fixed at `0` |
| `panel_condition_trend_scale` | `0.01431 ± 0.00662` | `0.01471 ± 0.00646` | `0.01469 ± 0.00647` |

The model treats these trends as fixed offsets. Its posterior intervals omit the trend standard errors and uncertainty in transport from Texas to the national study period. The shared prevalence-trend term adds another restriction; setting it to zero makes a stronger assumption.

The registry includes pregnancy outcomes beyond live births and diagnoses after birth. Its rates use a live-birth denominator. This is not exactly the certificate estimand, even for controls selected for low fetal loss or termination. Trends may also reflect changes in ascertainment or diagnosis timing.

Limb reduction sums upper- and lower-limb categories. Cases with both can be counted twice. A stable overlap share would preserve a trend, but that stability is another assumption.

Gastroschisis declined in the Texas series. This supports testing a nonzero trend rather than assuming a flat control. It does not establish how much of the national certificate decline is recording.

See [the panel note](20260804-dsp010-anomaly-panel-recording-factor.md) for the implemented model and its remaining restrictions. New fits must use the September numerical and validation changes.

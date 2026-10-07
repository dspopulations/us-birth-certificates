> [!NOTE]
> AI-assisted revision by Codex (GPT-6), 7 October 2026.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

# Local validation studies and conditional national rescaling

**Original date:** 4 August 2026. Historical research note, revised for clarity in October 2026.

This compared the settings of two validation studies with national recorded prevalence. It did not refit a model. It corrects earlier notes that attributed a Down syndrome recording sensitivity near 0.40 to Boulet.

## What was measured locally

| | Boulet 2011 | Salemi 2017 |
|---|---|---|
| setting | metro Atlanta, MACDP | Florida, FBDR + enhanced surveillance |
| births | 1995–2005 | 2007–2011 |
| certificate | 1989 revision | 2003 revision |
| DS sensitivity | `18.1%` (113/625) | `24.6%` (364/1478), 95% CI 22.4–26.8 |
| DS PPV | `97.4%` (113/116) | `87.3%` (84.1–90.5) |

[Boulet and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC3056031/) report 18.1% Down syndrome sensitivity in metropolitan Atlanta, rather than the 23% pooled sensitivity across six defects. [Salemi and colleagues](https://onlinelibrary.wiley.com/doi/10.1111/ppe.12326) studied Florida and the revised certificate. The Down syndrome figures above and the confirmed/pending values below are retained from the August extraction; the public abstract gives pooled results and does not by itself verify those rows.

The August note extracted 7.0% sensitivity for confirmed entries and 17.7% for pending entries, with PPVs of 89.6% and 86.4%. These refer to certificate channels at filing, not independent certainty about every entry. The older Atlanta certificate had no equivalent confirmed/pending sub-field.

Neither study measures national recording in 2016 to 2024. Other validation populations also exist, including [the Tennessee Medicaid study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11506645/). Population, era, case definition and ascertainment must match before transporting any rate.

## National recorded-count comparison

| window | recorded | covered births | per 10,000 |
|---|---|---|---|
| 1995–2005 | 19,772 | 43,057,164 | `4.59` |
| 2007–2011, 2003-revision area | 6,984 | 14,465,860 | `4.83` |
| 2016–2024 | 17,809 | 33,527,704 | `5.31` |

| | study area | national | factor | `s` measured | `s` national |
|---|---|---|---|---|---|
| Boulet / metro Atlanta | `2.22` | `4.59` | `2.068` | `0.181` | **`0.374`** |
| Salemi / Florida | `3.72` | `4.83` | `1.297` | `0.246` | **`0.319`** |

| | study-area true prevalence | project surveillance, same years | difference |
|---|---|---|---|
| Boulet / MACDP | `11.97` per 10,000 | `11.68` | `+2.4%` |
| Salemi / FBDR | `13.20` per 10,000 | `12.59` | `+4.8%` |

The national figures use the repository's covered-birth denominator and confirmed-or-pending definition. The Florida comparison uses the revised-certificate reporting area. The August calculation supplied a Florida denominator of about 1.12 million births from a separate source; that denominator and source definition should be checked before reusing the calculation.

The proposed rescaling was

```text
s_national = s_study * recorded_prevalence_national / recorded_prevalence_study
```

This is exact only under further restrictions, such as equal true prevalence and negligible false positives. More generally, recorded prevalence is `p_true * s + (1 - p_true) * f`. Differences in `p_true` or `f` change the rescaling.

The registry prevalences in the last table were within about 5% of the project's comparator. That is a consistency check, not proof of equality, compatible uncertainty, or nationally transferable recording. The rescaled values near 0.37 and 0.32 are conditional calculations, not newly measured national sensitivities.

The saved 2016 to 2024 state aggregate ranged from 1.44 to 13.43 recorded cases per 10,000 births among unsuppressed states. Florida was near the low end. Those later state figures cannot establish how much of an earlier study-area difference came from recording, and state recorded prevalence is not recording sensitivity.

The original claim that this calculation validated the model's recording anchor has been removed. Use the measured local rates as external context and test transport assumptions explicitly. See the [WONDER audit](20260803-cdc-wonder-state-level-feasibility.md) and [current model guide](../docs/models/README.md).

> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# Reference review and classifier extension through 2025

Reviewed on 6 November 2026.

The four current classifiers can include 2025 without filling the missing population-model references. They use birth-record features and the recorded Down syndrome label. They do not use surveillance prevalence, prenatal-reduction rates or recording weights. Their features, parameters and label rules are unchanged. Their fitting period extends from 2016–2024 to 2016–2025.

## Reference decisions

No supported annual 2025 values were found in the reviewed sources. The five imported lookup tables remain unchanged. Their missing 2025 estimates remain NULL. The [review record](../data/reference-review-2025.json) gives source checksums, coverage checks and the decision for each table.

| Imported reference | Stored years | Decision for 2025 |
| --- | --- | --- |
| Surveillance prevalence | 1989–2024 | Do not extend the existing carry-forward assumption. |
| Maternal-age prevalence | 1989–2018 | No supported annual replacement found. |
| Prenatal reduction | 1989–2024 | Do not extend the extrapolated tail without resolving its derivation. |
| Recording weights | 1989–2024 | No independent 2025 estimate of the share of cases recorded was found. |
| Ethnicity prevalence | 2000–2018 | No supported annual replacement found. |

The [CDC/NBDPN April 2026 brief](https://www.cdc.gov/birth-defects/media/pdfs/2026/04/Data-Brief-Birth-Defects-National-Estimates-2018-2022.pdf) covers 2018–2022. Its 11 included programmes cover 4,231,701 live births and ascertain all pregnancy outcomes. The brief defines prevalence using pooled reported cases divided by live births. It does not establish a live-birth-only numerator. These estimates cannot simply replace this project's live-birth probabilities.

The [new reference CSV](../data/us-births-nbdpn-trisomy21-2018-2022.csv) records the pooled Trisomy 21 estimate and five annual values from pages 13–14. Values and footnotes were checked against rendered pages. The pooled estimate is 16.1 per 10,000, with a 95% interval of 15.8–16.5. Annual values are 16.4, 16.5, 16.1, 16.3 and 15.3 for 2018–2022. This file is a separate source extraction. It is not loaded into the five annual lookup tables or extrapolated to 2025.

The [US population fact sheet updated on 17 December 2025](https://docs.downsyndromepopulation.org/factsheets/down-syndrome-population-usa-factsheet.pdf) still reports US estimates through mid-year 2018. Its latest surveillance window is 2016–2020. Its publication date does not make these 2025 estimates. The local de Graaf workbook is also unchanged. Its existing audit separates observed surveillance windows from projected recording fractions.

The 2025 database coverage check found no missing-key substitutions or derivation mismatches. Each unavailable reference-derived column remains NULL for all 3,619,990 records. Expected-count analyses must still pass the existing completeness checks before reporting population estimates.

## Classifier scope

The refit covers `usbc10_m1`, `usbc11_m1`, `usbc10_m1_cn` and `usbc11_m1_cn`. Each model retains its feature list, tuned parameters, target rule, seed 47, stratified 80/20 split and 200-round early-stopping window. No new tuning was run. The other historical variants keep their original periods.

The 2025 occurrence file contains 834 confirmed, 963 pending and 11,763 unknown recorded statuses. Confirmed-only models exclude pending records. The existing treatment of unknown status is retained. Monthly selection flags keep the existing quota and do not identify verified missed diagnoses.

Independent cohort checks count the raw recorded-status codes before using the derived label. They use `ca_down` where present and `ca_downs` otherwise. The imported 2016–2017 records use `ca_downs`, so a check of `ca_down` alone would omit those years' labels. The existing model loader already handles both fields.

All fits and writeback checks used a separate database. Before publication, checks compared all original source columns and reference tables with the validated import, then recomputed scores from the saved models. Publication preserved a full backup of the previous database and its predictions.

## Completed refit

The four refits were published to `data/us_births.db` on 6 November 2026 at 20:34 UTC. Every score was regenerated from the newly fitted models, including scores for 2016–2024.

| Model | Scores, 2016–2025 | Selected, 2016–2025 | Scores, 2025 | Selected, 2025 | Saved rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `usbc10_m1` | 37,209,272 | 29,441 | 3,619,990 | 2,699 | 1,128 |
| `usbc11_m1` | 37,209,272 | 29,441 | 3,619,990 | 2,699 | 868 |
| `usbc10_m1_cn` | 37,198,484 | 13,256 | 3,619,027 | 1,254 | 253 |
| `usbc11_m1_cn` | 37,198,484 | 13,256 | 3,619,027 | 1,254 | 90 |

All 146,485,338 original records remain present. Two aggregate row checksums matched the accepted import for each of its 37 years across all 349 original columns. Their types and all five reference tables are unchanged. The only added columns are the eight intended score and selection columns. Each model passed checks for score coverage, finite probabilities, eligible selections, monthly quotas and score ranking.

Scores recomputed from each saved model matched the stored scores exactly for 20,973 records per main model and 10,185 per confirmed-only model. These checks include all recorded cases and a fixed sample of other records across 2016–2025. They establish agreement between the saved models and database scores. They do not establish diagnostic accuracy or verify missed diagnoses.

The local run record is `output/classifier_refit_2025/20261106T184905Z/README.md`. It records settings, model metrics, validation results and the saved operational scripts. The previous database and its 2016–2024 predictions are preserved at `data/backups/before-2025-classifier-refit-20261106T203427Z/us_births.db`. Its SHA-256 matches the previous live database. The earlier fitted models and the original SAS backup were also rechecked and remain unchanged.

The published database SHA-256 is `76c155a9c5ace6c7fcce4cb62bafa847e434d4f36aca1546432edcae353b8e01`. The accepted import candidate remains unchanged and contains no classifier scores.

Ruff and spellcheck passed. All 502 default tests passed across the initial run and a cache-corrected rerun. Twenty tests needed that rerun because the sandbox prevented writes to the default PyTensor cache. Twelve slow tests were excluded by the default test selection.

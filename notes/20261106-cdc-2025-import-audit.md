> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# CDC 2025 layout, references and import validation

The 2025 import passed validation on 6 November 2026. Its separate database is `data/staging/natality/2025/us_births.db`. It contains 3,619,990 US occurrence records. The live `data/us_births.db` remains unchanged. Its SHA-256 hash before the import and after validation was `9a40368c72ec54a87687d622914d326797d2daf487127d1f025427e170b360ec`.

The import is accepted as a reading of the reviewed source. The project does not yet have the year-specific reference inputs needed for several 2025 modelled estimates. Those values remain NULL. No existing fit, prediction, year range or live database was replaced.

Use the [import guide](../docs/cdc-text-import.md#run-and-validate-2025) for commands. The [2024 audit](20261106-cdc-2024-import-audit.md) documents the preceding comparison with the saved SAS-derived database.

## Source and layout review

The [CDC portal](https://www.cdc.gov/nchs/data_access/VitalStatsOnline.htm) links the [US archive](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/natality/Nat2025us.zip) and [2025 guide](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2025.pdf). The local audit pinned their hashes in [sources.json](../config/natality/sources.json).

The archive contains `Nat2025PublicUS.c20260624.r20260922.txt`. It uses Deflate64 compression. The full stream contains 4,876,126,530 bytes and has CRC 4277420345. The records have the same physical framing as the reviewed 2024 archive: 1,330 documented positions, 15 blank padding bytes and two line-ending bytes. Every record's framing and padding passed the reader's checks. No expanded text or intermediate Parquet file was written.

The review compared all layout text on guide pages 11-40, then checked positions, widths, names and page references for every imported field. The existing 149 present fields retain their positions and domains, apart from the year value. Seven reporting flags replace filler:

| Source flag | Position | Meaning |
| --- | ---: | --- |
| `F_MBCntry` | 127 | Mother's birth country and nativity |
| `F_OTno` | 219 | Prior other terminations |
| `F_TBO` | 220 | Total birth order |
| `F_ILOO` | 221 | Interval since last other pregnancy outcome |
| `F_ILP` | 222 | Interval since last pregnancy |
| `F_ATTEND` | 439 | Attendant at birth |
| `F_LMPyy` | 512 | Year of last normal menses |

The [2025 specification](../config/natality/layouts/2025.json) contains 156 present fields, including 57 reporting flags, and 34 explicitly absent historical columns. It updates the field-to-flag links and removes links that the new guide no longer gives. Flags remain separate from source values; they do not erase values or change the study's cohort.

Two guide inconsistencies remain explicit. `ILP_R11` still repeats category codes in its printed list. The importer preserves its numeric codes without adding disputed labels. Page 34 also refers the last-menses month to flag position 513, while page 36 labels positions 513-516 as filler. The importer does not invent that flag. The last-menses month is outside the current import columns.

The quality notes on pages 72-75 describe reporting limits that differ by item and area. Examples relevant to these imported fields include the infertility-treatment flags for Georgia and North Carolina, nativity and WIC reporting for New York State, and attendant and spina bifida reporting for Virginia. These limits affect interpretation. Passing numerical checks does not establish diagnostic accuracy.

## One source-format exception

The first full attempt stopped at an out-of-domain token and left only an incomplete staging table. A separate full-file domain scan found one record with `1 `, a digit followed by a blank, in both `MRACE31` and `MRACE15`. Both fields normally encode that value as `01`. The related race recodes agree with numeric code 1. No other unexpected tokens were found among the 156 selected source fields.

The specification now permits that exact formatting exception only for those two fields in this pinned release. It requires exactly one occurrence in each field. The parser retains the original token in its frequency report and stores numeric code 1. It does not trim all fields or accept arbitrary space-padded numbers. Tests reject a different token, an unexpected count, a change of numeric meaning and use of the exception in 2024.

The restarted import passed these checks. Its report records both affected fields and their counts. The guide's separate statement of 225 imputed pluralities also agrees with the full source frequency. That statement concerns all occurrence records; the resident subset contains 124.

## Validation results

The live database ends at 2024, so no complete-row SAS comparison was available for 2025. The [40 published controls](../config/natality/controls/2025.json) were transcribed before importing. All matched exactly:

- 3,619,990 occurrence records, 3,612,012 resident births and 7,978 foreign-resident births.
- Race and non-Hispanic race counts from Table 4.
- Hispanic birth counts by maternal age group from Table 5.
- Congenital-anomaly counts and selected denominators from Table 32, including 1,789 births recorded with confirmed or pending Down syndrome among US residents.

The Table 32 checks use its reporting-area exclusions. They do not impose those exclusions on the stored data. Table 32 is on PDF page 156, although the contents list gives page 155. Suppressed cells and rounded rates were not treated as exact counts.

All 190 stored source columns matched the recorded source frequencies after the explicit conversions. Source-record numbers were unique and complete. Study derivations matched their checks, including the existing age-risk formula. Every reference join retained missing estimates as specified. The complete occurrence file has 834 confirmed and 963 pending Down syndrome records. These are recorded statuses, not independent diagnoses.

Matching frequencies and published counts provides less evidence about complete rows than the 2024 comparison. The validation report states that limitation. It also records the layout, source, guide, controls, code and reference hashes.

## Reference-data coverage

The five imported references contain no 2025 rows. The coverage report checks keys, missing values and the resulting derived values.

| Reference | Available years | Result for 2025 |
| --- | --- | --- |
| Surveillance prevalence | 1989-2024 | `p_ds_lb_wt` remains NULL |
| Maternal-age prevalence | 1989-2018 | `p_ds_lb_wt_mage` remains NULL |
| Prenatal reduction | 1989-2024 | `p_ds_lb_nt_reduc` remains NULL |
| Recording weights | 1989-2024 | `ds_case_weight` remains NULL, including non-cases |
| Ethnicity prevalence | 2000-2018 | No matching reference rows; probability placeholders remain unimplemented |

Each of the four derived columns above is NULL for all 3,619,990 records. Three existing probability placeholders also remain NULL. The age-based `p_ds_lb_nt` calculation is populated, but it remains a model input based on an external formula.

The descriptive report now checks its expected-count inputs before writing figures. It reports affected years and birth counts when values are missing. The current live database passes this check. The core reduction model already rejects missing years in its reduction-prior CSV. Existing fitting defaults, recording anchors and historical reports retain their previous year ranges. Extending those analyses requires a separate review of their external inputs and assumptions; this import does not copy 2024 estimates forward.

## Execution and checks

The run used batches of 50,000 records, two DuckDB threads and a 2 GB DuckDB memory limit on macOS arm64. Import through checkpoint took 70.0 seconds, including 0.6 seconds for derivations. Validation took 7.4 seconds. The candidate occupied about 227 MB. The sampled import memory peak was 2.38 GB; child-process coverage was incomplete, so this is not a strict upper bound. The comparison checks used about 2.02 GB. Download and the earlier diagnostic scan are outside these timings.

The ignored local reports beside the candidate are `us_births.import.json`, `us_births.frequencies.json`, `us_births.validation.json` and `us_births.coverage.json`. The baseline inventory and full-file domain audit are retained there too. The validation report has `passed: true`; reference readiness is separately false.

All 36 importer tests passed. The full default suite passed with 481 tests and 12 slow tests excluded. Ruff and spellcheck also passed. Other operating systems were not exercised in this session. Historical layout review, downstream integration and eventual replacement of the full database remain separate work.

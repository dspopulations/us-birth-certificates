> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# CDC source audit and 2024 import pilot

The full 2024 CDC import passed its source checks and comparison with the saved SAS-derived database on 6 November 2026. The live `data/us_births.db` was opened read-only. Its SHA-256 hash matched before and after the work.

The new candidate is `data/staging/natality/2024/us_births.db`. It contains one year. It has not replaced the live database. Use the [pilot guide](../docs/cdc-text-import.md) for commands and the [migration plan](../plans/2026-11-06-cdc-text-import-migration.md) for subsequent years.

## Source and layout findings

The [source catalogue](../config/natality/sources.json) records the CDC portal's US archive and guide links for all 37 years from 1989 to 2025. It includes the 2009 and 2010 addenda. Only 2024 has a reviewed layout. The other years cannot yet be imported through the new command.

The [2024 guide](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2024.pdf), page 9, gives 3,638,436 US occurrence records, including 3,628,934 births to US residents and 9,502 to foreign residents. The imported counts match each control exactly.

The downloaded archive contains one member, `Nat2024PublicUS.c20250512.r20250708.txt`. Its expanded size is 4,900,973,292 bytes. The member uses Deflate64, ZIP compression method 9. The pilot used the system `unzip` executable because Python's standard ZIP reader does not support that method. It checked the decompressor exit status, the full byte count and CRC value 2654719869.

The physical records differ from the guide's stated length. The guide specifies 1,330 positions. Each actual record contains those positions, another 15 blank bytes, and a carriage return followed by a line feed. The reader checked the extra padding and both separator bytes on every record. The [2024 layout](../config/natality/layouts/2024.json) records this exception for the pinned archive. It does not relax record-length checks for other years.

The layout accounts for all 133 existing import columns. It maps 99 to source fields and marks 34 historical columns as absent. It also imports 50 reporting flags. The mapping includes the explicit `IMP_PLUR` to `imp_plural` alias. All fields use checked positions and code domains. BMI uses an exact decimal with one place.

Two guide issues need care in later interpretation. The printed `ILP_R11` list repeats codes 00 and 01, then ends at 06. The 2015 and 2023 guides contain the same inconsistency. The 2024 source contains codes 00-08, 88 and 99, plus one blank. The importer preserves these codes without assigning category labels. The complete-row comparison confirms agreement with the existing extract, but does not resolve the printed labels. The layout records that limit.

Reporting flags also cannot be used as a blanket instruction to erase a value. Some records with `F_CA_DOWNS=0` have a populated Down syndrome status. The pilot retains both fields. It preserves the current study's recorded-status definition. Any new restriction based on reporting flags needs a separate cohort decision.

## Differences from the existing database

The following counts describe source values retained by the new route. They do not represent newly identified diagnoses.

| Field | Affected records | Explanation |
| --- | ---: | --- |
| `dob_tt` | 3,298,951 | The old unsigned eight-bit conversion could not hold codes above 255. This includes the not-stated code as well as time values. |
| `no_risks` | 1,307,982 | The old range check converted code 0 to NULL. |
| `no_congen` | 9,637 | The old range check converted code 9 to NULL. |
| `bmi` | 2,913,005 | The old route rounded these source decimals through a 16-bit floating-point representation. |
| `imp_plural` | 1,694 | The old importer omitted the `IMP_PLUR` alias, so it lost the source's imputation flag. |

For comparison only, the validation code reproduces the old missing values and rounds the old BMI values back to the source's one decimal place. It does not apply those losses to the new database.

After that documented normalisation, there were zero unmatched rows in either direction for each of the 12 months. The comparison covered all 133 imported columns and 15 derived study columns. It retained duplicate multiplicity. Age, race/origin and recorded-status analysis cells also matched exactly. There were no additional unexplained column-distribution differences.

The comparison used the saved SAS-derived database. The local annual SAS and Parquet files were absent, so the pilot did not perform a fresh SAS import. Its independent checks were the CDC layout, code domains, archive integrity and occurrence/residence control counts.

## Resources and reproducibility

These timings describe this machine and run, not a promised limit for other years or platforms. The run used 50,000 records per batch, two DuckDB threads and a 2 GB DuckDB memory limit.

| Measurement | Result |
| --- | ---: |
| Read, parse, insert and check the full source | 79.3 seconds |
| Calculate shared study derivations | 1.3 seconds |
| Import through final checkpoint | 81.1 seconds |
| Compare with the saved database | 33.3 seconds |
| Sampled import resident memory | 2.53 GB |
| Sampled comparison resident memory | 2.04 GB |
| Candidate database size | About 253 MB |

The import memory report marks child-process coverage as incomplete. Its sampled peak is therefore not a strict upper bound for the whole process tree. The comparison report includes child processes. No expanded text or intermediate Parquet file was written. Download time is outside these timings because the archive was already cached.

The ignored local reports are `us_births.import.json`, `us_births.frequencies.json` and `us_births.validation.json` beside the candidate database. They record source, guide, layout, code and reference hashes, package versions, source-code frequencies and comparison results. `baseline-inventory.json` records the original schema and annual counts.

The live database's SHA-256 hash remained `9a40368c72ec54a87687d622914d326797d2daf487127d1f025427e170b360ec`. The reviewed archive hash is recorded in the source catalogue. These hashes identify the files used; they are not publisher authentication checks.

## Checks and remaining scope

All 22 new importer tests passed. They cover explicit positions, source codes, short reads, truncated and malformed records, archive completeness, failed imports, safe retries, path protection and complete-row comparisons. One test preserves all column totals while changing which values occur together, then checks that the row comparison detects the change.

The repository's default suite passed with 467 tests and 12 slow tests excluded. The first sandboxed run could not write PyTensor's compile cache. The rerun with cache access passed. The shared derivation move preserved 14 function bodies unchanged. The other existing functions gained an explicit reference directory or reference-key checks. The live data comparison also checked the resulting calculations for 2024.

The full pilot ran on macOS arm64. Synthetic tests have not yet been run on the other configured platforms in this session. The remaining migration includes the 2025 layout and external-input coverage, historical layouts, optional exports and eventual controlled replacement of the full database. Those steps remain separate from acceptance of this one-year pilot.

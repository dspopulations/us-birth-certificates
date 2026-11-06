> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# CDC historical layout and import audit

This audit extends the [2024 pilot](20261106-cdc-2024-import-audit.md) and [2025 review](20261106-cdc-2025-import-audit.md) to the remaining 35 years. The source review was performed on 6 November 2026. All work uses separate candidate databases. The saved `data/us_births.db` remains the comparison baseline.

## Sources and layout scope

The [source catalogue](../config/natality/sources.json) pins all 37 archive hashes and the guide hashes. The links come from the US column of the [CDC portal](https://www.cdc.gov/nchs/data_access/VitalStatsOnline.htm). The catalogue records the page used for each annual occurrence and residence control. [Expanded annual layouts](../config/natality/layouts/) give every project import column an explicit mapping or absence reason. Reporting flags are retained separately.

The review checked annual position tables and code domains, then scanned every archive member. Guide extraction was a drafting aid. Visual checks resolved scanned tables and disputed entries, including the 1993 counts, the 1997 Down syndrome position, the end of the 2004 layout and the 2015 race table.

The import retains historical feature coverage. It does not populate old sex, birthweight, Apgar or gestation fields merely because native CDC fields resemble later project variables. It also retains deliberate omissions in the former extracts, including eight native clinical fields in 2005, education fields in 2009–2010, combined paternal age in 2012–2013 and MRACE6 in 2014–2017. The annual absence reasons distinguish those decisions from a field actually absent from the guide. Adding such fields needs a separate definition and cohort review.

The old 1989–2002 layouts use `DATAYEAR` and the 2003 revision uses `DOB_YY`. Maternal race codes change in 1992; the maternal-age range changes in 1997. Maternal age in 2003 continues to use `MAGER41 + 13`. The 2003 `UMEDUC` source maps to the project's `dmeduc` column. Down syndrome field names and their project aliases are recorded by year, rather than inferred from comments about SAS availability.

The [2009 addendum](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2009_Addendum.pdf) and [2010 addendum](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2010_Addendum.pdf) release additional fields and refer to the [2011 guide](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2011.pdf) for their descriptions. Both layouts therefore pin all three documents. Addendum fields and flags are checked at the referenced positions.

## Physical records and guide discrepancies

All records end with a carriage return and a line feed. The selected members have these lengths before those two separator bytes.

| Years | Bytes per physical record | Finding |
| --- | ---: | --- |
| 1989–2002 | 350 | Old certificate layout |
| 2003 | 1,350 | First year of the revised certificate |
| 2004–2005 | 1,500 | The complete 2004 layout ends at 1,500 despite an earlier 1,350-byte statement |
| 2006–2013 | 775 | Shorter revised-certificate layout |
| 2014–2015 | 1,345 | Includes the old-certificate Down syndrome field at position 1,345 |
| 2016–2025 | 1,345 | Guide length 1,330 plus 15 checked blank bytes |

The 2014 file-characteristics section still says 775, but the complete layout extends to 1,345. The 2005 and 2013 year-field descriptions retain the preceding year's value. The archive records all contain the correct year. Several guides print MRACE6 as length two even though its position and adjacent field allow one byte; the 2023 guide corrects that width. These discrepancies are recorded in the annual specifications.

The 2015 ZIP uses PPMd, compression method 98. Its member is `Nat2015PublicUS.c20160517.r20160907.txt`, with 5,372,823,351 expanded bytes. The import used `bsdtar`, then checked the complete member size and CRC. Deflate64 files use an available supported decompressor. Ordinary Deflate files use Python's ZIP reader.

The 2000 release has 4,058,814 resident births in its guide and archive. This audit uses that release-specific count, rather than substituting a later published total.

## Counted source exceptions

These values are present in the pinned files but do not follow the printed guide or the usual byte format. Their counts are required by the layout. They are not claims about the true age or race of a parent.

| Year | Source value or position | Count | Treatment |
| --- | --- | ---: | --- |
| 2004 | Carriage-return byte at position 1,422 | 184 | Permit only this byte inside the unselected `FRACE1E` field. No selected field or separator changes. |
| 2006 | `UFAGECOMB=09` | 2 | Preserve numeric 9 despite the guide's lower bound of 10. |
| 2009 | `UFAGECOMB=09` | 1 | Same treatment, with a separate annual count. |
| 2015 | `MRACE31=99` and `MRACE15=99` | 2,923 each | Preserve both unlisted raw codes. Use the available bridged race for the harmonised variable. |
| 2017 | `FAGECOMB` contains `1` followed by a space | 1 | Convert to numeric 1 without inventing a replacement age. |
| 2025 | `MRACE31` and `MRACE15` contain `1` followed by a space | 1 each | Remove the formatting difference while preserving numeric 1, as in the pilot. |

Every affected 2015 record has a valid `MBRACE` and an imputation flag of 1 or 2. The old type constraint lost the unlisted 99 to NULL, allowing `MBRACE` to supply the harmonised race. The new derivation permits that fallback for 99 while retaining both raw codes. The full 2015 comparison confirms that derived race and analysis cell counts remain unchanged. The [2015 guide](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2015.pdf), PDF page 9, does not define 99 for these two fields.

Blank race fields in 2014 are documented non-reporting values and remain NULL. Code 9 in the 2003–2013 `MRACEHISP` scheme is documented unknown origin and is retained. Neither is treated as a source exception.

## Comparison rules

Each annual validator checks stored source frequencies, unique source-record numbers, deterministic IDs, reference coverage and complete-row multiplicity. The historical row comparison includes all 133 project import columns and the derived study fields. It groups work by maternal age modulo 12, with an extra group for missing age. Thus years with no imported birth month still receive a complete comparison.

The comparison retains the five documented [2024 adjustments](20261106-cdc-2024-import-audit.md). It also accounts for old marital-status strings such as `1.0`, the old loss of `MRACEHISP=9`, and the old loss of the 2015 race codes 99. These normalisations affect comparison expressions only. The candidate retains the CDC source values.

The baseline has 59 rows with no year and partially populated source fields. Its valid-year totals are short by five records in 2005 and 27 each in 2018 and 2019. The allowed differences are tied to the exact baseline SHA-256 `9a40368c72ec54a87687d622914d326797d2daf487127d1f025427e170b360ec`. Validation requires every valid-year baseline row to appear in the candidate after the documented normalisations, with exactly the stated additional records. No baseline row may disappear, and no analysis cell may lose records. Missing-year baseline rows are not copied into the candidate. The total for 1989–2024 remains 142,865,348, equal to the baseline total including its 59 missing-year rows. This check does not establish a record-by-record link between those malformed rows and the additional annual records.

The annual published totals count US occurrences. Resident and foreign-resident totals are checked separately. These are births recorded in the source, not new diagnoses identified by this project.

## Annual results

All 37 imports passed source and stored-field validation. All 36 overlapping years passed the complete-row comparison under the rules above. There are no unexplained row differences. The 2025 import passed all forty published controls. The [aggregate validation CSV](../data/cdc-natality-import-validation-1989-2025.csv) records per-year hashes, counts, timings and measured memory. Detailed local reports remain under `data/staging/natality/<year>/`.

| Year | US occurrences | US residents | Foreign residents | Additional records against baseline |
| --- | ---: | ---: | ---: | ---: |
| 1989 | 4,045,693 | 4,040,958 | 4,735 | 0 |
| 1990 | 4,162,917 | 4,158,212 | 4,705 | 0 |
| 1991 | 4,115,342 | 4,110,907 | 4,435 | 0 |
| 1992 | 4,069,428 | 4,065,014 | 4,414 | 0 |
| 1993 | 4,004,523 | 4,000,240 | 4,283 | 0 |
| 1994 | 3,956,925 | 3,952,767 | 4,158 | 0 |
| 1995 | 3,903,012 | 3,899,589 | 3,423 | 0 |
| 1996 | 3,894,874 | 3,891,494 | 3,380 | 0 |
| 1997 | 3,884,329 | 3,880,894 | 3,435 | 0 |
| 1998 | 3,945,192 | 3,941,553 | 3,639 | 0 |
| 1999 | 3,963,465 | 3,959,417 | 4,048 | 0 |
| 2000 | 4,063,823 | 4,058,814 | 5,009 | 0 |
| 2001 | 4,031,531 | 4,025,933 | 5,598 | 0 |
| 2002 | 4,027,376 | 4,021,726 | 5,650 | 0 |
| 2003 | 4,096,092 | 4,089,950 | 6,142 | 0 |
| 2004 | 4,118,907 | 4,112,052 | 6,855 | 0 |
| 2005 | 4,145,619 | 4,138,349 | 7,270 | 5 |
| 2006 | 4,273,225 | 4,265,555 | 7,670 | 0 |
| 2007 | 4,324,008 | 4,316,233 | 7,775 | 0 |
| 2008 | 4,255,156 | 4,247,694 | 7,462 | 0 |
| 2009 | 4,137,836 | 4,130,665 | 7,171 | 0 |
| 2010 | 4,007,105 | 3,999,386 | 7,719 | 0 |
| 2011 | 3,961,220 | 3,953,590 | 7,630 | 0 |
| 2012 | 3,960,796 | 3,952,841 | 7,955 | 0 |
| 2013 | 3,940,764 | 3,932,181 | 8,583 | 0 |
| 2014 | 3,998,175 | 3,988,076 | 10,099 | 0 |
| 2015 | 3,988,733 | 3,978,497 | 10,236 | 0 |
| 2016 | 3,956,112 | 3,945,875 | 10,237 | 0 |
| 2017 | 3,864,754 | 3,855,500 | 9,254 | 0 |
| 2018 | 3,801,534 | 3,791,712 | 9,822 | 27 |
| 2019 | 3,757,582 | 3,747,540 | 10,042 | 27 |
| 2020 | 3,619,826 | 3,613,647 | 6,179 | 0 |
| 2021 | 3,669,928 | 3,664,292 | 5,636 | 0 |
| 2022 | 3,676,029 | 3,667,758 | 8,271 | 0 |
| 2023 | 3,605,081 | 3,596,017 | 9,064 | 0 |
| 2024 | 3,638,436 | 3,628,934 | 9,502 | 0 |
| 2025 | 3,619,990 | 3,612,012 | 7,978 | No baseline year |

The combined total is 146,485,338 records. Import time from cached sources ranged from 30.7 to 171.0 seconds per year on this macOS arm64 machine. The largest measured import memory was 2.97 GiB. Imports used 50,000-record batches, two DuckDB threads and a 2 GB DuckDB memory limit. The historical batch ran two annual jobs at once. These figures exclude the earlier download and guide-review work.

Validation timing includes stored-field checks and the database comparison. Its memory monitor covers the comparison phase. The local reports identify whether child-process memory was included. These measurements describe this run, not resource guarantees.

## Combined candidate and downstream checks

The combined candidate is `data/staging/natality/us_births.db`. It contains 146,485,338 records and 349 columns, including the union of annual reporting flags. Its size is 7,392,473,088 bytes (6.88 GiB). Combination and its final checks took 403.2 seconds; measured peak resident memory was 4.23 GiB. This exceeds the 2 GB DuckDB limit because that setting is not a limit on total process memory.

The build reopened the completed database and matched every year's row count and two aggregate checksums over all stored columns to the annual inputs. Its SHA-256 is `216a4a4cc7f80678554a452cd67abfd8d0dbf2fbf2879970a8f9561159cc74b2`. The complete local result, including annual provenance, is `data/staging/natality/us_births.build.json`.

The full `selection.data.prepare_cells` query for 2016–2024 returned 33,498,320 eligible births, compared with 33,498,266 in the baseline. The increase is exactly the 54 additional year-labelled source records from 2018 and 2019, spread across 49 cells. No cell lost records. The recorded numerator remains 17,776, with no cell-level numerator changes. This query has its own exclusions for unknown status and missing clinical inputs; its denominator is not the occurrence count.

A separate database with 41,530 source records exercised the classifier queries across 2003–2025. The confirmed-plus-pending query returned 26,168 rows; the confirmed-only query returned 25,869. Both returned 58 columns and unique source-derived IDs. The prediction writer then stored known test scores for 25,869 rows in that sample database. Every score matched its intended ID. This checks data access and writeback, not the quality of a fitted classifier.

The optional Parquet export and SPSS readback also passed on 100 source records. The SPSS writer initially treated decimal BMI as text; it now converts BMI to double precision and retains missing values and source codes. A regression test checks the numeric SPSS type and values.

The consumer checks left both the combined candidate and live baseline hashes unchanged. Their local aggregate report is `data/staging/natality/consumer-checks.json`. The default test suite passed **502 tests**, with 12 slow posterior checks excluded. Ruff, formatting, spellcheck, locked environment setup and whitespace checks passed. No posterior fit was changed or run as part of this import migration.

The accepted candidate has not replaced the live database. Old prediction columns are not copied. IDs identify source-record positions within a pinned release, so predictions must be regenerated in an analysis copy. None of the five reference tables has 2025 rows; unsupported probabilities and case weights remain NULL. Numerical import validation does not settle those reference-data decisions or the study's diagnostic assumptions.

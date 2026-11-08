> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# CDC natality import migration

## Decision and scope

Use the CDC's annual US natality ZIP files as the source for 1989–2025. Cache the compressed archives, read the text members in batches and write typed columns directly into DuckDB. Parquet is an optional export. The [current commands](../docs/cdc-text-import.md) replace the former SAS and mandatory Parquet chain.

The [CDC portal](https://www.cdc.gov/nchs/data_access/VitalStatsOnline.htm) supplies US and territory files separately. This migration includes US occurrences, including births to foreign residents. It does not add territories or change the study's cohort definitions.

The migration built annual candidates and a separate combined candidate under `data/staging/natality/`. The accepted candidate was copied to the live analysis database on 6 November 2026. The saved SAS-derived baseline remains under `data/backups/before-cdc-20261106T164938Z/` for historical comparisons. The importer still writes only separate candidates.

## Implemented controls

The [source catalogue](../config/natality/sources.json) records exact URLs, hashes, member information and guide counts. Thirty-seven expanded [annual layouts](../config/natality/layouts/) record positions, widths, code domains, flags, guide pages and explicit decisions about absent project fields. The 2009 and 2010 layouts include the addenda and the 2011 guide those addenda cite.

The reader validates the complete member, including its checksum, record boundaries, padding, field tokens and embedded year. It preserves raw unknown codes and reporting flags. Exceptions for values outside the printed guide are tied to exact source tokens or byte positions and counted. The importer does not infer corrected ages or race labels for those values.

The database retains the project's existing historical feature coverage. Native fields omitted by the former extract remain explicitly outside the import where adding them would change that coverage. Shared derivations continue to distinguish recorded status from modelled estimates. Missing external inputs remain NULL.

Annual validation checks stored fields against source frequencies and compares complete rows with the baseline for 1989–2024. It retains duplicate multiplicity and accounts for documented conversion losses. The 2025 validator uses published controls and independent derivation checks because no baseline records exist for that year.

Combination requires a successful validation tied to each input's exact database hash. It commits one year with its metadata, resumes completed years after interruption and checks the combined rows after reopening. It replaces only its own candidate output after validation. Prediction columns from the old database are not copied.

Optional Parquet exports come from read-only DuckDB queries. `make_all.py` uses this path for its established subset. The SPSS exporter can consume explicitly requested Parquet exports. The SAS downloader and intermediate preparation scripts have been removed. Historical validation reads the saved DuckDB baseline, supplied explicitly with `--baseline`.

## Evidence

The [2024 pilot audit](../notes/20261106-cdc-2024-import-audit.md) records the initial full-source comparison and five conversion losses. The [2025 audit](../notes/20261106-cdc-2025-import-audit.md) records the layout review, forty published controls and reference-data gaps. The [historical audit](../notes/20261106-cdc-historical-import-audit.md) records the remaining layouts, source exceptions, annual comparisons and combined build.

These checks establish agreement with the pinned files, the stated controls and the documented study definitions. They do not establish diagnostic accuracy, validate prenatal-reduction assumptions or justify carrying reference estimates forward.

## Remaining decisions

Regenerate predictions and reports against the CDC record IDs in the analysis copy. Keep the accepted import candidate unchanged so its recorded file hash remains useful.

All five external lookup tables lack 2025 rows. Review their sources and intended extrapolations before extending modelled expected counts to 2025. Earlier reference gaps, including the age series after 2018 and blank ethnicity values for 2015 and 2017, remain explicit. The importer does not resolve these research choices.

A later CDC release needs a fresh hash and layout review, even when its filename and record length stay the same. Continue to verify yearly code meanings against the guide. Field availability in the old SAS extract is not evidence that a field is absent from the CDC source.

> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# CDC text import

The importer supports the 37 US natality files for 1989–2025. It streams each compressed text member into typed Arrow batches and writes them directly to DuckDB. It does not create expanded text files or require Parquet. Territories are outside this import.

The annual [source catalogue](../config/natality/sources.json) pins the archive, guide and addendum hashes. Each [layout](../config/natality/layouts/) records positions, widths, code domains, blank meanings, reporting flags and guide pages. Every project import column has an explicit presence or absence decision. Equal record lengths do not establish equal layouts.

The importer leaves the live analysis database at `data/us_births.db` unchanged. Import and combination refuse that path, `data/us_births_temp.db`, the selected comparison database, and symbolic or hard links to those files. There is no automatic replacement step.

## Build the candidate

Use Python 3.14 through the locked environment, from the repository root. `--baseline` is required. For historical validation, select the saved pre-CDC DuckDB database. The local backup is shown below; use its actual path if stored elsewhere. The importer reads no SAS files.

```bash
uv sync --locked
uv run python scripts/import_natality.py --years 1989-2025 \
  --baseline data/backups/before-cdc-20261106T164938Z/us_births.db
```

The command downloads each archive and its documentation into `data/raw/natality/<year>/`. Existing cache files must match the audited hashes. A changed release requires a new source and layout review. Downloads use temporary files, retries and hash checks.

Each year is imported and validated under `data/staging/natality/<year>/`. After every annual validation passes, the command combines the annual databases into `data/staging/natality/us_births.db`. A comma-separated selection such as `--years 1989,2003-2005,2025` also works.

The defaults are 50,000 records per batch, two DuckDB threads and a 2 GB DuckDB memory limit. Python, Arrow and decompression use memory in addition to the DuckDB limit. Reports give elapsed time and measured resident memory, with a flag stating whether child processes were included.

Most archives use compression supported by Python. Deflate64 archives need `7zz`, `7z`, or an `unzip` build with both `USE_DEFLATE64` and `ZIP64_SUPPORT`. The 2015 archive uses PPMd and needs `7zz`, `7z`, or `bsdtar`. The importer checks the external command's exit status, the expanded byte count and the ZIP CRC checksum. It never extracts an archive into the working directory.

## Outputs and validation

| Annual output | Contents |
| --- | --- |
| `us_births.db` | Source columns, reporting flags, study derivations, reference tables and build metadata |
| `us_births.import.json` | Source controls, hashes, package versions, timings and memory measurements |
| `us_births.frequencies.json` | Counts of raw field tokens, including blanks, before conversion |
| `us_births.validation.json` | Validation result tied to the exact candidate database hash |
| `us_births.coverage.json` | Additional reference-coverage report for 2025 |

These paths are ignored by Git. They contain local data and audit artefacts, not publication outputs.

Source checks cover every record's length, separator, padding, selected fields and embedded year. They check occurrence and residence counts against the guide. A small number of source exceptions have explicit positions or tokens and exact expected counts. A changed count fails the import. The [historical audit](../notes/20261106-cdc-historical-import-audit.md) explains them.

The validator checks each stored source column against its raw token frequencies, checks unique source-record numbers and deterministic IDs, and checks derived values against available reference tables. Fields deliberately absent from an annual layout must remain NULL.

For 1989–2024, it also compares every imported column's distribution and complete rows across imported and derived study columns. The comparison preserves duplicate counts and does not assume shared row order. Thirteen age partitions, including missing age, limit memory use. Analysis counts by age, race/origin and recorded status provide a further check.

Comparison expressions account for documented losses in the old route, while the new database retains the source values. These include birth times above 255, risk and anomaly codes, BMI precision, the plurality flag alias, historical race/origin code 9, 2015 race code 99 and marital-status strings ending in `.0`. The saved baseline also lacks five year-labelled records in 2005 and 27 each in 2018 and 2019. Those allowances apply only to its exact pinned hash. Every valid-year baseline row must still match, and only the specified additional source records are allowed.

For 2025, the saved baseline has no records. Validation instead checks forty [published controls](../config/natality/controls/2025.json), stored source fields, IDs, independent derivations and reference coverage. Each control retains its published population and reporting-flag exclusions. See the [2025 audit](../notes/20261106-cdc-2025-import-audit.md).

A `source_validated` build has passed source checks. Its separate validation report must also have `passed: true`. Numerical agreement does not establish diagnostic accuracy or validate the external assumptions used in prevalence estimates.

## Combination and retries

The combined build accepts only annual databases with matching successful validation reports, source layouts and reference hashes. It fills flags absent in particular years with NULL. It copies each year and its provenance in one transaction. Counts and two aggregate row checksums cover every copied column and are checked again after reopening the combined file.

The build uses `us_births.building.db` until these checks pass. It then replaces only the separate candidate output. `us_births.build.json` records the combined result and every annual input. An interrupted combination resumes already committed years without inserting them twice. A changed input requires `--restart`; an unmanaged database is refused even with that option.

```bash
# Select the saved comparison database for these commands.
natality_baseline=data/backups/before-cdc-20261106T164938Z/us_births.db

# Import and validate one year.
uv run python scripts/import_natality.py --year 2004 --baseline "$natality_baseline"

# Rebuild an incomplete or changed annual candidate from the cached archive.
uv run python scripts/import_natality.py --year 2004 --restart \
  --baseline "$natality_baseline"

# Repeat validation of all annual candidates, then combine them.
uv run python scripts/import_natality.py --years 1989-2025 --validate-only \
  --baseline "$natality_baseline"

# Combine annual candidates whose exact files already passed validation.
uv run python scripts/import_natality.py --years 1989-2025 --combine-only \
  --baseline "$natality_baseline"
```

`--staging-dir` changes the annual-output root. For a range, `--output` selects the combined database and `--cache` selects a root with annual subdirectories. For one year, both options refer directly to that year's output or cache. `--batch-size`, `--memory-limit` and `--threads` set resource limits.

An unchanged annual build can be reused. Changes to its source, layout, import code or references require a restart. A restart reads the archive from the beginning; it does not seek to a compressed byte offset. The previous accepted annual table remains until the replacement passes source checks and commits.

The numeric `id` is `year * 1000000000 + source_record`. It identifies a record within the exact source release, not a person across releases. Old prediction IDs cannot be used to transfer predictions. Regenerate predictions against the new build in a separate analysis copy. Changing a validated candidate invalidates its file hash.

## Optional exports and analysis

Parquet is now an explicit export. By default it contains the 133 project import columns. Add `--all-columns` for study derivations, source-record IDs and reporting flags. DuckDB writes the export without loading a whole year into pandas. The writer checks its row count before replacing the destination.

```bash
uv run python scripts/export_natality.py --years 2014-2024 --annual-dir data
uv run python scripts/export_natality.py --years 2014-2024 --output data/us_births_all.parquet
```

Existing files require `--overwrite`. `--database` selects another read-only source. `scripts/make_all.py` keeps its established 2014–2024 output name and now exports from the combined CDC candidate. `scripts/export_spss.py` can consume these optional annual and combined Parquet files. The SPSS reader converts decimal BMI values to numeric values so SPSS does not treat BMI as text. Its pandas/SPSS step still needs memory for a whole input file. Record-level exports remain subject to the NCHS data restrictions.

All five external reference tables lack 2025 rows. The importer leaves unsupported year-specific probabilities and case weights NULL. It does not carry 2024 values forward. The age-risk formula remains available, but it is not an observed prevalence estimate. Expected-count reports reject missing inputs. The current recorded-status classifiers include 2025; population estimates still require the missing external inputs. See the [workflow guide](modelling-workflow.md) for model year ranges.

The live analysis database was replaced with a copy of the accepted CDC candidate on 6 November 2026. Keep the saved pre-CDC database for historical validation and the accepted candidate for read-only checks. Pass the candidate path explicitly to analysis commands that accept a database path. The [migration record](../plans/2026-11-06-cdc-text-import-migration.md) records the completed import work and remaining reference-data research.

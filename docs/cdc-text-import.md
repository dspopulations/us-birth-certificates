> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# CDC text import pilot

The CDC import reads the annual ZIP into a separate DuckDB database. It does not write an expanded text file or intermediate Parquet files. The supported years are 2024 and 2025. The source catalogue lists 1989-2025, but the command rejects years whose layouts have not been reviewed.

The live database at `data/us_births.db` remains the analysis source. The pilot command has no database replacement step. It refuses that path, `data/us_births_temp.db`, the selected comparison database, and symbolic or hard links to those files.

## Run the 2024 pilot

Use Python 3.14 through the locked project environment, from the repository root:

```bash
uv sync --locked
uv run python scripts/import_natality.py --year 2024
```

The defaults are 50,000 records per batch, two DuckDB threads and a 2 GB DuckDB memory limit. Python, Arrow and the decompressor also use memory. The run measures resident memory, which is the memory the operating system holds for the process. Its report states whether it could also measure child processes.

Both reviewed archives use Deflate64 compression. Python's standard ZIP module cannot decode that method. Install 7-Zip with its `7zz` or `7z` command, or use an `unzip` build that reports both `USE_DEFLATE64` and `ZIP64_SUPPORT` when run with `unzip -v`. The importer detects these tools. It streams the selected member to Python and checks the decompressor's exit status, the byte count and the ZIP CRC checksum. It uses Python directly for supported ZIP methods.

The command downloads the archive and guide into `data/raw/natality/2024/`. A download must match the hash recorded during the source audit. An existing file with a different hash causes an error. It is not silently accepted or overwritten. New source releases need a new audit and recorded hash.

The default outputs are:

| Path under `data/staging/natality/2024/` | Contents |
| --- | --- |
| `us_births.db` | Imported columns, reporting flags, study derivations, reference tables and build metadata. |
| `us_births.import.json` | Source controls, file and code hashes, package versions, timing and memory measurements. |
| `us_births.frequencies.json` | Counts of each source code, including blanks, before conversion. |
| `us_births.validation.json` | Read-only comparison with the existing database, including differences and its unchanged-file check. |

These directories are ignored by Git. The database and detailed local reports are not publication outputs.

Use an explicit baseline or resource settings if needed:

```bash
uv run python scripts/import_natality.py --year 2024 \
  --baseline data/us_births.db \
  --output data/staging/natality/2024/us_births.db \
  --batch-size 50000 --memory-limit 2GB --threads 2
```

## Validation and retries

The reader checks record boundaries, padding, field domains, embedded years, source integrity and the guide's occurrence and residence counts. It holds incomplete rows in a staging table. Failed imports do not become accepted analysis tables.

The importer then calculates the existing study definitions. It uses the shared functions in [preparation.py](../src/dspopulations_us_birth_certificates/preparation.py), also used by the legacy preparation script. The pilot skips the legacy type conversions and compaction routine. Missing reference years remain missing. Duplicate reference keys cause an error.

For 2024, the comparison opens both databases read-only. It compares every imported column's distribution, then compares complete rows across all imported columns and the derived study variables. It preserves duplicate counts and does not depend on row order. It divides this work by month to limit memory use. It also compares age, race/origin and recorded-status analysis cells.

The comparison accounts for five documented differences from the legacy route. Birth times above 255, `NO_RISKS=0` and `NO_CONGEN=9` were converted to missing values. BMI passed through a 16-bit floating-point value. The old importer did not map `IMP_PLUR` to `imp_plural`. The new database retains these values. Only the comparison expressions reproduce the old losses or restore the source precision. Any unexplained difference fails the comparison.

The source check and the later validation have separate results. A database marked `source_validated` has passed the source checks. The validation report must also contain `passed: true` before the pilot can be accepted. Successful import does not change the live database.

An unchanged completed import can be reused. Source, layout, code or reference changes require an explicit restart. The pilot currently holds one year per output database.

```bash
# Restart an incomplete or changed pilot in its separate database.
uv run python scripts/import_natality.py --year 2024 --restart

# Repeat the comparison without importing again.
uv run python scripts/import_natality.py --year 2024 --validate-only
```

A restart rereads the cached archive from its beginning. It does not resume at an arbitrary compressed byte offset. The importer replaces an earlier accepted pilot table only inside a transaction after the new source checks pass. It refuses to modify an existing database that lacks its pilot metadata.

The numeric `id` is `year * 1000000000 + source_record`. This is deterministic within the exact source release. It is not a permanent birth identifier. Neither these IDs nor old prediction IDs justify joining records across releases. The build metadata records the source hash.

## Run and validate 2025

```bash
uv run python scripts/import_natality.py --year 2025
```

This uses `data/raw/natality/2025/` and writes a separate candidate and reports under `data/staging/natality/2025/`. It has the same resource settings and path protections as the 2024 command.

The current database has no 2025 records. The reviewed source catalogue selects validation against published controls for this year. It does not treat a skipped SAS comparison as a match. The validator checks:

- All 190 stored source columns against frequencies recorded before conversion, including 34 columns absent from the 2025 source.
- Unique source-record numbers, deterministic IDs, year and row counts.
- Forty [published controls](../config/natality/controls/2025.json) transcribed from the guide, including residence, race/origin, maternal age and congenital-anomaly counts. The controls retain the published population and reporting-flag exclusions.
- Study derivations, including pending versus confirmed status, age, race/origin and the existing age-risk formula.
- Every external lookup's year and key coverage, and agreement between the available references and derived values.
- The live database's unchanged hash, size and modification time.

The extra `us_births.coverage.json` report lists all five imported reference tables. None has 2025 rows. The importer leaves `p_ds_lb_wt`, `p_ds_lb_wt_mage`, `p_ds_lb_nt_reduc` and `ds_case_weight` NULL for 2025. Three existing probability placeholders also remain NULL. The age-based `p_ds_lb_nt` calculation is available because its formula does not require a year-specific lookup. This does not make it an observed prevalence estimate.

Import acceptance and model-input coverage are separate results. A passing validation report does not authorise filling missing references or extending fitted models. The expected-count section of the descriptive report now fails with the affected years when inputs are missing. The core reduction model already rejects years absent from its reduction-prior CSV. Existing fitting defaults and frozen analyses retain their original year ranges.

Use `--validate-only` to repeat these checks. Use `--restart` after reviewed source, layout, code or reference changes. The command has no promotion step.

## Review another year

The catalogue is [sources.json](../config/natality/sources.json). Its links were taken from the US column of the CDC portal, separately from the territory column. A listed URL does not imply that its file has been validated.

The [2024 specification](../config/natality/layouts/2024.json) and [2025 specification](../config/natality/layouts/2025.json) record field positions, types, code domains, blank meanings, reporting flags and guide pages. Each gives every current import column an explicit presence or absence decision. They also distinguish the guide's record length from the observed blank padding in each exact archive. The 2025 specification adds seven reporting flags and updates field-to-flag links after a comparison of all layout pages.

Before enabling another year, review its guide, archive member, framing and code meanings. Add a separate specification and synthetic tests. Do not set `layout_status` to `reviewed` based on equal record lengths or a copied layout. The 2024 specification includes a note about the guide's inconsistent `ILP_R11` code list. It retains the source codes without assigning labels to the disputed categories.

See the [migration plan](../plans/2026-11-06-cdc-text-import-migration.md) for the remaining historical imports, research decisions on missing external inputs and eventual database replacement. No performance result from the pilot establishes diagnostic accuracy or validates the model's external assumptions.

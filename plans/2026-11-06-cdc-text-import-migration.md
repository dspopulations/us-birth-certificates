> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# Proposed migration to CDC natality text files

Prepared on 6 November 2026. The [source audit and 2024 pilot](../notes/20261106-cdc-2024-import-audit.md) and [2025 layout, reference review and import](../notes/20261106-cdc-2025-import-audit.md) now record the first implementations and their validation. The live database remains unchanged. Historical imports, wider integration and database replacement remain outstanding.

## Recommendation and scope

Replace the NBER SAS source with the CDC's annual US natality ZIP files. Keep the compressed archives locally, read their text members in batches, and write typed columns directly into a new DuckDB database. Make intermediate Parquet files optional.

The 2024 pilot provides a comparison with the existing SAS route. The 2025 import uses published controls because the saved database has no 2025 records. Next, migrate 1989-2023. The completed route should support all 37 years from 1989 to 2025. Import the US files only. Keep territories outside this migration unless the study scope changes.

Treat three concerns separately. The reader extracts source fields. The year-specific mappings give those fields their correct names, types and meanings. The existing derivations produce the study's combined variables and statistical inputs. A change of file format should not silently change the study population or its definitions.

The main work is checking the mappings for every year. Removing Parquet does not remove that work. Nor does agreement with the existing database prove that either route interpreted the source correctly.

## Evidence and current dependencies

The [CDC portal](https://www.cdc.gov/nchs/data_access/VitalStatsOnline.htm) lists a 2025 US ZIP of about 226 MB and its user guide. It supplies US and territory files separately. The proposed route would no longer depend on the release of a converted SAS file.

The [2025 guide](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2025.pdf), page 9, specifies fixed records of 1,330 positions and 3,619,990 US records. The guide gives 3,612,012 births to US residents and 7,978 to foreign residents. The full import matches these controls. Its actual member contains another 15 blank bytes per record and CRLF separators, for 4,876,126,530 bytes. The [2025 audit](../notes/20261106-cdc-2025-import-audit.md) records the archive identity and checked exception to the guide's length.

The [2011 guide](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2011.pdf), in its file characteristics section, specifies 775 positions per record. A single layout copied across years would therefore fail. Even equal record lengths would not establish equal field definitions.

The current [preparation guide](../docs/data-preparation.md) and scripts show these dependencies:

| Current component | Consequence for migration |
| --- | --- |
| `download_data.py` fetches SAS files for 1989-2024 and guides from 2000 onwards | Add CDC archive metadata and obtain the missing older guides and relevant addenda. Replace the existing rule that skips any file already present. |
| `import_parquet.py` reads a whole SAS year into pandas and selects `IMPORTED_VARS` | Replace full-year reads with batches. Account explicitly for every imported column in each year. |
| `combine_parquet.py`, `prepare_parquet.py` and `duckdb_create.py` build several intermediate files | Move typing and validation into the new import path. Keep Parquet only where a consumer needs it. |
| `duckdb_prepare.py` narrows types, adds combined variables and loads five reference CSVs | Separate source typing from derivation. Preserve the defined calculations except for documented corrections. |
| `make_all.py` and `export_spss.py` consume annual Parquet files | Provide equivalent exports from DuckDB or adapt these callers before retiring the old outputs. |
| Predictions use an `id` assigned from database row order | A rebuilt database must not inherit predictions by matching old numeric IDs. |

## 1. Record the source and layout for every year

Create a versioned source catalogue, proposed at `config/natality/sources.json`. Each year must identify the exact US archive URL, guide URLs, addenda, release information where available, and the expected control counts with their population definitions. Discover links from the portal, then record them explicitly. Do not construct historical URLs from a guessed naming pattern.

Create reviewed layout specifications under `config/natality/layouts/`. Each year must resolve to one complete specification. Years may share a base layout only after their guides have been compared. Produce an expanded annual specification so reviewers can inspect each year without following a long chain of overrides.

Each field specification should include:

- The CDC field name, start position and width. Store positions as the guide prints them, starting at 1. Convert them to Python slices in one tested function.
- The corresponding project column, if imported, and any alias used by NBER's SAS conversion.
- The source type, permitted codes or ranges, decimal notation or implied scale, and intended Arrow and DuckDB types.
- The meaning of blanks, unknown codes and not-applicable codes for that field and year.
- The applicable certificate revision, reporting flag and population restrictions where relevant.
- The guide page, any addendum that overrides it, and the reason for each project mapping.

Keep archive member identity, expected record length, encoding and separator rules with the annual specification. Classify every existing `IMPORTED_VARS` entry as present, present under another name, or absent with evidence. A missing mapping must fail validation. It must not silently produce an all-NULL column.

Use the CDC guides as the authority. NBER import programs and SAS metadata can help explain name changes, but must not define the new layout by themselves. PDF extraction may produce a draft specification. Check positions, widths and code lists against the actual tables before accepting it.

Review every year, with extra tests around the following boundaries already flagged by [variables.py](../src/dspopulations_us_birth_certificates/variables.py):

| Years or boundary | Main checks |
| --- | --- |
| 1989-2002, including 1996/1997 | Old year fields, maternal-age categories, Down syndrome codes and older sex, birthweight, Apgar and gestation fields. |
| 2002/2003/2004 | Transition file, maternal-age source, education aliases and coexistence of certificate revisions. |
| 2004-2007 | Delivery-method recodes and the `CA_DOWN` / `CA_DOWNS` naming changes. |
| 2008-2011 | Field removals, reporting flags and the 2009 and 2010 addenda. |
| 2013/2014 and 2015/2016 | Race and Hispanic-origin meanings, revised layouts and certificate coverage. |
| 2017/2018 and 2019/2020 | Hispanic-origin detail, bridged race and plurality categories. |
| 2024/2025 | Full field comparison, control totals and current data-quality notes. |

These are review priorities, not presumed layout families. Some comments in `variables.py` describe availability in the existing SAS extract. Check them before treating them as statements about the CDC text files. Do not automatically map older fields into later variables merely because their labels resemble each other.

## 2. Download a durable compressed source

Download each archive to an ignored local cache, proposed at `data/raw/natality/<year>/`. Use a temporary filename until the transfer completes. Add timeouts, retries and checks for incomplete transfers. Resume a partial HTTP transfer only if the server supports ranges and the source version can be checked; otherwise restart it.

Record the final URL, retrieval time, response metadata, byte length and SHA-256 digest. A digest identifies the exact bytes used; a locally computed digest alone does not prove that the publisher supplied those bytes. Compare a publisher checksum if one exists. Retain a changed archive as a distinct source version rather than silently replacing the previous one.

Inspect the ZIP directory. Select the documented data member explicitly and reject missing or ambiguous matches. Check its uncompressed size and compression method. Read it through to the end and require its ZIP integrity check to pass before accepting the year. Save guide hashes and layout hashes with the archive hash.

The compressed file is the restart point. Python's [ZIP API](https://docs.python.org/3.14/library/zipfile.html) supports reading a member as a binary stream, but opening a ZIP for reading requires a seekable source. This means the reader can move backwards and forwards within the file. A local archive meets that requirement and allows another parsing attempt without another download. Streaming directly from HTTP is not needed for the first version.

Extend `.gitignore` to cover the dedicated raw-data and staging directories, partial downloads, DuckDB journal files and private diagnostic files. The existing rules do not cover all these outputs. Keep specifications and safe aggregate checks in version control. Follow the repository's restrictions on record-level data and the [NCHS data user agreement](https://www.cdc.gov/nchs/policy/data-user-agreement.html).

## 3. Parse bounded batches into DuckDB

Use this proposed flow:

```text
CDC ZIP cached on disk
  -> binary reader for the selected text member
  -> annual layout and strict field checks
  -> typed Arrow batches held in memory
  -> annual staging table in a new DuckDB database
  -> validated imported columns
  -> combined variables and reference joins
  -> validated replacement database
```

Arrow batches hold a limited number of rows in memory. They are not intermediate files. DuckDB supports inserting from Arrow objects through SQL; see its [Arrow import documentation](https://duckdb.org/docs/2.0/guides/python/import_arrow).

Implement the reader under `src/dspopulations_us_birth_certificates/ingest/`, with separate modules for source metadata, layouts, parsing, database writes and validation. Add one command, proposed as `scripts/import_natality.py`, with year selection, source-cache and output paths, batch size, memory budget, restart and validation options. Call `init_script()` in `main()` and use `file_io.py` for JSON and CSV outputs consumed elsewhere.

The parser should:

- Read binary blocks of bounded size and preserve unfinished records across blocks. Support the verified separator format for each source, including a final record without a newline where permitted. Do not assume that all files are line-delimited.
- Validate the full record length before extracting fields. Remove only documented record separators. General whitespace stripping would remove meaningful trailing spaces.
- Slice the exact field positions. Do not infer column widths, split on spaces or use CSV delimiter detection.
- Decode with the declared encoding and reject unexpected bytes. Apply field-specific numeric and category checks before conversion.
- Check the embedded year against the selected source year. Investigate missing or conflicting years rather than overwriting them with the archive's year.
- Preserve documented unknown, pending and not-applicable codes. A blank can have a specific meaning, such as an unset imputation flag. Define its handling per field instead of using a global blank-to-NULL rule.
- Retain the reporting flags needed to interpret study fields, including the Down syndrome flag. Record structural absence separately in the layout metadata even where the analysis table uses NULL.
- Report malformed records and invalid codes with year, source member, record number and field name. Keep any detailed local evidence private. An unexplained parsing error blocks acceptance of the year.

Start with an adjustable batch size of roughly 50,000-100,000 records and one database writer. Treat this as a benchmark starting point, not a memory guarantee. Use bulk inserts with explicit column names and a fixed schema. Avoid per-row SQL calls and full-year pandas frames. If the first parser is too slow, benchmark vectorised byte extraction or another explicit-width reader against the same correctness checks.

Give DuckDB a disk-backed temporary directory and a memory limit below available RAM. Measure the whole process because Python and Arrow also allocate memory. DuckDB can spill many operations to disk, but some query combinations can still exhaust memory. Its [performance guide](https://duckdb.org/docs/2.0/guides/performance/how_to_tune_workloads) describes these limits. Budget for the old database, candidate database, ZIP cache, staging tables, journal and temporary space.

## 4. Preserve meaning and correct proven conversion errors

Retain the project's combined definitions for year, maternal age, race, Hispanic origin and recorded Down syndrome. Keep extraction rules separate from these derivations so each can be tested independently.

For example, the [2025 guide](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/UserGuide2025.pdf), page 39, places `CA_DOWN` at position 552 and its reporting flag at 558. Its C, P, N and U codes distinguish confirmed, pending, no and unknown. Blanks also occur for non-reporting areas. Preserve these distinctions before deriving the project's recorded-status indicators.

Resolve the following code-level concerns during the schema review:

- `prepare_parquet.py` permits `DOB_TT` up to 9999, but `duckdb_prepare.py` then tries to cast it to an unsigned eight-bit integer, whose maximum is 255. Choose a type that holds the documented source values.
- `prepare_parquet.py` excludes `NO_RISKS=0` and `NO_CONGEN=9`, although those codes appear in the current guide. Apply year-specific code lists.
- The same script restricts raw `MRACEHISP` to 1-8 despite the repository's documented older scheme including 9. Check that boundary against the older guides.
- BMI passes through a 16-bit float. Establish the precision needed to preserve the source decimals and avoid an unnecessary lossy conversion.

These findings come from code and documentation, not a measured audit of the current database. Quantify affected counts in the pilot. Maintain a change register with the old result, new result, source evidence and affected outputs. Exact agreement with a known lossy conversion is not an acceptance requirement. Every other unexplained difference is a reason to investigate.

Do not broaden historical feature coverage as an incidental part of the reader change. For example, adding old sex or gestation fields to later analysis variables needs an explicit mapping decision and separate output checks.

## 5. Make reruns and database replacement safe

Build into a new database path. Import each year into a staging table that holds its rows until validation finishes. Save batches as needed to bound resource use. Mark a year complete only after the entire member, integrity checks, record counts and field checks pass. Transfer its accepted rows and completion metadata together in a transaction, so both changes succeed or neither does. Incomplete staging tables must never be visible as accepted analysis data.

Record a source version and record number for every row. Preserve a numeric `id` for current callers, but assign it deterministically within the build and record the build identifier. A source record number is an address within one exact file, not a permanent person identifier. CDC reissues can change it. Do not deduplicate rows that happen to have equal selected values; they may be different births.

Skip a completed year only when its source hash, layout hash and relevant import-code version match. A normal retry discards the incomplete staging table and rereads that year from its cached ZIP. It retains completed years. Resume within a compressed member can wait until there is evidence that it is needed.

Refactor derivation so it accepts an explicit database connection and can rebuild a chosen year's derived columns safely. Validate that lookup tables have unique join keys. Do not rerun the present destructive compaction routine on the live database.

After all checks pass, checkpoint and close the candidate, then reopen it for final checks. Preserve the old database under a versioned backup name. Promote the candidate only with readers and writers closed, using a tested replacement and recovery procedure on each supported platform. A failed promotion must leave a usable old database or a documented route to restore it.

Old predictions and fitted results remain tied to their old database build. Regenerate them where needed. Future annual imports can preserve existing rows, but replacement of a source year must invalidate that year's dependent outputs.

## 6. Validate in layers

| Layer | Required evidence |
| --- | --- |
| Layout | Every requested year has a reviewed guide and specification. Field spans fit the record. Unexpected overlaps fail; documented aliases or subfields have explicit reasons. Every project column has an availability decision. |
| Parser | Synthetic records test positions, first and last fields, blank meanings, leading zeros, decimal scale, special codes, separators, short reads and truncated records. Expected results must be checked independently of the code that generates a fixture. |
| Archive and loading | Tests cover incomplete and corrupt ZIPs, unexpected members, wrong years, interrupted database writes and retries without duplicate rows. Include a ZIP64 case because expanded members can exceed 4 GiB. |
| Annual counts | Full file counts match the guide's occurrence controls. Resident counts use the matching residence definition. Parsed, staged and accepted counts reconcile exactly. |
| Field distributions | Compare code frequencies, missingness, minimum and maximum values, and reporting-flag combinations for every imported field. Check cross-tabulations of recorded status with age, race, origin and year. |
| SAS comparison | Compare every overlapping year after explicit type and name normalisation. Compare rows across all imported columns and retain duplicate counts. Do not assume row order is shared by the two sources. |
| Derived data | Compare combined variables, recorded counts, analysis cell counts and reference joins. Record exact count differences and justified tolerances for floating-point results. |
| Downstream use | Exercise descriptive queries, classifier input selection, Bayesian cell preparation, prediction writeback and optional exports against the candidate. |

Use row digests to locate differences efficiently, then compare actual fields for differing groups. Preserve duplicate multiplicity. Matching aggregate totals or digests alone is weaker evidence than an explained field comparison. When SAS and CDC releases differ, establish the source-version difference before calling it a parser defect.

Published tables provide independent checks only when their population, missing-data rules and reporting-area exclusions match the query. Do not change the study's cohort just to make a published total match. Review each guide's quality notes as well as its layout.

Save an aggregate validation report and a build manifest with source, guide, layout, code and reference-file hashes; package versions; row counts; timings; peak process memory; disk use; and all accepted deviations. Test the parser and small database workflow on Linux x86-64 and arm64, macOS arm64 and Windows AMD64. Keep full historical comparisons as local integration checks so CI does not fetch or publish microdata.

## 7. Handle 2025 analysis inputs explicitly

The new reader can support 2025 before every model is ready to use it. The existing [reference tables](../docs/data-preparation.md#lookup-tables) stop at 2024 or earlier. Missing joins currently leave derived values NULL, including `ds_case_weight` for an unmatched year.

The 2025 import now calculates supported derivations and writes a coverage report for every imported external lookup. Unsupported estimates remain NULL. Do not extend prevalence, prenatal-reduction or recording weights by copying 2024 values without a separate, documented research decision.

The descriptive expected-count report now rejects incomplete inputs and names the affected years. The core reduction model already rejects years without reduction priors. Before extending other consumers, review their hard-coded ranges and external anchors. Keep existing fitted analyses attached to their original year ranges and database build.

## Delivery order and acceptance

| Step | Deliverable | Condition for moving on |
| --- | --- | --- |
| 1. Source and schema audit | Annual source catalogue, initial 2024/2025 layouts, current database inventory and conversion change register | All pilot fields mapped to guide evidence; baseline schema and counts saved. |
| 2. 2024 pilot | Downloader, batch reader, Arrow-to-DuckDB writer and aggregate comparison report | Full-year control counts pass; no unexplained field differences; measured memory, time and disk use fit the chosen machine. |
| 3. 2025 import | Independently reviewed 2025 layout, candidate data and lookup-coverage report | Source checks pass; unsupported model inputs remain explicit. |
| 4. Historical migration | Reviewed specifications and imports for 1989-2023 | Every year passes source checks and comparison, including all transition years. |
| 5. Integration | Safe derivation, build identities, restart behaviour and optional exports | Downstream checks pass; interrupted-run and recovery tests pass. |
| 6. Replacement | Full 1989-2025 candidate, validation report, backup and updated preparation instructions | All deviations explained; old database recoverable; new build usable by existing callers. |

Set performance targets after the full 2024 pilot. Measure download, decompression, parsing, database insertion, derivation and validation separately. A quick initial sample is useful for debugging but cannot establish full-year resource use or archive integrity. Benchmark a Parquet staging option only if it offers a measured benefit for recovery or repeated work.

Retire the SAS import and mandatory Parquet chain after the full historical comparison and downstream checks. Retain local comparison sources until the migration is accepted. Update `docs/data-preparation.md`, the workflow guide and affected script references to describe the implemented commands. Use the existing shared dependency stack; propose new shared needs upstream if the pilot establishes a gap.

The implementation is complete when all requested years have reproducible source mappings, no unexplained loss or duplication, bounded measured resource use, safe reruns, and a validated database that the existing analyses can read. Import validation establishes that we read the files as specified. It does not validate diagnoses or the external assumptions used to estimate expected Down syndrome counts.

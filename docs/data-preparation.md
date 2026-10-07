> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Data preparation

The pipeline converts annual NCHS/NVSS natality SAS files for 1989–2024 into Parquet files and a DuckDB database. Stage 3 writes the type-constrained imported columns to `data/us_births.parquet`. Stage 5 adds harmonised and calculated columns to `data/us_births.db`. **The Parquet file does not contain the final database's derived columns.**

## Setup and source data

Run from the repository root:

```bash
uv sync --locked
uv run python scripts/download_data.py
```

The downloader fetches 36 annual SAS files from NBER and 27 NCHS guide/addendum PDFs. It skips existing files. It does not retry downloads or verify checksums. An interrupted download can leave a partial file that later runs skip; check suspect files and replace incomplete downloads.

Natality microdata uses the [NCHS Data Use Agreement](https://www.cdc.gov/nchs/data_access/restrictions.htm). Raw files, guide PDFs, Parquet files and DuckDB databases are gitignored. Do not commit or publish raw or derived record-level data. Small aggregate and reference CSVs are tracked in `data/`; that directory is not wholly ignored.

## Run the pipeline

```bash
uv run python scripts/import_parquet.py
uv run python scripts/combine_parquet.py
uv run python scripts/prepare_parquet.py
uv run python scripts/duckdb_create.py
uv run python scripts/duckdb_prepare.py
```

| Stage | Input | Output | Main operation |
| --- | --- | --- | --- |
| Import | `data/*.sas7bdat` | `us_births_<year>.parquet` | Read each year and reindex to `IMPORTED_VARS` |
| Combine | Annual `us_births_<four-digit-year>.parquet` files | `us_births_combined.parquet` | Concatenate with Polars |
| Prepare | Combined Parquet | `us_births.parquet` | Constrain types in batches |
| Create | Prepared Parquet | `us_births_temp.db` | Create the `us_births` table |
| Derive | Temporary database and five CSVs | `us_births.db` | Narrow types, add and populate columns, compact |

The final stage also replaces `us_births_temp.db` with a copy of the compacted database. There is no single script for this core chain.

## Reruns and resource use

- Import skips an annual output if it exists. Its year parser uses every digit in the SAS filename, so source names must contain only the four-digit year as their digits.
- Import reads one whole year's SAS file into pandas. Allow enough memory for the largest year's frame. Missing source columns become NULL after reindexing; reindexing does not apply the declared dtypes.
- Combine selects only filenames matching `us_births_<four-digit-year>.parquet`, sorted by name. It excludes combined and derived outputs. Its relaxed schema union can still hide unexpected column differences.
- Preparation streams batches. Invalid integer parses, non-integers and values outside specified ranges become NULL. The script prints invalid-value counts. It runs through `main()`, not on import.
- Database creation replaces the temporary database. The derivation stage adds `id` without `IF NOT EXISTS`, so do not rerun stage 5 directly on its completed database. Recreate the temporary database through stage 4 first.
- Compaction deletes the old final database before copying the replacement. Back up an existing database if it contains predictions or other work you need. A failure can leave the temporary database as the only usable copy.
- Rebuilding changes row identifiers and removes later prediction columns. Regenerate predictions and analyses against the rebuilt database.

The full dataset has about 143 million rows. Runtime depends on hardware, storage and available memory; old session timings are not a resource guarantee.

## Derived columns (stage 5)

Derivations run in order because later columns use earlier ones. `scripts/duckdb_prepare.py` defines the SQL. `variables.py` defines column names and source codes. The [coding reference](../previous/us-birth-certificates/data-preparation.md) lists historical race and origin codes.

- **`id`** (`BIGINT`): `row_number() OVER ()::BIGINT` keyed on `rowid`, populated just before the `year` derivation.
- **`year`**: `COALESCE(dob_yy, datayear)` - `dob_yy` (2003 revision) preferred, `datayear` (1989 revision) fallback.
- **`ca_down_c`** (`VARCHAR`; value domain C/P/N/U): (1) if `COALESCE(ca_down, ca_downs) IS NOT NULL` -> `UPPER(COALESCE(ca_down, ca_downs))` (`ca_down` preferred); else (2) `uca_downs` 1->'C', 2->'N', 9->'U'; else (3) `downs` 1->'C', 2->'N', 9->'U'; else NULL. `downs=8` (not on certificate) is not matched and falls through to NULL.
- **`down_ind`** (0/1/NULL): `UPPER(ca_down_c) IN ('C','P')` -> 1; `='N'` -> 0; else `downs=1` -> 1, `downs=2` -> 0; else `uca_downs=1` -> 1, `uca_downs=2` -> 0; else NULL. `ca_down_c='P'` (pending) counts as a case; `ca_down_c='U'` is not caught in the first branch and falls through to the `downs`/`uca_downs` branches (and may end NULL).
- **`mage_c`**: `COALESCE(mager, dmage, mage36 + 13, mager41 + 13)` - `mager` (2004+ single-year) preferred, `dmage` (<=2002 single-year) next, then the `mage36` recode +13 (<=2002), then the `mager41` recode +13 (2003-only; same 41-category coding as `mage36`). The `mager41` fallback recovers 2003, which carries none of the first three (see the cross-year coding section below). `mage36`/`mager41` code 01 ("Under 15") maps via +13 to age 14 - a lower-bound approximation, immaterial above the lowest analytic age boundary (20).
- **`p_ds_lb_nt`**: Morris double-logistic in `mage_c`, `1 / (1 + exp(7.33 - 4.211 / (1 + exp(-0.2815 * (mage_c - 37.23)))))` - maternal-age probability of a DS live birth absent terminations.
- **`p_ds_lb_wt`**: per-year surveillance prevalence from `us-births-surveillance-prevalence-1989-2024.csv` (36 rows) loaded into DuckDB table `prevalence_year`, joined on `year`. Values rise from `0.001038` (1989); the trailing value `0.001324215` appears for **2018-2024** (the last 7 entries are identical - intentional carry-forward of the last estimated year).
- **`mrace_c`** (1-5): `mrace15` (1,2,3 keep; 4-14 -> 4; **15 "More than one race" -> 5**) -> else `mracerec` (1-4 keep) -> else `mbrace` (1-digit 1-4 keep; 2-digit 01-03 -> 1/2/3, 04-14 -> 4, **bridged-multiple 21-24 -> 5**; Puerto Rico 0 -> NULL) -> else `mrace` (1,2,3 keep; 4-78 -> 4) -> else NULL. Category **5 "More than one race"** is only identifiable from MRACE15=15 (2014+) and MBRACE 21-24 (2003-2013, unreachable); MRACEREC and the 1989-cert MRACE carry no multi-race code, so 1989-2013 multi-race is folded into single-race categories. Unmatched values within a selected source branch resolve to NULL. MRACEREC/MRACE15 precede MBRACE, so the MBRACE branch is in practice only the fallback for Puerto Rico 2014-2019.
- **`mhisp_c`** (0-5): `mhisp_r` (0,1,2,3 keep; 4-5 -> 4; 9 -> 5) -> else `mhispx` (0,1,2,3 keep; 4-6 -> 4; 9 -> 5) -> else `umhisp` (0,1,2,3 keep; 4-5 -> 4; 9 -> 5) -> else `orracem` (1,2,3 keep; 6-8 -> 0 non-Hispanic; 4-5 -> 4; 9 -> 5) -> else NULL.
- **`mracehisp_c`** (1-6 or NULL): **reconstructed from `mhisp_c` + `mrace_c`** - deliberately not the raw NCHS `mracehisp` field, which is dual-coded across eras and absent pre-2003. `mhisp_c BETWEEN 1 AND 4` -> 5 (Hispanic); `mhisp_c = 5` (origin unknown) -> NULL (the row's race is discarded); non-Hispanic multi-race (`mrace_c = 5`) -> **6 (NH more than one race)**; `mhisp_c = 0` or NULL -> `mrace_c` (non-Hispanic race 1-4). Note the asymmetry: explicit "origin unknown" (5) drops to NULL, whereas *absent* origin (NULL) keeps the race as non-Hispanic. The selection model and `derive_recording_rates` now give code 6 (NH multi-race) its own race group (`race_idx` 6); only NULL/other races fall to the "Unknown" cell (idx 5). De Graaf has no multi-race anchor, so idx 6 carries the same weak `s(race, year)` fallback as Unknown.
- **`p_ds_lb_wt_mage`**: from `us_births_est_prevalence_age` (maternal-age CSV) joined on `year`; per row `mage_c < 35` -> `p_ds_lb_wt_lt35_sv` else `p_ds_lb_wt_gte35_sv`. The CSV covers 1989-2018, so 2019-2024 rows stay NULL.
- **`p_ds_lb_nt_reduc`**: `p_ds_lb_nt * (1 - r.reduction)` from `reduction_rate_year` joined on `year` (renamed from `p_ds_lb_wt_mage_reduc`; the multiplicand is `p_ds_lb_nt`, the Morris no-terminations risk - the old `_mage` suffix was a misnomer).
- **`ds_case_weight`** (`DOUBLE`): from `ds_case_weights` joined on `year`. When `down_ind=1`, selected by `mracehisp_c`: 1 -> `nhw`, 2 -> `nhb`, 3 -> `ai_an`, 4 -> `as_pi`, 5 -> `his`, else (`down_ind=1` with `mracehisp_c` NULL/other) -> `total`; otherwise 0 (non-cases and `down_ind != 1` get weight 0). Rows with `mracehisp_c=NULL` (origin unknown) fall to the `total` branch.

Of the seven declared `p_ds_lb_*` columns, only **four are actually populated** by stage-5 `UPDATE`s - `p_ds_lb_nt`, `p_ds_lb_wt`, `p_ds_lb_wt_mage`, and `p_ds_lb_nt_reduc`. The other three (`p_ds_lb_nt_mage`, `p_ds_lb_wt_ethn`, `p_ds_lb_nt_ethn`) are added as `DOUBLE` columns but have no `UPDATE`, so they remain NULL throughout (placeholders for downstream work - the ethnicity prevalence table is loaded but never joined). All computed columns are NULL between the two halves of stage 5, and per-year joins silently leave rows NULL when a year has no matching lookup row.

## Lookup tables

The five CSVs are read from `data/` and loaded into DuckDB tables. Run from the repository root. Missing lookup years leave derived values NULL.

| File | Columns | Year coverage | Feeds |
| --- | --- | --- | --- |
| `us-births-surveillance-prevalence-1989-2024.csv` | `year`, `p_ds_lb_wt` (36 rows; 2018 value carried forward to 2024) | 1989-2024 | `p_ds_lb_wt` (via table `prevalence_year`) |
| `us-births-estimated-prevalence-maternal-age-1989-2018.csv` | `year`, `p_ds_lb_wt_lt35_sv`, `p_ds_lb_wt_gte35_sv`, `p_ds_lb_nt_lt35_sv`, `p_ds_lb_nt_gte35_sv` (30 rows; only the two `p_ds_lb_wt_*` columns are consumed) | 1989-2018 | `p_ds_lb_wt_mage` (via table `us_births_est_prevalence_age`) |
| `us-births-reduction-rates-1989-2024.csv` | `year`, `reduction` (36 rows) | 1989-2024 | `p_ds_lb_nt_reduc` (via table `reduction_rate_year`) |
| `us-births-ds-rec-weights.csv` | `year`, `nhw`, `nhb`, `his`, `as_pi`, `ai_an`, `total` (36 rows; referenced by name in the SQL) | 1989-2024 | `ds_case_weight` (via table `ds_case_weights`) |
| `us-births-estimated-prevalence-ethnicity-2000-2018.csv` | `year`, `mracehisp_c`, `prevalence` (95 rows, long format, exactly 5 `mracehisp_c` codes 1-5 per year) | 2000-2018 | none in this script - loaded standalone as table `us_births_est_prevalence_ethnicity` for downstream use |

Notes:
- The two `p_ds_lb_nt_*` columns in the maternal-age CSV are loaded but never joined (the `_nt` reduction path uses `us_births.p_ds_lb_nt` directly).
- In the ethnicity CSV, years **2015 and 2017** have blank `prevalence` values for all five codes, which pandas reads as NA.
- Year-coverage mismatch: the maternal-age and ethnicity tables stop at 2018, while reduction and rec-weights run to 2024. For 2019-2024 rows, `p_ds_lb_wt_mage` stays NULL while `p_ds_lb_nt_reduc` and `ds_case_weight` are populated.

## External estimates and their limits

The probability and weight columns are statistical inputs, not NCHS diagnostic labels.

- `p_ds_lb_nt` uses the Morris age-risk formula. SQL obtains its parameters from `chance.MORRIS_PARAMS`, which also supports the Python calculation.
- `p_ds_lb_wt` carries its 2018 prevalence value forward through 2024. A change in numeric precision around 2014/2015 suggests different source vintages, but does not establish their provenance.
- The reduction CSV extrapolates the tail. The fitting code gives wider priors from 2020; the [family review](../notes/20260803-dsp-core-model-family-review.md) finds that 2019 also appears extrapolated. Source derivation and the intended tail need reconciliation.
- The age and ethnicity lookup series stop at 2018. Ethnicity values are also blank for 2015 and 2017.
- `ds_case_weight` derives from recording-rate assumptions. It is not an independent expected-count benchmark.

The reduction series appears numerically consistent with `1 - surveillance_prevalence / age_expected_prevalence` in the grounded years. This is a reconstruction, not a documented original derivation. Changing the age-risk curve without revisiting that denominator can make the inputs inconsistent. The [source workbook note](../notes/20260803-degraaf-surveillance-workbook-extraction.md) explains the separate surveillance anchor used by DSP007–DSP010.

The tracked `us-births-degraaf-prevalence-recording-2000-2024.csv` is a reference extraction, not a stage-5 input. It distinguishes raw surveillance prevalence from regression-filled estimates that reuse certificate counts. See the [corrected extraction note](../notes/20260628-degraaf-corrected-prevalence-extraction.md).

## Cross-year coding and missing values

Consult the source guides, `variables.py` and the [historical coding reference](../previous/us-birth-certificates/data-preparation.md) before changing any derivation.

- Maternal age in 2003 comes from `mager41 + 13`. Code 1 represents "under 15" and maps to 14. Later single-age codes also pool the endpoints 10–12 and 50+.
- The revised certificate phased in through 2015; national coverage was complete in 2016. A field's first appearance or high coverage in 2014 does not imply full revised-certificate coverage.
- Raw `mracehisp` codes have different meanings before and after 2014. Use the harmonised `mracehisp_c` for cross-era comparisons. More than one race is separately identifiable from 2014; its earlier coding is bridged.
- `down_ind` can be NULL for unknown or absent status. The descriptive report counts those births in its all-birth denominator but not its recorded numerator. Bayesian cell preparation excludes unknown status. State the cohort rule when comparing outputs.
- Maternal education uses different schemes across certificate eras. Maternal marital status uses `mar` in 2003–2013 and `dmar` in the surrounding years.
- The extract does not import the old `CSEX`, `DBIRWT`, `FMAPS` or `GESTAT10` fields. Later sex, birthweight, Apgar and gestation variables therefore cannot simply be extended backwards. `DMETH_REC` also changes coding across years.

A June 2026 database check found 59 records with no year or other substantive source fields. That is a historical build result, not a check performed on every build. Reports should exclude missing year from time series and inspect unexpected missingness rather than assuming the same count. The [June validation note](../notes/20260627-data-prep-adjustments-validation.md) records source-guide checks and resolved changes.

## Side outputs

`uv run python scripts/make_all.py` builds `data/us_births_all.parquet` from 2014–2024 annual files. Despite its name, it is a modelling subset, not the core pipeline.

`uv run python scripts/export_spss.py` exports 2014–2022 annual files and `us_births_all.parquet` to SPSS and ZIP files. It leaves intermediate `.sav` files and refuses to overwrite an existing ZIP. These record-level outputs retain the natality data restrictions.

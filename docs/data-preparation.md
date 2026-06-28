> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# Data preparation

This document describes the pipeline that turns raw NCHS/NVSS natality SAS microdata (one file per year, 1989-2024) into a single analysis-ready dataset. The pipeline converts SAS to per-year parquet, combines the years, applies tight type constraints, loads the result into DuckDB, and derives the harmonised and computed columns the study depends on. The final artefacts are a DuckDB database at `data/us_births.db` and a matching `data/us_births.parquet`. The combined dataset is approximately **142.9 million birth records** across the 36 years.

## Prerequisites

### Conda environment

The pipeline runs in a conda environment named `dspop-us-birth-certificates` (Python **3.14**, channel `conda-forge`). Create and activate it from the repo root:

```bash
conda env create -f environment.yml
conda activate dspop-us-birth-certificates
```

The package itself installs editable (`-e ./`) as part of `environment.yml`, and the sibling `dse_research_utils` installs editable from `-e ../../dseinternational/research/src/python` (the `research` repo must be cloned alongside this one).

Key data-pipeline packages (note the channel split): `polars>=1.40.1`, `pyarrow>=23.0.1`, and `pandas>=3.0.1` come from conda-forge, while `duckdb>=1.5.2`, `pyreadstat>=1.3.4` (reads `.sas7bdat`), `fastparquet`, and `truststore` come from the `pip:` subsection. `truststore` is imported at module top in the download script and `truststore.inject_into_ssl()` runs on import, so the environment must be active or the download script fails immediately.

### Source data via `scripts/download_data.py`

Run the downloader from the repo root:

```bash
python scripts/download_data.py
```

It creates `data/` if absent (`os.makedirs('data')`) and downloads, in this order:

1. **27 NCHS user-guide PDFs** from `ftp.cdc.gov` (`UserGuide2005`-`UserGuide2024` plus `Nat2000doc`-`Nat2004doc` for the 2000-2004 years). Some filenames are irregular: `UserGuide2019-508.pdf` and `UserGuide2018-508.pdf` carry a `-508` 508-compliance suffix, and `UserGuide2010_Addendum.pdf` / `UserGuide2009_Addendum.pdf` are separate addendum files alongside `UserGuide2010.pdf` / `UserGuide2009.pdf`.
2. **36 natality SAS microdata files** `natality{year}us.sas7bdat` for every year 1989-2024, from `data.nber.org`.

Every file is written under `data/`. The script **skips any file already present** (`if not os.path.exists(filename)`), so re-running fetches only what is missing and an interrupted download can be resumed. It uses `urllib.request.urlretrieve` over HTTPS. There is no progress bar, retry, or checksum verification, so a partially-written file from an interrupted run will be treated as present and skipped - delete suspect partial files before re-running. These are large files (full US natality microdata across 36 years); expect substantial download time and disk use.

> [!IMPORTANT]
> Raw natality microdata is governed by the [NCHS Data Use Agreement](https://www.cdc.gov/nchs/data_access/restrictions.htm). The `data/` directory is **gitignored** and raw (or derived) records must **never** be committed or published. Despite what `README.md` says for the `data/` directory, the data is subject to the NCHS DUA, **not** CC BY 4.0.

> [!NOTE]
> An older `previous/us-birth-certificates/readme.md` instructs `python ./prepare/download_data.py`; that path no longer exists. The script lives at `scripts/download_data.py`.

### Lookup CSVs at the repo root

Stage 5 reads five lookup CSVs by `./`-relative path. They must sit at the repo root (the first four were moved here out of `previous/us-birth-certificates/`; the surveillance-prevalence series was externalised from an inline table):

- `us-births-estimated-prevalence-maternal-age-1989-2018.csv`
- `us-births-reduction-rates-1989-2024.csv`
- `us-births-ds-rec-weights.csv`
- `us-births-estimated-prevalence-ethnicity-2000-2018.csv`
- `us-births-surveillance-prevalence-1989-2024.csv`

## Pipeline overview

All scripts are run **from the repo root** (paths are CWD-relative). There is currently **no single orchestrator script** for the core chain - run the five stages in order:

```
1. import_parquet.py    data/*.sas7bdat                 -> data/us_births_{year}.parquet
2. combine_parquet.py   data/*.parquet                  -> data/us_births_combined.parquet
3. prepare_parquet.py   data/us_births_combined.parquet -> data/us_births.parquet
4. duckdb_create.py     data/us_births.parquet          -> data/us_births_temp.db
5. duckdb_prepare.py    data/us_births_temp.db + 5 CSVs -> data/us_births.db
```

## Running the pipeline

With the environment active and from the repo root:

```bash
python scripts/import_parquet.py
python scripts/combine_parquet.py
python scripts/prepare_parquet.py
python scripts/duckdb_create.py
python scripts/duckdb_prepare.py
```

## Stages in detail

### 1. `import_parquet.py`

**Command:** `python scripts/import_parquet.py`

**Inputs:**
- `data/*.sas7bdat` (raw NCHS natality microdata, one file per year; year encoded in the filename digits)
- `dspopulations_us_birth_certificates.variables.IMPORTED_VARS` (133-entry ordered column list, derived from the `IMPORTED` dict in `src/dspopulations_us_birth_certificates/variables.py`)

**Outputs:**
- `data/us_births_{year}.parquet` (one zstd-compressed parquet per source year, columns reindexed to the 133 `IMPORTED_VARS` in that exact order)

**What it does:**
- Globs `data/*.sas7bdat` (non-recursive), derives each file's year by concatenating all digits in the filename stem (`''.join(filter(str.isdigit, file.stem))`), then `int()`.
- Adds a year to the work set only if `data/us_births_{year}.parquet` does not already exist (resumable, idempotent), and processes newest year first (sources sorted descending).
- Reads each year fully into a single pandas DataFrame via `pd.read_sas(..., format='sas7bdat', encoding='latin-1')` (no chunking).
- Reindexes to exactly the 133 `IMPORTED_VARS` columns in order: source columns not in the list are dropped; list columns absent from that year's SAS are created as all-NA.
- Writes `df.to_parquet(..., compression='zstd')`, calling `gc.collect()` after each year to release the prior DataFrame.

**Notes/gotchas:**
- `reindex` selects/orders/pads columns only - it does **not** apply the `IMPORTED` dict dtypes. Dtype coercion is a separate later step.
- Silent column drop/pad: a renamed or missing source variable for a given year becomes an all-NA column with no warning.
- Fragile year parsing: any extra digits in a filename (version suffix, 2-digit year) yield a wrong year. Filenames must contain only the 4-digit year as their digits.
- The skip check is existence-only: a truncated parquet from an interrupted run is treated as done. Delete the partial file to force reprocessing.
- Memory and runtime: each year's SAS (~2-4 GB) is read whole into pandas, but in practice this is only ~50 s/year, so a full 1989-2024 import is ~30-40 minutes (not multi-hour). Peak memory is one year's DataFrame at a time, with `gc.collect()` between years.

### 2. `combine_parquet.py`

**Command:** `python scripts/combine_parquet.py`

**Inputs:**
- `data/*.parquet` (every parquet in `data/`; on a clean first run these are the per-year files)

**Outputs:**
- `data/us_births_combined.parquet`

**What it does:**
- Sets `src_dir = pathlib.Path("data")` and deletes any prior output with `out_parquet.unlink(missing_ok=True)` so a stale combined file is never re-globbed.
- Globs `list(src_dir.glob("*.parquet"))` (non-recursive); raises `FileNotFoundError` if the glob is empty.
- Builds lazy frames (`pl.scan_parquet`), concatenates with `pl.concat(lfs, how="diagonal_relaxed")` - `diagonal` takes the union of columns (missing filled with null), `relaxed` coerces mismatched dtypes to a supertype.
- Streams the result via `combined.sink_parquet(out_parquet.as_posix())`.

**Notes/gotchas:**
- The `data/*.parquet` glob is **indiscriminate**: it picks up any parquet in `data/`, including derived outputs (`us_births.parquet`, `us_births_all.parquet`) from later stages. On a clean first run only the per-year files exist, but re-running after later stages without cleaning `data/` would fold those derived files back in. The just-written combined file is excluded only because it is deleted first; other derived parquets are not.
- `diagonal_relaxed` silently absorbs schema drift across years, which can mask unexpected column/dtype differences.
- `sink_parquet` streams the lazy plan, so an error in a single input file surfaces at sink time, not at scan construction.
- Glob ordering is filesystem-dependent, so combined row order is not guaranteed chronological.

### 3. `prepare_parquet.py`

**Command:** `python scripts/prepare_parquet.py`

**Inputs:**
- `data/us_births_combined.parquet` (read as a pyarrow dataset, scanned with `batch_size=2_097_152`, `use_threads=True`)
- `dspopulations_us_birth_certificates.variables.Variables` (imported as `vars`) - supplies the column-name constants used as keys in `uint8_specs`, `uint16_specs`, `string_cols`, `float16_cols`

**Outputs:**
- `data/us_births.parquet` (`ParquetWriter` with `compression='zstd'`, `use_dictionary=True`, `write_statistics=True`; `row_group_size=500_000` is passed on each `writer.write_table(...)` call, not to the constructor)
- A module-level `stats` dict accumulating `parse_invalid` / `non_integer` / `range_invalid` counts (kept in memory only, never persisted)
- stdout: per-column lines, `Warning: Unspecified column ...` lines, then `Done.`

**What it does:**
- Routes each column by membership in this order: `uint8_specs` -> `uint16_specs` -> `string_cols` -> `float16_cols` -> else pass-through (with a printed warning).
- uint columns go through `constrain_and_cast_uint_robust` with `non_integer='null'`, `range_invalid='null'`: for string input it trims whitespace, maps empty to null, applies `_NUMERIC_RE = r'^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$'` (non-matches nulled, counted `parse_invalid`), casts to float64 (`safe=False`); nulls non-integers (`f != floor(f)`, counted `non_integer`); casts to the target uint dtype; then nulls out-of-range values when a min/max bound is set (counted `range_invalid`).
- `string_cols` and `float16_cols` use a plain `cast_to` (`pc.cast`, `safe=False`) to `pa.string()` / `pa.float16()`.
- Streams the scan: each `RecordBatch` -> `process_batch` -> single-batch `pa.Table`; the `ParquetWriter` is created lazily from the first output table's schema, then `writer.write_table(table, row_group_size=500_000)` per batch; `writer.close()` in a `finally`.

Example range specs: uint16 `DATAYEAR`/`BIRYR`/`DOB_YY`=(1989, None), `DOB_TT`/`DBWT`=(0, 9999), `DWGT_R`=(100, 999), `PWGT_R`=(75, 999); uint8 `DOB_MM`=(1, 12), `MAGER`=(12, 50), `DOWNS`=(0, 255), `UCA_DOWNS`=(1, 9), `NO_CONGEN`=(0, 1), `DPLURAL`=(1, 4), `APGAR5`=(0, 99). Columns with `(None, None)` (e.g. `DMAGE`, `MRACE`, `MAR`, `DMEDUC`) still get robust parse + non-integer nulling + the dtype cast but no range clamping. `string_cols` covers `SEX`, marital flags (`MAR_P`, `DMAR`), the `AB_*`, `CA_*` (including `CA_DOWN`/`CA_DOWNS`), `RF_*`, `LD_*` flags, `BFED`, `WIC`; `float16_cols = [vars.BMI]`.

**Notes/gotchas:**
- **No `if __name__ == '__main__'` guard:** the entire rewrite of `data/us_births.parquet` runs at **module import**. Importing the module to reuse `process_batch` or `constrain_and_cast_uint_robust` triggers the full conversion as a side effect.
- Relative `in_path`/`out_path`, so it must run from the repo root, and the combined parquet must already exist.
- `stats` is a module global, never written out or printed - the accumulated invalid counts are lost on exit unless inspected interactively.
- Unspecified columns pass through unchanged (only a stdout warning), so anything not in the four collections is not type-constrained.
- Casts are `safe=False` (non-raising); correctness depends entirely on the regex + non-integer + range logic, not on cast-time validation. Reduce `batch_size` if memory-constrained.

### 4. `duckdb_create.py`

**Command:** `python scripts/duckdb_create.py`

**Inputs:**
- `data/us_births.parquet`

**Outputs:**
- `data/us_births_temp.db` (intermediate DuckDB) with table `us_births`

**What it does:**
- `create_temp_db()` (called from the `__main__` guard) sets `src_dir = pathlib.Path('data')`, `source_parquet = data/us_births.parquet`, `out_db_temp = data/us_births_temp.db`.
- Deletes any existing temp DB (`out_db_temp.unlink(missing_ok=True)`), connects a fresh DuckDB via `duckdb.connect(out_db_temp.as_posix())`.
- Runs `CREATE TABLE us_births AS SELECT * FROM read_parquet(?)`, parameterised positionally with `source_parquet.as_posix()`. `SELECT *` carries every parquet column through unchanged.
- Wraps the work in `try/finally` that always calls `con.close()`; no explicit commit, no transformation logic.

**Notes/gotchas:**
- This is an **intermediate** DB, not the final artefact; stage 5 produces `data/us_births.db`.
- The `unlink` is destructive: any existing `data/us_births_temp.db` is deleted unconditionally at the start of every run.
- `CREATE TABLE` has no `IF NOT EXISTS`/`OR REPLACE`, but is safe because the file is always freshly created after the unlink.
- A failure mid-`CREATE` can leave a partially-written/empty temp DB on disk.

### 5. `duckdb_prepare.py`

**Command:** `python scripts/duckdb_prepare.py`

**Inputs:**
- `data/us_births_temp.db` (table `us_births` from stage 4)
- The four repo-root lookup CSVs (read with `pd.read_csv(...).convert_dtypes()`)

**Outputs:**
- `data/us_births.db` (compacted final DB)
- `data/us_births_temp.db` (overwritten at the end with a copy of the compacted `us_births.db`)

**What it does (two halves):**

*First half - type narrowing and empty computed columns.* `combine_all()` opens the temp DB and reshapes `us_births` in place via four helpers that interpolate `vars.<NAME>` constants into DDL:
- `alter_column_type` -> plain `ALTER ... TYPE` (in-place retype, no cast).
- `alter_try_cast_column_type` -> `ALTER ... TYPE ... USING TRY_CAST(...)` (lenient: non-fitting values become NULL).
- `alter_cast_column_type` -> hard `USING CAST(...)` - **defined but never called** here.
- `add_column` -> `ADD COLUMN IF NOT EXISTS` (idempotent).

It narrows roughly 130 imported columns: `USMALLINT` for large year/count/weight fields (`DATAYEAR`, `BIRYR`, `DOB_YY`, `DBWT`, `DWGT_R`, `BMI_R`, `PWGT_R`), `UTINYINT` for small categorical/recode/age fields (including the race/Hispanic chains `MBRACE`/`MRACE`/`MRACEREC`/`MRACE31`/`MRACE6`/`MRACE15`/`MRACEIMP` and `ORMOTH`/`ORRACEM`/`UMHISP`/`MHISPX`/`MHISP_R`/`MRACEHISP`, plus father equivalents), `VARCHAR` for Y/N/U flag fields (`SEX`, `MAR_P`, `DMAR`, `AB_*`, `CA_*` incl. `CA_DOWN`/`CA_DOWNS`, `RF_*`, `LD_*`, `BFED`, `WIC`), and `FLOAT` for `BMI`. `TRY_CAST` is reserved for columns whose source values may be dirty or out of range (e.g. `DOB_MM`, `MAGER14`, `MRACEHISP`, `MAR`, `DMEDUC`, `MEDUC`, `FAGECOMB`, `DBWT`, `DWGT_R`, `NO_CONGEN`, `DPLURAL`, ...). It then `ADD`s 14 empty computed columns (`year`, `mage_c`, `mrace_c`, `mhisp_c`, `mracehisp_c`, `down_ind`, `ca_down_c`, and the seven `p_ds_lb_*` probability columns as `DOUBLE`) plus a `BIGINT id` column added with a bare `ADD COLUMN` (no `IF NOT EXISTS`).

*Second half - per-row derivations and compaction.* A sequence of in-place `UPDATE`s populates the computed columns (see [Derived columns](#derived-columns-stage-5)), joining the five CSVs on `year`. Finally, after `con.close()`, the script deletes `data/us_births.db`, `ATTACH`es both DBs and runs `COPY FROM DATABASE temp_db TO db` to compact into a fresh file, then `shutil.copy2` copies `us_births.db` back over `us_births_temp.db` so both files end identical and compacted.

**Notes/gotchas:**
- **CWD requirement:** the four `pd.read_csv` calls use `./`-relative paths and the CSVs must be present at the repo root, or the script raises `FileNotFoundError` mid-stage.
- Choice of `TRY_CAST` vs plain `ALTER TYPE` is load-bearing: `TRY_CAST` silently nulls non-fitting values (can mask data issues); plain retype errors if a stored value does not fit the narrower type.
- Target types are unsigned and small (`UTINYINT` caps at 255, `USMALLINT` at 65535); width choices assume NVSS recodes/sentinels stay in range.
- `id` is added without `IF NOT EXISTS`, so re-running stage 5 on a DB that already has `id` errors on that statement.
- The compaction deletes `data/us_births.db` first, so a crash between the unlink and the copy can leave only the temp DB.

## Derived columns (stage 5)

Each derivation is an in-place `UPDATE` executed **in order**; later columns depend on earlier ones, so reordering breaks correctness. The race/Hispanic precedence matches `previous/us-birth-certificates/data-preparation.md`; the underlying column-name constants live in `src/dspopulations_us_birth_certificates/variables.py`.

- **`id`** (`BIGINT`): `row_number() OVER ()::BIGINT` keyed on `rowid`, populated just before the `year` derivation.
- **`year`**: `COALESCE(dob_yy, datayear)` - `dob_yy` (2003 revision) preferred, `datayear` (1989 revision) fallback.
- **`ca_down_c`** (`VARCHAR`; value domain C/P/N/U): (1) if `COALESCE(ca_down, ca_downs) IS NOT NULL` -> `UPPER(COALESCE(ca_down, ca_downs))` (`ca_down` preferred); else (2) `uca_downs` 1->'C', 2->'N', 9->'U'; else (3) `downs` 1->'C', 2->'N', 9->'U'; else NULL. `downs=8` (not on certificate) is not matched and falls through to NULL.
- **`down_ind`** (0/1/NULL): `UPPER(ca_down_c) IN ('C','P')` -> 1; `='N'` -> 0; else `downs=1` -> 1, `downs=2` -> 0; else `uca_downs=1` -> 1, `uca_downs=2` -> 0; else NULL. `ca_down_c='P'` (pending) counts as a case; `ca_down_c='U'` is not caught in the first branch and falls through to the `downs`/`uca_downs` branches (and may end NULL).
- **`mage_c`**: `COALESCE(mager, dmage, mage36 + 13, mager41 + 13)` - `mager` (2004+ single-year) preferred, `dmage` (<=2002 single-year) next, then the `mage36` recode +13 (<=2002), then the `mager41` recode +13 (2003-only; same 41-category coding as `mage36`). The `mager41` fallback recovers 2003, which carries none of the first three (see [Known data-quality issues](#known-data-quality-issues)). `mage36`/`mager41` code 01 ("Under 15") maps via +13 to age 14 - a lower-bound approximation, immaterial above the lowest analytic age boundary (20).
- **`p_ds_lb_nt`**: Morris double-logistic in `mage_c`, `1 / (1 + exp(7.33 - 4.211 / (1 + exp(-0.2815 * (mage_c - 37.23)))))` - maternal-age probability of a DS live birth absent terminations.
- **`p_ds_lb_wt`**: per-year surveillance prevalence from `us-births-surveillance-prevalence-1989-2024.csv` (36 rows) loaded into DuckDB table `prevalence_year`, joined on `year`. Values rise from `0.001038` (1989); the trailing value `0.001324215` appears for **2018-2024** (the last 7 entries are identical - intentional carry-forward of the last estimated year).
- **`mrace_c`** (1-5): `mrace15` (1,2,3 keep; 4-14 -> 4; **15 "More than one race" -> 5**) -> else `mracerec` (1-4 keep) -> else `mbrace` (1-digit 1-4 keep; 2-digit 01-03 -> 1/2/3, 04-14 -> 4, **bridged-multiple 21-24 -> 5**; Puerto Rico 0 -> NULL) -> else `mrace` (1,2,3 keep; 4-78 -> 4) -> else NULL. Category **5 "More than one race"** is only identifiable from MRACE15=15 (2014+) and MBRACE 21-24 (2003-2013, unreachable); MRACEREC and the 1989-cert MRACE carry no multi-race code, so 1989-2013 multi-race is folded into single-race categories. Unknown/out-of-range -> NULL. MRACEREC/MRACE15 precede MBRACE, so the MBRACE branch is in practice only the fallback for Puerto Rico 2014-2019.
- **`mhisp_c`** (0-5): `mhisp_r` (0,1,2,3 keep; 4-5 -> 4; 9 -> 5) -> else `mhispx` (0,1,2,3 keep; 4-6 -> 4; 9 -> 5) -> else `umhisp` (0,1,2,3 keep; 4-5 -> 4; 9 -> 5) -> else `orracem` (1,2,3 keep; 6-8 -> 0 non-Hispanic; 4-5 -> 4; 9 -> 5) -> else NULL.
- **`mracehisp_c`** (1-6 or NULL): **reconstructed from `mhisp_c` + `mrace_c`** - deliberately not the raw NCHS `mracehisp` field, which is dual-coded across eras and absent pre-2003. `mhisp_c BETWEEN 1 AND 4` -> 5 (Hispanic); `mhisp_c = 5` (origin unknown) -> NULL (the row's race is discarded); non-Hispanic multi-race (`mrace_c = 5`) -> **6 (NH more than one race)**; `mhisp_c = 0` or NULL -> `mrace_c` (non-Hispanic race 1-4). Note the asymmetry: explicit "origin unknown" (5) drops to NULL, whereas *absent* origin (NULL) keeps the race as non-Hispanic. The selection model and `derive_recording_rates` route code 6 to the "Unknown" race cell (same as NULL) - giving multi-race its own group is a follow-up.
- **`p_ds_lb_wt_mage`**: from `us_births_est_prevalence_age` (maternal-age CSV) joined on `year`; per row `mage_c < 35` -> `p_ds_lb_wt_lt35_sv` else `p_ds_lb_wt_gte35_sv`. The CSV covers 1989-2018, so 2019-2024 rows stay NULL.
- **`p_ds_lb_nt_reduc`**: `p_ds_lb_nt * (1 - r.reduction)` from `reduction_rate_year` joined on `year` (renamed from `p_ds_lb_nt_reduc`; the multiplicand is `p_ds_lb_nt`, the Morris no-terminations risk - the old `_mage` suffix was a misnomer).
- **`ds_case_weight`** (`DOUBLE`): from `ds_case_weights` joined on `year`. When `down_ind=1`, selected by `mracehisp_c`: 1 -> `nhw`, 2 -> `nhb`, 3 -> `ai_an`, 4 -> `as_pi`, 5 -> `his`, else (`down_ind=1` with `mracehisp_c` NULL/other) -> `total`; otherwise 0 (non-cases and `down_ind != 1` get weight 0). Rows with `mracehisp_c=NULL` (origin unknown) fall to the `total` branch.

Of the seven declared `p_ds_lb_*` columns, only **four are actually populated** by stage-5 `UPDATE`s - `p_ds_lb_nt`, `p_ds_lb_wt`, `p_ds_lb_wt_mage`, and `p_ds_lb_nt_reduc`. The other three (`p_ds_lb_nt_mage`, `p_ds_lb_wt_ethn`, `p_ds_lb_nt_ethn`) are added as `DOUBLE` columns but have no `UPDATE`, so they remain NULL throughout (placeholders for downstream work - the ethnicity prevalence table is loaded but never joined). All computed columns are NULL between the two halves of stage 5, and per-year joins silently leave rows NULL when a year has no matching lookup row.

## Lookup tables

The five CSVs are read with `pd.read_csv(...).convert_dtypes()` (relative `./` path, repo root) and loaded into DuckDB tables. The four moved out of `previous/` carry a UTF-8 BOM that the pandas C parser strips (so the first column parses as `year`); the surveillance-prevalence CSV was written without a BOM.

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

## Provenance of model parameters

The probability and weight series are external statistical estimates, not NCHS codings; their in-repo provenance is currently thin and should be recorded (source, vintage, method):

- **`p_ds_lb_nt`** uses the Morris et al. maternal-age-specific Down-syndrome live-birth risk model (double-logistic; constants 7.33, 4.211, 0.2815, 37.23). The same formula is implemented in `src/dspopulations_us_birth_certificates/chance.py` - consider calling it from one place to avoid drift.
- **`p_ds_lb_wt`** (`us-births-surveillance-prevalence-1989-2024.csv`, table `prevalence_year`) is per-year surveillance prevalence. The **2018 value is carried forward unchanged through 2024** (seven identical `0.001324215` entries), and a precision discontinuity at 2014/2015 suggests two spliced source vintages.
- **`us-births-reduction-rates-1989-2024.csv`** is **linearly extrapolated for 2020-2024** (constant +0.005965/yr).
- **`us-births-estimated-prevalence-maternal-age-1989-2018.csv`** and **`us-births-estimated-prevalence-ethnicity-2000-2018.csv`** stop at 2018, so the age-/ethnicity-adjusted estimates are NULL beyond their coverage (and the ethnicity series is loaded but not yet joined).

These carry-forward / extrapolation assumptions are intentional stopgaps; they should be made explicit (a named constant and a comment) rather than buried in repeated literals.

## Side outputs (not part of the core chain)

These scripts form a separate side branch and are **not** part of the SAS -> parquet -> DuckDB chain.

### `make_all.py`

```bash
python scripts/make_all.py
```

`merge_years()` reads the 11 per-year parquets for **2014-2024** (hardcoded), applies `variables.set_imported_column_types(df)` to each (not `set_all_column_types`, which would `KeyError` because COMPUTED columns do not exist in per-year parquets yet), concatenates with `pd.concat(...)`, and writes `data/us_births_all.parquet` using `engine='fastparquet'`. The name is **misleading**: it builds only a 2014-2024 modelling subset, not the full pipeline. `pd.concat` does not reset the index, so the output carries non-unique per-year row indices.

### `export_spss.py`

```bash
python scripts/export_spss.py
```

`export_spss()` computes one run timestamp (`%Y%m%d_%H%M`) shared by all outputs, then for each source reads the parquet and writes an SPSS `.sav` via `pyreadstat.write_sav`, compressing it into a `.zip` (`ZipFile(..., 'x', ZIP_DEFLATED)`). Sources are the **2014-2022** per-year parquets plus `data/us_births_all.parquet` (10 in total - note no 2023/2024 per-year files, unlike `make_all.py`). It must run after `make_all.py` (it consumes `us_births_all.parquet`). The intermediate `.sav` is left on disk alongside the `.zip`; the `'x'` (exclusive-create) mode raises `FileExistsError` if a target `.zip` already exists (possible only on reruns within the same minute).

## Operational notes

- **Run order matters.** The five core stages are strictly sequential; each consumes the previous stage's output.
- **`combine_parquet` globs `data/*.parquet` indiscriminately.** Clean any derived parquet outputs (`us_births.parquet`, `us_births_all.parquet`) from `data/` before re-running stage 2, or they will be folded into the combined file.
- **`prepare_parquet` executes at import** (no `__main__` guard) - importing it triggers the full rewrite of `data/us_births.parquet`.
- **Stage 5 needs the repo root as CWD and the four CSVs present**, or it fails mid-run with `FileNotFoundError`.
- **Time and memory.** Measured end-to-end on the full 1989-2024 dataset (~142.9 M rows): import ~30-40 min (~50 s/year), combine ~30 s, prepare ~6 min, `duckdb_create` ~9 min, `duckdb_prepare` ~13 min (retypes + derivations + compaction) - roughly **~1 hour total**. `import_parquet` reads ~90 GB of SAS one year at a time, so available RAM must comfortably exceed the largest single year.
- `data/` is gitignored and DUA-restricted - never commit or publish any raw or derived microdata.

### Verifying source SAS files

NBER downloads have no checksum, so a **truncated `.sas7bdat`** is a real risk - an interrupted download leaves a short file that `import_parquet` only discovers mid-run as `ValueError: failed to read complete page from file (read N of M bytes)`. A complete sas7bdat is exactly `header_length + (page_count x page_length)` bytes, so a fast integrity check is:

```python
from pandas.io.sas.sas7bdat import SAS7BDATReader
r = SAS7BDATReader(path, encoding="latin-1")
ok = (os.path.getsize(path) - r.header_length) % r._page_length == 0  # 0 => complete
```

A truncated file can be completed with a **resumable HTTP Range download** (continue from the existing byte offset) rather than re-fetching the whole ~2.7 GB file; the NBER server honours `Range` (returns `206 Partial Content`).

## Known data-quality issues

- **2003 maternal age - recovered via `mager41`.** The 2003 natality file uses the intermediate 2003 schema and stores maternal age as `mager41` (plus `mager14`/`umagerpt`), not the `mager`/`dmage`/`mage36` of adjacent years. Before this was handled, `mage_c` - and everything derived from it (`p_ds_lb_nt`, `p_ds_lb_wt_mage`, `p_ds_lb_nt_reduc`) - was NULL for all ~4.1 M 2003 records. `mager41` carries single-year age in the **same 41-category coding as `mage36`** (1 = under 15 … 41 = 54; `umagerpt` is all-99/unknown in 2003), so `mager41` is now in `IMPORTED_VARS` (and the `prepare_parquet`/`duckdb_prepare` type maps) and `mage_c` falls back to `mager41 + 13` for 2003. After the fix, 2003 `mage_c` is populated for all 4,096,092 records (mean 27.4) and total `mage_c` nulls across 1989-2024 drop to 59.
- **Expected 2003 cert-transition nulls.** Many other columns are all-NULL for 2003 by design: 1989-cert items (`datayear`, `biryr`, `dmage`, `mage36`, `ormoth`, `orracem`, `dmar`, `downs`, father `dfage`/`orfath`/…) end after 2002, while the new 2003-cert items (`mager`, `fagecomb`, the `ab_*`/`ca_*`/`rf_*`/`ld_*` checkboxes) only phase in from 2004 as states adopt the revised certificate. Race/Hispanic and the Down-syndrome indicator **are** captured for 2003 (via `mracerec`/`mbrace` and `uca_downs`), so `mrace_c`, `mracehisp_c`, and `down_ind` are populated for 2003.

## Outputs reference

Key files produced under `data/` (all gitignored):

- `data/us_births_{year}.parquet` - per-year zstd parquet (stage 1)
- `data/us_births_combined.parquet` - all years concatenated (stage 2)
- `data/us_births.parquet` - type-constrained (stage 3)
- `data/us_births_temp.db` - intermediate DuckDB (stage 4; overwritten at the end of stage 5 with a copy of the final DB)
- `data/us_births.db` - **final** compacted DuckDB (stage 5)
- `data/us_births_all.parquet` - 2014-2024 modelling subset (side branch, `make_all.py`)

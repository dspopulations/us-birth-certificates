# AGENTS.md

This file provides guidance to agentic coding tools (Codex, Cursor, Aider, and similar) when working with code in this repository.

> **Keep in sync:** `CLAUDE.md`, `AGENTS.md`, and `.github/copilot-instructions.md` share the same body content. When you change one, update the other two so every assistant sees the same guidance.

## Project purpose

This repository hosts an exploratory study of factors associated with recorded births of babies with Down syndrome in US birth certificate data.

Read plans/readme.md to learn about project plans.

## Environment and commands

Python **3.14** via conda (env name: `dspop-us-birth-certificates`):

```bash
conda env create -f environment.yml   # create env
conda activate dspop-us-birth-certificates
```

The package itself is installed editable (`-e ./`) as part of `environment.yml`. `pyproject.toml` uses hatchling; version lives in `src/dspopulations_us_birth_certificates/__init__.py`. Import name is `dspopulations_us_birth_certificates` (distribution name `dspopulations-us-birth-certificates`).

- Lint: `ruff check` (targets `py314`, line-length 88; `E501` and `E741` intentionally ignored — long lines and single-letter math names like `l`, `X`, `y` are accepted).
- Format: `ruff format`.
- Tests: `pytest` (config in `pyproject.toml`: `testpaths = ["tests"]`, `-q`). The `tests/` directory does not yet exist — create it when adding tests.
- Spellcheck (markdown and `docs/**/*.qmd`): `npm run spellcheck`. Dictionary at `config/spellcheck/allow-en.txt`; language is **en-GB**.

## Notebooks

Jupytext pairing is configured: `formats = "ipynb,py:percent"`. `.ipynb` files are **gitignored** — only the paired `.py` percent-format files are committed. When creating or editing notebooks, keep the `.py` counterpart in sync.

Matplotlib style for notebooks: `notebook.mplstyle` at repo root.

## Repository layout

- `src/dspopulations_us_birth_certificates/` — the installable package. Ported utility modules live here: `variables.py`, `data_utils.py`, `ml_utils.py`, `plot_utils.py`, `stats_utils.py`, `repl_utils.py`, `chance.py`, `comparison.py`, `training.py`. New library code belongs here too.
- `scripts/` — standalone data-pipeline scripts (run from the repo root): `download_data.py`, `import_parquet.py`, `combine_parquet.py`, `prepare_parquet.py`, `duckdb_create.py`, `duckdb_prepare.py`, `make_all.py`, `export_spss.py`.
- `notebooks/` — jupytext-paired exploratory notebooks (both `.py:percent` and `.ipynb`; only the `.py` is committed).
- `previous/us-birth-certificates/` — historical artefacts kept as a reference: `data-preparation.md`, `readme.md`, derived CSV summaries, and the (gitignored) `output/`. Source code has been moved out to `src/`, `scripts/`, and `notebooks/`.
- `data/` — **gitignored**. Holds raw `.sas7bdat` files, NCHS user-guide PDFs, and derived `.parquet` / DuckDB files. Never commit anything from here.

## Data access and handling

- Raw natality microdata is governed by the [NCHS Data Use Agreement](https://www.cdc.gov/nchs/data_access/restrictions.htm). Do not publish raw records, row-level extracts, or any output that could enable re-identification.
- Download script pattern lives at `scripts/download_data.py` (pulls SAS files from `data.nber.org` and user guides from `ftp.cdc.gov`). It uses `truststore.inject_into_ssl()` to work around SSL verification issues.
- The canonical pipeline converts SAS → parquet (per-year) → DuckDB / combined parquet. See `scripts/{import_parquet,combine_parquet,duckdb_create,duckdb_prepare,prepare_parquet}.py`.

## Harmonising variables across years

NVSS codings change across years — this is the main source of non-obvious complexity in the pipeline. Before adding or modifying any variable-derivation code, consult `previous/us-birth-certificates/data-preparation.md`, which documents the canonical merge rules for (and `src/dspopulations_us_birth_certificates/variables.py` for the current Python implementation):

- **Race** (`MRACE` 1989–2013 → `MRACEREC` → `MBRACE` → `MRACE15`/`MRACE6` → combined `mrace_c`).
- **Hispanic origin** (`ORRACEM` 1989–2002 → `UMHISP`/`MRACEHISP` → `MHISP_R` → `MHISPX` → combined `mhisp_c`).

Preserve these merge definitions when porting logic — they encode editorial decisions that affect comparability across the full 1989–2024 series.

## Licensing

Dual-license repo — be aware when adding files:
- **Code** → AGPL-3.0-or-later (`LICENSE`). AGPL's network-service clause applies to any hosted deployment.
- **Docs / reports / papers** → CC BY 4.0 (expected at `docs/LICENSE`).
- **Data** → subject to NCHS DUA, *not* CC BY despite what `README.md` currently says for the `data/` directory.

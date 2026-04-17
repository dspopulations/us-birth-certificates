# GitHub Copilot instructions

This file provides guidance to GitHub Copilot (and Copilot Chat / Copilot coding agent) when working with code in this repository.

> **Keep in sync:** `CLAUDE.md`, `AGENTS.md`, and `.github/copilot-instructions.md` share the same body content. When you change one, update the other two so every assistant sees the same guidance.

## Project purpose

This repository hosts an exploratory study of factors associated with recorded births of babies with Down syndrome in US birth certificate data.

Analysis of trends in live births of babies with Down syndrome in the US, using the CDC/NCHS National Vital Statistics System (NVSS) natality files (1989–2024). Source data is `.sas7bdat` published by NBER; Down syndrome is coded via `DOWNS` (1989–2002), `UCA_DOWNS` (2003), then `CA_DOWNS` (2004+).

## Environment and commands

Python **3.14** via conda (env name: `dspop-us-birth-certificates`):

```bash
conda env create -f environment.yml   # create env
conda activate dspop-us-birth-certificates
```

The package itself is installed editable (`-e ./`) as part of `environment.yml`. `pyproject.toml` uses hatchling; version lives in `src/dspopulations-us-birth-certificates/__init__.py`.

- Lint: `ruff check` (targets `py314`, line-length 88; `E501` and `E741` intentionally ignored — long lines and single-letter math names like `l`, `X`, `y` are accepted).
- Format: `ruff format`.
- Tests: `pytest` (config in `pyproject.toml`: `testpaths = ["tests"]`, `-q`). The `tests/` directory does not yet exist — create it when adding tests.
- Spellcheck (markdown and `docs/**/*.qmd`): `npm run spellcheck`. Dictionary at `config/spellcheck/allow-en.txt`; language is **en-GB**.

## Notebooks

Jupytext pairing is configured: `formats = "ipynb,py:percent"`. `.ipynb` files are **gitignored** — only the paired `.py` percent-format files are committed. When creating or editing notebooks, keep the `.py` counterpart in sync.

Matplotlib style for notebooks: `notebook.mplstyle` at repo root.

## Repository layout

- `src/dspopulations-us-birth-certificates/` — the installable package. Currently a skeleton (only `__init__.py`); this is where new library code belongs.
- `previous/us-birth-certificates/` — the prior iteration of this project, kept as a **reference implementation**. It contains the working data pipeline, variable definitions, modelling notebooks, and utilities (`variables.py`, `data_utils.py`, `ml_utils.py`, `plot_utils.py`, `stats_utils.py`, `experiment_runner.py`, `training.py`, etc.). Treat this as source material to port/refactor into `src/` rather than code to modify in place. Its `output/` is gitignored.
- `data/` — **gitignored**. Holds raw `.sas7bdat` files, NCHS user-guide PDFs, and derived `.parquet` / DuckDB files. Never commit anything from here.

## Data access and handling

- Raw natality microdata is governed by the [NCHS Data Use Agreement](https://www.cdc.gov/nchs/data_access/restrictions.htm). Do not publish raw records, row-level extracts, or any output that could enable re-identification.
- Download script pattern lives at `previous/us-birth-certificates/download_data.py` (pulls SAS files from `data.nber.org` and user guides from `ftp.cdc.gov`). It uses `truststore.inject_into_ssl()` to work around SSL verification issues.
- The canonical pipeline converts SAS → parquet (per-year) → DuckDB / combined parquet. See `previous/us-birth-certificates/{import_parquet,combine_parquet,duckdb_create,duckdb_prepare,prepare_parquet}.py`.

## Harmonising variables across years

NVSS codings change across years — this is the main source of non-obvious complexity in the pipeline. Before adding or modifying any variable-derivation code, consult `previous/us-birth-certificates/data-preparation.md`, which documents the canonical merge rules for:

- **Race** (`MRACE` 1989–2013 → `MRACEREC` → `MBRACE` → `MRACE15`/`MRACE6` → combined `mrace_c`).
- **Hispanic origin** (`ORRACEM` 1989–2002 → `UMHISP`/`MRACEHISP` → `MHISP_R` → `MHISPX` → combined `mhisp_c`).

Preserve these merge definitions when porting logic — they encode editorial decisions that affect comparability across the full 1989–2024 series.

## Licensing

Dual-license repo — be aware when adding files:
- **Code** → AGPL-3.0-or-later (`LICENSE`). AGPL's network-service clause applies to any hosted deployment.
- **Docs / reports / papers** → CC BY 4.0 (expected at `docs/LICENSE`).
- **Data** → subject to NCHS DUA, *not* CC BY despite what `README.md` currently says for the `data/` directory.

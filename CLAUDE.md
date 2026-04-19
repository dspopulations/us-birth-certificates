# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Keep in sync:** `CLAUDE.md`, `AGENTS.md`, and `.github/copilot-instructions.md` share the same body content. When you change one, update the other two so every assistant sees the same guidance.

## Project purpose

This repository hosts an exploratory study of factors associated with recorded births of babies with Down syndrome in US birth certificate data.

@plans/readme.md

## Environment and commands

Python **3.14** via conda (env name: `dspop-us-birth-certificates`):

```bash
conda env create -f environment.yml   # create env
conda activate dspop-us-birth-certificates
```

The package itself is installed editable (`-e ./`) as part of `environment.yml`. `pyproject.toml` uses hatchling; version lives in `src/dspopulations_us_birth_certificates/__init__.py`. Import name is `dspopulations_us_birth_certificates` (distribution name `dspopulations-us-birth-certificates`).

- Lint: `ruff check`
- Format: `ruff format`
- Tests: `pytest` (config in `pyproject.toml`: `testpaths = ["tests"]`, `-q`). The `tests/` directory does not yet exist — create it when adding tests.
- Spellcheck (markdown and `docs/**/*.qmd`): `npm run spellcheck`. Dictionary at `config/spellcheck/allow-en.txt`; language is **en-GB**.

## Notebooks

Jupytext pairing is configured: `formats = "ipynb,py:percent"`. `.ipynb` files are **gitignored** — only the paired `.py` percent-format files are committed. When creating or editing notebooks, keep the `.py` counterpart in sync.

Matplotlib style for notebooks: `notebook.mplstyle` at repo root.

## Repository layout

- `src/dspopulations_us_birth_certificates/` — the installable package
- `scripts/` — standalone data-pipeline scripts (run from the repo root)
- `notebooks/` — jupytext-paired exploratory notebooks (both `.py:percent` and `.ipynb`; only the `.py` is committed)
- `previous/us-birth-certificates/` — historical artefacts kept as a reference
- `data/` — **gitignored**. Holds raw `.sas7bdat` files, NCHS user-guide PDFs, and derived `.parquet` / DuckDB files. Never commit anything from here.

## Data access and handling

- Raw natality microdata is governed by the [NCHS Data Use Agreement](https://www.cdc.gov/nchs/data_access/restrictions.htm). Do not publish raw records.
- Download script pattern lives at `scripts/download_data.py`
- The pipeline converts SAS → parquet (per-year) → DuckDB / combined parquet.

## Harmonising variables across years

NVSS codings change across years — this is the main source of non-obvious complexity in the pipeline. Before adding or modifying any variable-derivation code, consult `previous/us-birth-certificates/data-preparation.md` and `src/dspopulations_us_birth_certificates/variables.py`:

- **Race** (`MRACE` 1989–2013 → `MRACEREC` → `MBRACE` → `MRACE15`/`MRACE6` → combined `mrace_c`).
- **Hispanic origin** (`ORRACEM` 1989–2002 → `UMHISP`/`MRACEHISP` → `MHISP_R` → `MHISPX` → combined `mhisp_c`).

## Licensing

Dual-license repo — be aware when adding files:
- **Code** → AGPL-3.0-or-later (`LICENSE`). AGPL's network-service clause applies to any hosted deployment.
- **Docs / reports / papers** → CC BY 4.0 (expected at `docs/LICENSE`).
- **Data** → subject to NCHS DUA, *not* CC BY despite what `README.md` currently says for the `data/` directory.

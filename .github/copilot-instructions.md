# GitHub Copilot instructions

This file provides guidance to GitHub Copilot (and Copilot Chat / Copilot coding agent) when working with code in this repository.

> **Keep in sync:** `CLAUDE.md`, `AGENTS.md`, and `.github/copilot-instructions.md` share the same body content. When you change one, update the other two so every assistant sees the same guidance.

## Project purpose

This repository hosts an exploratory study of factors associated with recorded births of babies with Down syndrome in US birth certificate data.

Read plans/readme.md to learn about project plans.

## Disclosing AI-assisted contributions

Any content drafted with the help of an LLM-based AI tool **must be clearly labelled as AI-assisted**. This applies to **document drafts, pull requests, issues, and comments on pull requests or issues**. Prefix the content with a GitHub-style note callout naming the tool and model used, for example:

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

Substitute the actual tool and model you are using (for example `Codex`, `Cursor`, `Aider`, or `GitHub Copilot`). Keep the label at the very top of the draft, PR/issue body, or comment. Do not remove an existing disclosure label when editing AI-assisted content.

The `> [!NOTE]` alert syntax above is GitHub-flavoured Markdown and is the right form for PRs, issues, and their comments. **It does not render in Quarto.** For Quarto documents (`docs/**/*.qmd`), use a Quarto callout div instead, matching the existing callouts in those docs:

```markdown
::: {.callout-note title="AI-assisted"}
Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).
:::
```

## Environment and commands

Python **3.14** via conda (env name: `dspop-us-birth-certificates`):

```bash
conda env create -f environment.yml   # create env
conda activate dspop-us-birth-certificates
```

The package itself is installed editable (`-e ./`) as part of `environment.yml`. `pyproject.toml` uses hatchling; version lives in `src/dspopulations_us_birth_certificates/__init__.py`. Import name is `dspopulations_us_birth_certificates` (distribution name `dspopulations-us-birth-certificates`).

## Shared utilities (`dse_research_utils`)

Notebooks and scripts reference a shared external package (`dse_research_utils`) from the sibling [`research`](https://github.com/dseinternational/research) repository for environment setup, plot styling, and metadata reporting. Import paths start with `dse_research_utils.*`.

- `environment.yml` installs it editable from a relative path: `-e ../../dseinternational/research/src/python` (note the `../../` — this repo lives under `dspopulations/`, not `dseinternational/`). The sibling repo must be cloned alongside this one.
- Scripts call `dse_research_utils.environment.setup.init_script()` at the top of `main()` to apply the default matplotlib style.
- Notebooks call `dse_research_utils.environment.setup.init_workbook()` (style + environment summary) followed by `dse_research_utils.metadata.packages.report_package_versions(PACKAGE_LIST)` for reproducibility.
- Plotting code imports `dse_research_utils.plot.styles` and uses its `FIGSIZE_*`, `COLOUR_*`, `DPI_*` constants instead of hardcoded literals.
- The project-wide `PACKAGE_LIST` (used for version reporting) is re-exported from `dspopulations_us_birth_certificates`.
- `src/.../repl_utils.py` is a thin compatibility shim that delegates to the shared library — new code should import from `dse_research_utils` directly.

- Lint: `ruff check`
- Format: `ruff format`
- Tests: `pytest` (config in `pyproject.toml`: `testpaths = ["tests"]`, default `-q -m 'not slow'`). Tests marked `@pytest.mark.slow` fit real Bayesian models with enough draws to support posterior-quality assertions — invoke with `pytest -m slow` when you need to run them (locally, not in CI).
- Spellcheck (markdown and `docs/**/*.qmd`): `npm run spellcheck`. Dictionary at `config/spellcheck/allow-en.txt`; language is **en-GB**.

**Before creating a PR, always run both `ruff check src tests scripts` and `npm run spellcheck` and resolve any findings.** Fix real lint errors; for false-positive unknown-word flags from cspell, add the term to `config/spellcheck/allow-en.txt` rather than rewording the prose.

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

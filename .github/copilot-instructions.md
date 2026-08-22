# Repository assistant instructions

This file provides guidance to agentic coding tools (Codex, Cursor, Aider, and similar) when working with code in this repository.

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

Python **3.14** via [uv](https://docs.astral.sh/uv/). There is no conda layer: PyMC 6 compiles with the Numba backend by default, so no C toolchain or BLAS is needed and every package in the scientific stack ships a CPython 3.14 wheel.

```bash
uv sync          # create/refresh .venv from uv.lock (uv provisions Python from .python-version)
uv run pytest    # run anything inside that environment
```

Supported platforms are linux-x86_64, linux-aarch64, macOS-arm64 and win-amd64 (see `[tool.uv] environments` in `pyproject.toml`). **Windows contributors no longer need WSL.** Intel macOS is not supported — numba publishes no macOS x86-64 wheels. GPU acceleration remains an opt-in `jax[cuda]` overlay.

Two system-level prerequisites are not Python packages: the LLVM OpenMP runtime on macOS (`brew install libomp`), which the `lightgbm`/`xgboost` wheels link against, and the Graphviz `dot` binary for the notebook graph-plotting paths.

`uv sync` installs this package editable. Dependency layout: the scientific stack is inherited from `dse-research-utils` extras rather than restated (see the comment above `[project.dependencies]`), repo-only runtime needs go in `[project.dependencies]`, and tooling goes in the `dev` `[dependency-groups]` entry — do not reintroduce `[project.optional-dependencies]` or split test/modelling/data-preparation dependencies across several extras. `pyproject.toml` uses hatchling; version lives in `src/dspopulations_us_birth_certificates/__init__.py`. Import name is `dspopulations_us_birth_certificates` (distribution name `dspopulations-us-birth-certificates`).

`uv.lock` is committed. Regenerate it with `uv lock` whenever you change dependencies, and commit the result — CI runs `uv sync --locked`, which fails on a stale lockfile.

## Shared utilities (`dse_research_utils`)

Notebooks and scripts reference a shared external package (`dse_research_utils`) from the sibling [`research`](https://github.com/dseinternational/research) repository for environment setup, plot styling, and metadata reporting. Import paths start with `dse_research_utils.*`.

- `pyproject.toml` resolves it from the public git tag `v0.11.2` via `[tool.uv.sources]`, with the extras `[boosting,columnar,dependence,graphs,io,jax,notebook,tuning]`. Those extras are where the scientific stack comes from — add a package to the right extra upstream rather than re-declaring it here. A commented local-dev override in the same block points at a sibling checkout instead — `../../dseinternational/research/src/python` (note the `../../` — this repo lives under `dspopulations/`, not `dseinternational/`), which must be cloned alongside this one.
- Scripts call `dse_research_utils.environment.setup.init_script()` at the top of `main()` to apply the default matplotlib style.
- Notebooks call `dse_research_utils.environment.setup.init_workbook()` (style + environment summary) followed by `dse_research_utils.metadata.packages.report_package_versions(PACKAGE_LIST)` for reproducibility.
- Plotting code imports `dse_research_utils.plot.styles` and uses its `FIGSIZE_*`, `COLOUR_*`, `DPI_*` constants instead of hardcoded literals.
- The project-wide `PACKAGE_LIST` (used for version reporting) is re-exported from `dspopulations_us_birth_certificates`.
- `src/.../repl_utils.py` is a thin compatibility shim that delegates to the shared library — new code should import from `dse_research_utils` directly.

- Lint: `uv run ruff check`
- Format: `uv run ruff format`
- Tests: `uv run pytest` (config in `pyproject.toml`: `testpaths = ["tests"]`, default `-q -m 'not slow'`). Tests marked `@pytest.mark.slow` fit real Bayesian models with enough draws to support posterior-quality assertions — invoke with `uv run pytest -m slow` when you need to run them (locally, not in CI).
- Spellcheck (markdown and `docs/**/*.qmd`): `npm run spellcheck`. Dictionary at `config/spellcheck/allow-en.txt`; language is **en-GB**.

**Before creating a PR, always run both `uv run ruff check src tests scripts` and `npm run spellcheck` and resolve any findings.** Fix real lint errors; for false-positive unknown-word flags from cspell, add the term to `config/spellcheck/allow-en.txt` rather than rewording the prose.

## Notebooks

Jupytext pairing is configured: `formats = "ipynb,py:percent"`. `.ipynb` files are **gitignored** — only the paired `.py` percent-format files are committed. When creating or editing notebooks, keep the `.py` counterpart in sync.

Matplotlib style for notebooks: `notebook.mplstyle` at repo root.

## Repository layout

- `src/dspopulations_us_birth_certificates/` — the installable package
- `scripts/` — standalone data-pipeline scripts (run from the repo root)
- `notebooks/` — jupytext-paired exploratory notebooks (both `.py:percent` and `.ipynb`; only the `.py` is committed)
- `previous/us-birth-certificates/` — historical artefacts kept as a reference
- `data/` — mostly gitignored. Raw `.sas7bdat` files, NCHS user-guide PDFs, derived `.parquet` files, and DuckDB files must never be committed. Small derived/reference CSVs may be tracked when they are aggregate, non-record-level inputs to the analysis.

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

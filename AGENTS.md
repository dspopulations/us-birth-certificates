> [!NOTE]
> AI-assisted update by Claude Code (Opus 5.5).

> [!NOTE]
> AI-assisted update by Codex (GPT-6).

# Repository assistant instructions

Keep `AGENTS.md`, `CLAUDE.md` and `.github/copilot-instructions.md` identical when changing these instructions.

## Purpose and scope

This repository studies births recorded with Down syndrome in US birth certificate data. Read `plans/readme.md` for the aims, `docs/modelling-workflow.md` for commands and `notes/readme.md` for the status of research notes.

Distinguish recorded diagnoses, classifier-selected births and modelled expected counts. Do not describe `ds_pred_missing*` as verified missed cases. Estimates of prevalence, recording and prenatal reduction depend on external evidence and assumptions. Numerical validation does not establish those assumptions.

## Writing

Use plain language, short words, active voice and one idea per sentence. Explain technical terms and check claims against code or cited evidence. Avoid emotive judgements, filler, em dashes and unsupported claims. Use sentence-case headings and straight quotes. Keep dated evidence separate from current instructions. Remove completed task lists and superseded advice when their useful findings are recorded elsewhere.

## AI disclosure

Label AI-assisted document drafts, PRs, issues and their comments at the top. Name the actual tool and model. Keep existing disclosures when editing.

For Markdown, use:

```markdown
> [!NOTE]
> AI-assisted revision by Codex (GPT-6).
```

For Quarto, place a callout after the YAML front matter:

```markdown
::: {.callout-note title="AI-assisted"}
AI-assisted revision by Codex (GPT-6).
:::
```

## Environment and checks

Use Python 3.14 through uv from the repository root:

```bash
uv sync --locked
uv run ruff check src tests scripts
pnpm install --frozen-lockfile
pnpm run spellcheck
uv run pytest
```

Run both lint and spellcheck before creating a PR and resolve findings. Spellcheck uses en-GB and `config/spellcheck/allow-en.txt`. Add legitimate unknown terms to that dictionary. Do not rewrite accurate prose to evade false positives.

The default pytest run excludes `slow` tests. Run `uv run pytest -m slow` when changes require posterior-quality or parameter-recovery checks. Format changed Python files with `uv run ruff format`.

The configured platforms are Linux x86-64 and arm64, macOS arm64 and Windows AMD64. macOS needs `brew install libomp` for boosting libraries. Graph plots need Graphviz `dot`; report rendering needs Quarto. These are separate system dependencies.

Keep `uv.lock` committed. After changing dependencies, run `uv lock` and `uv sync --locked`. CI rejects a stale lockfile. Put repository runtime needs in `[project.dependencies]` and tooling in the `dev` dependency group. The scientific stack comes from `dse-research-utils` extras; add shared needs upstream rather than duplicating them here. Do not restore separate optional-dependency groups for testing, modelling or preparation.

The package uses hatchling. Its version is in `src/dspopulations_us_birth_certificates/__init__.py`. The import name is `dspopulations_us_birth_certificates`.

## Shared utilities and artefacts

`pyproject.toml` selects `dse-research-utils` from public tag `v0.18.0`. Its source block has a commented local override at `../../dseinternational/research/src/python`. See `docs/shared-utilities.md` for adapter contracts.

- Scripts call `dse_research_utils.environment.setup.init_script()` in `main()`.
- Notebooks use `init_workbook()` and report versions from the project `PACKAGE_LIST`.
- Plotting uses the shared `FIGSIZE_*` and `DPI_*` constants and the design-token colours.
- Use `CHART_COLOURS` for at most six series and sequential or diverging steps for ordered values. Draw text and reference lines in `TEXT_COLOUR` or `MUTED_TEXT_COLOUR`.
- Take a quantity's colour from `plot_colours.py`, so that it matches across figures.
- New code imports shared functions directly rather than through `repl_utils.py`.
- Keep project-specific decisions in local adapters. Do not duplicate shared implementations.
- Use `file_io.py` for JSON and CSV artefacts read by reports or other processes. It replaces files atomically and preserves the intended permissions.

## Data and variable rules

Read `docs/data-preparation.md`, `previous/us-birth-certificates/data-preparation.md` and `variables.py` before changing derivations. Variable names and codes change across years. Check source user guides and tests at those boundaries.

Run preparation scripts from the repository root. Raw SAS files, user-guide PDFs, Parquet files and DuckDB databases are gitignored and must not be committed. Small aggregate and reference CSVs may be tracked. Do not publish raw or derived record-level natality data; the NCHS Data Use Agreement applies.

## Notebooks and reports

Jupytext pairs notebooks as `ipynb,py:percent`. Only the `.py` files are committed; `.ipynb` files are gitignored. Keep local pairs in sync when editing notebooks. `notebook.mplstyle` defines the local notebook style.

Quarto files in `docs/` are templates. Fit and analysis scripts copy them into run directories containing the required artefacts. Render the copied file. `docs/report/` is a report scaffold, not a completed study report.

## Licences

- Code uses AGPL-3.0-or-later; see `LICENSE` and the package metadata.
- Documentation, reports and papers use CC BY 4.0, as declared in `README.md`.
- Data uses its source terms, including the NCHS Data Use Agreement. It is not covered by CC BY 4.0.

# Characteristics of babies born with Down syndrome -- an exploratory data analysis of US birth certificate data

> [!WARNING]
> This is work in progress. All data and models are preliminary.

**This repository hosts an exploratory study of factors associated with recorded births of babies with Down syndrome in US birth certificate data.**

Accurate assessments of the numbers and characteristics of babies born with Down syndrome is important for planning healthcare, education and social support services. They are also important for ascertaining the consequences of changing medical technologies and practices, social policies and attitudes, and individual behaviour over time, and for projecting future trends.

This study aims to provide updated estimates of the numbers of babies born with Down syndrome in the U.S. and to explore factors influencing births and how they may be changing over time.

Recorded Down syndrome births in these data are systematically under-reported, and not at random. Because recording depends on the same characteristics we want to measure, we do not attempt to identify *which* individual births were missed; instead we estimate the *number* of missed cases and its distribution at the population level, using a structural Bayesian selection model, and use the machine-learning strand to characterise the recording process rather than to recover individuals. See [current plans](./plans/readme.md) for the detailed aims and the methodological rationale.

## Analyses

A Bayesian selection model sits alongside the LightGBM classifier:

- **`dspopulations_us_birth_certificates.selection`** — three-stage
  selection model decomposing observed recording into baseline
  livebirth rate × screening/termination pass-through × birth-
  certificate sensitivity. Driver: `scripts/fit_selection_model.py`.
  Template: `docs/models/selection/index.qmd`. Design notes:
  [`notes/20260622-predictors-bayesian-model.md`](./notes/20260622-predictors-bayesian-model.md)
  (with supporting design notes under `notes/`).
  Variant-comparison aggregator: `scripts/compare_selection_variants.py`.

## Getting started

### Clone repository

```bash
git clone https://github.com/dspopulations/us-birth-certificates.git
```

### Prerequisites

#### Fitting models

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install it ([instructions](https://docs.astral.sh/uv/getting-started/installation/)), then create the environment from the repository root:

```bash
uv sync
```

uv provisions the Python interpreter itself from `.python-version` (**3.14**), resolves from the committed `uv.lock`, and installs this package editable. Prefix commands with `uv run` — for example `uv run pytest` or `uv run python scripts/fit_model.py`.

Supported platforms are Linux (x86-64 and arm64), Apple Silicon macOS, and native Windows. Intel macOS is not supported: numba publishes no macOS x86-64 wheels.

Two system-level prerequisites are not Python packages and are not installed by `uv sync`:

- **macOS only** — LLVM's OpenMP runtime, which the `lightgbm` and `xgboost` wheels link against: `brew install libomp`.
- **Graph plotting only** — the Graphviz `dot` binary, alongside the Python bindings: `brew install graphviz`, `apt install graphviz`, or `winget install Graphviz.Graphviz`.

#### Creating reports

TODO

## Data preparation

The pipeline that turns the raw NCHS/NVSS natality SAS microdata (1989–2024) into the analysis-ready `data/us_births.db` DuckDB database (and matching `data/us_births.parquet`) is documented in [docs/data-preparation.md](./docs/data-preparation.md). Source data is fetched with `scripts/download_data.py` and is subject to the [NCHS Data Use Agreement](https://www.cdc.gov/nchs/data_access/restrictions.htm). Raw records, NCHS user-guide PDFs, Parquet files, and DuckDB files are gitignored and must never be committed; small aggregate/reference CSVs under `data/` may be tracked when they are non-record-level inputs to the analysis.

## License

All source code in this repository is licensed under the GNU Affero General Public License v3.0 **(AGPL-3.0-only)**. See `LICENSE`.

Some other artifacts are licensed under other licenses:

- **Code**: GNU Affero General Public License v3.0 (AGPL-3.0) — see `LICENSE`.
- **Documentation, reports and papers**: Creative Commons Attribution 4.0 International (CC BY 4.0) — see `docs/LICENSE`.
- **Data**: subject to the original data source terms, including the NCHS Data Use Agreement for natality microdata. Data are not covered by the repository's code or documentation licences.

AGPL-3.0 requires that if you modify and run this software to provide a network service, you must offer the corresponding source code to users of that service.

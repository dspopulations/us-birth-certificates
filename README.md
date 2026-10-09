> [!NOTE]
> AI-assisted update by Codex (GPT-6).

# Down syndrome births in US birth certificate data

This repository studies the numbers and characteristics of babies recorded with Down syndrome in US birth certificates from 1989 to 2024. It also estimates expected population counts under explicit assumptions about prenatal reduction and certificate recording.

> [!WARNING]
> This study is in progress. Models and estimates are preliminary.

Birth certificates miss some Down syndrome diagnoses. Recording also varies across groups and clinical circumstances. A model trained on the recorded checkbox predicts that checkbox, which reflects both true prevalence and recording. Its high-scoring unrecorded births are not verified missed cases.

The Bayesian models estimate aggregate counts with external age-risk, surveillance and recording information. Their estimates depend on those sources and on model assumptions. They do not identify individual missed cases. See the [study aims](plans/readme.md) and [model inventory](docs/models/README.md).

## Documentation

- [Data preparation](docs/data-preparation.md) describes inputs, commands, derived variables and limits.
- [Modelling workflow](docs/modelling-workflow.md) describes fitting, diagnostics and reports.
- [Model inventory](docs/models/README.md) explains the DSP001–DSP010 accounting models.
- [Selection package](src/dspopulations_us_birth_certificates/selection/README.md) describes the separate three-stage model.
- [Research notes](notes/readme.md) separates historical fits, source checks and proposals from current guidance.
- [Shared utilities](docs/shared-utilities.md) records the dependency and local adapter contracts.
- [Assistant instructions](AGENTS.md) sets contribution rules.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/dspopulations/us-birth-certificates.git
cd us-birth-certificates
uv sync --locked
```

uv installs Python 3.14 from `.python-version` and the packages in `uv.lock`. It installs this project in editable mode. Run Python commands through `uv run` from the repository root.

The configured platforms are Linux x86-64 and arm64, Apple Silicon macOS, and Windows AMD64. Intel macOS is excluded from the lockfile's platform set.

On macOS, install the OpenMP runtime for LightGBM and XGBoost:

```bash
brew install libomp
```

Graph plots also require the Graphviz `dot` program. HTML reports require the [Quarto CLI](https://quarto.org/docs/get-started/). Neither program is installed by `uv sync`.

### Plot fonts

The shared style uses Noto Sans and Noto Sans Math. Install both on machines that render figures. Without them, matplotlib uses fallback fonts and the layout may differ.

```bash
# macOS
brew install --cask font-noto-sans font-noto-sans-math
# Debian or Ubuntu
sudo apt install fonts-noto-core
```

On Windows, install [Noto Sans](https://fonts.google.com/noto/specimen/Noto+Sans) and [Noto Sans Math](https://fonts.google.com/noto/specimen/Noto+Sans+Math). Restart notebook kernels after installing fonts. If matplotlib retains its old font list, remove `fontlist-*.json` from the directory returned by:

```bash
uv run python -c "import matplotlib; print(matplotlib.get_cachedir())"
```

The local `notebook.mplstyle` uses smaller text and narrower fonts than the shared script style.

## Checks

Install Node.js 24 and [pnpm](https://pnpm.io/installation). Use the pnpm version pinned in `package.json`. Before opening a PR, install the locked spellchecker packages and run:

```bash
uv run ruff check src tests scripts
pnpm install --frozen-lockfile
pnpm run spellcheck
```

Run `uv run pytest` for code changes. The default suite excludes slow model fits; use `uv run pytest -m slow` when the change requires those checks.

## Data and licences

Raw natality records are subject to the [NCHS Data Use Agreement](https://www.cdc.gov/nchs/data_access/restrictions.htm). Do not commit or publish raw records or derived record-level data. Small aggregate and reference CSVs may be tracked. See [data preparation](docs/data-preparation.md) for the pipeline.

- Code uses AGPL-3.0-or-later, as declared in `pyproject.toml` and `package.json`. The licence text is in [LICENSE](LICENSE).
- Documentation, reports and papers use [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Data retains its source terms. The code and documentation licences do not cover natality microdata.

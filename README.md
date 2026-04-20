# Characteristics of babies born with Down syndrome -- an exploratory data analysis of US birth certificate data

> [!WARNING]
> This is work in progress. All data and models are preliminary.

**This repository hosts an exploratory study of factors associated with recorded births of babies with Down syndrome in US birth certificate data.**

Accurate assessments of the numbers and characteristics of babies born with Down syndrome is important for planning healthcare, education and social support services. They are also important for ascertaining the consequences of changing medical technologies and practices, social policies and attitudes, and individual behaviour over time, and for projecting future trends.

This study aims to provide updated estimates of the numbers of babies born with Down syndrome in the U.S. and to explore factors influencing births and how they may be changing over time.

- [Current plans](./plans/readme.md)

## Analyses

Two Bayesian pipelines sit alongside the LightGBM classifier:

- **`dspopulations_us_birth_certificates.bayes`** — HSGP cell-model
  pipeline (time + maternal age smooths). Driver: `scripts/fit_bayes_model.py`.
  Template: `docs/models/m1-year-age/index.qmd`.
- **`dspopulations_us_birth_certificates.selection`** — three-stage
  selection model decomposing observed recording into baseline
  livebirth rate × screening/termination pass-through × birth-
  certificate sensitivity. Driver: `scripts/fit_selection_model.py`.
  Template: `docs/models/selection/index.qmd`. Design doc:
  [`plans/docs/bayesian_selection_model.md`](./plans/docs/bayesian_selection_model.md).
  Variant-comparison aggregator: `scripts/compare_selection_variants.py`.

## Getting started

### Clone repository

```bash
git clone https://github.com/dspopulations/us-birth-certificates.git
```

### Prerequisites

#### Fitting models

To fit models, a recent Python installation is required. Some of our dependencies are best installed from [conda-forge](https://conda-forge.org/), for which either [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [Miniforge](https://conda-forge.org/download/) is required.

Then, to install Python dependencies, from the repository root:

```bash
conda env update -f environment.yml
```

#### Creating reports

TODO

## License

All source code in this repository is licensed under the GNU Affero General Public License v3.0 **(AGPL-3.0-only)**. See `LICENSE`.

Some other artifacts are licensed under other licenses:

- **Code**: GNU Affero General Public License v3.0 (AGPL-3.0) — see `LICENSE`.
- **Documentation, reports and papers**: Creative Commons Attribution 4.0 International (CC BY 4.0) — see `docs/LICENSE`.
- **Data**: Creative Commons Attribution 4.0 International (CC BY 4.0) — see `data/LICENSE` for details.

AGPL-3.0 requires that if you modify and run this software to provide a network service, you must offer the corresponding source code to users of that service.

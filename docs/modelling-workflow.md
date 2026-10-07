> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# Modelling and report workflow

Run commands from the repository root after `uv sync --locked`. Prepare `data/us_births.db` as described in [data preparation](data-preparation.md). These commands can require large amounts of memory and time; profiles set defaults, not guarantees of convergence.

## Recorded-status classifiers

LightGBM model definitions live in `src/dspopulations_us_birth_certificates/models/`. The `usbc10` family includes demographic and clinical features; `usbc11` uses clinical features and maternal age. Variants with `_cn` train on confirmed cases and omit pending cases. Other variants count confirmed and pending as positive.

```bash
uv run python scripts/fit_model.py --model-id usbc10_m1 --profile reporting --render
uv run python scripts/tune_model.py --help
uv run python scripts/compare_variants.py --help
```

The fit script uses a deterministic stratified training/validation split. When tuning is enabled, tuning, early stopping and reported validation metrics use the same split. Those metrics are tuning diagnostics, not performance on an untouched test set. The old refactor proposal for five-fold cross-validation is not implemented.

The CLI writes to `output/fit_model_<profile>/<timestamp>/` by default when a profile is supplied, or `output/fit_model/<timestamp>/` otherwise. `--output-dir` overrides the location. Outputs include configuration, a manifest, metrics, the fitted model, diagnostic tables and plots, and a copied Quarto template.

`--write-predictions` changes the DuckDB database. It writes scores and quota-based `ds_pred_missing*` flags for the selected model. A flag identifies a selected high-scoring birth, not a validated missed diagnosis. The quota makes the selected count depend on the recorded count.

## Aggregate Bayesian models

The [model inventory](models/README.md) describes DSP001–DSP010 and their assumptions.

```bash
uv run python scripts/fit_core_reduction_model.py DSP004 --profile reporting --render
uv run python scripts/fit_core_reduction_model.py DSP008 --years 2004-2024 --profile reporting --render
uv run python scripts/fit_core_reduction_model.py --help
```

DSP007–DSP010 require the extracted surveillance anchor. DSP010 also requires the anomaly panel and its condition table. See the [source extraction note](../notes/20260803-degraaf-surveillance-workbook-extraction.md) and [panel note](../notes/20260804-dsp010-anomaly-panel-recording-factor.md) for source requirements.

Core runs write to `output/selection_core_reduction/<model_id>/<timestamp>/`. Inspect `validation.json`, `manifest.json`, the saved configuration and predictive checks before using estimates. An 89% interval on an expected count does not describe all uncertainty in the unknown realised count. Old fits without the current provenance and validation remain unvalidated.

```bash
uv run python scripts/audit_anchored_chain_health.py --strict
uv run python scripts/compare_core_reduction_models.py --help
uv run python scripts/compare_core_reduction_sensitivities.py --help
```

Model comparisons must use compatible cohorts, likelihoods and external-data assumptions. Do not pool draws from calibration scenarios or label their outer range as a credible interval. September changes to the surveillance likelihood and calibration solver require new fits; see the [code-review fixes](../notes/20260905-dsp-code-review-fixes.md).

## Three-stage selection model

This is a separate model family in the same package, not a DSP identifier. It splits the recorded probability into baseline livebirth risk, screening/termination pass-through and recording sensitivity. The [package README](../src/dspopulations_us_birth_certificates/selection/README.md) documents its API and priors.

```bash
uv run python scripts/fit_selection_model.py --variant C --spec full --profile reporting --render
uv run python scripts/compare_selection_variants.py --help
```

## Descriptive and classifier comparisons

```bash
uv run python scripts/analyse_descriptive.py --render
uv run python scripts/analyse_predicted.py --years 2016-2024 --render
uv run python scripts/compare_predicted.py --help
```

The descriptive report queries recorded births. The predicted and comparison reports describe classifier-selected cohorts. Their distributions cannot be treated as the characteristics of all missed Down syndrome births.

## Report rendering

Install the Quarto CLI separately. `--render` copies the appropriate template into the run directory and renders it with the run's tables and configuration. To render a copied template again:

```bash
uv run quarto render output/<run-directory>/index.qmd --to html
```

Replace the example path with the actual run directory. Templates in `docs/` do not contain study results by themselves. `docs/report/` currently contains only an overview scaffold; it does not assemble the run reports into a complete book.

The data queries and calculations are in the Python analysis modules. Preserve saved run configurations, manifests and source hashes when reviewing historical reports.

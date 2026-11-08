> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# Modelling and report workflow

Run commands from the repository root after `uv sync --locked`. Prepare `data/us_births.db` as described in [data preparation](data-preparation.md). These commands can require large amounts of memory and time; profiles set defaults, not guarantees of convergence.

The [CDC text import](cdc-text-import.md) builds annual candidates and a combined 1989–2025 database at `data/staging/natality/us_births.db`. The importer does not publish that candidate. Use a separate analysis copy when writing predictions, since a write invalidates the accepted candidate hash. For read-only candidate checks, use `--duckdb-path data/staging/natality/us_births.db` with `fit_model.py` without `--write-predictions`.

The [2025 reference review](../notes/20261106-2025-reference-review-and-classifiers.md) distinguishes classifier inputs from external population-model inputs. Missing 2025 surveillance, reduction and recording references still prevent complete expected-count analyses. The recorded-status classifiers do not use those references.

## Output locations

Set `DSPOP_OUTPUT_ROOT` to move model outputs to a scratch disk:

```bash
export DSPOP_OUTPUT_ROOT=/mnt/scratch/usbc-output
uv run python scripts/fit_selection_model.py --variant C --spec full --profile reporting
```

This writes the run to `/mnt/scratch/usbc-output/selection/C/full/<timestamp>/`. An unset or empty variable keeps the existing `output/` default. Relative roots are relative to the working directory. The resolver expands `~` in the environment variable and in the new root options.

The fitting commands (`fit_model.py`, `fit_selection_model.py` and `fit_core_reduction_model.py`) accept `--output-root` to override the environment variable. Their `--output-dir` option still selects an exact run directory and takes precedence over both. Subdirectories remain the same:

| Command | Path below the resolved root |
| --- | --- |
| `fit_model.py` | `fit_model/<timestamp>/`, or `fit_model_<profile>/<timestamp>/` with a profile |
| `tune_model.py` | `tuning/<model_id>/` |
| `fit_selection_model.py` | `selection/<variant>/<spec>/<timestamp>/` |
| `fit_core_reduction_model.py` | `selection_core_reduction/<model_id>/<timestamp>/` |
| `run_all_selection_variants.py` | Selection runs as above, with logs in `selection/_run_logs/` |

The batch command accepts `--output-root` and checks that same root when using `--skip-existing`. Selection analysis scripts that call `latest_fit_dir()` also discover runs under the environment root. An explicit `latest_fit_dir(root=...)` still names the selection directory itself.

Existing command-specific options retain their meanings. For `tune_model.py`, `--output-root` is the tuning directory itself, so the command appends only `<model_id>/`. For `compare_selection_variants.py`, `--root` is the selection directory itself; otherwise `--output-root` or the environment root supplies its `selection/` directory. Comparison results go alongside those fits unless `--output-dir` selects an exact directory. For `compare_variants.py --by-model-id`, `--models-root` takes precedence over `<resolved root>/models/`; `--output` continues to select the comparison CSV, with `comparison.csv` in the working directory as its default.

Commands print the selected locations. Fit manifests or run configurations record the absolute run directory; tuning also records its absolute root. Copied Quarto reports read their artefacts relative to the run directory, so they can be rendered outside the repository's `output/` tree.

## Recorded-status classifiers

LightGBM model definitions live in `src/dspopulations_us_birth_certificates/models/`. The `usbc10` family includes demographic and clinical features; `usbc11` uses clinical features and maternal age. Variants with `_cn` train on confirmed cases and omit pending cases. Other variants count confirmed and pending as positive.

The four current variants (`usbc10_m1`, `usbc11_m1` and their `_cn` counterparts) use 2016–2025. Historical variants retain 2016–2024. Named models take their year range from the definition; `--start-year` and `--end-year` apply only to fits without `--model-id`.

```bash
uv run python scripts/fit_model.py --model-id usbc10_m1 --profile reporting --render
uv run python scripts/tune_model.py --help
uv run python scripts/compare_variants.py --help
```

The fit script uses a deterministic stratified training/validation split. When tuning is enabled, tuning, early stopping and reported validation metrics use the same split. Those metrics are tuning diagnostics, not performance on an untouched test set.

Use the separate nested evaluation command for a fresh training procedure. It starts with all original USBC10 candidate features and ignores saved tuned parameters and historical feature pruning. Each outer test fold stays out of parameter tuning, early stopping and category learning. Inner folds choose the parameters and number of boosting rounds. The final model in each outer fold uses all its training rows and that fixed number of rounds.

```bash
uv run python scripts/evaluate_classifier.py --start-year 2025 --end-year 2025
uv run python scripts/evaluate_classifier.py --label confirmed --start-year 2025 --end-year 2025
```

The defaults use five outer folds, three inner folds and 20 fresh trials per outer fold. Outputs go to `nested_classifier/<label>/<timestamp>/` under the resolved output root. `--output-dir` selects an exact new directory. The command saves settings, trial histories and aggregate metrics without writing predictions to the database. Random folds estimate performance within the selected period. They do not assess transport to future years or validate past M1 fits. See the [November evaluation and sensitivity evidence](../notes/20261108-model-review-evidence.md) for the completed, smaller search on 2025 records.

The CLI writes to `output/fit_model_<profile>/<timestamp>/` by default when a profile is supplied, or `output/fit_model/<timestamp>/` otherwise. `--output-dir` overrides the location. Outputs include configuration, a manifest, metrics, the fitted model, diagnostic tables and plots, and a copied Quarto template.

To refit with saved parameters, pass `--no-optimize --prior-best-params <params.json>` and the intended `--model-id`. The parameters file must match that model. The 2025 refit retains the existing features and parameters while expanding the fitted cohort; it does not reuse the earlier fitted trees.

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
uv run python scripts/run_selection_sensitivity.py
```

The sensitivity command fits the full three-stage model against the same recorded counts in every scenario. It changes baseline prevalence, recording, termination or false positives one family at a time. These stress tests are specified in `selection/sensitivity.py`; they are not confidence ranges from external studies. Defaults use four chains with 1,500 tuning and 1,500 retained draws per chain. Results go to `selection_sensitivity/<timestamp>/` under the resolved output root. `--scenarios` selects named cases and `--output-dir` selects an exact new directory.

Each case saves numerical validation, annual and total estimates, replicated-count checks and recording-anchor diagnostics. The command combines cells with identical model probabilities, which leaves the posterior unchanged. Clinical-stratum checks need the original cells and are not part of this study. A failed numerical check remains visible and gives a non-zero exit status after all cases finish. Read the [evidence note](../notes/20261108-model-review-evidence.md) before interpreting the estimates.

## Descriptive and classifier comparisons

For the reviewed 2018–2025 trends in confirmed and pending Down syndrome
records by maternal age and race and ethnicity, run:

```bash
uv run python scripts/analyse_recorded_ds_trends.py
```

The command reads the database without changing it and writes a new run under
`output/recorded_ds_age_dependent/`. It fits separate age-dependent models for
the two statuses, simulates the group-difference tests, and saves aggregate
tables and PNG/SVG figures. `--counts-csv` accepts the retained aggregate
counts in `notes/data/20261108-recorded-ds-trends/aggregate_counts.csv` instead
of querying the database. `--output-dir` selects an exact new directory.
Read the [research note](../notes/20261108-recorded-ds-age-dependent-trends.md)
for the exploratory findings and limits. These models describe recorded
status, not true prevalence or recording sensitivity.

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

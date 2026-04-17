# Refactor plan — `notebooks/00010-predictors-10-c.py`

Status: **in progress** (tracked in https://github.com/dspopulations/us-birth-certificates/issues — to be filed).
Last updated: 2026-04-17.

This document captures the agreed approach for turning
`notebooks/00010-predictors-10-c.py` into a reusable, testable, reproducible
modelling pipeline with Quarto reporting. Patterns and naming follow the
sister repositories `V:/dev/dseinternational/language-reading-predictors`
(closest match — LightGBM + SHAP + Quarto, declarative model definitions)
and `V:/dev/dseinternational/vocabulary-growth` (pipeline orchestration,
CLI entry points), with generic utilities layered over
`V:/dev/dseinternational/research` (`dse_research_utils`) where it makes
sense.

## Goals

- **Reusability** — one pipeline class; model variants expressed as declarative
  dataclasses with chained feature-selection history.
- **Agentic coding** — small files with clear single responsibilities; every
  pipeline step callable from a notebook or a test.
- **Testing & CI** — synthetic fixture exercises the pipeline in under a
  minute; ruff + pytest + cspell on every PR.
- **Feature selection** — captured as data (`SelectionStep` entries), not
  ad-hoc code in a notebook; selection provenance serialised with every run.
- **Model comparison** — paired per-fold comparison of classification
  metrics (AP, log-loss, Brier, P@K) across variants.
- **Documentation** — Quarto reports per run, composite book for the paper;
  `docs/` gets a proper home for reporting.
- **Reproducibility** — every run writes a manifest (git SHA, env, package
  versions, config snapshot, input counts, seed) and a full artefact bundle
  (model, predictions, metrics, plots, report) under
  `output/models/<model_id>-<run_config>/`.

## Decisions on open questions

1. **Cross-validation.** Move from the single stratified 80/20 split to
   5-fold `StratifiedKFold` now. Given positive rate ≈ 1:800, per-fold AP
   is noisy; paired per-fold comparison across variants is what makes
   the feature-selection decisions defensible. A single held-out split is
   retained only as a final "reporting" evaluation after the variant has
   been chosen.
2. **Full-data prediction + DuckDB writeback.** Split out into
   `scripts/write_predictions.py` — takes a model run directory and upserts
   predictions by `id` in a transaction. Keeps the fit pipeline pure and
   lets us rescore prior years without retraining.

## Target layout

```
src/dspopulations_us_birth_certificates/
├─ variables.py                     (existing — schema; extend with DEFAULT_PREDICTORS group)
├─ data_utils.py                    (existing — load_predictors_data)
├─ ml_utils.py                      (existing — extend: stratified fold helper, scorer registry)
├─ plot_utils.py                    (existing)
├─ stats_utils.py                   (existing)
├─ manifest.py                      (new — write_manifest: git SHA, env, versions, config)
├─ reporting.py                     (new — render_quarto_report wrapper)
├─ tuning.py                        (new — run_optuna_study; writes best_params.json + trials.csv)
├─ models/
│  ├─ __init__.py                   (exposes MODELS registry)
│  ├─ common.py                     (RunConfig, ModelConfig, ModelFitContext,
│  │                                 SelectionStep, ShapScatterSpec)
│  ├─ base_model.py                 (ModelDefinition base; __init_subclass__ auto-registers;
│  │                                 selection_steps chained through MRO)
│  ├─ base_pipeline.py              (EstimatorPipeline abstract: prepare → split →
│  │                                 train → metrics → importance → SHAP → calibration →
│  │                                 plots → save → report)
│  ├─ lgbm_pipeline.py              (LGBMClassifierPipeline)
│  └─ usbc10.py                     (USBC10_M0 base + USBC10_M1, USBC10_M2 variants)
└─ explain/
   ├─ shap_analysis.py              (TreeExplainer wrapper; bar / beeswarm / scatter helpers)
   └─ calibration.py                (tail_calibration_table, precision_recall_at_k)

scripts/
├─ fit_model.py                     (argparse: model_id [all] --config {dev,test,reporting}
│                                     [--render] [--write-predictions])
├─ tune_model.py                    (argparse: model_id --n-trials N --timeout S)
├─ compare_variants.py              (reads per-variant cv_scores.csv; paired comparison)
├─ write_predictions.py             (upserts a trained model's full-data predictions into DuckDB)
└─ …existing pipeline scripts…

notebooks/
├─ 00010-predictors-10-c.py         (shrinks to ~30 lines — demo/smoke usage of USBC10_M2)
└─ …

docs/
├─ refactor-plan.md                 (this file)
├─ models/usbc10/index.qmd          (template — pure lookup from run output_dir)
└─ report/                          (book-format Quarto project, execute.freeze: true)

tests/
├─ conftest.py                      (synthetic 5k-row fixture with known base rate)
├─ test_variables.py                (mrace_c/mhisp_c harmonisation at cross-coding years)
├─ test_models_registry.py          (unique ids; SelectionStep shape; variant_of resolves)
├─ test_lgbm_pipeline_smoke.py      (fit USBC10_M0 at "dev"; assert artefacts land)
└─ test_calibration.py              (P@K, tail calibration edge cases)

.github/workflows/
└─ ci.yml                           (ruff + pytest + cspell; opt-in smoke-fit job)
```

## Data contracts

### `ModelDefinition` class attributes (declarative, immutable)
- `model_id: str`, `variant_of: str | None`
- `target_var`, `numeric_features`, `categorical_features`, `removed_features`
- `params`, `base_params`, `train_config`
- `year_range: tuple[int, int]`, `include_unknown: bool`
- `selection_steps: list[SelectionStep]` — date, rationale, features_removed,
  metrics_before, metrics_after (populated from the parent variant's manifest)
- `shap_scatter_specs: list[ShapScatterSpec]` — `(x, colour_by, description)`
  so the ad-hoc SHAP scatter calls become data, not code
- `notes: str`

### `RunConfig` presets
| name       | n_trials | num_boost_round | early_stop | cv_splits | shap            |
|------------|----------|-----------------|------------|-----------|-----------------|
| dev        |       10 |             500 |         50 |         3 | skip            |
| test       |       50 |           10000 |        200 |         5 | subsample 5000  |
| reporting  |      200 |           50000 |        200 |         5 | full            |

### `ModelFitContext` (mutable; threaded through pipeline steps)
Carries `config`, `run_config`, `X/y_train_full`, per-fold models, final-fit
model, `best_iteration`, validation predictions, metrics, permutation
importance, SHAP explanation, output directory.

## Per-run output layout

```
output/models/<model_id>-<run_config>/
├─ manifest.json
├─ config.json
├─ metrics.json
├─ cv_scores.csv
├─ model.txt
├─ predictions_valid.parquet
├─ feature_importance_gain.csv
├─ permutation_importance.csv
├─ shap_importance.csv
├─ calibration_tail.csv
├─ precision_recall_at_k.csv
├─ plots/
│   ├─ roc.{png,svg}  pr.{png,svg}  perm_importance.{png,svg}
│   ├─ dendrogram.{png,svg}  correlation_heatmap.{png,svg}
│   ├─ shap_bar.{png,svg}  shap_beeswarm.{png,svg}
│   └─ shap_scatter_<x>_by_<colour>.{png,svg}
└─ index.qmd                       (rendered to report.html when --render)
```

## Migration sequence

Each step is a reviewable PR on its own branch; later steps may stack on
earlier ones while reviews are in flight.

1. **Scaffolding (this PR)** — empty module skeletons with typed public API,
   `tests/` with skipped stubs, `docs/` Quarto placeholders, CI workflow,
   `.gitignore` updates. No behaviour change to existing code.
2. **Lift inline helpers** — move `precision_recall_at_k`,
   `plot_precision_recall_at_k`, `tail_calibration_table` out of the
   notebook into `explain/calibration.py`; unit test each. Collapse the
   repeated SHAP plotting loops into `explain/shap_analysis.py`.
   Notebook still runs end-to-end, ~300 lines shorter.
3. **Config layer** — `RunConfig`, `ModelConfig`, `ModelFitContext`,
   `SelectionStep`, `ShapScatterSpec` in `models/common.py`.
4. **Pipeline** — `EstimatorPipeline`, `LGBMClassifierPipeline`; wire
   the notebook to instantiate the pipeline for Model 0. Smoke test green.
5. **Variants** — port Model 1 and Model 2 as `USBC10_M1(USBC10_M0)` and
   `USBC10_M2(USBC10_M1)` with `SelectionStep` entries. Notebook collapses
   to ~30 lines.
6. **Tuning + comparison scripts** — `tune_model.py`, `compare_variants.py`.
7. **Quarto templates + `reporting.py`** — first render of
   `docs/models/usbc10/index.qmd` from a run's artefacts.
8. **Manifest + predictions writeback** — `manifest.py`,
   `scripts/write_predictions.py`; retire the inline DuckDB code.
9. **CI smoke-fit job** — enable the fit-on-fixture job.
10. **Upstream candidates** — open an issue against
    `V:/dev/dseinternational/research` proposing
    `base_model.py`, `base_pipeline.py`, `manifest.py`,
    `reporting.py::render_quarto_report`, and classification plotting
    helpers as additions to `dse_research_utils`.

## Shared vs project-local code

**Upstream candidates** (generic across studies — move when stable):
- `models/base_model.py` — `ModelDefinition` + `SelectionStep` + registry
- `models/base_pipeline.py` — `EstimatorPipeline`
- `manifest.py` — reproducibility manifest writer
- `reporting.py::render_quarto_report` — Quarto subprocess helper
- Classification plotting (ROC / PR / P@K) currently in `plot_utils.py`

**Project-local** (study-specific):
- `variables.py`, `data_utils.py` — NVSS schema and harmonisation
- `models/usbc10.py` — the model definitions themselves
- `models/lgbm_pipeline.py` — until another project wants to share it

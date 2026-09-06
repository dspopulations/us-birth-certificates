> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Shared utility compatibility with v0.13.0

This records the consumer checks for [issue #111](https://github.com/dspopulations/us-birth-certificates/issues/111), completed on 6 September 2026. The audit used the [tagged migration guide](https://github.com/dseinternational/research/blob/v0.13.0/docs/migrating-to-0.13.md) and the upstream source diff from v0.12.5 to v0.13.0.

## Dependency and installation

`pyproject.toml` selects the public v0.13.0 tag. `uv.lock` resolves it to commit `458cc41b1dc33f4c0204919253ac92251c61b2bb`. The existing extras are unchanged. No shared dependency floors were copied into this repository, and no other package versions changed in the lockfile.

`uv lock` and `uv sync --locked` succeeded on macOS arm64 with Python 3.14. The installed distribution reports version `0.13.0`; its `direct_url.json` records the same commit and requested tag. The three repository assistant instruction files now name the new tag.

## Call-site audit

The audit searched source, scripts, paired notebook Python files and Quarto documents for shared imports and the changed helper names. An additional Python syntax-tree scan resolved the direct shared imports against the installed package.

| Shared utility | Consumer and check |
| --- | --- |
| Console primitives and key/value tables | `src/dspopulations_us_birth_certificates/cli_output.py` delegates banners, sections, formatting and tables to the library. The new test captures the real shared console and checks headings, data counts, parameters and metrics. The changed upstream `dataframe_table` is not called here. |
| Environment initialisation | Scripts call `init_script`; notebooks use `repl_utils.print_environment_info`, which calls `init_workbook`. Tests execute both paths and check that the shared font size replaces an initial test value. |
| Package metadata | The notebook shim reports every entry in `PACKAGE_LIST`. The test checks that each installed version appears and that none is reported as missing. Modelling scripts use the same shared metadata module. |
| Plot styles | Descriptive and predicted analyses, selection diagnostics and reporting scripts use shared sizes, colours and palettes. The new descriptive test runs an in-memory DuckDB query on six synthetic rows and writes three sets of PNG, SVG and CSV files. It checks the totals and plotted rates. The rate figure was also inspected visually. |
| Local report generation | Existing tests exercise the selection diagnostic CLI and its PNG/SVG/CSV outputs, report-template handling and the LightGBM pipeline with synthetic fixtures. These passed in the full suite. |

The syntax-tree scan encountered an existing malformed string at line 33 of `notebooks/00011-predictors-11.py`, also present in the starting commit. That file was inspected as text and has no direct shared-library imports. Its legacy `experiment_runner` path was not executed. Full historical notebooks were not run because they require their data and analysis environments; this audit checks their shared imports and the current notebook setup shim.

## Migration changes that do not apply

| Upstream change | Evidence in this consumer |
| --- | --- |
| HSGP boundary, basis size and centring | No calls to `build_hsgp_1d`, `build_tau_modifier` or HSGP constructors were found in source, scripts or notebooks. There is no shared HSGP design to migrate or refit. |
| Diagnostic null extrema and `scan_completed` | This repository does not read the shared `diagnostics_summary.json` format or call shared sampling-quality helpers. `selection/diagnostics.py` obtains ArviZ summaries and applies its own health checks; consumers read local CSV summaries and validation records. |
| LOO thresholds, non-finite counts and relative efficiency | No calls to `loo_summary_row` or `reff_or_default` were found. The core-model comparison explicitly records that it does not compare raw LOO or WAIC values. |
| Upload URLs and raw `relative_paths` | No shared storage imports or `BlobUploadResult` consumers were found. There is no local upload wrapper to migrate. |
| Closing plot collections | No calls to `save_plotcollection` were found. The local descriptive path uses `plot_utils.save_fig` and closes its own figure. The new test also checks that an unrelated figure remains open. |
| Regression, ROPE, interval and inverse-logit contracts | No imports of the changed shared statistical helpers were found. The project uses local statistics and interval modules. Symbolic model expressions call `pm.math.invlogit`, not the changed numeric shared transform. |
| Other shared plotting and report-data changes | No imports of shared Gaussian-process plots, graph plots or report-data helpers were found. Local graph and report code does not delegate to those changed modules. |

## Verification and limits

The checks ran against the locked v0.13.0 installation.

- `MPLBACKEND=Agg uv run pytest tests/test_shared_utilities.py -ra`: 3 passed.
- `MPLBACKEND=Agg uv run pytest -ra`: 329 passed, 12 deselected, no skipped tests, 114 warnings.
- `uv run ruff check src tests scripts`: passed.
- `uv run ruff format --check tests/test_shared_utilities.py`: passed.
- `npm run spellcheck`: passed.

The 12 deselected cases carry the `slow` marker. They cover parameter recovery and convergence in `test_selection_parameter_recovery.py`, and four joint simulation fits in `test_core_review_regressions.py`. They were not run because no model construction, sampling or statistical calculation path changed in this upgrade. The default suite still ran model construction, small fits and diagnostic rendering.

The suite emitted warnings from scientific libraries and plotting, including deprecated SHAP colour methods, layout changes and the treatment of PyMC potentials during prior predictive sampling. Passing these compatibility checks does not establish the statistical quality of saved research fits. Linux and Windows installation were not run locally; the existing CI matrix covers those systems.

## Effect on saved analyses

No model definitions, saved draws, analysis data or reported estimates were changed. The audit found no use of the corrected shared statistical helpers, so this dependency upgrade does not require model refits or revised numerical results. Report and figure generation passed with synthetic inputs; complete saved analyses and published reports were not regenerated or compared byte for byte. Existing model validation requirements still apply to any research results used for reporting.

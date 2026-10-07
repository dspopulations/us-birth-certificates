> [!NOTE]
> AI-assisted revision by Claude Code (Opus 5.5).

> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# Shared utility contracts

`pyproject.toml` resolves `dse-research-utils` from public tag `v0.18.0` in the [research repository](https://github.com/dseinternational/research). `uv.lock` records the release commit and transitive versions. Install with `uv sync --locked`.

This package supplies the scientific stack through its core dependencies and the `boosting`, `columnar`, `dependence`, `graphs`, `io`, `jax`, `notebook` and `tuning` extras. Keep repository-only dependencies and development tools in this repository's metadata. The commented local source override points to `../../dseinternational/research/src/python`.

## Local adapters

Shared functions perform generic calculations. Local adapters retain the study's decisions and output formats.

| Local helper | Shared function or module | Local contract |
| --- | --- | --- |
| `intervals.equal_tail_interval` | `statistics.array_intervals.equal_tail_interval` | `0 < prob < 1`, explicit NaN policy, scalar full reduction; `posterior_mean_eti` uses the mean |
| `feature_groups.feature_groups_from_linkage` | `ml.feature_groups.feature_groups_from_linkage` | `cluster_NN` identifiers, ordering by first feature, default cut 0.30, empty and singleton handling |
| `stats_utils` and pipeline linkage | `ml.feature_groups.linkage_from_dissimilarity` | Distance-correlation dissimilarity, average linkage and singleton shape `(0, 4)` |
| `ml_utils.group_permutation_importance` | `ml.permutation.heldout_permutation_deltas` | Donor-plan order, absent-feature handling, positive-class column, scorer direction, table schema and ranking |
| `manifest` and `selection.io` | `metadata.provenance` | Selected git fields, package list, source and input hashes, manifest keys; selection also retains raw working-tree status |
| `file_io` | `storage.files.atomic_write` and mode probe | Atomic JSON/CSV replacement, encoding and umask-derived permissions |
| `plot_utils.save_fig` and SHAP figure helper | `plot.io.save_styled_figure` | PNG/SVG/CSV names, DPI, tight bounds, unindexed CSV and caller-owned figure closing |
| Selection validation | Public energy diagnostic and diagnostic-table reductions | Strict R-hat cutoff, inclusive effective-sample-size and energy cutoffs, constant exemptions and failure for missing diagnostics |

Change study behaviour in the adapter rather than duplicating a shared implementation. For new generic operations, import the shared function directly. The `repl_utils.py` module remains a notebook compatibility shim.

## Compatibility details

The v0.14 adapter migration changed some edge cases. Interval inputs are converted to float64. Empty arrays return NaN bounds. NaN omission suppresses reduction warnings. An SVG-backend failure warns while retaining other figure outputs. Detached manifests use `branch: null`. Group permutation preserves nullable column dtypes.

The v0.15 upgrade changed Optuna's default search behaviour. An old tuned parameter set remains usable, but it does not become a result of a new search merely because the dependency changed. Record sampler version and settings when comparing or resuming studies.

The v0.17 adoption uses public diagnostic helpers and the shared file-mode probe. Validation thresholds remain local. Preserve historical manifests and environments; do not rewrite them to look current.

The v0.18 upgrade moves figures to the DSE design-token colours. Series take `CHART_COLOURS`, which is also the default property cycle. Images take the sequential scale, or the diverging scale for signed values. `plot_colours.py` names colours by role, so recorded births, estimated births, reduction and the selection variants keep one colour across figures. `categorical_palette()` now raises for more than six categories. The predicted-analysis groupings with seven to eleven categories therefore name their earlier `tab10` or `viridis` colormaps until they are redesigned. Saved figures change only when their scripts run again.

## Upgrade evidence

The superseded release-specific guides are available in Git history. Their checks were run at the time of each upgrade, not as part of the current documentation review.

| Release | Recorded work |
| --- | --- |
| v0.13.0 | Import, environment, plot and report compatibility audit; [issue 111](https://github.com/dspopulations/us-birth-certificates/issues/111) |
| v0.14.0 | Local adapter adoption and differential-output checks; [issue 113](https://github.com/dspopulations/us-birth-certificates/issues/113) |
| v0.15.0 | Numerical-stack upgrade and fast/slow tests; [issue 115](https://github.com/dspopulations/us-birth-certificates/issues/115) |
| v0.17.0 | Public diagnostic and permission helpers; [PR 130](https://github.com/dspopulations/us-birth-certificates/pull/130) |
| v0.18.0 | Design-token plot colours and role colours; [PR 133](https://github.com/dspopulations/us-birth-certificates/pull/133) |

For future upgrades, read the tagged upstream migration guide, check actual consumers and run checks appropriate to the changed paths. Dependency compatibility alone does not validate saved research fits or establish that new outputs match old ones.

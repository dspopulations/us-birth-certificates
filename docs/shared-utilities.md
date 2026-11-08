> [!NOTE]
> AI-assisted revision by Claude Code (Opus 5.5).

> [!NOTE]
> AI-assisted revision by Codex (GPT-6).

# Shared utility contracts

`pyproject.toml` resolves `dse-research-utils` from public tag `v0.19.4` in the [research repository](https://github.com/dseinternational/research). `uv.lock` records the release commit and transitive versions. Install with `uv sync --locked`.

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

The [v0.19.3 upgrade](https://github.com/dseinternational/research/blob/v0.19.3/docs/migrating-to-0.19.md) raises the pandas and Polars minimums. The resolved environment also updates PyMC, PyTensor and JAX. The selected extras and the shared NumPy and PyTensor compatibility bounds remain in place. DuckDB stays at the already locked version 2.0.0. Its database files are not readable by DuckDB 1.x.

The v0.19.3 palette puts orange before green. With the existing role mapping, estimated counts, external references and selection variant B now use green; termination and variant C use orange. Race-group colours follow the new default order. Recorded counts remain pink. Review legends when comparing newly rendered figures with older outputs.

The current import streams CDC text into typed Arrow batches and writes them to DuckDB. DuckDB also writes the optional Parquet exports, including `scripts/make_all.py`. The SAS and Polars combination scripts have been removed, along with the unused `fastparquet` dependency. `truststore` still supplies trusted certificates for CDC downloads, and `pyreadstat` still supports SPSS exports through the shared `columnar` extra.

The [v0.19.4 upgrade](https://github.com/dseinternational/research/blob/v0.19.4/docs/migrating-to-0.19.md) changes only two versions in this repository's lock: `dse-research-utils` from 0.19.3 to 0.19.4 and pandas from 3.1.0 to 3.1.1. The shared release resolves to `bd916566081579e19031cabb637475a26cbd62aa`. PyMC 6.4.0, PyTensor 3.3.7, Numba 0.68.0, PyArrow 26.0.1 and h5netcdf 1.8.2 were already installed. The selected extras, compatibility bounds and plot colours are unchanged.

Pandas 3.1.1 fixes missing datetime values in Arrow timestamp conversion and unintended data sharing after forward or backward fill. The synthetic comparison on 8 November 2026 retained source codes, missing values, schemas, annual counts and all 98 grouped selection counts across the upgrade. At the time, it covered the Polars Parquet combination path, PyArrow and DuckDB reads, and a legacy fastparquet round trip. The Polars combination and fastparquet paths have since been removed. This checks the tested inputs, not every historical file.

The [PyMC 6.4 release](https://github.com/pymc-devs/pymc/releases/tag/v6.4.0) fixes LKJ correlation and covariance sampling and density calculations. The review found no LKJ distributions in current or historical tracked project code. The core model's multivariate normal observation uses a supplied correlation matrix, not an LKJ distribution. No stored fit was identified as needing a refit for that specific fix. The compatibility check opened a historical PyMC 6.0.1 trace and read three stored posterior values with the current netCDF reader. File readability does not establish that a fit meets current model or validation requirements; the [September refit guidance](../notes/20260905-dsp-code-review-fixes.md) still applies.

## Upgrade evidence

The superseded release-specific guides are available in Git history. Checks apply to the versions and inputs tested at each upgrade.

| Release | Recorded work |
| --- | --- |
| v0.13.0 | Import, environment, plot and report compatibility audit; [issue 111](https://github.com/dspopulations/us-birth-certificates/issues/111) |
| v0.14.0 | Local adapter adoption and differential-output checks; [issue 113](https://github.com/dspopulations/us-birth-certificates/issues/113) |
| v0.15.0 | Numerical-stack upgrade and fast/slow tests; [issue 115](https://github.com/dspopulations/us-birth-certificates/issues/115) |
| v0.17.0 | Public diagnostic and permission helpers; [PR 130](https://github.com/dspopulations/us-birth-certificates/pull/130) |
| v0.18.0 | Design-token plot colours and role colours; [PR 133](https://github.com/dspopulations/us-birth-certificates/pull/133) |
| v0.19.4 | Synthetic data comparison, nutpie sampling, trace save/load and copied report render; [issue 139](https://github.com/dspopulations/us-birth-certificates/issues/139) |

For future upgrades, read the tagged upstream migration guide, check actual consumers and run checks appropriate to the changed paths. Dependency compatibility alone does not validate saved research fits or establish that new outputs match old ones.

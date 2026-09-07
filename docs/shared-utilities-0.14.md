> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Shared utility adoption in v0.14.0

This records the upgrade and adapter work for [issue #113](https://github.com/dspopulations/us-birth-certificates/issues/113), completed on 7 September 2026. It follows the [0.14.0 upgrade notes](https://github.com/dseinternational/research/blob/v0.14.0/docs/migrating-to-0.14.md) and the [combined migration guide](https://github.com/dseinternational/research/blob/v0.14.0/docs/consolidation-migration.md). The earlier [v0.13.0 compatibility audit](shared-utilities-0.13.md) still describes the dependency-only checks; this upgrade also replaces local implementations.

## Dependency and installation

`pyproject.toml` selects the public v0.14.0 tag. `uv.lock` resolves it to commit `50390f4f9cd19033e3d0dc9bd050b383bd413c36`, replacing `458cc41b1dc33f4c0204919253ac92251c61b2bb`. The existing extras are unchanged and no other package changed in the lockfile — the release adds no dependency requirements. `uv lock` and `uv sync --locked` succeeded on macOS arm64 with Python 3.14, and the installed distribution reports `0.14.0`. The three repository assistant instruction files name the new tag.

## What now delegates

Each local helper became a thin adapter. The scientific and formatting decisions stay in this repository; the calculation moves upstream.

| Local helper | Shared function | What stays local |
| --- | --- | --- |
| `intervals.equal_tail_interval` | `statistics.array_intervals.equal_tail_interval` | The coverage restriction `0 < prob < 1` (the shared helper also accepts `prob=1`), the NaN policy (`nan=False` propagates, `nan=True` omits), and the scalar return for a full reduction. `posterior_mean_eti` still summarises with the **mean**, not the median. |
| `feature_groups.feature_groups_from_linkage` | `ml.feature_groups.feature_groups_from_linkage` | The `cluster_NN` identifiers, numbered by each group's first feature, and the 0.30 default cut. Zero- and one-feature sets still need no well-formed linkage. |
| `stats_utils.distance_corr_dissimilarity_linkage`, `models.base_pipeline._distance_corr_linkage` | `ml.feature_groups.linkage_from_dissimilarity` | Distance-correlation dissimilarity itself, average linkage, and the `(0, 4)` linkage for a single feature. The condensed matrix is still returned where callers use it. |
| `ml_utils.group_permutation_importance` | `ml.permutation.heldout_permutation_deltas` | Donor plans drawn group-by-group then repeat-by-repeat from `default_rng(random_state)`; dropping absent features and skipping emptied groups before any draw; the positive-class probability column; the average-precision scorer; `higher_is_better`; and the output schema, descending rank and `ddof=1` spread (0.0 for one repeat). |
| `manifest._git_info`, `manifest._package_versions` | `metadata.provenance.git_snapshot`, `package_versions` | The three manifest git fields and the tracked-package list. |
| `manifest.write_manifest`, `selection.io.save_artefacts`, `selection.io.save_summary` | `storage.files.atomic_write`, via the new `file_io` module | The manifest fields, JSON indentation, the explicit UTF-8 encoding on selection artefacts, the summary CSV index, and the file mode. |
| `selection.io` artefact, source, input and lockfile hashes | `metadata.provenance.sha256_file` | Which files are hashed, in what order, and under which manifest keys. |
| `plot_utils.save_fig`, `explain.shap_analysis._save_fig` | `plot.io.save_styled_figure` | The PNG/SVG/CSV trio at one stem, the tight bounding box, the caller's DPI, the unindexed CSV, and leaving the figure open for the caller to close. |

`selection.io.fit_manifest` keeps its own `git` helper. Its `worktree_changes` field records the raw `git status --porcelain` listing, which `git_snapshot` deliberately does not expose, so delegating it would change the field rather than preserve it.

The new `src/dspopulations_us_birth_certificates/file_io.py` adds one project decision on top of `atomic_write`: the temporary file starts owner-only, so the wrapper restores the umask-derived mode a plain `write_text` would have produced (0o644 in the usual case) before the file is renamed into place.

## Comparing old and new outputs

A differential harness recorded 43 behaviours — return values, dtypes, raised exception types and emitted warnings — against the helpers before and after the adapters, on the same synthetic inputs. **Thirty-nine are byte-for-byte identical**, including:

- every interval over axis `0`, `1`, `2`, `-1`, `(0, 1)` and `None`, with and without NaN omission, and every `posterior_mean_eti` summary;
- the feature groups, their identifiers, member order and cut behaviour at three thresholds, plus the empty and singleton cases;
- the dissimilarity and linkage matrices from both linkage call sites, including the one-feature case;
- six grouped-permutation runs — complete tables, group identifiers, feature lists, baseline scores, means, `ddof=1` spreads, ranks, column dtypes and the empty-group schema — across seeds, repeat counts, `predict`/`predict_proba` and all-absent groups;
- the git and package-version dictionaries;
- the saved figure filenames, the CSV bytes, the PNG size, and the figure still being open afterwards.

The remaining four recorded entries differ, in three ways. All are in `intervals.equal_tail_interval` and all follow from the shared reduction's documented contract:

| Change | Recorded entries | Effect here |
| --- | --- | --- |
| Samples are converted to float64, so float32 input no longer returns float32 bounds (differences of order 1e-8 on a float32 array). | 1 | **No effect on any project number.** PyTensor's `floatX` is float64 and no module in `src`, `scripts` or `tests` produces float32; every caller passes float64 posterior draws. |
| NaN-omitting reductions no longer emit a `RuntimeWarning`, including for an entirely omitted slice. | 2 | Identical NaN and non-NaN bounds, two fewer warnings. |
| An empty array returns NaN bounds instead of raising `IndexError`. | 1 | `posterior_mean_eti` already returned NaN for an all-NaN sample, so NaN-for-no-data is the existing policy; the exception was an artefact of `np.quantile`. The mean is still calculated separately, so an empty sample gives `{mean: nan, lo: nan, hi: nan}` rather than a silently narrowed population. |

One further difference is a matter of code rather than measured output: `save_fig` now drops the SVG sibling with a warning instead of raising if the SVG backend fails, so a backend failure no longer costs the PNG and CSV already written. The recorded figure outputs — filenames, CSV bytes, PNG size and the figure still being open — are unchanged.

Two behaviours change only outside the paths this project exercises. A detached HEAD now records `branch: null` in the model manifest instead of the literal string `"HEAD"`; the SHA is unaffected. And a permutation of a nullable or Arrow-backed column now keeps its dtype instead of being flattened by `to_numpy()` — the explanation set built by `build_explain_set` contains only float64 and categorical columns, so no current run is affected.

`write_manifest` output was compared field by field: identical top-level, git, environment, package, fingerprint and seed keys, identical byte length, identical two-space indentation and identical `0o644` mode.

## Areas of the release that do not apply

| Shared addition | Why it is not adopted here |
| --- | --- |
| `storage.directories.promote_directory` | No staging-tree promotion exists in this repository; run directories are written in place. |
| `report.readers`, `ReportData.read_summary`/`value_at` | No shared report-data consumers. Reports read local CSV and JSON through their own loaders. |
| `ml.permutation.pooled_oof_permutation_deltas` | There is no out-of-fold evaluation to pool: `EstimatorPipeline.cross_validate` raises `NotImplementedError` and the pipeline uses one train/validation split. |
| `statistics.models.hsgp_design` and `create_hsgp` | No HSGP models, and no saved HSGP geometry to validate or refit. |
| Likelihood-factor aggregation, predictive summaries, report assets and upload inventory | Not imported. The selection model computes its own likelihood and predictive quantities. |

## Verification

All checks ran against the locked v0.14.0 installation on macOS arm64, Python 3.14.

- `MPLBACKEND=Agg uv run pytest -ra`: 367 passed, 12 deselected, 114 warnings, in both random and fixed order. The baseline before the adapters was 329 passed with the same 114 warnings; the 38 new cases are `tests/test_shared_adapters.py`.
- `MPLBACKEND=Agg uv run pytest -m slow -p no:randomly -ra`: 12 passed. These fit real Bayesian models and assert posterior quality, covering the interval and artefact-writing paths end to end. They were run because `intervals` and `selection/io` both changed.
- `uv run ruff check src tests scripts`: passed.
- `uv run ruff format --check`: passed on every changed file.
- `npm run spellcheck`: passed.

`tests/test_shared_adapters.py` covers the acceptance criteria directly: interval axis permutations against a quantile reference, empty and all-missing slices, nonfinite inputs, the retained coverage restriction, the mean-based summary; cluster identifiers, member order, thresholds and degenerate feature sets; donor-plan reproducibility, score direction, the baseline over the exact explanation sample, the custom scorer, the standard-deviation convention, absent-feature handling, preserved categorical and nullable metadata and an untouched input frame; the manifest git and package fields; and atomic writes keeping their permissions and leaving the previous file intact when a writer fails. Selection artefacts are checked as a set: the three JSON files, their modes, the manifest hashing the configs written before it, and a summary CSV that still carries its index. One test rebuilds a whole grouped-importance result by hand from `default_rng(0)`, which pins the random-number consumption order.

The LightGBM pipeline smoke test exercises the delegated permutation path against a real trained booster with categorical features, so the adapter is checked in the pipeline and not only in isolation.

## Effect on saved analyses

No model definition, saved draws, analysis data or reported estimate changed. Every numerical helper produced identical values on identical inputs, and the one contract change that could alter numbers — float64 conversion — cannot reach a project call site, because nothing here produces float32. This upgrade therefore requires no refit and no revised results.

Complete saved analyses and published reports were not regenerated or compared byte for byte, and Linux and Windows installation were not run locally; the CI matrix covers those systems. Passing these checks does not establish the statistical quality of saved research fits, and the existing model-validation requirements still apply to any result used for reporting.

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Shared utility compatibility with v0.15.0

This records the upgrade for [issue #115](https://github.com/dspopulations/us-birth-certificates/issues/115), completed on 9 September 2026. It follows the [0.15.0 upgrade notes](https://github.com/dseinternational/research/blob/v0.15.0/docs/migrating-to-0.15.md). Unlike [v0.14.0](shared-utilities-0.14.md), which replaced local implementations with shared ones, this release changes no library API — it lifts the NumPy ceiling this repository has inherited since 0.11.0 and raises floors underneath it. The adapters introduced in 0.14.0 are untouched.

## Dependency and installation

`pyproject.toml` selects the public v0.15.0 tag. `uv.lock` resolves it to commit `e818caf52c0a73aed4ba5286bdfdc58c7867ea69`, replacing `50390f4f9cd19033e3d0dc9bd050b383bd413c36`. The existing extras (`boosting,columnar,dependence,graphs,io,jax,notebook,tuning`) are unchanged. `uv lock` and `uv sync --locked` succeeded on macOS arm64 with Python 3.14, and the installed distribution reports `0.15.0`. The three repository assistant instruction files name the new tag.

The lockfile still holds 196 packages. Ten moved:

| Package | 0.14.0 lock | 0.15.0 lock | Why |
| --- | --- | --- | --- |
| `dse-research-utils` | 0.14.0 | 0.15.0 | the tag bump itself |
| `numpy` | 2.4.6 | 2.5.3 | the lifted ceiling — see below |
| `numba` | 0.66.0 | 0.67.0 | 0.67.0 is what relaxes to `numpy<2.6` |
| `llvmlite` | 0.48.0 | 0.49.0 | pinned by numba 0.67.0 |
| `pytensor` | 3.3.0 | 3.3.1 | shared floor raised to `>=3.3.1`; 3.3.1 admits `numba<=0.67.0` |
| `pymc` | 6.3.1 | 6.3.2 | reachable once pytensor 3.3.1 is in the solve |
| `preliz` | 0.27.1 | 0.28.0 | shared floor raised |
| `pytensor-distributions` | 0.2.0 | 0.3.2 | pulled by preliz 0.28.0 |
| `optuna` | 4.9.0 | 5.0.0 | `tuning` floor raised — see below |
| `optuna-integration` | 4.9.0 | 5.0.0 | as above |

The other floors this release raised were already satisfied by the 0.14.0 solve and did not move the lock: `statsmodels` 0.15.0, `arviz-stats` 1.3.2, `arviz-plots` 1.3.1, `xgboost` 3.4.1, `polars` 1.44.1, `pyreadstat` 1.3.6, `orjson` 3.12.0 and `networkx` 3.6.1.

`numpy`, `numba`, `pymc` and `llvmlite` needed an explicit `uv lock --upgrade-package`; a plain re-lock keeps a satisfying pin rather than widening one. Taking the lift is the point of the release, so the lockfile records the stack the upgrade notes describe — **numpy 2.5.3 / numba 0.67.0 / pytensor 3.3.1 / pymc 6.3.2**.

## Both Dependabot ignore rules moved

`.github/dependabot.yml` carries two ceilings inherited from `dse-research-utils` and pytensor. Both have moved up one step, and both rules move with them, in this same change:

| Rule | Was | Now |
| --- | --- | --- |
| `numpy` | `>=2.5.0` | `>=2.6.0` |
| `numba` | `>0.66.0` | `>0.67.0` |

The `numba` rule is unique to this repository among the three consumers, so it is the one most easily missed. Leaving either behind means Dependabot keeps proposing an individually-plausible widening that no resolver can satisfy. The surrounding comment now names pytensor 3.3.1 and numba 0.67.0 as the chain holding the new ceiling.

## Optuna 5.0 changes the search, not the code

This repository takes the `tuning` extra, so the floor move from 4.9 to 5.0 reaches real call sites: `src/dspopulations_us_birth_certificates/tuning.py`, `scripts/fit_model.py`, and the paired notebooks under `notebooks/`.

Nothing here uses an API that 5.0 removed or deprecated. Every study is single-objective (`create_study(direction="maximize")`) with an explicitly constructed `TPESampler(seed=...)` and `HyperbandPruner()`, pruned through `optuna.integration.LightGBMPruningCallback`. There is no `optuna.multi_objective`, no `constraints_func`, no parameter-importance evaluator, and no `RDBStorage` or `JournalStorage` — `tuning.run_optuna_study` persists a study by pickling it to `study.pkl`, so the UTC timestamp normalisation does not apply. A smoke run of that exact surface on 5.0 — search space, pruning callback, `study.optimize`, `cli_output.print_optuna_summary`, `trials_dataframe`, and a pickle round-trip — completed with no errors and no deprecation warnings.

**The search itself is different.** Because the samplers here are constructed without `multivariate=` or `constant_liar=`, they take the new 5.0 defaults, and `TPESampler._is_multivariate` returns `True` for a single-objective study: 4.x ran independent TPE with constant-liar off, 5.x runs multivariate TPE with constant-liar on. Same seed, same space, different algorithm.

Two consequences for this project:

- `DEFAULT_PRIOR_BEST_PARAMS` in `scripts/fit_model.py` came from a 4.x search. It remains a perfectly good starting point — it is a set of hyperparameters, not a claim about a sampler — but it is not a 5.x result and should not be reported as one.
- No `study.pkl`, `trials.csv` or `best_params.json` artefact is tracked in the repository or present in `output/`, so nothing stored needs re-reading under the new sampler. Any future study resumed or compared across this boundary belongs to the sampler that produced it, and a 4.x study and a 5.x re-run are not interchangeable evidence.

Neither sampler default was pinned back to its 4.x value. Doing so would freeze a search strategy upstream has deliberately replaced; whether to re-tune on 5.0 is a research decision, not an upgrade one.

## `statsmodels` 0.15.0 reaches nothing here

The upgrade notes single out `statsmodels` 0.15.0 as the largest library move after Optuna and ask consumers to check it against any regression output they report. A search of `src/`, `scripts/`, `tests/`, `notebooks/`, `docs/` and `previous/` finds no import of `statsmodels` and no reference to it outside the inherited dependency metadata. It arrives transitively through the shared core, and this repository reports no regression output that depends on it. Nothing to check.

## Verification on the new stack

A floor change does not refit a model, so the numerical work was re-run rather than assumed.

| Check | Result |
| --- | --- |
| `uv lock`, `uv sync --locked` | Clean on macOS arm64, Python 3.14 |
| `uv run pytest` | 367 passed, 12 deselected |
| `uv run pytest -m slow` | 12 passed — the convergence gate, run in full |
| `uv run ruff check src tests scripts` | All checks passed |
| `npm run spellcheck` | 53 files, 0 issues |

The slow suite is the part that matters for a numerical-stack change. `test_joint_simulation_fit_numerical_health` fits DSP003, DSP008, DSP009 and DSP010 through nutpie at 4 chains × 1000 draws and requires `validate_fit` to return `passed` — free-variable, divergence and energy checks together with `convergence_health` — then checks the calibration ranks. `test_sampler_converged` hard-fails on R-hat blow-up, and the parameter-recovery tests check 89% ETI coverage for six array parameters and a posterior-mean tolerance for the scalar intercept. All twelve pass on numpy 2.5.3 / numba 0.67.0 / pytensor 3.3.1 / pymc 6.3.2, with no threshold relaxed.

No saved fit was compared against a new one across this upgrade. The implementation is unchanged, but the sampler ran on a different compiled stack, and the upgrade notes are explicit that identity has to be re-established before such a comparison means anything.

"""Fit a LightGBM model for Down syndrome live-birth prediction.

Thin CLI over ``LGBMClassifierPipeline`` in the package. The pipeline
owns data loading, splitting, training, metrics, permutation, SHAP,
and artefact saving. This script just resolves configuration from CLI
flags and profile presets, runs optional Optuna tuning, and invokes the
pipeline.

When Optuna tuning is enabled, the current implementation uses the same
deterministic stratified split for tuning, early stopping, and reported
validation metrics. Treat those metrics as tuning-set diagnostics rather
than as an untouched test-set estimate.

Configuration profiles
----------------------
Pick a profile with ``--profile {dev,test,reporting}``. Presets are
sourced from ``RunConfig.from_name()`` in the package so the CLI and
the library agree on what each name means.

- ``dev``: fast inner loop — 2-year slice, 10 Optuna trials, 500 boost
  rounds, SHAP off. Use while developing feature sets.
- ``test``: moderate fidelity — 5-year slice, 50 Optuna trials, 10 000
  boost rounds, SHAP on a 5 000-row subsample. Use for pre-PR
  validation.
- ``reporting``: full run — 2016–2024, 200 Optuna trials, 50 000 boost
  rounds, full permutation + SHAP. Use for publication-quality numbers.

Individual flags always override profile defaults.

Examples
--------
    python scripts/fit_model.py --profile dev
    python scripts/fit_model.py --profile reporting

    # Smoke test — seconds rather than minutes
    python scripts/fit_model.py --profile dev --no-optimize \\
        --num-boost-round 200 --no-permutation --no-shap
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import dse_research_utils.environment.setup as setup
import dse_research_utils.metadata.packages as package_metadata
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

from dspopulations_us_birth_certificates import PACKAGE_LIST, cli_output, data_utils
from dspopulations_us_birth_certificates.models import (
    MODELS,
    ModelConfig,
    RunConfig,
)
from dspopulations_us_birth_certificates.models.lgbm_pipeline import (
    LGBMClassifierPipeline,
)
from dspopulations_us_birth_certificates.variables import Variables as vars

optuna.logging.set_verbosity(optuna.logging.WARNING)


DEFAULT_NUMERIC: tuple[str, ...] = (
    vars.YEAR,
    vars.DBWT,
    vars.WTGAIN,
    vars.BMI,
    vars.MAGE_C,
    vars.FAGECOMB,
)

DEFAULT_CATEGORICAL: tuple[str, ...] = (
    vars.BFACIL3,
    vars.SEX,
    vars.PRECARE,
    vars.GESTREC10,
    vars.RF_PDIAB,
    vars.RF_GDIAB,
    vars.RF_PHYPE,
    vars.RF_GHYPE,
    vars.RF_EHYPE,
    vars.RF_PPTERM,
    vars.RF_INFTR,
    vars.RF_FEDRG,
    vars.RF_ARTEC,
    vars.LD_INDL,
    vars.LD_AUGM,
    vars.ME_PRES,
    vars.DMETH_REC,
    vars.APGAR5,
    vars.APGAR10,
    vars.AB_AVEN1,
    vars.AB_AVEN6,
    vars.AB_NICU,
    vars.AB_SURF,
    vars.AB_ANTI,
    vars.AB_SEIZ,
    vars.CA_ANEN,
    vars.CA_MNSB,
    vars.CA_CCHD,
    vars.CA_CDH,
    vars.CA_OMPH,
    vars.CA_GAST,
    vars.CA_LIMB,
    vars.CA_CLEFT,
    vars.CA_CLPAL,
    vars.CA_HYPO,
    vars.CA_DISOR,
    vars.MEDUC,
    vars.MRACEHISP,
    vars.FEDUC,
    vars.FRACEHISP,
    vars.PAY_REC,
    vars.WIC,
)

DEFAULT_BASE_PARAMS: dict = {
    "objective": "binary",
    "metric": ["average_precision", "binary_logloss"],
    "boosting_type": "gbdt",
    "max_bin": 255,
    "scale_pos_weight": 1,
    "force_col_wise": True,
}

DEFAULT_TARGET_VAR = "ca_down_c_p_n"

DEFAULT_PRIOR_BEST_PARAMS: dict = {
    "learning_rate": 0.009461164726049449,
    "num_leaves": 180,
    "min_data_in_leaf": 756,
    "min_gain_to_split": 0.9285634625013361,
    "feature_fraction": 0.9239582799934513,
    "bagging_fraction": 0.9185684081749333,
    "bagging_freq": 2,
    "lambda_l1": 0.0005836073944757167,
    "lambda_l2": 0.6142323696066677,
}


_CLI_PROFILE_YEAR_RANGES: dict[str, tuple[int, int]] = {
    "dev": (2023, 2024),
    "test": (2020, 2024),
    "reporting": (2016, 2024),
}


def _profile_defaults(name: str) -> dict[str, object]:
    """Translate a RunConfig preset into argparse defaults for fit_model."""
    rc = RunConfig.from_name(name)
    start_year, end_year = _CLI_PROFILE_YEAR_RANGES[name]
    return {
        "start_year": start_year,
        "end_year": end_year,
        "optimize": rc.n_trials > 0,
        "optimize_trials": rc.n_trials,
        "num_boost_round": rc.num_boost_round,
        "early_stopping_rounds": rc.early_stopping_rounds,
        "plots": True,
        "permutation": rc.shap_mode != "skip",
        "shap": rc.shap_mode != "skip",
    }


@dataclass
class FitConfig:
    start_year: int = 2016
    end_year: int = 2024
    include_unknown: bool = True
    training_split: float = 0.8
    num_boost_round: int = 50_000
    early_stopping_rounds: int = 200
    optimize_trials: int = 200
    select_hyperparameters: bool = True
    random_seed: int = 47
    num_threads: int | None = None
    output_dir: Path = field(default_factory=lambda: Path("output/fit_model"))
    drop_features: list[str] = field(default_factory=list)
    prior_best_params_path: Path | None = None
    save_plots: bool = True
    run_permutation: bool = True
    run_shap: bool = True
    write_predictions: bool = False
    duckdb_path: Path = Path("data/us_births.db")
    profile: str | None = None
    load_model: Path | None = None
    model_id: str | None = None
    render: bool = False


def parse_args(argv: list[str] | None = None) -> FitConfig:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--profile", choices=list(RunConfig.preset_names()), default=None)
    pre_ns, _ = pre.parse_known_args(argv)
    profile_defaults = _profile_defaults(pre_ns.profile) if pre_ns.profile else {}

    p = argparse.ArgumentParser(
        description="Fit a LightGBM model for Down syndrome live-birth prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--profile",
        choices=list(RunConfig.preset_names()),
        default=None,
        help=(
            "Configuration profile preset — sourced from RunConfig.from_name() "
            "in the package. Individual flags override profile defaults."
        ),
    )

    p.add_argument(
        "--model-id",
        choices=sorted(MODELS.keys()) or None,
        default=None,
        help=(
            "Named model variant from the registry (e.g. usbc10_m0, "
            "usbc10_m1, usbc10_m2). When set, the definition supplies "
            "target_var, numeric/categorical features, base_params, "
            "year_range, include_unknown, selection_history, "
            "shap_scatter_specs, and notes; --drop-features, --start-year, "
            "--end-year, and --include-unknown are therefore ignored. Omit "
            "to use the CLI's ad-hoc feature set."
        ),
    )

    p.add_argument("--start-year", type=int, default=2016)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument(
        "--include-unknown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat records with unknown DS status as negatives.",
    )

    p.add_argument("--training-split", type=float, default=0.8)
    p.add_argument("--num-boost-round", type=int, default=50_000)
    p.add_argument("--early-stopping-rounds", type=int, default=200)
    p.add_argument("--random-seed", type=int, default=47)
    p.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="LightGBM threads (default: physical cores - 2).",
    )

    p.add_argument(
        "--optimize-trials",
        type=int,
        default=200,
        help="Number of Optuna trials (ignored with --no-optimize).",
    )
    p.add_argument(
        "--optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Optuna search; when disabled, use --prior-best-params.",
    )
    p.add_argument(
        "--prior-best-params",
        type=Path,
        default=None,
        help="JSON file with best params to use when --no-optimize is set.",
    )

    p.add_argument(
        "--drop-features",
        nargs="*",
        default=[],
        help="Feature names to drop from the initial set before training.",
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/fit_model/<timestamp>).",
    )
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--permutation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute permutation importance (slow).",
    )
    p.add_argument(
        "--shap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute SHAP values and plots (slow).",
    )

    p.add_argument(
        "--write-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-row predictions back to the DuckDB database.",
    )
    p.add_argument("--duckdb-path", type=Path, default=Path("data/us_births.db"))

    p.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help=(
            "Path to a saved LightGBM model.txt. When set, skips Optuna and "
            "training; re-runs metrics / permutation / SHAP / plots against "
            "the loaded booster. Useful for regenerating diagnostics."
        ),
    )

    p.add_argument(
        "--render",
        action="store_true",
        default=False,
        help=(
            "Invoke `quarto render` on the per-run index.qmd after fitting. "
            "Without this flag the template is copied but not rendered."
        ),
    )

    if profile_defaults:
        p.set_defaults(**profile_defaults)

    ns = p.parse_args(argv)

    out_dir = ns.output_dir or (
        Path(f"output/fit_model_{ns.profile}" if ns.profile else "output/fit_model")
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    return FitConfig(
        start_year=ns.start_year,
        end_year=ns.end_year,
        include_unknown=ns.include_unknown,
        training_split=ns.training_split,
        num_boost_round=ns.num_boost_round,
        early_stopping_rounds=ns.early_stopping_rounds,
        optimize_trials=ns.optimize_trials,
        select_hyperparameters=ns.optimize,
        random_seed=ns.random_seed,
        num_threads=ns.num_threads,
        output_dir=out_dir,
        drop_features=list(ns.drop_features),
        prior_best_params_path=ns.prior_best_params,
        save_plots=ns.plots,
        run_permutation=ns.permutation,
        run_shap=ns.shap,
        write_predictions=ns.write_predictions,
        duckdb_path=ns.duckdb_path,
        profile=ns.profile,
        load_model=ns.load_model,
        model_id=ns.model_id,
        render=ns.render,
    )


@dataclass(frozen=True)
class _FeatureSpec:
    """Target/feature/year-range resolution shared by tuning and model-config building."""

    target_var: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    year_range: tuple[int, int]
    include_unknown: bool
    confirmed_only: bool


def _resolve_feature_spec(config: FitConfig) -> _FeatureSpec:
    """Resolve target/features/year-range from a ``--model-id`` or CLI ad-hoc args.

    Single source for the model-id-vs-ad-hoc decision that both
    ``_build_model_config`` and ``_load_xy_for_tuning`` need, so the two
    can't drift out of sync with each other.
    """
    if config.model_id is not None:
        definition = MODELS[config.model_id]
        return _FeatureSpec(
            target_var=definition.target_var,
            numeric=tuple(definition.numeric_features),
            categorical=tuple(definition.categorical_features),
            year_range=definition.year_range,
            include_unknown=definition.include_unknown,
            confirmed_only=definition.confirmed_only,
        )
    # --drop-features is ignored under --model-id (see CLI help), so it only
    # applies to the ad-hoc feature lists here.
    return _FeatureSpec(
        target_var=DEFAULT_TARGET_VAR,
        numeric=tuple(f for f in DEFAULT_NUMERIC if f not in config.drop_features),
        categorical=tuple(
            f for f in DEFAULT_CATEGORICAL if f not in config.drop_features
        ),
        year_range=(config.start_year, config.end_year),
        include_unknown=config.include_unknown,
        confirmed_only=False,
    )


def _build_model_config(config: FitConfig, params: dict) -> ModelConfig:
    """Build a ModelConfig from either a named variant or CLI ad-hoc args.

    When ``config.model_id`` is set, the definition from ``MODELS`` provides
    the base model identity and metadata from ``definition.to_config()``,
    including ``base_params``, ``selection_history``, ``shap_scatter_specs``,
    and ``notes`` — fields with no ad-hoc equivalent, so this branch stays
    separate from ``_resolve_feature_spec``. Tuned ``params`` and the CLI's
    ``training_split`` / ``num_threads`` still override the definition's
    training defaults because those are tuning-time knobs, not part of the
    variant's identity.
    """
    if config.model_id is not None:
        definition = MODELS[config.model_id]
        base = definition.to_config()
        train_config = dict(base.train_config)
        train_config["training_split"] = config.training_split
        train_config.update(_evaluation_split_metadata(config))
        if config.num_threads is not None:
            train_config["num_threads"] = config.num_threads
        return ModelConfig(
            model_id=base.model_id,
            variant_of=base.variant_of,
            target_var=base.target_var,
            numeric_features=base.numeric_features,
            categorical_features=base.categorical_features,
            base_params=dict(base.base_params),
            params=dict(params) if params else dict(base.params),
            train_config=train_config,
            year_range=base.year_range,
            include_unknown=base.include_unknown,
            confirmed_only=base.confirmed_only,
            selection_history=base.selection_history,
            shap_scatter_specs=base.shap_scatter_specs,
            notes=base.notes,
            predictions_column=base.predictions_column,
            missing_flag_column=base.missing_flag_column,
        )

    spec = _resolve_feature_spec(config)
    train_config: dict = {
        "training_split": config.training_split,
        "verbosity": 1,
        "log_period": 10,
        **_evaluation_split_metadata(config),
    }
    if config.num_threads is not None:
        train_config["num_threads"] = config.num_threads
    return ModelConfig(
        model_id=f"fit_model_{config.profile or 'default'}",
        variant_of=None,
        target_var=spec.target_var,
        numeric_features=spec.numeric,
        categorical_features=spec.categorical,
        base_params=dict(DEFAULT_BASE_PARAMS),
        params=dict(params),
        train_config=train_config,
        year_range=spec.year_range,
        include_unknown=spec.include_unknown,
        selection_history=(),
        shap_scatter_specs=(),
        notes=f"Run produced by scripts/fit_model.py with profile={config.profile!r}.",
    )


def _evaluation_split_metadata(config: FitConfig) -> dict[str, object]:
    """Describe how the train/validation split should be interpreted."""
    tuning_uses_split = bool(
        config.select_hyperparameters and config.load_model is None
    )
    role = "early_stopping_and_reported_metrics"
    if tuning_uses_split:
        role = "optuna_tuning_early_stopping_and_reported_metrics"
    return {
        "evaluation_split_role": role,
        "hyperparameter_tuning_uses_validation_split": tuning_uses_split,
        "validation_independent_of_hyperparameter_tuning": not tuning_uses_split,
    }


def _build_run_config(config: FitConfig) -> RunConfig:
    base = (
        RunConfig.from_name(config.profile, random_seed=config.random_seed)
        if config.profile
        else RunConfig.from_name("reporting", random_seed=config.random_seed)
    )
    shap_mode = base.shap_mode if config.run_shap else "skip"
    return replace(
        base,
        num_boost_round=config.num_boost_round,
        early_stopping_rounds=config.early_stopping_rounds,
        n_trials=config.optimize_trials,
        shap_mode=shap_mode,
    )


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    categorical: list[str],
    model_config: ModelConfig,
    config: FitConfig,
) -> dict:
    """Run an Optuna search against a stratified split. Returns best params."""
    cli_output.section("Hyperparameter search (Optuna)")
    cli_output.info(
        f"Trials=[bold]{config.optimize_trials}[/bold], "
        f"num_boost_round=[bold]{config.num_boost_round}[/bold], "
        f"early_stopping_rounds=[bold]{config.early_stopping_rounds}[/bold]"
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=1 - config.training_split,
        stratify=y,
        random_state=config.random_seed,
    )
    cli_output.print_split_summary(X_train, X_valid, y_train, y_valid)
    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical, free_raw_data=False
    )
    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical,
        reference=train_data,
        free_raw_data=False,
    )
    n_cores = joblib.cpu_count(only_physical_cores=True)
    base_params: dict = {
        **model_config.base_params,
        "seed": config.random_seed,
        "num_threads": config.num_threads or max(1, n_cores - 2),
        "verbosity": -1,
    }

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "feature_pre_filter": False,
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.75, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 32, 512, log=True),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 500, 10000, log=True
            ),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        params = {**base_params, **trial_params}
        pruning_cb = optuna.integration.LightGBMPruningCallback(
            trial, "average_precision"
        )
        gbm = lgb.train(
            params,
            train_data,
            num_boost_round=config.num_boost_round,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=config.early_stopping_rounds),
                lgb.log_evaluation(period=0),
                pruning_cb,
            ],
        )
        return gbm.best_score["valid_0"]["average_precision"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_seed),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(objective, n_trials=config.optimize_trials, show_progress_bar=True)
    cli_output.section("Tuning complete")
    cli_output.print_optuna_summary(study)
    return study.best_params


def load_prior_best_params(path: Path | None) -> dict:
    if path is None:
        return dict(DEFAULT_PRIOR_BEST_PARAMS)
    with path.open() as f:
        return json.load(f)


def _recover_loaded_params(model_path: Path) -> dict:
    """Return the hyperparameters used by a saved booster.

    Prefers a sibling ``best_params.json`` (the per-run artefact fit_model
    writes next to ``model.txt``), since that file captures tuned values
    directly. Falls back to parsing ``booster.params`` from the saved
    ``model.txt``; returns an empty dict if neither is available so the
    caller can still proceed.
    """
    sibling = model_path.parent / "best_params.json"
    if sibling.is_file():
        try:
            with sibling.open() as f:
                return json.load(f)
        except OSError, json.JSONDecodeError:
            pass

    try:
        booster = lgb.Booster(model_file=str(model_path))
    except Exception:  # noqa: BLE001 - surface any LightGBM failure as empty
        return {}
    params = getattr(booster, "params", None) or {}
    return dict(params)


def _load_xy_for_tuning(
    config: FitConfig,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the predictors frame once for tuning (outside the pipeline).

    Resolves target/features/year-range via ``_resolve_feature_spec`` so
    tuning operates on the same data the pipeline will see.
    """
    spec = _resolve_feature_spec(config)
    df = data_utils.load_predictors_data(
        from_year=spec.year_range[0],
        to_year=spec.year_range[1],
        include_unknown=spec.include_unknown,
        confirmed_only=spec.confirmed_only,
        db_path=str(config.duckdb_path),
    )
    categorical = list(spec.categorical)
    features = categorical + list(spec.numeric)
    X = df[features].copy()
    y = df[spec.target_var].replace({pd.NA: 0, np.nan: 0}).astype(np.int32)
    X[categorical] = X[categorical].astype("category")
    return X, y, categorical


_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_ident(name: str, kind: str) -> None:
    """Guard: column names are interpolated into SQL, so reject exotic chars."""
    if not _SAFE_IDENT.match(name):
        raise ValueError(f"Invalid {kind} column name: {name!r}")


def write_predictions_to_duckdb(
    df: pd.DataFrame,
    gbm: lgb.Booster,
    features: list[str],
    categorical: list[str],
    duckdb_path: Path,
    *,
    predictions_column: str,
    missing_flag_column: str,
) -> None:
    import duckdb

    _check_ident(predictions_column, "predictions")
    _check_ident(missing_flag_column, "missing flag")

    X_full = df[features].copy()
    X_full[categorical] = X_full[categorical].astype("category")
    df = df.copy()
    df[predictions_column] = gbm.predict(X_full, num_iteration=gbm.best_iteration)

    tmp_pred = f"_{predictions_column}_tmp"
    tmp_flag = "_ds_pred_missing_ids"
    _check_ident(tmp_pred, "temp prediction")

    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            f"ALTER TABLE us_births ADD COLUMN IF NOT EXISTS "
            f"{predictions_column} DOUBLE;"
        )
        con.execute(
            f"ALTER TABLE us_births ADD COLUMN IF NOT EXISTS "
            f"{missing_flag_column} BOOLEAN DEFAULT FALSE"
        )
        con.execute(f"UPDATE us_births SET {predictions_column} = NULL")
        con.execute(f"UPDATE us_births SET {missing_flag_column} = FALSE")
        con.execute(f"DROP TABLE IF EXISTS {tmp_pred}")
        con.execute(f"CREATE TABLE {tmp_pred} (id BIGINT, {predictions_column} DOUBLE)")
        con.execute(
            f"INSERT INTO {tmp_pred} (id, {predictions_column}) "
            f"SELECT id, {predictions_column} FROM df"
        )
        con.execute(
            f"""
            UPDATE us_births b
            SET {predictions_column} = p.{predictions_column}
            FROM {tmp_pred} p
            WHERE b.id = p.id;
            """
        )
        # Flag likely missing DS cases as ``<missing_flag_column>`` using a
        # year×month quota of ceil(1.5 × recorded), picking the top
        # non-recorded births by ``<predictions_column>``. Multiplier 1.5
        # encodes the ~60% under-reporting rate observed across surveillance
        # studies (recorded ≈ 40%, missed ≈ 60% → total ≈ recorded + 1.5 ×
        # recorded = 2.5 × recorded). Downstream analyses can then select
        # R = down_ind=1, R' = down_ind=1 OR <missing_flag_column>.
        #
        # TODO: revisit per-year multipliers calibrated against
        # surveillance-based live-birth estimates — the 60% rate is close to
        # constant but varies slightly by year. Uniform 1.5× is a sufficient
        # v1 approximation.
        con.execute(f"DROP TABLE IF EXISTS {tmp_flag}")
        con.execute(
            f"""
            CREATE TABLE {tmp_flag} AS
            WITH year_month_quota AS (
                SELECT year, dob_mm,
                       CAST(CEIL(COUNT(*) * 1.5) AS BIGINT) AS n_select
                FROM us_births
                WHERE id IN (SELECT id FROM {tmp_pred})
                  AND down_ind = 1
                  AND {predictions_column} IS NOT NULL
                GROUP BY year, dob_mm
            ),
            ranked AS (
                SELECT b.id,
                       ROW_NUMBER() OVER (
                           PARTITION BY b.year, b.dob_mm
                           ORDER BY b.{predictions_column} DESC
                       ) AS rn,
                       q.n_select
                FROM us_births b
                JOIN year_month_quota q
                  ON q.year = b.year AND q.dob_mm = b.dob_mm
                WHERE b.id IN (SELECT id FROM {tmp_pred})
                  AND b.down_ind = 0
                  AND b.{predictions_column} IS NOT NULL
            )
            SELECT id FROM ranked WHERE rn <= n_select
            """
        )
        con.execute(
            f"""
            UPDATE us_births b
            SET {missing_flag_column} = TRUE
            FROM {tmp_flag} t
            WHERE b.id = t.id
            """
        )
        con.execute(f"DROP TABLE {tmp_flag}")
        con.execute(f"DROP TABLE IF EXISTS {tmp_pred}")
    finally:
        con.close()


def _resolve_best_params(config: FitConfig) -> dict:
    """Resolve the LightGBM hyperparameters to fit with.

    One of three mutually exclusive paths: reuse a loaded model's own
    params (``--load-model``), run an Optuna search (``--optimize``, the
    default), or load a prior ``best_params.json`` (``--no-optimize``).
    """
    if config.load_model is not None:
        cli_output.section("Loading saved model")
        cli_output.info(
            f"Loading [blue]{config.load_model}[/blue]; skipping tuning + fit."
        )
        # Surface the hyperparameters actually used by the loaded booster so
        # ``best_params.json``, ``config.json``, and the manifest describe the
        # run faithfully. Prefer a sibling ``best_params.json`` if present
        # (richer than whatever LightGBM echoes back via booster.params).
        best_params = _recover_loaded_params(config.load_model)
        cli_output.print_params("Recovered params", best_params)
        return best_params

    if config.select_hyperparameters:
        X, y, categorical = _load_xy_for_tuning(config)
        ad_hoc = _build_model_config(config, params={})
        return optimize_hyperparameters(X, y, categorical, ad_hoc, config)

    best_params = load_prior_best_params(config.prior_best_params_path)
    cli_output.section("Using prior best params")
    cli_output.print_params("Prior best params", best_params)
    return best_params


def _run_post_fit_steps(
    pipeline: LGBMClassifierPipeline,
    config: FitConfig,
    model_config: ModelConfig,
    df: pd.DataFrame,
) -> None:
    """Metrics, importance, SHAP, artefacts, manifest, report, and optional DuckDB writeback."""
    pipeline.compute_metrics()
    if config.run_permutation:
        pipeline.permutation_importance_analysis()
    pipeline.shap_analysis()
    pipeline.save_artefacts(save_plots=config.save_plots)
    pipeline.write_manifest()
    pipeline.report(render=config.render)

    if not config.write_predictions:
        return
    cli_output.section("Write predictions to DuckDB")
    gbm = pipeline.context.final_model
    features = list(model_config.categorical_features) + list(
        model_config.numeric_features
    )
    write_predictions_to_duckdb(
        df,
        gbm,
        features,
        list(model_config.categorical_features),
        config.duckdb_path,
        predictions_column=model_config.predictions_column,
        missing_flag_column=model_config.missing_flag_column,
    )
    cli_output.success(
        f"Wrote predictions to {config.duckdb_path} "
        f"(columns {model_config.predictions_column}, "
        f"{model_config.missing_flag_column})"
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    setup.init_script()
    np.random.seed(config.random_seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    cli_output.print_run_header(
        command="fit_model",
        profile=config.profile,
        output_dir=config.output_dir,
        model_id=config.model_id,
    )

    cli_output.section("Environment")
    package_metadata.report_package_versions(list(PACKAGE_LIST))

    cli_output.section("Fit configuration")
    cli_output.print_fit_config(config)

    # --load-model regenerates diagnostics from a saved booster rather than
    # tuning + fitting; see _resolve_best_params for the other two paths
    # (Optuna search, or a prior best_params.json).
    best_params = _resolve_best_params(config)

    (config.output_dir / "best_params.json").write_text(
        json.dumps(best_params, indent=2)
    )

    model_config = _build_model_config(config, params=best_params)
    run_config = _build_run_config(config)

    cli_output.section("Resolved model + run configuration")
    cli_output.print_model_config(model_config)
    cli_output.print_run_config(run_config)

    pipeline = LGBMClassifierPipeline(
        config=model_config, run_config=run_config, output_dir=config.output_dir
    )

    df = pipeline.load_data(db_path=str(config.duckdb_path))
    pipeline.prepare_features(df)

    if config.load_model is not None:
        pipeline.load_final_model(config.load_model)
    else:
        pipeline.train_final()

    _run_post_fit_steps(pipeline, config, model_config, df)

    # Persist the full resolved config alongside artefacts for reproducibility.
    (config.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **{
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in config.__dict__.items()
                },
            },
            indent=2,
        )
    )

    cli_output.section("Done")
    cli_output.success(f"Artefacts in {config.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

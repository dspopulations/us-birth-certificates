"""Fit a LightGBM model for Down syndrome live-birth prediction.

Thin CLI over ``LGBMClassifierPipeline`` in the package. The pipeline
owns data loading, splitting, training, metrics, permutation, SHAP,
and artefact saving. This script just resolves configuration from CLI
flags and profile presets, runs optional Optuna tuning, and invokes the
pipeline.

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
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

from dspopulations_us_birth_certificates import data_utils, repl_utils
from dspopulations_us_birth_certificates.models import (
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


def parse_args(argv: list[str] | None = None) -> FitConfig:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--profile", choices=list(RunConfig.preset_names()), default=None
    )
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
    )


def _build_model_config(config: FitConfig, params: dict) -> ModelConfig:
    numeric = tuple(f for f in DEFAULT_NUMERIC if f not in config.drop_features)
    categorical = tuple(
        f for f in DEFAULT_CATEGORICAL if f not in config.drop_features
    )
    train_config: dict = {
        "training_split": config.training_split,
        "verbosity": 1,
        "log_period": 10,
    }
    if config.num_threads is not None:
        train_config["num_threads"] = config.num_threads
    return ModelConfig(
        model_id=f"fit_model_{config.profile or 'default'}",
        variant_of=None,
        target_var="ca_down_c_p_n",
        numeric_features=numeric,
        categorical_features=categorical,
        base_params=dict(DEFAULT_BASE_PARAMS),
        params=dict(params),
        train_config=train_config,
        year_range=(config.start_year, config.end_year),
        include_unknown=config.include_unknown,
        selection_history=(),
        shap_scatter_specs=(),
        notes=f"Run produced by scripts/fit_model.py with profile={config.profile!r}.",
    )


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
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=1 - config.training_split,
        stratify=y,
        random_state=config.random_seed,
    )
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
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(
        objective, n_trials=config.optimize_trials, show_progress_bar=True
    )
    print(f"Best AP: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def load_prior_best_params(path: Path | None) -> dict:
    if path is None:
        return dict(DEFAULT_PRIOR_BEST_PARAMS)
    with path.open() as f:
        return json.load(f)


def _load_xy_for_tuning(
    config: FitConfig,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the predictors frame once for tuning (outside the pipeline)."""
    df = data_utils.load_predictors_data(
        from_year=config.start_year,
        to_year=config.end_year,
        include_unknown=config.include_unknown,
        db_path=str(config.duckdb_path),
    )
    numeric = [f for f in DEFAULT_NUMERIC if f not in config.drop_features]
    categorical = [f for f in DEFAULT_CATEGORICAL if f not in config.drop_features]
    features = categorical + numeric
    X = df[features].copy()
    y = df["ca_down_c_p_n"].replace({pd.NA: 0, np.nan: 0}).astype(np.int32)
    X[categorical] = X[categorical].astype("category")
    return X, y, categorical


def write_predictions_to_duckdb(
    df: pd.DataFrame,
    gbm: lgb.Booster,
    features: list[str],
    categorical: list[str],
    duckdb_path: Path,
) -> None:
    import duckdb

    X_full = df[features].copy()
    X_full[categorical] = X_full[categorical].astype("category")
    df = df.copy()
    df["p_ds_lb_pred_01"] = gbm.predict(X_full, num_iteration=gbm.best_iteration)

    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            "ALTER TABLE us_births ADD COLUMN IF NOT EXISTS p_ds_lb_pred_01 DOUBLE;"
        )
        con.execute("DROP TABLE IF EXISTS ds_lb_pred_01")
        con.execute("CREATE TABLE ds_lb_pred_01 (id BIGINT, p_ds_lb_pred DOUBLE)")
        con.execute(
            "INSERT INTO ds_lb_pred_01 (id, p_ds_lb_pred) "
            "SELECT id, p_ds_lb_pred_01 FROM df"
        )
        con.execute(
            """
            UPDATE us_births b
            SET p_ds_lb_pred_01 = p.p_ds_lb_pred
            FROM ds_lb_pred_01 p
            WHERE b.id = p.id;
            """
        )
        con.execute("DROP TABLE IF EXISTS ds_lb_pred_01")
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    np.random.seed(config.random_seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    repl_utils.print_environment_info()
    print(f"\nOutput directory: {config.output_dir}\n")

    # Tuning (still inline — moves to scripts/tune_model.py in step 6). Skip
    # when --load-model is set: we're regenerating diagnostics, not fitting.
    if config.load_model is not None:
        print(f"Loading saved model from {config.load_model}; skipping tuning + fit.")
        best_params = {}
    elif config.select_hyperparameters:
        X, y, categorical = _load_xy_for_tuning(config)
        ad_hoc = _build_model_config(config, params={})
        best_params = optimize_hyperparameters(X, y, categorical, ad_hoc, config)
    else:
        best_params = load_prior_best_params(config.prior_best_params_path)
        print(f"Using prior best params: {best_params}")

    (config.output_dir / "best_params.json").write_text(
        json.dumps(best_params, indent=2)
    )

    model_config = _build_model_config(config, params=best_params)
    run_config = _build_run_config(config)

    pipeline = LGBMClassifierPipeline(
        config=model_config, run_config=run_config, output_dir=config.output_dir
    )

    df = pipeline.load_data(db_path=str(config.duckdb_path))
    pipeline.prepare_features(df)

    if config.load_model is not None:
        pipeline.load_final_model(config.load_model)
    else:
        pipeline.train_final()

    pipeline.compute_metrics()
    if config.run_permutation:
        pipeline.permutation_importance_analysis()
    pipeline.shap_analysis()
    pipeline.save_artefacts(save_plots=config.save_plots)
    pipeline.write_manifest()

    if config.write_predictions:
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
        )

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

    print(f"\nDone. Artefacts in {config.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

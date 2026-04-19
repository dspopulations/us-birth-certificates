"""Fit a LightGBM model for Down syndrome live-birth prediction.

Runs the same pipeline as `notebooks/00014-predictors-12.py` from the
command line: load predictors from DuckDB, stratified train/validation
split, optional Optuna hyperparameter search, LightGBM training with
early stopping, validation metrics, and optional permutation /
SHAP / correlation diagnostics.

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

Individual flags always override profile defaults. For a true smoke
test, combine ``--profile dev`` with ``--no-optimize`` and a small
``--num-boost-round``.

Examples
--------
    python scripts/fit_model.py --profile dev
    python scripts/fit_model.py --profile test
    python scripts/fit_model.py --profile reporting

    # Smoke test — seconds rather than minutes
    python scripts/fit_model.py --profile dev --no-optimize \\
        --num-boost-round 200 --no-permutation --no-shap
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from dspopulations_us_birth_certificates import (
    data_utils,
    ml_utils,
    plot_utils,
    repl_utils,
    stats_utils,
)
from dspopulations_us_birth_certificates.explain import calibration, shap_analysis
from dspopulations_us_birth_certificates.models import RunConfig
from dspopulations_us_birth_certificates.variables import Variables as vars

optuna.logging.set_verbosity(optuna.logging.WARNING)


DEFAULT_NUMERIC: list[str] = [
    vars.YEAR,
    vars.DBWT,
    vars.WTGAIN,
    vars.BMI,
    vars.MAGE_C,
    vars.FAGECOMB,
]

DEFAULT_CATEGORICAL: list[str] = [
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
]

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


# Built-in configuration profiles are sourced from the package's RunConfig
# presets (see src/.../models/common.py), so the CLI and the library agree on
# what each name means. Year range is profile-specific to the CLI (RunConfig
# is data-agnostic) and lives in this mapping.
_CLI_PROFILE_YEAR_RANGES: dict[str, tuple[int, int]] = {
    "dev": (2023, 2024),
    "test": (2020, 2024),
    "reporting": (2016, 2024),
}


def _profile_defaults(name: str) -> dict[str, object]:
    """Translate a RunConfig preset into argparse defaults for fit_model.

    Keeps the CLI a thin view of the shared presets: only the runtime knobs
    that ``fit_model.py`` actually exposes are returned.
    """
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
        "shap_subsample_size": rc.shap_subsample_size,
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
    shap_subsample_size: int | None = None
    write_predictions: bool = False
    duckdb_path: Path = Path("data/us_births.db")
    profile: str | None = None


def parse_args(argv: list[str] | None = None) -> FitConfig:
    # Pre-parse to pick up --profile first, so profile defaults can be
    # applied to the main parser before it runs. Any explicit CLI flag
    # still wins because argparse gives CLI values priority over defaults.
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

    # Profile selector (re-declared on the main parser so it shows in --help)
    p.add_argument(
        "--profile",
        choices=list(RunConfig.preset_names()),
        default=None,
        help=(
            "Configuration profile preset — sourced from RunConfig.from_name() "
            "in the package. Individual flags override profile defaults."
        ),
    )

    # Data selection
    p.add_argument("--start-year", type=int, default=2016)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument(
        "--include-unknown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat records with unknown DS status as negatives.",
    )

    # Train / eval
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

    # Hyperparameter search
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
        help="Run Optuna search; when disabled, use --prior-best-params (or built-in defaults).",
    )
    p.add_argument(
        "--prior-best-params",
        type=Path,
        default=None,
        help="JSON file with best params to use when --no-optimize is set.",
    )

    # Features
    p.add_argument(
        "--drop-features",
        nargs="*",
        default=[],
        help="Feature names to drop from the initial set before training.",
    )

    # Output
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
        "--shap-subsample-size",
        type=int,
        default=None,
        help=(
            "If set, restrict the SHAP explanation set to roughly this many "
            "rows (all positives are kept; the negative budget is split "
            "evenly between random and hard negatives). Defaults to the "
            "profile's ``shap_subsample_size``; None means the full default "
            "explanation set."
        ),
    )

    # DB writeback
    p.add_argument(
        "--write-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-row predictions back to the DuckDB database.",
    )
    p.add_argument("--duckdb-path", type=Path, default=Path("data/us_births.db"))

    # Apply profile defaults before parsing. set_defaults only changes
    # defaults, so flags passed on the command line still win.
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
        shap_subsample_size=ns.shap_subsample_size,
        write_predictions=ns.write_predictions,
        duckdb_path=ns.duckdb_path,
        profile=ns.profile,
    )


def build_xy(
    df: pd.DataFrame, drop_features: list[str]
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    numeric = [f for f in DEFAULT_NUMERIC if f not in drop_features]
    categorical = [f for f in DEFAULT_CATEGORICAL if f not in drop_features]
    features = categorical + numeric
    X = df[features].copy()
    y = df["ca_down_c_p_n"].replace({pd.NA: 0, np.nan: 0}).astype(np.int32)
    X[categorical] = X[categorical].astype("category")
    return X, y, categorical


def base_lgbm_params(config: FitConfig, num_threads: int) -> dict:
    return {
        "objective": "binary",
        "metric": ["average_precision", "binary_logloss"],
        "boosting_type": "gbdt",
        "max_bin": 255,
        "scale_pos_weight": 1,
        "force_col_wise": True,
        "seed": config.random_seed,
        "num_threads": num_threads,
        "verbosity": 1,
    }


def load_prior_best_params(path: Path | None) -> dict:
    if path is None:
        return dict(DEFAULT_PRIOR_BEST_PARAMS)
    with path.open() as f:
        return json.load(f)


def optimize_hyperparameters(
    train_data: lgb.Dataset,
    valid_data: lgb.Dataset,
    base_params: dict,
    config: FitConfig,
) -> dict:
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
        params = {**base_params, **trial_params, "verbosity": -1}
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


def train_final_model(
    train_data: lgb.Dataset,
    valid_data: lgb.Dataset,
    params: dict,
    config: FitConfig,
) -> lgb.Booster:
    full_params = {**params, "feature_pre_filter": True}
    return lgb.train(
        full_params,
        train_data,
        num_boost_round=config.num_boost_round,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            early_stopping(stopping_rounds=config.early_stopping_rounds),
            log_evaluation(period=10),
        ],
    )


def evaluate(
    gbm: lgb.Booster,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    features: list[str],
    output_dir: Path,
) -> np.ndarray:
    p_valid = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

    metrics_df, fpr, tpr, _thr, _tp, _fp, _npos = ml_utils.get_metrics(
        y_valid, p_valid, K=10_000, thr=0.01
    )
    metrics_df.to_csv(output_dir / "validation_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))

    df_imp_gain = pd.DataFrame(
        {
            "feature": features,
            "importance_gain": gbm.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance_gain", ascending=False)
    df_imp_gain.to_csv(output_dir / "feature_importance_gain.csv", index=False)

    # Precision@K / Recall@K curve
    pr_at_k = calibration.precision_recall_at_k(np.asarray(y_valid), p_valid)
    pr_at_k.to_csv(output_dir / "precision_recall_at_k.csv", index=False)

    # Top-tail calibration (predicted vs observed rates in the top fraction)
    tail = calibration.tail_calibration_table(np.asarray(y_valid), p_valid)
    tail.to_csv(output_dir / "calibration_tail.csv", index=False)

    # Log-loss / Brier at raw probabilities
    summary = {
        "best_iteration": int(gbm.best_iteration),
        "mean_predicted_prob": float(np.asarray(p_valid).mean()),
        "log_loss": float(log_loss(y_valid, p_valid, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_valid, p_valid)),
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    return p_valid, fpr, tpr


def run_diagnostics(
    gbm: lgb.Booster,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical: list[str],
    config: FitConfig,
    fpr: np.ndarray,
    tpr: np.ndarray,
) -> None:
    model_idx = 0  # single-model CLI
    if config.save_plots:
        plot_utils.plot_roc_curve(
            fpr,
            tpr,
            model_idx,
            save=True,
            output_dir=str(config.output_dir),
            file_name="roc_curve",
        )
        plot_utils.plot_precision_recall_curve(
            fpr,
            tpr,
            model_idx,
            save=True,
            output_dir=str(config.output_dir),
            file_name="precision_recall_curve",
        )

    if not (config.run_permutation or config.run_shap):
        return

    explain_kwargs: dict[str, int] = {}
    if config.shap_subsample_size is not None and config.shap_subsample_size > 0:
        # Split the negative budget roughly evenly between random and hard
        # negatives; all positives are always kept.
        n_pos = int((np.asarray(y_valid) == 1).sum())
        neg_budget = max(0, config.shap_subsample_size - n_pos)
        half = neg_budget // 2
        explain_kwargs = {
            "n_neg_rand": half,
            "n_neg_hard": neg_budget - half,
        }

    X_eval, y_eval = ml_utils.build_explain_set(
        gbm, X_valid, y_valid, categorical, **explain_kwargs
    )
    model_wrapped = ml_utils.LGBMEstimator(gbm)

    if config.run_permutation:
        result = permutation_importance(
            model_wrapped,
            X_eval,
            y_eval,
            scoring=ml_utils.ap_scorer,
            n_repeats=20,
            n_jobs=8,
            random_state=config.random_seed,
        )
        perm_df = pd.DataFrame(
            {
                "feature": X_eval.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        perm_df.to_csv(config.output_dir / "permutation_importance.csv", index=False)

        if config.save_plots:
            plot_utils.plot_permutation_importances(
                result,
                X_eval,
                model_idx,
                save=True,
                output_dir=str(config.output_dir),
                file_name="permutation_importances",
            )

    if config.run_shap:
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform

        distance, corr = stats_utils.distance_corr_dissimilarity(X_eval)
        condensed = squareform(distance, checks=True)
        dist_linkage = hierarchy.linkage(condensed, method="average")
        dendro_labels = X_eval.columns.to_list()
        if config.save_plots:
            dendro = plot_utils.plot_dendrogram(
                dist_linkage,
                dendro_labels,
                model_idx,
                save=True,
                output_dir=str(config.output_dir),
                file_name="dendrogram",
            )
            plot_utils.plot_correlation_heatmap(
                corr,
                dendro,
                label_threshold=0.3,
                model_idx=model_idx,
                save=True,
                output_dir=str(config.output_dir),
                file_name="correlation_heatmap",
            )

        explanation = shap_analysis.compute_explanation(gbm, X_eval)
        shap_analysis.shap_importance(explanation, X_eval.columns).to_csv(
            config.output_dir / "shap_importance.csv", index=False
        )
        if config.save_plots:
            shap_analysis.plot_bar(
                explanation,
                model_idx=model_idx,
                max_display=40,
                save=True,
                output_dir=str(config.output_dir),
                file_stem="shap_bar",
                show=False,
            )
            shap_analysis.plot_beeswarm(
                explanation,
                model_idx=model_idx,
                max_display=40,
                save=True,
                output_dir=str(config.output_dir),
                file_stem="shap_beeswarm",
                show=False,
            )


def write_predictions_to_duckdb(
    df: pd.DataFrame,
    gbm: lgb.Booster,
    features: list[str],
    categorical: list[str],
    config: FitConfig,
) -> None:
    import duckdb

    X_full = df[features].copy()
    X_full[categorical] = X_full[categorical].astype("category")
    df = df.copy()
    df["p_ds_lb_pred_01"] = gbm.predict(X_full, num_iteration=gbm.best_iteration)

    con = duckdb.connect(str(config.duckdb_path))
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

    n_cores = joblib.cpu_count(only_physical_cores=True)
    num_threads = config.num_threads or max(1, n_cores - 2)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    repl_utils.print_environment_info()
    print(f"\nOutput directory: {config.output_dir}\n")

    # Persist the invocation config so runs are reproducible
    (config.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **{
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in config.__dict__.items()
                },
                "num_threads": num_threads,
            },
            indent=2,
        )
    )

    df = data_utils.load_predictors_data(
        from_year=config.start_year,
        to_year=config.end_year,
        include_unknown=config.include_unknown,
        db_path=str(config.duckdb_path),
    )

    X, y, categorical = build_xy(df, config.drop_features)
    features = X.columns.to_list()
    print(f"Using {len(features)} features: {features}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=1 - config.training_split,
        stratify=y,
        random_state=config.random_seed,
    )
    print(
        f"Train: {len(y_train)} rows ({int((y_train == 1).sum())} positive); "
        f"Valid: {len(y_valid)} rows ({int((y_valid == 1).sum())} positive)"
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

    base_params = base_lgbm_params(config, num_threads)

    if config.select_hyperparameters:
        best = optimize_hyperparameters(train_data, valid_data, base_params, config)
    else:
        best = load_prior_best_params(config.prior_best_params_path)
        print(f"Using prior best params: {best}")

    (config.output_dir / "best_params.json").write_text(json.dumps(best, indent=2))

    params = {**base_params, **best}

    gbm = train_final_model(train_data, valid_data, params, config)
    gbm.save_model(
        str(config.output_dir / "model.txt"), num_iteration=gbm.best_iteration
    )

    p_valid, fpr, tpr = evaluate(gbm, X_valid, y_valid, features, config.output_dir)

    run_diagnostics(gbm, X_valid, y_valid, categorical, config, fpr, tpr)

    if config.write_predictions:
        write_predictions_to_duckdb(df, gbm, features, categorical, config)

    print(f"\nDone. Artefacts in {config.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

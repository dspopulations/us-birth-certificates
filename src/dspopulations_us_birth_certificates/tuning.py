"""Optuna hyperparameter tuning harness.

Drives an Optuna study against a ``ModelDefinition`` using the same
train/valid split as the fit pipeline. Writes ``best_params.json``,
``trials.csv``, and a picklable ``study.pkl`` under
``output/tuning/<model_id>/``.

Updates to ``ModelDefinition.params`` are a deliberate, reviewable commit
by the author — this module never mutates a model class.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

from dspopulations_us_birth_certificates import data_utils
from dspopulations_us_birth_certificates.models.base_model import ModelDefinition
from dspopulations_us_birth_certificates.models.common import RunConfig

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_lgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    """Return the LightGBM hyperparameter search space currently in use.

    Centralised so the search space is defined once and can be referenced
    from tests, manifests, and documentation. The ``feature_pre_filter``
    override is required to let Optuna change ``min_data_in_leaf`` across
    trials without rebuilding the ``lgb.Dataset``.
    """
    return {
        "feature_pre_filter": False,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.75, log=True),
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


def _load_xy(
    definition: type[ModelDefinition], db_path: str | None
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cfg = definition.to_config()
    kwargs: dict[str, Any] = dict(
        from_year=cfg.year_range[0],
        to_year=cfg.year_range[1],
        include_unknown=cfg.include_unknown,
    )
    if db_path is not None:
        kwargs["db_path"] = db_path
    df = data_utils.load_predictors_data(**kwargs)

    numeric = list(cfg.numeric_features)
    categorical = list(cfg.categorical_features)
    features = categorical + numeric
    X = df[features].copy()
    y = df[cfg.target_var].replace({pd.NA: 0, np.nan: 0}).astype(np.int32)
    X[categorical] = X[categorical].astype("category")
    return X, y, categorical


def run_optuna_study(
    definition: type[ModelDefinition],
    run_config: RunConfig,
    *,
    n_trials: int | None = None,
    timeout: int | None = None,
    output_root: Path = Path("output/tuning"),
    db_path: str | None = None,
) -> optuna.Study:
    """Run an Optuna study and persist its artefacts.

    Returns the completed ``optuna.Study``. ``best_params`` is written to
    ``output_root/<model_id>/best_params.json`` for manual copy into the
    definition. ``trials.csv`` and ``study.pkl`` are written alongside.

    ``n_trials`` defaults to ``run_config.n_trials``; pass an explicit
    value to override. ``timeout`` (seconds) is an optional budget cap.
    """
    out_dir = output_root / definition.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    n_trials = n_trials if n_trials is not None else run_config.n_trials

    X, y, categorical = _load_xy(definition, db_path=db_path)
    cfg = definition.to_config()

    training_split = cfg.train_config.get("training_split", 0.8)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=1 - training_split,
        stratify=y,
        random_state=run_config.random_seed,
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
    num_threads = cfg.train_config.get("num_threads") or max(1, n_cores - 2)
    base_params: dict = {
        **cfg.base_params,
        "seed": run_config.random_seed,
        "num_threads": num_threads,
        "verbosity": -1,
    }

    def objective(trial: optuna.Trial) -> float:
        params = {**base_params, **suggest_lgbm_params(trial)}
        pruning_cb = optuna.integration.LightGBMPruningCallback(
            trial, "average_precision"
        )
        gbm = lgb.train(
            params,
            train_data,
            num_boost_round=run_config.num_boost_round,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=run_config.early_stopping_rounds),
                lgb.log_evaluation(period=0),
                pruning_cb,
            ],
        )
        return gbm.best_score["valid_0"]["average_precision"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=run_config.random_seed),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # Persist artefacts
    (out_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    with (out_dir / "study.pkl").open("wb") as f:
        pickle.dump(study, f)

    logger.info(
        "Tuning complete for %s: best AP=%.6f over %d trials",
        definition.model_id,
        study.best_value,
        len(study.trials),
    )
    return study

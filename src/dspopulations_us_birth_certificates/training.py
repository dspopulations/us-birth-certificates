"""Training pipeline: data splitting, Optuna hyperparameter tuning, LightGBM training."""

from __future__ import annotations

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_features(
    df: pd.DataFrame,
    config,
    drop_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Select features, encode categoricals, and build X/y arrays.

    Returns (X, y, features_list, categorical_list).
    """
    features = list(config.all_features)
    categorical = list(config.categorical_features)

    if drop_features:
        features = [f for f in features if f not in drop_features]
        categorical = [c for c in categorical if c not in drop_features]

    X = df[features].copy()
    y = df["ca_down_c_p_n"].replace({pd.NA: 0, np.nan: 0}).astype(np.int32)
    X[categorical] = X[categorical].astype("category")

    return X, y, features, categorical


def create_datasets(
    X: pd.DataFrame,
    y: pd.Series,
    categorical: list[str],
    config,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    lgb.Dataset,
    lgb.Dataset,
]:
    """Train/test split and build LightGBM Dataset objects.

    Returns (X_train, X_valid, y_train, y_valid, train_data, valid_data).
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=1 - config.training_split,
        stratify=y,
        random_state=config.random_seed,
    )

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical,
        free_raw_data=False,
    )
    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical,
        reference=train_data,
        free_raw_data=False,
    )

    return X_train, X_valid, y_train, y_valid, train_data, valid_data


def print_balance(y_train: pd.Series, y_valid: pd.Series):
    """Print class balance for training and validation sets."""
    for name, y in [("Training", y_train), ("Validation", y_valid)]:
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        print(
            f"{name} set: {neg} negatives, {pos} positives, "
            f"probability positive {pos / neg:.8f}"
        )


def optimize_hyperparameters(
    train_data: lgb.Dataset,
    valid_data: lgb.Dataset,
    config,
) -> dict:
    """Run Optuna hyperparameter search and return best parameters.

    If ``config.select_hyperparameters`` is False, returns
    ``config.prior_best_params`` directly.
    """
    if not config.select_hyperparameters:
        return dict(config.prior_best_params)

    n_cores = joblib.cpu_count(only_physical_cores=True)
    base_params = {
        **config.base_params,
        "seed": config.random_seed,
        "num_threads": max(1, n_cores - 2),
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
    study.optimize(objective, n_trials=config.optimize_trials, show_progress_bar=True)

    print(f"Best trial AP: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def train_model(
    train_data: lgb.Dataset,
    valid_data: lgb.Dataset,
    params: dict,
    config,
    output_dir: str | None = None,
    model_idx: int = 0,
) -> lgb.Booster:
    """Train a LightGBM model with early stopping.

    Optionally saves the model to ``output_dir``.
    """
    n_cores = joblib.cpu_count(only_physical_cores=True)
    full_params = {
        **config.base_params,
        **params,
        "feature_pre_filter": True,
        "seed": config.random_seed,
        "num_threads": max(1, n_cores - 2),
        "verbosity": 1,
    }

    gbm = lgb.train(
        full_params,
        train_data,
        num_boost_round=config.num_boost_round,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.early_stopping_rounds),
            lgb.log_evaluation(period=10),
        ],
    )

    if output_dir:
        gbm.save_model(
            f"{output_dir}/model_{model_idx}.txt",
            num_iteration=gbm.best_iteration,
        )

    return gbm

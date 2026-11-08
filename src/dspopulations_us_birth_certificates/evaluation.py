"""Nested evaluation of a fresh recorded-status classifier training procedure.

Outer test folds never supply tuning scores, stopping decisions or categories.
The input feature list must be specified before evaluation. This module does
not validate feature lists or parameters previously selected using these data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from dspopulations_us_birth_certificates.tuning import suggest_lgbm_params


@dataclass(frozen=True)
class NestedCVConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    n_trials: int = 20
    num_boost_round: int = 2000
    early_stopping_rounds: int = 50
    random_seed: int = 67
    num_threads: int = 4

    def __post_init__(self):
        for name in ("outer_folds", "inner_folds"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least 2")
        for name in (
            "n_trials",
            "num_boost_round",
            "early_stopping_rounds",
            "num_threads",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


def fold_frames(train: pd.DataFrame, test: pd.DataFrame, categorical: tuple[str, ...]):
    """Learn category vocabularies from training rows; unseen test levels are missing."""
    train, test = train.copy(), test.copy()
    for name in categorical:
        train[name] = train[name].astype("category").cat.remove_unused_categories()
        known = test[name].isin(train[name].cat.categories)
        test[name] = test[name].where(known).astype(train[name].dtype)
    return train, test


def binary_metrics(y, probabilities) -> dict:
    return {
        "n": len(y),
        "n_positive": int(np.sum(y)),
        "recorded_fraction": float(np.mean(y)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "mean_predicted_probability": float(np.mean(probabilities)),
    }


def nested_cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    categorical: tuple[str, ...],
    base_params: dict,
    config: NestedCVConfig,
    on_fold: Callable[[dict, optuna.Study], None] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Tune separately within each outer training set, then score its test fold.

    Inner validation folds choose parameters and stopping rounds. The outer
    model uses the median selected inner stopping round and all outer training
    rows, with no early stopping on its test fold. Each row is scored once.
    Fold dispersion is descriptive, not a confidence interval from independent
    repeats. No saved tuned parameters are accepted by this interface.
    """
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must have aligned rows")
    if X.empty or not X.columns.is_unique or not X.index.is_unique:
        raise ValueError(
            "Evaluation needs nonempty data and unique columns and row indices"
        )
    labels = np.asarray(y)
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError(
            "Evaluation requires both binary classes and no missing labels"
        )
    counts = np.bincount(labels.astype(int), minlength=2)
    if counts.min() < config.outer_folds:
        raise ValueError("Each class must have at least outer_folds rows")
    if not set(categorical).issubset(X.columns):
        raise ValueError("Categorical columns must be present in X")
    # Only fixed objective settings are accepted. Tuned registry parameters
    # must not enter a fold under aliases that override the fresh search.
    allowed = {
        "objective",
        "metric",
        "boosting_type",
        "max_bin",
        "scale_pos_weight",
        "force_col_wise",
    }
    if set(base_params) - allowed:
        raise ValueError("base_params must contain only fixed objective settings")
    outer = StratifiedKFold(
        config.outer_folds, shuffle=True, random_state=config.random_seed
    )
    predictions = np.full(len(y), np.nan)
    results = []
    for fold, (training, testing) in enumerate(outer.split(X, y)):
        train_y = y.iloc[training]
        if (
            np.bincount(np.asarray(train_y, dtype=int), minlength=2).min()
            < config.inner_folds
        ):
            raise ValueError(
                "Each outer training class must have at least inner_folds rows"
            )
        inner = StratifiedKFold(
            config.inner_folds, shuffle=True, random_state=config.random_seed + fold + 1
        )
        inner_indices = list(inner.split(training, train_y))
        common = {
            **base_params,
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "num_threads": config.num_threads,
            "seed": config.random_seed + fold,
            "deterministic": True,
            "force_col_wise": True,
        }

        def objective(
            trial, common=common, inner_indices=inner_indices, training=training
        ):
            params = {**common, **suggest_lgbm_params(trial)}
            scores, rounds = [], []
            for fitting, validation in inner_indices:
                fit_rows, validation_rows = training[fitting], training[validation]
                fit_X, valid_X = fold_frames(
                    X.iloc[fit_rows], X.iloc[validation_rows], categorical
                )
                fit_data = lgb.Dataset(
                    fit_X, label=y.iloc[fit_rows], categorical_feature=list(categorical)
                )
                valid_data = lgb.Dataset(
                    valid_X, label=y.iloc[validation_rows], reference=fit_data
                )
                booster = lgb.train(
                    params,
                    fit_data,
                    num_boost_round=config.num_boost_round,
                    valid_sets=[valid_data],
                    callbacks=[
                        lgb.early_stopping(config.early_stopping_rounds, verbose=False)
                    ],
                )
                stopping_round = booster.best_iteration or config.num_boost_round
                scores.append(
                    average_precision_score(
                        y.iloc[validation_rows],
                        booster.predict(valid_X, num_iteration=stopping_round),
                    )
                )
                rounds.append(stopping_round)
            trial.set_user_attr("inner_best_iterations", rounds)
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.random_seed + fold),
        )
        study.optimize(objective, n_trials=config.n_trials)
        rounds = max(
            1, int(np.median(study.best_trial.user_attrs["inner_best_iterations"]))
        )
        train_X, test_X = fold_frames(X.iloc[training], X.iloc[testing], categorical)
        booster = lgb.train(
            {**common, **study.best_params, "feature_pre_filter": False},
            lgb.Dataset(train_X, label=train_y, categorical_feature=list(categorical)),
            num_boost_round=rounds,
        )
        scores = booster.predict(test_X, num_iteration=rounds)
        if not np.all(np.isfinite(scores) & (scores >= 0) & (scores <= 1)):
            raise RuntimeError("Non-finite or invalid outer test predictions")
        predictions[testing] = scores
        row = {
            "fold": fold + 1,
            "n_training": len(training),
            "selected_rounds": rounds,
            "inner_average_precision": study.best_value,
            **binary_metrics(y.iloc[testing], scores),
        }
        results.append(row)
        if on_fold is not None:
            on_fold({**row, "selected_params": study.best_params}, study)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Every row must receive an outer test prediction")
    return pd.DataFrame(results), {
        **binary_metrics(y, predictions),
        "evaluation": "nested_cross_validation",
        "test_used_for_tuning": False,
        "test_used_for_early_stopping": False,
        "saved_tuned_parameters_reused": False,
        "target": "recorded_checkbox",
    }

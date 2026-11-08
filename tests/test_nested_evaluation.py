"""Check that outer test rows never enter tuning or early stopping."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dspopulations_us_birth_certificates import evaluation
from dspopulations_us_birth_certificates.evaluation import NestedCVConfig, fold_frames


def test_categories_are_learned_only_from_training_rows():
    frame = pd.DataFrame({"category": pd.Categorical(["common", "train", "unseen"])})
    train, test = fold_frames(frame.iloc[:2], frame.iloc[2:], ("category",))
    assert list(train["category"].cat.categories) == ["common", "train"]
    assert test["category"].isna().all()
    assert frame["category"].iloc[2] == "unseen"


def test_nested_folds_keep_test_rows_out_of_every_training_decision(monkeypatch):
    rng = np.random.default_rng(67)
    X = pd.DataFrame({"number": rng.normal(size=240), "category": np.arange(240) % 3})
    y = pd.Series((X["number"] > 0).astype(int), index=X.index)
    config = NestedCVConfig(
        outer_folds=3,
        inner_folds=2,
        n_trials=2,
        num_boost_round=4,
        early_stopping_rounds=2,
        num_threads=1,
    )
    real_train = evaluation.lgb.train
    calls = []

    class Booster:
        def __init__(self, model, call):
            self.model, self.call = model, call
            self.best_iteration = model.best_iteration

        def predict(self, data, **kwargs):
            self.call["prediction_rows"] = set(data.index)
            return self.model.predict(data, **kwargs)

    def train(params, data, **kwargs):
        call = {
            "training_rows": set(data.data.index),
            "validation_rows": set().union(
                *(set(d.data.index) for d in kwargs.get("valid_sets", []))
            ),
            "rounds": kwargs["num_boost_round"],
            "early_stopping": bool(kwargs.get("callbacks")),
        }
        assert call["training_rows"].isdisjoint(call["validation_rows"])
        calls.append(call)
        return Booster(real_train(params, data, **kwargs), call)

    monkeypatch.setattr(evaluation.lgb, "train", train)
    monkeypatch.setattr(
        evaluation,
        "suggest_lgbm_params",
        lambda trial: {
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
            "min_data_in_leaf": 5,
        },
    )
    folds, metrics = evaluation.nested_cross_validate(
        X,
        y,
        categorical=("category",),
        base_params={"objective": "binary"},
        config=config,
    )
    all_test_rows = []
    group_size = config.n_trials * config.inner_folds + 1
    for index in range(config.outer_folds):
        inner = calls[index * group_size : (index + 1) * group_size - 1]
        final = calls[(index + 1) * group_size - 1]
        test_rows = final["prediction_rows"]
        assert final["training_rows"].isdisjoint(test_rows)
        assert not final["validation_rows"]
        assert final["early_stopping"] is False
        assert final["rounds"] == folds.loc[index, "selected_rounds"]
        for call in inner:
            assert (
                call["training_rows"] | call["validation_rows"]
                == final["training_rows"]
            )
            assert test_rows.isdisjoint(call["training_rows"] | call["validation_rows"])
            assert call["prediction_rows"] == call["validation_rows"]
        all_test_rows.extend(test_rows)
    assert sorted(all_test_rows) == list(X.index)
    assert folds["n"].sum() == len(X)
    assert metrics["n_positive"] == int(y.sum())
    assert metrics["test_used_for_tuning"] is False
    assert metrics["test_used_for_early_stopping"] is False
    assert metrics["saved_tuned_parameters_reused"] is False
    assert np.isfinite(
        list(
            metrics[key]
            for key in ("average_precision", "roc_auc", "log_loss", "brier_score")
        )
    ).all()


def test_nested_evaluation_rejects_reused_tuned_parameters():
    X = pd.DataFrame({"x": range(40)})
    y = pd.Series([0, 1] * 20)
    with pytest.raises(ValueError, match="fixed objective settings"):
        evaluation.nested_cross_validate(
            X,
            y,
            categorical=(),
            base_params={"learning_rate": 0.1},
            config=NestedCVConfig(),
        )


@pytest.mark.parametrize(
    "change",
    [{"outer_folds": 1}, {"inner_folds": 1}, {"n_trials": 0}, {"num_threads": 0}],
)
def test_invalid_nested_settings(change):
    with pytest.raises(ValueError):
        replace(NestedCVConfig(), **change)


def test_nested_evaluation_rejects_insufficient_classes():
    X = pd.DataFrame({"x": range(10)})
    y = pd.Series([1] + [0] * 9)
    with pytest.raises(ValueError, match="at least outer_folds"):
        evaluation.nested_cross_validate(
            X, y, categorical=(), base_params={}, config=NestedCVConfig()
        )

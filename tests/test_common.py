"""Tests for ``models.common`` — RunConfig and ModelConfig serialisation."""

from __future__ import annotations

import json
from datetime import date

import pytest

from dspopulations_us_birth_certificates.models import (
    ModelConfig,
    RunConfig,
    SelectionStep,
    ShapScatterSpec,
)


def test_run_config_from_name_dev() -> None:
    rc = RunConfig.from_name("dev")
    assert rc.name == "dev"
    assert rc.n_trials == 10
    assert rc.num_boost_round == 500
    assert rc.early_stopping_rounds == 50
    assert rc.cv_splits == 3
    assert rc.shap_mode == "skip"
    assert rc.shap_subsample_size is None


def test_run_config_from_name_test() -> None:
    rc = RunConfig.from_name("test")
    assert rc.n_trials == 50
    assert rc.num_boost_round == 10_000
    assert rc.cv_splits == 5
    assert rc.shap_mode == "subsample"
    assert rc.shap_subsample_size == 5_000


def test_run_config_from_name_reporting() -> None:
    rc = RunConfig.from_name("reporting")
    assert rc.n_trials == 200
    assert rc.num_boost_round == 50_000
    assert rc.shap_mode == "full"


def test_run_config_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown RunConfig preset"):
        RunConfig.from_name("prod")  # type: ignore[arg-type]


def test_run_config_seed_override() -> None:
    rc = RunConfig.from_name("dev", random_seed=123)
    assert rc.random_seed == 123


def test_run_config_preset_names_semantic_order() -> None:
    # Semantic progression dev → test → reporting matches the CLI docstring
    # and argparse ``--help`` output.
    assert RunConfig.preset_names() == ("dev", "test", "reporting")


def test_run_config_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    rc = RunConfig.from_name("dev")
    with pytest.raises(FrozenInstanceError):
        rc.n_trials = 999  # type: ignore[misc]


def _make_config() -> ModelConfig:
    return ModelConfig(
        model_id="test_m0",
        variant_of=None,
        target_var="ca_down_c_p_n",
        numeric_features=("year", "mage_c"),
        categorical_features=("sex", "rf_pdiab"),
        base_params={"objective": "binary"},
        params={"learning_rate": 0.01},
        train_config={"training_split": 0.8},
        year_range=(2016, 2024),
        include_unknown=True,
        selection_history=(
            SelectionStep(
                step_date=date(2026, 4, 17),
                rationale="initial baseline",
                features_removed=(),
                features_added=("year", "mage_c"),
            ),
            SelectionStep(
                step_date=date(2026, 4, 18),
                rationale="drop weakest predictors",
                features_removed=("ca_cdh", "apgar10"),
                metrics_before={"ap": 0.421},
                metrics_after={"ap": 0.418},
            ),
        ),
        shap_scatter_specs=(ShapScatterSpec("year", "mage_c", "age-year interaction"),),
        notes="Baseline model for smoke tests.",
    )


def test_model_config_to_dict_is_json_serialisable() -> None:
    cfg = _make_config()
    d = cfg.to_dict()
    # Round-trip through JSON to prove everything is serialisable.
    text = json.dumps(d)
    reloaded = json.loads(text)

    assert reloaded["model_id"] == "test_m0"
    assert reloaded["numeric_features"] == ["year", "mage_c"]
    assert reloaded["year_range"] == [2016, 2024]
    # Dates became ISO strings
    assert reloaded["selection_history"][0]["step_date"] == "2026-04-17"
    assert reloaded["selection_history"][1]["features_removed"] == ["ca_cdh", "apgar10"]
    assert reloaded["shap_scatter_specs"][0]["x_feature"] == "year"


def test_model_config_to_dict_preserves_tuples_as_lists() -> None:
    cfg = _make_config()
    d = cfg.to_dict()
    assert isinstance(d["categorical_features"], list)
    assert isinstance(d["selection_history"], list)
    assert isinstance(d["shap_scatter_specs"], list)

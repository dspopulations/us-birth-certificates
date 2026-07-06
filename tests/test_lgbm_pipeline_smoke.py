"""End-to-end smoke test for ``LGBMClassifierPipeline``.

Runs the pipeline end-to-end against ``synthetic_predictors_frame`` using
a minimal ad-hoc ``ModelConfig`` and the ``dev`` ``RunConfig`` preset,
then asserts that the expected artefacts land and that the trained model
beats the base-rate AP.

The USBC10-variant assertions land in step 5; the manifest assertions in
step 8.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from dspopulations_us_birth_certificates.models import ModelConfig, RunConfig
from dspopulations_us_birth_certificates.models.lgbm_pipeline import (
    LGBMClassifierPipeline,
)

from .conftest import SYNTHETIC_CATEGORICAL, SYNTHETIC_NUMERIC, SYNTHETIC_TARGET


def _smoke_config() -> ModelConfig:
    return ModelConfig(
        model_id="smoke_m0",
        variant_of=None,
        target_var=SYNTHETIC_TARGET,
        numeric_features=SYNTHETIC_NUMERIC,
        categorical_features=SYNTHETIC_CATEGORICAL,
        base_params={
            "objective": "binary",
            "metric": ["average_precision", "binary_logloss"],
            "boosting_type": "gbdt",
            "force_col_wise": True,
            "verbosity": -1,
        },
        params={
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
        },
        train_config={"training_split": 0.8, "verbosity": -1, "log_period": 0},
        year_range=(2020, 2024),
        include_unknown=True,
        selection_history=(),
        shap_scatter_specs=(),
        notes="Smoke-test config.",
    )


def test_fit_on_synthetic_fixture(
    synthetic_predictors_frame: pd.DataFrame, tmp_output_dir: Path
) -> None:
    """End-to-end pipeline run on the synthetic fixture.

    Skips ``load_data`` (DuckDB not available in CI) by driving individual
    pipeline methods. ``permutation`` and ``shap`` are skipped to keep the
    test under a few seconds.
    """
    model_config = _smoke_config()
    # Use dev preset but shrink further for test speed and disable SHAP.
    base_rc = RunConfig.from_name("dev", random_seed=0)
    run_config = replace(
        base_rc,
        num_boost_round=50,
        early_stopping_rounds=10,
        shap_mode="skip",
    )

    pipeline = LGBMClassifierPipeline(
        config=model_config, run_config=run_config, output_dir=tmp_output_dir
    )
    pipeline.prepare_features(synthetic_predictors_frame)
    pipeline.train_final()
    pipeline.compute_metrics()
    # Skip permutation_importance_analysis (slow) and shap_analysis (mode=skip).
    pipeline.save_artefacts()

    # ---- artefacts ----
    assert (tmp_output_dir / "model.txt").is_file()
    assert (tmp_output_dir / "config.json").is_file()
    assert (tmp_output_dir / "metrics.json").is_file()
    assert (tmp_output_dir / "predictions_valid.parquet").is_file()
    assert (tmp_output_dir / "precision_recall_at_k.csv").is_file()
    assert (tmp_output_dir / "calibration_tail.csv").is_file()
    assert (tmp_output_dir / "feature_importance_gain.csv").is_file()
    for stem in ("roc_curve", "precision_recall_curve"):
        for ext in (".png", ".svg", ".csv"):
            assert (tmp_output_dir / "plots" / f"{stem}{ext}").is_file(), (
                f"missing plots/{stem}{ext}"
            )

    # ROC CSV has the arrays actually plotted, sorted by fpr.
    roc_data = pd.read_csv(tmp_output_dir / "plots" / "roc_curve.csv")
    assert list(roc_data.columns) == ["fpr", "tpr"]
    assert len(roc_data) > 1

    pr_data = pd.read_csv(tmp_output_dir / "plots" / "precision_recall_curve.csv")
    assert set(pr_data.columns) == {"recall", "precision"}

    # ---- model quality ----
    metrics = json.loads((tmp_output_dir / "metrics.json").read_text())
    assert "average_precision" in metrics
    assert metrics["n_valid"] > 0
    assert metrics["n_positive_valid"] > 0
    assert metrics["evaluation_split_role"] == "early_stopping_and_reported_metrics"
    assert metrics["validation_independent_of_hyperparameter_tuning"] is True
    base_rate = metrics["n_positive_valid"] / metrics["n_valid"]
    # AP should beat the base rate by a comfortable margin given the
    # injected signal (num_age and cat_risk both drive the target).
    assert metrics["average_precision"] > base_rate * 2, (
        f"Expected AP > 2× base rate={base_rate:.4f}; "
        f"got {metrics['average_precision']:.4f}"
    )

    # ---- config round-trip ----
    cfg_json = json.loads((tmp_output_dir / "config.json").read_text())
    assert cfg_json["model_id"] == "smoke_m0"
    assert cfg_json["numeric_features"] == list(SYNTHETIC_NUMERIC)


def test_pipeline_preserves_train_valid_shapes(
    synthetic_predictors_frame: pd.DataFrame, tmp_output_dir: Path
) -> None:
    """``prepare_features`` produces correctly shaped train/valid splits."""
    model_config = _smoke_config()
    run_config = RunConfig.from_name("dev", random_seed=0)
    pipeline = LGBMClassifierPipeline(
        config=model_config, run_config=run_config, output_dir=tmp_output_dir
    )
    pipeline.prepare_features(synthetic_predictors_frame)

    ctx = pipeline.context
    assert ctx.X_train is not None
    assert ctx.X_valid is not None
    n_total = len(synthetic_predictors_frame)
    assert len(ctx.X_train) + len(ctx.X_valid) == n_total
    assert list(ctx.X_train.columns) == list(SYNTHETIC_CATEGORICAL) + list(
        SYNTHETIC_NUMERIC
    )
    # Stratified split keeps the positive rate approximately constant.
    rate_train = float(ctx.y_train.mean())
    rate_valid = float(ctx.y_valid.mean())
    assert abs(rate_train - rate_valid) < 0.01


def test_grouped_permutation_importance_outputs(
    synthetic_predictors_frame: pd.DataFrame,
    tmp_output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grouped importance is derived and saved alongside permutation output."""

    class _FakePermutationResult:
        def __init__(self, n_features: int) -> None:
            self.importances_mean = np.linspace(0.01, 0.001, n_features)
            self.importances_std = np.zeros(n_features)
            self.importances = np.tile(self.importances_mean, (3, 1)).T

    def _fake_permutation_importance(
        estimator: Any,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs: Any,
    ) -> _FakePermutationResult:
        return _FakePermutationResult(X.shape[1])

    monkeypatch.setattr(
        "dspopulations_us_birth_certificates.models.base_pipeline."
        "permutation_importance",
        _fake_permutation_importance,
    )
    monkeypatch.setattr(
        "dspopulations_us_birth_certificates.models.base_pipeline."
        "_distance_corr_linkage",
        lambda X_eval: (
            np.zeros((X_eval.shape[1], X_eval.shape[1])),
            np.eye(X_eval.shape[1]),
            np.empty((0, 4)),
        ),
    )
    monkeypatch.setattr(
        "dspopulations_us_birth_certificates.models.base_pipeline."
        "feature_groups.feature_groups_from_linkage",
        lambda names, linkage: {
            f"cluster_{i:02d}": [name] for i, name in enumerate(names, start=1)
        },
    )

    model_config = _smoke_config()
    base_rc = RunConfig.from_name("dev", random_seed=0)
    run_config = replace(
        base_rc,
        num_boost_round=50,
        early_stopping_rounds=10,
        shap_mode="skip",
    )
    pipeline = LGBMClassifierPipeline(
        config=model_config, run_config=run_config, output_dir=tmp_output_dir
    )
    pipeline.prepare_features(synthetic_predictors_frame)
    pipeline.train_final()
    pipeline.compute_metrics()
    pipeline.permutation_importance_analysis()
    pipeline.save_artefacts(save_plots=False)

    grouped = pipeline.context.permutation_importance["group_permutation_importance"]
    assert {"group", "features", "dimension_hint", "importance_mean"}.issubset(
        grouped.columns
    )
    assert len(grouped) > 0

    grouped_path = tmp_output_dir / "grouped_permutation_importance.csv"
    groups_path = tmp_output_dir / "feature_groups.csv"
    assert grouped_path.is_file()
    assert groups_path.is_file()

    grouped_csv = pd.read_csv(grouped_path)
    assert {"candidate_stage", "interpretation_flag"}.issubset(grouped_csv.columns)


def test_cross_validate_not_implemented_in_step4(
    synthetic_predictors_frame: pd.DataFrame, tmp_output_dir: Path
) -> None:
    """``cross_validate`` is a follow-up; step 4 raises explicitly."""
    pipeline = LGBMClassifierPipeline(
        config=_smoke_config(),
        run_config=RunConfig.from_name("dev"),
        output_dir=tmp_output_dir,
    )
    with pytest.raises(NotImplementedError, match="cross_validate"):
        pipeline.cross_validate()

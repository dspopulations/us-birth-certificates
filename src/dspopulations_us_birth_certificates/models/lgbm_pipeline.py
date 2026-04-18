"""LightGBM classification pipeline.

Concrete ``EstimatorPipeline`` that trains a binary LightGBM booster.
Handles categorical dtypes, ``lgb.Dataset`` construction, early stopping,
and SHAP ``TreeExplainer`` integration (via ``explain.shap_analysis``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.models.base_pipeline import (
    EstimatorPipeline,
)


class LGBMClassifierPipeline(EstimatorPipeline):
    """Binary-classification pipeline backed by ``lightgbm.train``."""

    def _full_params(self) -> dict[str, Any]:
        cfg = self.context.config
        n_cores = joblib.cpu_count(only_physical_cores=True)
        num_threads = cfg.train_config.get("num_threads") or max(1, n_cores - 2)
        return {
            **cfg.base_params,
            **cfg.params,
            "seed": self.context.run_config.random_seed,
            "num_threads": num_threads,
            "verbosity": cfg.train_config.get("verbosity", 1),
        }

    def configure_estimator(self) -> dict[str, Any]:
        """LightGBM uses parameter dicts rather than estimator objects; return it."""
        return self._full_params()

    def train_fold(self, fold_idx: int) -> Any:
        raise NotImplementedError(
            "train_fold lands with cross_validate in a follow-up PR."
        )

    def train_final(self) -> lgb.Booster:
        ctx = self.context
        if ctx.X_train is None:
            raise RuntimeError("prepare_features() must run before train_final().")

        categorical = list(ctx.config.categorical_features)
        train_data = lgb.Dataset(
            ctx.X_train,
            label=ctx.y_train,
            categorical_feature=categorical,
            free_raw_data=False,
        )
        valid_data = lgb.Dataset(
            ctx.X_valid,
            label=ctx.y_valid,
            categorical_feature=categorical,
            reference=train_data,
            free_raw_data=False,
        )

        rc = ctx.run_config
        params = self._full_params()
        params.setdefault("feature_pre_filter", True)

        gbm = lgb.train(
            params,
            train_data,
            num_boost_round=rc.num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=rc.early_stopping_rounds),
                lgb.log_evaluation(period=ctx.config.train_config.get("log_period", 10)),
            ],
        )

        ctx.final_model = gbm
        ctx.best_iteration = int(gbm.best_iteration)
        return gbm

    def _predict_valid(self) -> np.ndarray:
        ctx = self.context
        return ctx.final_model.predict(
            ctx.X_valid, num_iteration=ctx.best_iteration
        )

    def _save_final_model(self, path: Path) -> None:
        self.context.final_model.save_model(
            str(path), num_iteration=self.context.best_iteration
        )

    def _gain_importance(self) -> pd.DataFrame | None:
        ctx = self.context
        if ctx.final_model is None:
            return None
        features = ctx.final_model.feature_name()
        importance = ctx.final_model.feature_importance(importance_type="gain")
        return pd.DataFrame(
            {"feature": features, "importance_gain": importance}
        ).sort_values("importance_gain", ascending=False)

    def load_final_model(self, path: str | Path) -> lgb.Booster:
        """Load a previously saved LightGBM booster into the context.

        Use this to regenerate diagnostics (metrics, SHAP, plots) without
        retraining — e.g. via ``scripts/fit_model.py --load-model <path>``.
        ``best_iteration`` is taken from the booster's tree count since
        ``save_model(..., num_iteration=best)`` already truncates the file.
        """
        booster = lgb.Booster(model_file=str(path))
        self.context.final_model = booster
        self.context.best_iteration = booster.num_trees()
        return booster

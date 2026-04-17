"""LightGBM classification pipeline.

Concrete ``EstimatorPipeline`` that trains a binary LightGBM booster.
Handles categorical dtypes, ``lgb.Dataset`` construction, early stopping,
and SHAP ``TreeExplainer`` integration.

Implementation is populated in refactor step 4.
"""

from __future__ import annotations

from typing import Any

from dspopulations_us_birth_certificates.models.base_pipeline import (
    EstimatorPipeline,
)


class LGBMClassifierPipeline(EstimatorPipeline):
    """Binary-classification pipeline backed by ``lightgbm.train``."""

    def configure_estimator(self) -> Any:
        raise NotImplementedError("populated in refactor step 4")

    def train_fold(self, fold_idx: int) -> Any:
        raise NotImplementedError("populated in refactor step 4")

    def train_final(self) -> Any:
        raise NotImplementedError("populated in refactor step 4")

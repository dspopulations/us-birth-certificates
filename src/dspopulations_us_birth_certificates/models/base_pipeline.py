"""Abstract pipeline that orchestrates one model run.

``EstimatorPipeline`` defines the sequence of steps — each a method that
reads and writes ``self.context`` (a ``ModelFitContext``). Subclasses
(``LGBMClassifierPipeline``, future sklearn wrappers) override
``configure_estimator``, ``train_fold``, and ``train_final``.

Every step is individually callable so a notebook or a test can drive the
pipeline at any granularity. ``fit()`` is the full-run convenience.

Implementation is populated in refactor step 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from dspopulations_us_birth_certificates.models.common import (
    ModelConfig,
    ModelFitContext,
    RunConfig,
)

if TYPE_CHECKING:
    import pandas as pd


class EstimatorPipeline(ABC):
    """Orchestrates one model run end to end."""

    def __init__(self, config: ModelConfig, run_config: RunConfig) -> None:
        self.context = self._build_context(config, run_config)

    def _build_context(
        self, config: ModelConfig, run_config: RunConfig
    ) -> ModelFitContext:
        """Build an empty ``ModelFitContext`` rooted at the run's output dir."""
        raise NotImplementedError("populated in refactor step 4")

    # ---- data ----------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load the harmonised predictors frame for the configured year range."""
        raise NotImplementedError("populated in refactor step 4")

    def prepare_features(self, df: pd.DataFrame) -> None:
        """Populate ``X_train/y_train/X_valid/y_valid`` on the context."""
        raise NotImplementedError("populated in refactor step 4")

    # ---- training ------------------------------------------------------------

    @abstractmethod
    def configure_estimator(self) -> Any:
        """Return a fresh, unfitted estimator configured for this variant."""

    @abstractmethod
    def train_fold(self, fold_idx: int) -> Any:
        """Train one fold's estimator and append it to ``context.fold_models``."""

    @abstractmethod
    def train_final(self) -> Any:
        """Fit the final estimator on all training data for artefact export."""

    def cross_validate(self) -> None:
        """Run k-fold CV using ``run_config.cv_splits``."""
        raise NotImplementedError("populated in refactor step 4")

    # ---- evaluation ----------------------------------------------------------

    def compute_metrics(self) -> None:
        """Compute AP, log-loss, Brier, ROC-AUC, P@K, R@K, tail calibration."""
        raise NotImplementedError("populated in refactor step 4")

    def permutation_importance_analysis(self) -> None:
        """Populate ``context.permutation_importance``."""
        raise NotImplementedError("populated in refactor step 4")

    def shap_analysis(self) -> None:
        """Populate ``context.shap_explanation`` subject to ``run_config.shap_mode``."""
        raise NotImplementedError("populated in refactor step 4")

    # ---- outputs -------------------------------------------------------------

    def save_artefacts(self) -> None:
        """Write ``model.txt``, predictions, metrics, importances, plots."""
        raise NotImplementedError("populated in refactor step 4")

    def write_manifest(self) -> None:
        """Delegate to ``manifest.write_manifest`` for this run."""
        raise NotImplementedError("populated in refactor step 4")

    def report(self, render: bool = False) -> None:
        """Copy the Quarto template into the run dir; render if requested."""
        raise NotImplementedError("populated in refactor step 4")

    # ---- convenience ---------------------------------------------------------

    def fit(self, render: bool = False) -> ModelFitContext:
        """Run every step in order and return the populated context."""
        raise NotImplementedError("populated in refactor step 4")

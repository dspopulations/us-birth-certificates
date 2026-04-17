"""SHAP TreeExplainer helpers.

Wraps the repetitive ``shap.TreeExplainer`` + ``shap.plots.bar /
beeswarm / scatter`` calls from the current notebook into reusable helpers
that take a ``ModelFitContext``.

Implementation is populated in refactor step 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import shap

    from dspopulations_us_birth_certificates.models.common import (
        ModelFitContext,
        ShapScatterSpec,
    )


def compute_explanation(context: ModelFitContext) -> shap.Explanation:
    """Populate and return ``context.shap_explanation``."""
    raise NotImplementedError("populated in refactor step 2")


def plot_bar(
    context: ModelFitContext,
    max_display: int = 40,
    output_dir: Path | None = None,
) -> None:
    """Save the SHAP bar plot for the final model."""
    raise NotImplementedError("populated in refactor step 2")


def plot_beeswarm(
    context: ModelFitContext,
    max_display: int = 40,
    output_dir: Path | None = None,
) -> None:
    """Save the SHAP beeswarm plot for the final model."""
    raise NotImplementedError("populated in refactor step 2")


def plot_scatter(
    context: ModelFitContext,
    spec: ShapScatterSpec,
    output_dir: Path | None = None,
) -> None:
    """Save one SHAP scatter plot described by ``spec``."""
    raise NotImplementedError("populated in refactor step 2")

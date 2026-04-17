"""Reproducibility manifest for a model run.

Captures the provenance needed to reconstitute a result:

- git SHA and whether the working tree was dirty
- runtime environment (platform, CPU, RAM, Python version)
- relevant package versions (lightgbm, optuna, shap, sklearn, numpy,
  pandas, scipy, xgboost if later)
- ``ModelConfig`` snapshot
- ``RunConfig`` snapshot
- input row count and positive count
- random seeds (model seed and split seed)
- final validation metrics

Implementation is populated in refactor step 8.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspopulations_us_birth_certificates.models.common import ModelFitContext


def write_manifest(context: ModelFitContext, output_dir: Path) -> Path:
    """Serialise a run manifest to ``output_dir/manifest.json``.

    Returns the path written. Existing manifests are overwritten.
    """
    raise NotImplementedError("populated in refactor step 8")

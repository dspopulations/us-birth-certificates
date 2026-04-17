"""Optuna hyperparameter tuning harness.

Drives an Optuna study against a ``ModelDefinition`` using the same data
pipeline as the fit pipeline. Writes ``best_params.json``, ``trials.csv``,
and a picklable ``study.pkl`` under ``output/tuning/<model_id>/``.

Updates to ``ModelDefinition.params`` are a deliberate, reviewable commit
by the author — this module never mutates a model class.

Implementation is populated in refactor step 6.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna

    from dspopulations_us_birth_certificates.models.base_model import ModelDefinition
    from dspopulations_us_birth_certificates.models.common import RunConfig


def run_optuna_study(
    definition: type[ModelDefinition],
    run_config: RunConfig,
    *,
    n_trials: int,
    timeout: int | None = None,
    output_root: Path = Path("output/tuning"),
) -> optuna.Study:
    """Run an Optuna study and persist its artefacts.

    Returns the completed ``optuna.Study``. ``best_params`` is written to
    ``output_root/<model_id>/best_params.json`` for manual copy into the
    definition.
    """
    raise NotImplementedError("populated in refactor step 6")


def suggest_lgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    """Return the LightGBM hyperparameter search space currently in use.

    Centralised so the search space is defined once and can be referenced
    from tests, manifests, and documentation.
    """
    raise NotImplementedError("populated in refactor step 6")

"""Dataclasses shared across model definitions and pipelines.

Public API:
    - RunConfig — speed / fidelity preset (``dev`` / ``test`` / ``reporting``).
    - SelectionStep — provenance of one feature-selection decision.
    - ShapScatterSpec — declarative spec for a SHAP scatter plot.
    - ModelConfig — serialisable snapshot of a ``ModelDefinition``.
    - ModelFitContext — mutable state threaded through pipeline steps.

Implementations are intentionally omitted in this scaffolding PR. Follow-up
work in step 3 of ``docs/refactor-plan.md`` populates the fields and methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

RunConfigName = Literal["dev", "test", "reporting"]


@dataclass(frozen=True)
class RunConfig:
    """Speed / fidelity preset chosen at CLI invocation time."""

    name: RunConfigName
    n_trials: int
    num_boost_round: int
    early_stopping_rounds: int
    cv_splits: int
    shap_mode: Literal["skip", "subsample", "full"]
    shap_subsample_size: int | None = None
    random_seed: int = 47

    @classmethod
    def from_name(cls, name: RunConfigName) -> RunConfig:
        """Return the preset configuration for ``name``."""
        raise NotImplementedError("populated in refactor step 3")


@dataclass(frozen=True)
class SelectionStep:
    """One feature-selection decision with reproducible provenance."""

    step_date: date
    rationale: str
    features_removed: tuple[str, ...] = ()
    features_added: tuple[str, ...] = ()
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ShapScatterSpec:
    """Declarative spec for an auto-generated SHAP scatter plot."""

    x_feature: str
    colour_by_feature: str | None
    description: str = ""


@dataclass
class ModelConfig:
    """Serialisable snapshot of a ``ModelDefinition`` at fit time."""

    model_id: str
    variant_of: str | None
    target_var: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    base_params: dict[str, Any]
    params: dict[str, Any]
    train_config: dict[str, Any]
    year_range: tuple[int, int]
    include_unknown: bool
    selection_history: tuple[SelectionStep, ...]
    shap_scatter_specs: tuple[ShapScatterSpec, ...]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        raise NotImplementedError("populated in refactor step 3")


@dataclass
class ModelFitContext:
    """Mutable state threaded through pipeline steps.

    Each step reads / writes fields on this object. Large arrays
    (``shap_values``, per-fold models) are stored on the context but not
    serialised — pipelines decide when to persist them to ``output_dir``.
    """

    config: ModelConfig
    run_config: RunConfig
    output_dir: Path
    X_train: Any = None
    y_train: Any = None
    X_valid: Any = None
    y_valid: Any = None
    fold_models: list[Any] = field(default_factory=list)
    fold_predictions: list[Any] = field(default_factory=list)
    final_model: Any = None
    best_iteration: int | None = None
    p_valid: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    permutation_importance: Any = None
    shap_explanation: Any = None

"""Dataclasses shared across model definitions and pipelines.

Public API:
    - RunConfig — speed / fidelity preset (``dev`` / ``test`` / ``reporting``).
    - SelectionStep — provenance of one feature-selection decision.
    - ShapScatterSpec — declarative spec for a SHAP scatter plot.
    - ModelConfig — serialisable snapshot of a ``ModelDefinition``.
    - ModelFitContext — mutable state threaded through pipeline steps.
    - prune_features — drop a set of names from a feature tuple, order preserved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

RunConfigName = Literal["dev", "test", "reporting"]

ShapMode = Literal["skip", "subsample", "full"]


def prune_features(
    source: tuple[str, ...], removed: tuple[str, ...]
) -> tuple[str, ...]:
    """Return ``source`` with every name in ``removed`` dropped, order preserved.

    Shared by the model-variant definitions (``usbc10.py``, ``usbc11.py``, and
    their confirmed-only variants) for deriving a pruned feature tuple from a
    parent variant's feature set plus a ``SelectionStep``'s removal list.
    """
    return tuple(f for f in source if f not in removed)


# Preset values for each named run configuration. See docs/refactor-plan.md
# for the rationale behind the specific numbers. Adjustments to these presets
# are a deliberate commit by the author — they are load-bearing for
# reproducibility of tuned hyperparameters.
_PRESETS: dict[str, dict[str, Any]] = {
    "dev": {
        "n_trials": 10,
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
        "cv_splits": 3,
        "shap_mode": "skip",
        "shap_subsample_size": None,
    },
    "test": {
        "n_trials": 50,
        "num_boost_round": 10_000,
        "early_stopping_rounds": 200,
        "cv_splits": 5,
        "shap_mode": "subsample",
        "shap_subsample_size": 5_000,
    },
    "reporting": {
        "n_trials": 200,
        "num_boost_round": 50_000,
        "early_stopping_rounds": 200,
        "cv_splits": 5,
        "shap_mode": "full",
        "shap_subsample_size": None,
    },
}


@dataclass(frozen=True)
class RunConfig:
    """Speed / fidelity preset chosen at CLI invocation time."""

    name: RunConfigName
    n_trials: int
    num_boost_round: int
    early_stopping_rounds: int
    cv_splits: int
    shap_mode: ShapMode
    shap_subsample_size: int | None = None
    random_seed: int = 47

    @classmethod
    def from_name(cls, name: RunConfigName, *, random_seed: int = 47) -> RunConfig:
        """Return the preset configuration for ``name``.

        ``random_seed`` overrides the default for the returned instance,
        since determinism is usually controlled at the CLI layer rather
        than baked into the preset.
        """
        if name not in _PRESETS:
            raise ValueError(
                f"Unknown RunConfig preset {name!r}. Valid names: {sorted(_PRESETS)}"
            )
        return cls(name=name, random_seed=random_seed, **_PRESETS[name])

    @classmethod
    def preset_names(cls) -> tuple[str, ...]:
        """Return valid preset names in semantic order (dev → test → reporting).

        The order matches ``_PRESETS`` insertion order so argparse's ``--help``
        output agrees with the documented progression in the CLI docstring.
        """
        return tuple(_PRESETS)


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
    confirmed_only: bool = False
    notes: str = ""
    # Names of the DuckDB columns that receive this model's predictions
    # and derived year×month predicted-missing flag. Each model variant
    # owns its own column pair so diagnostic re-runs don't overwrite a
    # previous model's predictions. Defaults reflect the usbc10 family.
    predictions_column: str = "p_ds_lb_pred_01"
    missing_flag_column: str = "ds_pred_missing"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation.

        Tuples become lists; ``date`` values become ISO strings so the
        result can be written straight to ``json.dumps``.
        """
        return {
            "model_id": self.model_id,
            "variant_of": self.variant_of,
            "target_var": self.target_var,
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "base_params": dict(self.base_params),
            "params": dict(self.params),
            "train_config": dict(self.train_config),
            "year_range": list(self.year_range),
            "include_unknown": self.include_unknown,
            "confirmed_only": self.confirmed_only,
            "selection_history": [
                {
                    **asdict(step),
                    "step_date": step.step_date.isoformat(),
                    "features_removed": list(step.features_removed),
                    "features_added": list(step.features_added),
                }
                for step in self.selection_history
            ],
            "shap_scatter_specs": [asdict(s) for s in self.shap_scatter_specs],
            "notes": self.notes,
            "predictions_column": self.predictions_column,
            "missing_flag_column": self.missing_flag_column,
        }


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

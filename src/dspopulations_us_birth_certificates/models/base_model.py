"""Base class for declarative model definitions.

Subclasses set class attributes (``model_id``, ``target_var``, ``params``,
``selection_steps``, ``shap_scatter_specs``) and are auto-registered into
the module-level ``MODELS`` dict via ``__init_subclass__``.

Feature-selection history is reconstructed by walking the MRO from the
root ancestor down to the concrete subclass and concatenating each class's
``selection_steps``. Variants therefore inherit the full decision history
of their parent without duplication.

Implementation is populated in refactor step 3 (see
``docs/refactor-plan.md``).
"""

from __future__ import annotations

from typing import Any, ClassVar

from dspopulations_us_birth_certificates.models.common import (
    ModelConfig,
    SelectionStep,
    ShapScatterSpec,
)

MODELS: dict[str, type[ModelDefinition]] = {}


class ModelDefinition:
    """Declarative definition of a model variant.

    Class attributes (set on subclasses):
        model_id: Unique identifier (e.g. ``"usbc10_m0"``).
        variant_of: ``model_id`` of the parent variant, if any.
        target_var: Column name of the binary target.
        numeric_features: Ordered tuple of numeric feature names.
        categorical_features: Ordered tuple of categorical feature names.
        base_params: LightGBM parameters that are invariant across variants
            (objective, metric, seed, threads).
        params: Tuned hyperparameters (from ``scripts/tune_model.py``).
        train_config: Training-loop parameters (num_boost_round,
            early_stopping_rounds, validation split fraction).
        year_range: ``(from_year, to_year)`` inclusive.
        include_unknown: Whether to include rows where ``ca_down_c_p_n``
            is missing / unknown.
        selection_steps: Feature-selection decisions introduced by *this*
            class. The full history across ancestors is reconstructed by
            ``selection_history``.
        shap_scatter_specs: SHAP scatter plots to produce for this variant.
        notes: Free-text rationale.
    """

    model_id: ClassVar[str] = ""
    variant_of: ClassVar[str | None] = None
    target_var: ClassVar[str] = ""
    numeric_features: ClassVar[tuple[str, ...]] = ()
    categorical_features: ClassVar[tuple[str, ...]] = ()
    base_params: ClassVar[dict[str, Any]] = {}
    params: ClassVar[dict[str, Any]] = {}
    train_config: ClassVar[dict[str, Any]] = {}
    year_range: ClassVar[tuple[int, int]] = (2016, 2024)
    include_unknown: ClassVar[bool] = True
    selection_steps: ClassVar[tuple[SelectionStep, ...]] = ()
    shap_scatter_specs: ClassVar[tuple[ShapScatterSpec, ...]] = ()
    notes: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.model_id:
            if cls.model_id in MODELS:
                raise ValueError(
                    f"Duplicate model_id {cls.model_id!r}: "
                    f"{MODELS[cls.model_id]} vs {cls}"
                )
            MODELS[cls.model_id] = cls

    @classmethod
    def selection_history(cls) -> tuple[SelectionStep, ...]:
        """Return the concatenated selection history across this class's MRO.

        Walks from the oldest ancestor to this class, concatenating each
        class's ``selection_steps`` — so a variant inherits its parent's
        decisions and then adds its own.
        """
        raise NotImplementedError("populated in refactor step 3")

    @classmethod
    def to_config(cls) -> ModelConfig:
        """Return an immutable ``ModelConfig`` snapshot of this definition."""
        raise NotImplementedError("populated in refactor step 3")

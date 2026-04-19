"""Base class + registry for Bayesian model definitions.

Subclasses of ``BayesModelDefinition`` auto-register into the module-level
``MODELS`` dict on import, mirroring the LightGBM side in
``models/base_model.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from dspopulations_us_birth_certificates.bayes.config import (
    BayesModelConfig,
    OutcomeName,
)

if TYPE_CHECKING:
    import pandas as pd
    import pymc as pm


MODELS: dict[str, type[BayesModelDefinition]] = {}


class BayesModelDefinition:
    """Declarative definition of a Bayesian model variant.

    Class attributes (set on subclasses):
        model_id: Unique identifier (e.g. ``"m1-year-age"``).
        dims: Ordered tuple of cell dimensions (e.g. ``("year", "mage_c")``).
            These are the SQL ``GROUP BY`` columns in ``load_cells``.
        year_range: ``(from_year, to_year)`` inclusive.
        outcomes: Outcome dataset names this model can be fit against.
            Defaults to both ``recorded`` and ``recorded_plus_predicted``.
        priors: Free-form dict of prior hyperparameters. Serialised into
            ``BayesModelConfig`` so fits are reproducible from artefacts.
        notes: Free-text description.

    Subclasses override ``build(cells)`` to return a ``pm.Model``.
    """

    model_id: ClassVar[str] = ""
    dims: ClassVar[tuple[str, ...]] = ()
    year_range: ClassVar[tuple[int, int]] = (2016, 2024)
    outcomes: ClassVar[tuple[OutcomeName, ...]] = (
        "recorded",
        "recorded_plus_predicted",
    )
    priors: ClassVar[dict[str, Any]] = {}
    notes: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.model_id:
            if cls.model_id in MODELS:
                raise ValueError(
                    f"Duplicate Bayes model_id {cls.model_id!r}: "
                    f"{MODELS[cls.model_id]} vs {cls}"
                )
            MODELS[cls.model_id] = cls

    @classmethod
    def build(cls, cells: pd.DataFrame) -> pm.Model:
        """Return a ``pm.Model`` constructed from the aggregated cell frame.

        ``cells`` has columns for each dim in ``cls.dims`` plus ``n_cell``
        (exposure) and ``y_cell`` (positive count). Implementations should
        use ``pm.Data`` for inputs so the model can be re-evaluated at
        new coords.
        """
        raise NotImplementedError

    @classmethod
    def to_config(
        cls, *, outcome: OutcomeName, outcome_params: dict[str, Any] | None = None
    ) -> BayesModelConfig:
        return BayesModelConfig(
            model_id=cls.model_id,
            dims=tuple(cls.dims),
            year_range=tuple(cls.year_range),
            outcome=outcome,
            outcome_params=dict(outcome_params or {}),
            priors=dict(cls.priors),
            notes=cls.notes,
        )

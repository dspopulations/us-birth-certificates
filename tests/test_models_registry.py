"""Structural invariants of the ``MODELS`` registry.

Ensures every ``ModelDefinition`` in the registry is well-formed:

- model_ids are unique
- every ``variant_of`` resolves to a registered model
- every SelectionStep carries a rationale and a date
- ``selection_history()`` chains through the MRO, so a descendant's
  history includes every ancestor's steps in order.
"""

from __future__ import annotations

from datetime import date

import pytest

from dspopulations_us_birth_certificates.models import MODELS, ModelDefinition


def test_registry_non_empty() -> None:
    assert MODELS, "MODELS registry should contain at least one definition"


def test_every_model_id_is_unique() -> None:
    ids = [cls.model_id for cls in MODELS.values()]
    assert len(ids) == len(set(ids))


def test_every_variant_of_resolves_in_registry() -> None:
    for cls in MODELS.values():
        if cls.variant_of is not None:
            assert cls.variant_of in MODELS, (
                f"{cls.model_id!r} declares variant_of={cls.variant_of!r} "
                f"but that id is not in MODELS"
            )


def test_every_selection_step_has_rationale_and_date() -> None:
    for cls in MODELS.values():
        for step in cls.selection_steps:
            assert step.rationale.strip(), (
                f"{cls.model_id}: SelectionStep missing rationale"
            )
            assert isinstance(step.step_date, date), (
                f"{cls.model_id}: SelectionStep.step_date must be a date"
            )


def test_selection_history_chains_through_mro() -> None:
    # USBC10_M2 inherits from USBC10_M1 inherits from USBC10_M0.
    # selection_history() walks oldest ancestor first.
    m0 = MODELS["usbc10_m0"]
    m1 = MODELS["usbc10_m1"]
    m2 = MODELS["usbc10_m2"]

    assert len(m0.selection_history()) == 1
    assert len(m1.selection_history()) == 2
    assert len(m2.selection_history()) == 3

    m2_history = m2.selection_history()
    assert m2_history[0].rationale == m0.selection_steps[0].rationale
    assert m2_history[1].rationale == m1.selection_steps[0].rationale
    assert m2_history[2].rationale == m2.selection_steps[0].rationale


def test_to_config_reflects_variant_state() -> None:
    m1 = MODELS["usbc10_m1"]
    m0 = MODELS["usbc10_m0"]

    m1_cfg = m1.to_config()
    m0_cfg = m0.to_config()

    # M1 inherits numeric features from M0 (no numeric removals in this variant).
    assert m1_cfg.numeric_features == m0_cfg.numeric_features
    # M1's categorical set is strictly smaller than M0's.
    assert len(m1_cfg.categorical_features) < len(m0_cfg.categorical_features)
    removed = set(m0_cfg.categorical_features) - set(m1_cfg.categorical_features)
    assert removed  # non-empty


def test_abstract_subclass_without_model_id_is_not_registered() -> None:
    """A subclass without model_id doesn't register (used for abstract helpers)."""
    before = dict(MODELS)

    class _AbstractHelper(ModelDefinition):
        """No model_id — not registered."""

    assert dict(MODELS) == before


def test_duplicate_model_id_raises() -> None:
    with pytest.raises(ValueError, match="Duplicate model_id"):

        class _ConflictingM0(ModelDefinition):
            model_id = "usbc10_m0"  # already taken

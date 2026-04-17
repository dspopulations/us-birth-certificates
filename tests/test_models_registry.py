"""Structural invariants of the ``MODELS`` registry.

These tests run as soon as the concrete model definitions land (step 5).
Until then they are skipped but still collected so CI fails if a test
disappears by accident.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="populated in refactor step 5")


def test_every_model_id_is_unique() -> None: ...


def test_every_variant_of_resolves_in_registry() -> None: ...


def test_every_selection_step_has_rationale_and_date() -> None: ...


def test_selection_history_chains_through_mro() -> None: ...

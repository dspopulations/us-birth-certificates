"""Pytest fixtures for the ``us-birth-certificates`` test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def synthetic_predictors_frame() -> object:
    """A small synthetic predictors frame matching the NVSS schema.

    Targets a realistic class imbalance (~1:800 positives) so smoke tests
    exercise the pipeline's rare-event handling without requiring real
    NVSS microdata.

    Implementation is populated in refactor step 4 alongside the first
    smoke test.
    """
    pytest.skip("fixture populated in refactor step 4")


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Iterator[Path]:
    """Per-test ``output/models/<run>/`` directory."""
    out = tmp_path / "output"
    out.mkdir()
    yield out

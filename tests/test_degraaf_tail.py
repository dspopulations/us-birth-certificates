"""Tests for the de Graaf column-I tail sensitivity splice.

The production prevalence anchor stays the default; ``apply_degraaf_tail`` only swaps
the 2020-2024 target for the five named de Graaf races. These guard that contract.
"""

from __future__ import annotations

import numpy as np

from dspopulations_us_birth_certificates.selection import degraaf_tail as DT
from dspopulations_us_birth_certificates.selection.recording_anchor import (
    ANCHOR_YEARS,
    PREV_RACE_YEAR,
)


def test_tail_block_shape_and_finite() -> None:
    assert DT.DEGRAAF_TAIL_PREV.shape == (5, 5)
    assert np.isfinite(DT.DEGRAAF_TAIL_PREV).all()
    assert DT.DEGRAAF_TAIL_YEARS == (2020, 2021, 2022, 2023, 2024)


def test_apply_preserves_shape_and_head_columns() -> None:
    out = DT.apply_degraaf_tail(PREV_RACE_YEAR)
    assert out.shape == PREV_RACE_YEAR.shape
    # 2016-2019 untouched (all rows, all columns before the tail).
    head_cols = [j for j, y in enumerate(ANCHOR_YEARS) if y < DT.DEGRAAF_TAIL_FIRST_YEAR]
    assert np.array_equal(
        out[:, head_cols], PREV_RACE_YEAR[:, head_cols], equal_nan=True
    )


def test_apply_replaces_named_race_tail_only() -> None:
    out = DT.apply_degraaf_tail(PREV_RACE_YEAR)
    tail_cols = [j for j, y in enumerate(ANCHOR_YEARS) if y >= DT.DEGRAAF_TAIL_FIRST_YEAR]
    # Named races (idx 0-4) tail equals de Graaf column I.
    assert np.allclose(out[:5, tail_cols], DT.DEGRAAF_TAIL_PREV)
    # Unknown / Multi-race rows (idx 5-6) are left as the production target (NaN here).
    assert np.array_equal(
        np.isnan(out[5:, :]), np.isnan(PREV_RACE_YEAR[5:, :])
    )


def test_does_not_mutate_input() -> None:
    before = PREV_RACE_YEAR.copy()
    DT.apply_degraaf_tail(PREV_RACE_YEAR)
    assert np.array_equal(PREV_RACE_YEAR, before, equal_nan=True)

"""Tests for NVSS variable harmonisation across coding-boundary years.

Targets the ``mrace_c`` (race) and ``mhisp_c`` (Hispanic origin) merge
rules documented in ``previous/us-birth-certificates/data-preparation.md``.
Cross-coding boundaries to pin down: 2002→2003 (ORRACEM → UMHISP),
2013→2014 (MRACEREC → MBRACE / MRACE15 / MRACE6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.data_utils import map_mhisp, map_mrace
from dspopulations_us_birth_certificates.variables import Variables as vars


def _row(**kwargs) -> pd.Series:
    """Build a row with NaN defaults and overrides from kwargs."""
    all_cols = {
        vars.MRACE: np.nan,
        vars.MRACEREC: np.nan,
        vars.MBRACE: np.nan,
        vars.MRACE15: np.nan,
        vars.ORRACEM: np.nan,
        vars.UMHISP: np.nan,
        vars.MHISP_R: np.nan,
        vars.MHISPX: np.nan,
    }
    all_cols.update(kwargs)
    return pd.Series(all_cols)


# ---- map_mrace -----------------------------------------------------------


def test_map_mrace_prefers_mrace15_when_available() -> None:
    row = _row(**{vars.MRACE15: 2, vars.MRACEREC: 3, vars.MBRACE: 4, vars.MRACE: 1})
    assert map_mrace(row) == 2


def test_map_mrace_mrace15_high_codes_collapse_to_4() -> None:
    # MRACE15 values 4..14 all map to the "Other" category (4).
    for v in (4, 7, 10, 14):
        assert map_mrace(_row(**{vars.MRACE15: v})) == 4


def test_map_mrace_falls_back_to_mracerec() -> None:
    # No MRACE15 → use MRACEREC.
    row = _row(**{vars.MRACEREC: 3, vars.MBRACE: 2, vars.MRACE: 1})
    assert map_mrace(row) == 3


def test_map_mrace_falls_back_to_mbrace() -> None:
    row = _row(**{vars.MBRACE: 2, vars.MRACE: 1})
    assert map_mrace(row) == 2


def test_map_mrace_legacy_mrace_high_codes_collapse_to_4() -> None:
    # MRACE values 4..78 all map to 4 in the legacy coding.
    for v in (4, 20, 77, 78):
        assert map_mrace(_row(**{vars.MRACE: v})) == 4


def test_map_mrace_returns_nan_when_all_missing() -> None:
    assert pd.isna(map_mrace(_row()))


# ---- map_mhisp -----------------------------------------------------------


def test_map_mhisp_prefers_mhispx_when_available() -> None:
    # MHISPX is the newest column; it wins the fallback chain.
    row = _row(
        **{
            vars.MHISPX: 2,
            vars.MHISP_R: 3,
            vars.UMHISP: 1,
            vars.ORRACEM: 9,
        }
    )
    assert map_mhisp(row) == 2


def test_map_mhisp_mhispx_high_codes_collapse_to_4() -> None:
    for v in (4, 5, 6):
        assert map_mhisp(_row(**{vars.MHISPX: v})) == 4


def test_map_mhisp_unknown_code_9_maps_to_5() -> None:
    assert map_mhisp(_row(**{vars.MHISPX: 9})) == 5


def test_map_mhisp_falls_back_to_mhisp_r() -> None:
    row = _row(**{vars.MHISP_R: 1, vars.UMHISP: 2, vars.ORRACEM: 3})
    assert map_mhisp(row) == 1


def test_map_mhisp_falls_back_to_umhisp() -> None:
    row = _row(**{vars.UMHISP: 3, vars.ORRACEM: 1})
    assert map_mhisp(row) == 3


def test_map_mhisp_orracem_non_hispanic_codes_become_zero() -> None:
    # ORRACEM values 6-8 indicate "non-Hispanic" and should map to 0.
    for v in (6, 7, 8):
        assert map_mhisp(_row(**{vars.ORRACEM: v})) == 0


def test_map_mhisp_orracem_hispanic_codes_preserved() -> None:
    for v in (1, 2, 3):
        assert map_mhisp(_row(**{vars.ORRACEM: v})) == v


def test_map_mhisp_returns_nan_when_all_missing() -> None:
    assert pd.isna(map_mhisp(_row()))

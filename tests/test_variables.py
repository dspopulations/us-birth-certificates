"""Tests for NVSS variable harmonisation across coding-boundary years.

Targets the ``mrace_c`` (race) and ``mhisp_c`` (Hispanic origin) merge
rules documented in ``previous/us-birth-certificates/data-preparation.md``.
Cross-coding boundaries to pin down: 2002→2003 (ORRACEM → UMHISP),
2013→2014 (MRACEREC → MBRACE / MRACE15 / MRACE6).

Implementation is populated in refactor step 5.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="populated in refactor step 5")


def test_mrace_c_preserves_pre_2014_codings() -> None: ...


def test_mrace_c_maps_2014_plus_codings() -> None: ...


def test_mhisp_c_preserves_pre_2003_codings() -> None: ...


def test_mhisp_c_maps_2003_plus_codings() -> None: ...

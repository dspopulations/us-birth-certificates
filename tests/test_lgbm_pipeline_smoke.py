"""End-to-end smoke test for ``LGBMClassifierPipeline``.

Fits ``USBC10_M0`` at the ``dev`` preset against ``synthetic_predictors_frame``
and asserts:

- ``model.txt`` exists in the run's output dir
- ``metrics.json`` contains ``average_precision`` and beats the base rate
- ``plots/roc.png`` and ``plots/pr.png`` are non-empty
- ``manifest.json`` carries a git SHA and package versions

Implementation is populated in refactor step 4 (artefact assertions) and
extended in step 8 (manifest assertions).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="populated in refactor step 4")


def test_fit_usbc10_m0_dev_on_synthetic_fixture() -> None: ...

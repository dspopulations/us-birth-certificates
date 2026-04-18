"""Pytest fixtures for the ``us-birth-certificates`` test suite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Synthetic schema used by the smoke fixture. Names are deliberately unrelated
# to the real NVSS columns so tests don't accidentally lean on live data.
SYNTHETIC_NUMERIC = ("num_age", "num_weight", "num_gain", "num_bmi")
SYNTHETIC_CATEGORICAL = (
    "cat_sex",
    "cat_risk",
    "cat_care",
    "cat_insurance",
    "cat_region",
)
SYNTHETIC_TARGET = "ca_down_c_p_n"


@pytest.fixture
def synthetic_predictors_frame() -> pd.DataFrame:
    """A small synthetic predictors frame with a known signal.

    5 000 rows with roughly 1:100 positives. The target is correlated with
    ``num_age`` and ``cat_risk`` so the LightGBM smoke run has something to
    learn, but the base rate stays rare enough to exercise the pipeline's
    rare-event code paths.
    """
    rng = np.random.default_rng(0)
    n = 5_000

    num_age = rng.integers(15, 50, size=n).astype(np.float64)
    num_weight = rng.normal(3_300, 600, size=n)
    num_gain = rng.integers(0, 50, size=n).astype(np.float64)
    num_bmi = rng.uniform(18, 40, size=n)

    cat_sex = rng.integers(0, 2, size=n).astype(np.int32)
    cat_risk = rng.integers(0, 2, size=n).astype(np.int32)
    cat_care = rng.integers(0, 5, size=n).astype(np.int32)
    cat_insurance = rng.integers(0, 4, size=n).astype(np.int32)
    cat_region = rng.integers(0, 10, size=n).astype(np.int32)

    # Latent score: older mothers + risk flag => higher probability
    logit = (
        -5.0  # base: rare event
        + 0.06 * (num_age - 25)
        + 1.2 * cat_risk
        + rng.normal(0, 0.5, size=n)
    )
    p_pos = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.uniform(0, 1, size=n) < p_pos).astype(np.int32)

    return pd.DataFrame(
        {
            "num_age": num_age,
            "num_weight": num_weight,
            "num_gain": num_gain,
            "num_bmi": num_bmi,
            "cat_sex": cat_sex,
            "cat_risk": cat_risk,
            "cat_care": cat_care,
            "cat_insurance": cat_insurance,
            "cat_region": cat_region,
            SYNTHETIC_TARGET: target,
        }
    )


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Iterator[Path]:
    """Per-test ``output/models/<run>/`` directory."""
    out = tmp_path / "output"
    out.mkdir()
    yield out

"""End-to-end smoke test for the ``m1-year-age`` Bayesian model.

Builds the PyMC model on a tiny synthetic cell frame and runs a 50-draw
fit — just enough to exercise the prior-predictive + NUTS + posterior-
predictive path without spending real time on convergence. Assertions
focus on shapes and artefact presence, not posterior quality.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dspopulations_us_birth_certificates.bayes import (
    MODELS,
    BayesFitContext,
    BayesRunConfig,
    sample,
    save_artefacts,
)


@pytest.fixture
def synthetic_cells() -> pd.DataFrame:
    """Year×age cells with a small, realistic-shape signal."""
    rng = np.random.default_rng(0)
    years = np.arange(2020, 2023)
    ages = np.arange(20, 30)
    records = []
    for year in years:
        for age in ages:
            n = int(rng.integers(300, 600))
            base_rate = 1e-4 * (age - 19)  # age 20 → 1e-4, age 29 → 1e-3
            y = int(rng.binomial(n, base_rate))
            records.append({"year": year, "mage_c": age, "n_cell": n, "y_cell": y})
    return pd.DataFrame(records)


def test_m1_builds_and_samples(synthetic_cells: pd.DataFrame, tmp_path: Path) -> None:
    definition = MODELS["m1-year-age"]
    model = definition.build(synthetic_cells)

    # Minimal config: 50 draws/tune, 1 chain, default NUTS (avoid nutpie
    # compile overhead in CI).
    base = BayesRunConfig.from_name("dev", random_seed=0)
    run_config = replace(
        base,
        draws=50,
        tune=50,
        chains=1,
        prior_predictive_samples=50,
        nuts_sampler="pymc",
    )

    idata = sample(model, config=run_config)

    # Shapes.
    assert "prior" in idata.groups()
    assert "prior_predictive" in idata.groups()
    assert "posterior" in idata.groups()
    assert "posterior_predictive" in idata.groups()
    assert idata.posterior.sizes["draw"] == 50
    assert idata.posterior.sizes["chain"] == 1
    assert "alpha" in idata.posterior.data_vars
    assert "p" in idata.posterior.data_vars
    assert idata.posterior["p"].sizes["cell"] == len(synthetic_cells)

    # Artefacts.
    context = BayesFitContext(
        config=definition.to_config(outcome="recorded"),
        run_config=run_config,
        output_dir=tmp_path,
        cells=synthetic_cells,
        model=model,
        idata=idata,
    )
    save_artefacts(context, tmp_path)
    assert (tmp_path / "idata.nc").is_file()
    assert (tmp_path / "cells.parquet").is_file()
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "run_config.json").is_file()

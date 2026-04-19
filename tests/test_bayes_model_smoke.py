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

pytest.importorskip("pymc")

from dspopulations_us_birth_certificates.bayes import (  # noqa: E402
    MODELS,
    BayesFitContext,
    BayesRunConfig,
    diagnostics,
    plots,
    sample,
    save_artefacts,
    save_prior_predictive_summary,
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

    # Prior predictive summary carries all the expected rows.
    pps = diagnostics.prior_predictive_summary(idata, synthetic_cells)
    expected_rows = {
        "baseline_rate",
        "cell_rate_mean",
        "y_cell_min_exposure",
        "y_cell_max_exposure",
        "year_trend_rate_ratio",
        "age_gradient_rate_ratio",
        "ls_year_coord_units",
        "ls_age_coord_units",
    }
    assert expected_rows.issubset(set(pps.index))
    assert list(pps.columns) == ["unit", "median", "hdi_lo", "hdi_hi"]
    assert pps["median"].notna().all()
    assert (pps.loc["baseline_rate", "median"] > 0) and (
        pps.loc["baseline_rate", "median"] < 1
    )

    save_prior_predictive_summary(pps, tmp_path)
    assert (tmp_path / "prior_predictive_summary.csv").is_file()

    # Prior-draws plot and prior/posterior overlay write artefacts.
    plots_dir = tmp_path / "plots"
    plots.plot_prior_draws(
        idata,
        synthetic_cells,
        coord_name="year",
        smooth_name="f_year",
        output_path=plots_dir / "prior_year_draws",
        n_draws=10,
    )
    plots.plot_prior_draws(
        idata,
        synthetic_cells,
        coord_name="mage_c",
        smooth_name="f_age",
        output_path=plots_dir / "prior_mage_c_draws",
        n_draws=10,
    )
    plots.plot_prior_posterior_overlay(
        idata,
        var_names=("alpha", "ls_year", "eta_year"),
        output_path=plots_dir / "prior_posterior_overlay",
    )
    for stem in ("prior_year_draws", "prior_mage_c_draws", "prior_posterior_overlay"):
        for ext in (".png", ".svg", ".csv"):
            assert (plots_dir / f"{stem}{ext}").is_file(), (
                f"missing {stem}{ext}"
            )

    # CSV companion for the prior-draws plot is long-format with the
    # right series labels and carries the coord column we asked for.
    prior_draws_csv = pd.read_csv(plots_dir / "prior_year_draws.csv")
    assert "year" in prior_draws_csv.columns
    assert "series" in prior_draws_csv.columns
    series = set(prior_draws_csv["series"].unique())
    assert {"hdi_lo", "hdi_hi", "median"}.issubset(series)
    assert any(s.startswith("draw_") for s in series)

    # Prior/posterior overlay CSV is tidy (variable, group, value).
    overlay_csv = pd.read_csv(plots_dir / "prior_posterior_overlay.csv")
    assert set(overlay_csv["group"].unique()) == {"prior", "posterior"}

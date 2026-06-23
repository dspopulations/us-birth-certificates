"""Smoke tests for ``selection.diagnostics``.

Fits a tiny-sample full-spec model once per module and verifies each of
the six diagnostic functions returns a matplotlib Figure with at least
one Axes. The fit is deliberately under-converged — these tests guard
against API / shape regressions, not posterior quality.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless-safe

pytest.importorskip("pymc")
pytest.importorskip("arviz")

from dspopulations_us_birth_certificates.selection import (  # noqa: E402
    TrueParams,
    build_model,
    diagnostics,
    simulate_cells,
    variant_C_default,
)

N_YEAR = 9


@pytest.fixture(scope="module")
def fitted() -> tuple[pd.DataFrame, object]:
    """Tiny synthetic full-spec fit (fast, under-converged on purpose)."""
    import pymc as pm

    truth = TrueParams.from_priors(
        variant_C_default(), n_year=N_YEAR, seed=0
    )
    cells = simulate_cells(
        truth,
        n_cells_per_month=3,
        n_year=N_YEAR,
        n_cells_mean=2000,
        seed=0,
    )
    model = build_model(
        cells, variant_C_default(), spec="full", n_year=N_YEAR
    )
    with model:
        idata = pm.sample(
            draws=80,
            tune=80,
            chains=2,
            target_accept=0.9,
            random_seed=0,
            progressbar=False,
            nuts_sampler="pymc",
        )
        idata.update(pm.sample_posterior_predictive(idata, random_seed=0))
    return cells, idata


def _assert_has_axes(fig) -> None:
    assert fig is not None
    axes = fig.get_axes()
    assert len(axes) >= 1


def test_identifiability_pairplot(fitted) -> None:
    _, idata = fitted
    fig = diagnostics.identifiability_pairplot(idata)
    _assert_has_axes(fig)

    table = diagnostics.identifiability_table(idata)
    assert {"race_idx", "correlation", "abs_correlation"}.issubset(table.columns)
    assert len(table) >= 1
    assert (table["abs_correlation"] >= 0).all()
    assert (table["abs_correlation"] <= 1).all()


def test_eta_term_year_trajectory_plot(fitted) -> None:
    _, idata = fitted
    fig = diagnostics.eta_term_year_trajectory_plot(idata)
    _assert_has_axes(fig)

    table = diagnostics.eta_term_year_trajectory_table(idata)
    assert len(table) == N_YEAR
    assert {"year_idx", "posterior_mean", "lo", "hi"}.issubset(table.columns)


def test_cchd_consistency_check(fitted) -> None:
    cells, idata = fitted
    fig = diagnostics.cchd_consistency_check(idata, cells)
    _assert_has_axes(fig)

    summary = diagnostics.cchd_consistency_summary(idata, cells)
    assert {"posterior_mean", "lo_95", "hi_95", "target", "target_in_95_ci"} == set(
        summary.columns
    )
    assert 0 <= summary["posterior_mean"].iloc[0] <= 1


def test_posterior_predictive_by_stratum(fitted) -> None:
    cells, idata = fitted
    for stratum in ("year_idx", "race_idx", "age_idx"):
        fig = diagnostics.posterior_predictive_by_stratum(
            idata, cells, stratum_col=stratum
        )
        _assert_has_axes(fig)


def test_posterior_predictive_bad_col(fitted) -> None:
    cells, idata = fitted
    with pytest.raises(KeyError):
        diagnostics.posterior_predictive_by_stratum(
            idata, cells, stratum_col="does_not_exist"
        )


def test_decomposition_by_race(fitted) -> None:
    cells, idata = fitted
    fig = diagnostics.decomposition_by_race(idata, cells)
    _assert_has_axes(fig)
    summary = fig._selection_data  # type: ignore[attr-defined]
    assert {"race", "true_livebirths", "recorded", "missed"}.issubset(
        summary.columns
    )
    assert (summary["recorded"] <= summary["true_livebirths"] + 1e-6).all()


def test_age_curve_check(fitted) -> None:
    cells, idata = fitted
    fig = diagnostics.age_curve_check(idata, cells)
    _assert_has_axes(fig)

    table = diagnostics.age_curve_table(idata)
    assert {"age_band", "posterior_mean_per_1000", "morris_per_1000"}.issubset(
        table.columns
    )
    # Posterior means should land roughly near the Morris anchor given the
    # tight sigma=0.10 logit prior.
    diffs = np.log10(
        table["posterior_mean_per_1000"] / table["morris_per_1000"]
    )
    assert (diffs.abs() < 0.5).all(), (
        f"theta_LB posterior diverged from Morris beyond half a log10: {diffs.tolist()!r}"
    )


def test_summary_table_and_convergence_health(fitted) -> None:
    _, idata = fitted
    summary = diagnostics.summary_table(
        idata, var_names=("theta_lb_age", "s_race")
    )
    health = diagnostics.convergence_health(summary)
    for key in (
        "max_rhat",
        "min_ess",
        "rhat_threshold",
        "ess_threshold",
        "rhat_ok",
        "ess_ok",
        "all_ok",
    ):
        assert key in health

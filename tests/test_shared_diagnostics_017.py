# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Study-specific diagnostic rules retained with shared calculations."""

from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dspopulations_us_birth_certificates.selection import fit_validation
from dspopulations_us_birth_certificates.selection.diagnostics import convergence_health


@pytest.fixture
def valid_trace():
    rng = np.random.default_rng(42)
    shape = (4, 500)
    coordinates = {"chain": [3, 1, 7, 2], "draw": np.arange(500)}
    return xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"x": (("chain", "draw"), rng.normal(size=shape))}, coords=coordinates
            ),
            "sample_stats": xr.Dataset(
                {
                    "diverging": (("chain", "draw"), np.zeros(shape, dtype=bool)),
                    "energy": (("chain", "draw"), rng.normal(size=shape)),
                },
                coords=coordinates,
            ),
        }
    )


def test_transposed_energy_retains_values_and_chain_order(valid_trace):
    expected = np.asarray(az.bfmi(valid_trace.sample_stats["energy"]), dtype=float)
    valid_trace.sample_stats["energy"] = valid_trace.sample_stats["energy"].transpose(
        "draw", "chain"
    )
    model = SimpleNamespace(free_RVs=[SimpleNamespace(name="x")])

    _, health = fit_validation.validate_fit(valid_trace, model)

    np.testing.assert_allclose(health["bfmi"], expected)
    assert health["status"] == "passed"


@pytest.mark.parametrize(
    "energy_case", ["missing", "constant", "nonfinite", "extra_dimension"]
)
def test_unavailable_energy_never_passes_validation(valid_trace, energy_case):
    if energy_case == "missing":
        del valid_trace.sample_stats["energy"]
    elif energy_case == "constant":
        valid_trace.sample_stats["energy"] = xr.zeros_like(
            valid_trace.sample_stats["energy"]
        )
    elif energy_case == "nonfinite":
        valid_trace.sample_stats["energy"].values[0, 0] = np.nan
    else:
        valid_trace.sample_stats["energy"] = valid_trace.sample_stats[
            "energy"
        ].expand_dims(event=[0, 1])
    model = SimpleNamespace(free_RVs=[SimpleNamespace(name="x")])

    _, health = fit_validation.validate_fit(valid_trace, model)

    assert health["status"] == "failed"
    assert any("energy diagnostic" in failure for failure in health["failures"])


@pytest.mark.parametrize("bfmi, passed", [(0.3, True), (0.299999, False)])
def test_energy_cutoff_remains_inclusive(valid_trace, monkeypatch, bfmi, passed):
    monkeypatch.setattr(fit_validation, "bfmi_per_chain", lambda _trace: [bfmi] * 4)
    model = SimpleNamespace(free_RVs=[SimpleNamespace(name="x")])
    _, health = fit_validation.validate_fit(valid_trace, model)
    assert (health["status"] == "passed") is passed


@pytest.mark.parametrize(
    "rhat, ess, passed",
    [(1.009999, 400.0, True), (1.01, 400.0, False), (1.0, 399.999, False)],
)
@pytest.mark.parametrize("rhat_column", ["r_hat", "rhat"])
def test_unrounded_thresholds_and_legacy_column_names(rhat, ess, passed, rhat_column):
    summary = pd.DataFrame(
        {rhat_column: [rhat], "ess_bulk": [500.0], "ess_tail": [ess]}
    )
    health = convergence_health(summary)
    assert health["all_ok"] is passed
    assert health["max_rhat"] == rhat
    assert health["min_ess"] == ess


def test_nullable_missing_diagnostics_fail_except_for_known_constants():
    summary = pd.DataFrame(
        {
            "r_hat": pd.array([1.0, pd.NA], dtype="Float64"),
            "ess_bulk": pd.array([500.0, pd.NA], dtype="Float64"),
            "ess_tail": pd.array([500.0, pd.NA], dtype="Float64"),
        },
        index=["x", "fixed[0]"],
    )

    assert not convergence_health(summary)["all_ok"]
    assert convergence_health(summary, constant_names=("fixed",))["all_ok"]
    assert not convergence_health(summary.drop(columns="ess_tail"))["all_ok"]
    assert not convergence_health(summary.iloc[:0])["all_ok"]


def test_infinite_effective_sample_size_is_not_a_finite_diagnostic():
    summary = pd.DataFrame({"r_hat": [1.0], "ess_bulk": [np.inf], "ess_tail": [np.inf]})
    health = convergence_health(summary)
    assert health["ess_ok"]
    assert not health["finite_diagnostics"]
    assert not health["all_ok"]

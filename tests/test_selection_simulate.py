"""Tests for the forward simulator in ``selection.simulate``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.selection import (
    TrueParams,
    simulate_cells,
    variant_C_default,
)
from dspopulations_us_birth_certificates.selection.priors import (
    N_AGE,
    N_EDU,
    N_PAYER,
    N_RACE,
)

REQUIRED_COLS = (
    "year_idx",
    "age_idx",
    "race_idx",
    "edu_idx",
    "payer_idx",
    "preterm",
    "cchd",
    "nicu",
    "aven",
    "N_cell",
    "R_cell",
)

TRUE_COLS = (
    "true_theta_lb",
    "true_eta_detect",
    "true_eta_term",
    "true_eta",
    "true_s",
    "true_p_ds_lb",
    "true_p_recorded",
)


def _make_cells(seed: int = 0, n_cells_per_month: int = 30) -> pd.DataFrame:
    truth = TrueParams.from_priors(
        variant_C_default(),
        n_year=9,
        post_dobbs_year_start=6,
        seed=seed,
    )
    return simulate_cells(
        truth,
        n_cells_per_month=n_cells_per_month,
        n_year=9,
        post_dobbs_year_start=6,
        seed=seed,
    )


def test_simulate_produces_valid_cells() -> None:
    cells = _make_cells()
    for col in REQUIRED_COLS + TRUE_COLS:
        assert col in cells.columns, f"missing column {col!r}"
    # Integer indices lie in vocabulary ranges.
    assert cells["age_idx"].between(0, N_AGE - 1).all()
    assert cells["race_idx"].between(0, N_RACE - 1).all()
    assert cells["edu_idx"].between(0, N_EDU - 1).all()
    assert cells["payer_idx"].between(0, N_PAYER - 1).all()
    # Binary flags.
    for col in ("preterm", "cchd", "nicu", "aven"):
        assert set(cells[col].unique()).issubset({0, 1})
    # R <= N.
    assert (cells["R_cell"] <= cells["N_cell"]).all()
    assert (cells["N_cell"] >= 0).all()


def test_recorded_rate_in_expected_range() -> None:
    """Total recorded rate should land in the plausible real-world band."""
    cells = _make_cells()
    rate = cells["R_cell"].sum() / cells["N_cell"].sum()
    # Real-world range for 2016-2024 US data is ~9e-4 to 1.2e-3 per
    # livebirth. The simulator draws from priors so we allow a wider band
    # but flag drift into implausible territory.
    assert 3e-4 < rate < 3e-3, f"Recorded rate out of range: {rate:.2e}"


def test_true_probabilities_stored() -> None:
    """All ``true_*`` columns are in [0, 1]."""
    cells = _make_cells()
    for col in TRUE_COLS:
        values = cells[col]
        assert values.between(0.0, 1.0).all(), f"{col} outside [0, 1]"


def test_rng_determinism() -> None:
    """Same seed -> identical outputs."""
    a = _make_cells(seed=7)
    b = _make_cells(seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_rng_different_seeds_differ() -> None:
    a = _make_cells(seed=7)
    b = _make_cells(seed=8)
    assert not a.equals(b)


def test_recorded_increases_with_age() -> None:
    """Older-mother cells should have higher recorded-DS rates in aggregate."""
    cells = _make_cells(n_cells_per_month=60)
    # N-weighted recorded rate by age band.
    grouped = cells.groupby("age_idx").apply(
        lambda g: g["R_cell"].sum() / max(g["N_cell"].sum(), 1)
    )
    # Monotone non-decreasing is the target, but with finite samples allow
    # Spearman rank correlation > 0.8 instead.
    ages = np.asarray(grouped.index)
    rates = np.asarray(grouped.values, dtype=float)
    # Sort by age, compute rank correlation with rates.
    order = np.argsort(ages)
    rho = np.corrcoef(
        np.argsort(np.argsort(ages[order])),
        np.argsort(np.argsort(rates[order])),
    )[0, 1]
    assert rho > 0.8, f"Age-rate Spearman rank correlation too low: {rho:.2f}"


def test_simulate_cell_dimensions() -> None:
    """``n_cells_per_month * 12 * n_year`` cells emitted."""
    cells = _make_cells(n_cells_per_month=10)
    assert len(cells) == 10 * 12 * 9


def test_eta_term_year_shape_matches_n_year() -> None:
    truth = TrueParams.from_priors(
        variant_C_default(),
        n_year=7,
        post_dobbs_year_start=4,
        seed=0,
    )
    assert truth.eta_term_year.shape == (7,)
    cells = simulate_cells(
        truth,
        n_cells_per_month=5,
        n_year=7,
        post_dobbs_year_start=4,
        seed=0,
    )
    assert "region_idx" not in cells.columns
    assert cells["year_idx"].between(0, 6).all()

"""Input contracts shared by DSP model construction and external inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def integer_array(value, name: str, *, minimum: int = 0) -> np.ndarray:
    """Validate before casting, so negative indices and fractions cannot wrap."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain finite integers")
    if (
        not np.all(np.isfinite(array))
        or np.any(array != np.floor(array))
        or np.any(array < minimum)
        or np.any(array >= 2**63)
    ):
        raise ValueError(f"{name} must contain finite integers >= {minimum}")
    return array.astype(np.int64)


def validate_cells(cells: pd.DataFrame, *, n_year: int, n_age: int) -> None:
    """Require integer counts and indices on a complete modelled year axis."""
    required = {"year_idx", "age_idx", "N_cell", "R_cell"}
    if missing := required - set(cells):
        raise ValueError(f"core cells missing columns: {sorted(missing)}")
    if cells.empty:
        raise ValueError("core model requires at least one age-year cell")
    for name, size in (("year_idx", n_year), ("age_idx", n_age)):
        integer_array([size], name + " dimension", minimum=1)
        index = integer_array(cells[name], name)
        if np.any(index >= size):
            raise ValueError(f"{name} values out of range [0, {size - 1}]")
    births = integer_array(cells["N_cell"], "N_cell")
    recorded = integer_array(cells["R_cell"], "R_cell")
    if np.any(recorded > births):
        raise ValueError("R_cell > N_cell in at least one core cell")
    if np.any(
        np.bincount(cells.year_idx.astype(int), weights=births, minlength=n_year) <= 0
    ):
        raise ValueError("every modelled year must contain positive births")


def probability_array(value, name: str) -> np.ndarray:
    """Require finite probabilities strictly inside the unit interval."""
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array) & (array > 0) & (array < 1)):
        raise ValueError(f"{name} must contain finite probabilities in (0, 1)")
    return array

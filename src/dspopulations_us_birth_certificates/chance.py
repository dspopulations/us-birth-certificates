"""Common computation/calculation utilities."""

import math

import numpy as np
import pandas as pd

# Morris et al. maternal-age-specific Down-syndrome live-birth risk (absent
# terminations), corrected double-logistic. Held as a single source of truth so
# the Python and SQL (scripts/duckdb_prepare.py) implementations cannot drift.
MORRIS_PARAMS: tuple[float, float, float, float] = (7.33, 4.211, 0.2815, 37.23)


def get_ds_lb_nt_probability(age: int) -> float:
    """
    Returns the chance that live born baby will have Down syndrome, given the mother's age.

    Parameters:
    - age: Maternal age in years.

    Returns:
    - Probability of Down syndrome in live born baby.

    Refs:
    - https://doi.org/10.1136/jms.9.1.2
    - https://doi.org/10.1258/096914105775220679 (corrected formula)
    """

    a, b, c, d = MORRIS_PARAMS
    return 1 / (1 + math.exp(a - b / (1 + math.exp(-c * (age - d)))))


def get_ds_lb_nt_probability_array(
    age: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """
    Returns the chance that live born baby will have Down syndrome, given the mother's age.

    Parameters:
    - age: Maternal age in years.

    Returns:
    - Probability of Down syndrome in live born baby. Returns a
      ``pd.Series`` when ``age`` is a ``Series``, otherwise a
      ``np.ndarray``.

    Refs:
    - https://doi.org/10.1136/jms.9.1.2
    - https://doi.org/10.1258/096914105775220679 (corrected formula)
    """

    a, b, c, d = MORRIS_PARAMS
    return 1 / (1 + np.exp(a - b / (1 + np.exp(-c * (age - d)))))


def ds_lb_nt_probability_sql(age_expr: str) -> str:
    """SQL (DuckDB) expression for the Morris Down-syndrome live-birth risk in
    terms of ``age_expr`` (a column name or SQL expression giving maternal age in
    years). Built from ``MORRIS_PARAMS`` so it stays in sync with the functions
    above.

    Refs:
    - https://doi.org/10.1136/jms.9.1.2
    - https://doi.org/10.1258/096914105775220679 (corrected formula)
    """

    a, b, c, d = MORRIS_PARAMS
    return f"1 / (1 + exp({a} - {b} / (1 + exp(-{c} * ({age_expr} - {d})))))"

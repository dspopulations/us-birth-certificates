"""Cell aggregation for Bayesian binomial models.

The Bayesian pipeline works on aggregated ``(year × age × ...)`` cells
rather than observation-level rows — with ~33M rows in 2016–2024 that's
the only tractable shape for MCMC. Each cell contributes one
``Binomial(n_cell, p_cell)`` term.

``load_cells`` is the single entry point: given an outcome spec (via name)
and a tuple of dim columns, it runs the outcome SQL against DuckDB and
aggregates into cells. A uniform schema is returned regardless of which
outcome construction is chosen — downstream models key off dim column
names only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from dspopulations_us_birth_certificates.bayes.outcomes import (
    OutcomeSpec,
    build_outcome_sql,
    outcome_spec_from_name,
)

DEFAULT_DB_PATH = Path("data/us_births.db")


def load_cells(
    *,
    outcome: str | OutcomeSpec,
    dims: tuple[str, ...],
    year_range: tuple[int, int],
    db_path: str | Path = DEFAULT_DB_PATH,
    outcome_params: dict[str, Any] | None = None,
    drop_na_dims: bool = True,
) -> pd.DataFrame:
    """Aggregate births into ``(dims)`` cells with ``n_cell`` and ``y_cell``.

    Args:
        outcome: Either a name (``"recorded"`` / ``"recorded_plus_predicted"``)
            or a fully-constructed ``OutcomeSpec``.
        dims: Dim columns to group by. ``year`` is implied and must appear
            first if included explicitly; if absent, it is prepended so that
            every fit has a time axis.
        year_range: ``(from_year, to_year)`` inclusive.
        db_path: Path to the DuckDB database (read-only connection).
        outcome_params: Extra kwargs when resolving ``outcome`` from a name
            (e.g. ``{"multiplier": 1.5}`` for ``recorded_plus_predicted``).
        drop_na_dims: If ``True`` (default), rows with a NULL value in any
            dim are dropped. Keep ``True`` unless you intend to add an
            explicit "unknown" level to the model.

    Returns:
        DataFrame with columns ``[*dims, n_cell, y_cell]``. Rows are sorted
        by dims. ``n_cell`` is ``int64``; ``y_cell`` is ``int64``.
    """
    if not dims:
        raise ValueError("dims must be non-empty")
    dims = tuple(dict.fromkeys(dims))  # de-dup, preserve order
    if "year" not in dims:
        dims = ("year",) + dims

    spec = (
        outcome
        if isinstance(outcome, OutcomeSpec)
        else outcome_spec_from_name(outcome, **(outcome_params or {}))
    )
    extra_cols = tuple(d for d in dims if d != "year")
    obs_sql = build_outcome_sql(spec, year_range=year_range, extra_columns=extra_cols)

    dim_list = ", ".join(dims)
    where_dims = (
        " AND ".join(f"{d} IS NOT NULL" for d in dims) if drop_na_dims else "TRUE"
    )
    query = f"""
        WITH obs AS ({obs_sql})
        SELECT
            {dim_list},
            COUNT(*) AS n_cell,
            SUM(is_case) AS y_cell
        FROM obs
        WHERE {where_dims}
        GROUP BY {dim_list}
        ORDER BY {dim_list}
    """

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(query).df()
    finally:
        con.close()

    df["n_cell"] = df["n_cell"].astype("int64")
    df["y_cell"] = df["y_cell"].astype("int64")
    return df

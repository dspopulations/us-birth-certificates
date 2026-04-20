"""SQL builders for the two outcome datasets used by the Bayesian models.

- ``recorded``: the observed positive indicator ``down_ind = 1``.
- ``recorded_plus_predicted``: recorded cases topped up with predicted-
  missing cases, flagged via a BOOLEAN column on ``us_births`` (default
  ``ds_pred_missing`` from the usbc10 family; override with
  ``flag_column`` to read usbc11's ``ds_pred_missing_02`` instead). The
  flag is populated by the prediction step in ``scripts/fit_model.py``
  using a year×month quota of ``ceil(1.5 × recorded)`` top non-recorded
  births by the matching predictions column. See that script for the
  multiplier rationale (~60% under-reporting).

Every builder returns a SQL expression that yields one row per birth in
scope with a binary ``is_case`` column alongside the dim columns.
``data.load_cells`` then aggregates into cells via
``SELECT dims..., SUM(is_case) AS y_cell, COUNT(*) AS n_cell``.

Exposure is always "every birth in range with ``down_ind`` known" — that
way ``n_cell`` counts the same denominator regardless of which outcome
construction is used, and ``y_cell`` differs only in how cases are
identified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

DEFAULT_FLAG_COLUMN = "ds_pred_missing"

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_ident(name: str, kind: str) -> None:
    """Column names are interpolated into SQL, so reject exotic chars."""
    if not _SAFE_IDENT.match(name):
        raise ValueError(f"Invalid {kind} column name: {name!r}")


@dataclass(frozen=True)
class OutcomeSpec:
    """Description of how to construct the positive indicator."""

    name: str
    params: dict[str, Any]


def recorded_spec() -> OutcomeSpec:
    """Outcome = ``down_ind = 1``."""
    return OutcomeSpec(name="recorded", params={})


def recorded_plus_predicted_spec(
    flag_column: str = DEFAULT_FLAG_COLUMN,
) -> OutcomeSpec:
    """Outcome = recorded DS births plus predicted-missing.

    Reads the persisted ``flag_column`` from ``us_births``, which the
    prediction pipeline populates using a year×month quota at a fixed
    multiplier of 1.5 (~60% under-reporting). The default reads the
    usbc10 family column (``ds_pred_missing``). Pass a different column
    (e.g. ``ds_pred_missing_02``) to sample against another variant's
    predicted-missing flag. For sensitivity runs at other multipliers,
    regenerate the column from ``fit_model.py``.
    """
    _check_ident(flag_column, "flag")
    return OutcomeSpec(
        name="recorded_plus_predicted",
        params={"flag_column": flag_column},
    )


def build_outcome_sql(
    spec: OutcomeSpec,
    *,
    year_range: tuple[int, int],
    extra_columns: tuple[str, ...],
) -> str:
    """Return a SQL expression yielding one row per in-scope birth.

    Columns: ``year``, ``is_case`` (0/1), plus the names in ``extra_columns``
    (e.g. ``mage_c``). Suitable for use as a sub-query:

        SELECT year, mage_c, SUM(is_case) AS y_cell, COUNT(*) AS n_cell
        FROM ({build_outcome_sql(...)})
        GROUP BY year, mage_c
    """
    from_year, to_year = year_range
    # Every birth in scope contributes to exposure. "In scope" means within
    # the year range and with down_ind known — we can't label an unknown.
    year_filter = (
        f"b.year >= {from_year} AND b.year <= {to_year} AND b.down_ind IS NOT NULL"
    )
    extra = ", ".join("b." + c for c in extra_columns)

    if spec.name == "recorded":
        case_expr = "CAST(b.down_ind AS INTEGER)"
    elif spec.name == "recorded_plus_predicted":
        flag_column = spec.params.get("flag_column", DEFAULT_FLAG_COLUMN)
        _check_ident(flag_column, "flag")
        case_expr = (
            f"CASE WHEN b.down_ind = 1 OR b.{flag_column} THEN 1 ELSE 0 END"
        )
    else:
        raise ValueError(f"Unknown outcome spec: {spec.name!r}")

    return f"""
        SELECT
            b.year AS year,
            {extra},
            {case_expr} AS is_case
        FROM us_births AS b
        WHERE {year_filter}
    """


def outcome_spec_from_name(name: str, **params: Any) -> OutcomeSpec:
    """Resolve a CLI-friendly outcome name to an ``OutcomeSpec``."""
    if name == "recorded":
        return recorded_spec()
    if name == "recorded_plus_predicted":
        return recorded_plus_predicted_spec(**params)
    raise ValueError(
        f"Unknown outcome name {name!r}. Valid: 'recorded', 'recorded_plus_predicted'"
    )

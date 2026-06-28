"""Aggregate NCHS natality (DuckDB) rows into selection-model cells.

``prepare_cells`` pulls per-birth rows from the project's ``us_births``
table and aggregates them into one row per distinct covariate cell,
returning a frame with the integer index columns expected by
:mod:`dspopulations_us_birth_certificates.selection.model` plus
``N_cell`` / ``R_cell`` totals.

Schema assumptions
------------------
The current ``data/us_births.db`` has these relevant columns (DuckDB
``DESCRIBE`` confirmed 2026-04):

    ``year`` (USMALLINT), ``mage_c`` (UTINYINT), ``mracehisp_c`` (UTINYINT:
    1 NH White, 2 NH Black, 3 NH AIAN, 4 NH Asian/PI/Other, 5 Hispanic,
    6 NH more than one race (its own group, race_idx 6), NULL Unknown),
    ``meduc`` (UTINYINT 1-8, 9 or NULL unknown),
    ``pay_rec`` (UTINYINT 1 Medicaid, 2 Private, 3 Self-pay, 4 Other, 9
    Unknown), ``gestrec10`` (UTINYINT 1-5 preterm, 6-10 term, 99/NULL
    unknown), ``ca_cchd`` / ``ab_nicu`` / ``ab_aven1`` (VARCHAR Y/N/U),
    ``down_ind`` (UTINYINT 0/1/NULL).

The DB does not carry a state/region column, so the model has no
region dimension — the termination effect is modelled with demographic
covariates plus a homoscedastic year term.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from dspopulations_us_birth_certificates.selection.priors import (
    N_AGE,
    N_EDU,
    N_PAYER,
    N_RACE,
)

DEFAULT_DB_PATH = Path("data/us_births.db")
DEFAULT_YEAR_RANGE: tuple[int, int] = (2016, 2024)


# --------------------------------------------------------------------------- #
# Coding maps                                                                 #
# --------------------------------------------------------------------------- #
#
# The SQL uses CASE ... ELSE on raw DB codes to emit the integer indices
# that line up with the vocabularies in ``selection.priors``. Keeping
# these as explicit dicts makes the tests straightforward — and lets you
# diff a schema change against the expected encoding.

AGE_BIN_EDGES: tuple[int, ...] = (20, 25, 30, 35, 40, 45)
# -> bins: [0,20), [20,25), [25,30), [30,35), [35,40), [40,45), [45, inf)

# ``mracehisp_c`` -> race_idx (see priors.RACE_LEVELS).
# 1 NH White -> 0, 2 NH Black -> 1, 3 NH AIAN -> 2, 4 NH Asian/PI/Other -> 3,
# 5 Hispanic -> 4, 6 NH more than one race -> 6 (its own group), NULL -> 5 (Unknown).
# (The race_case SQL routes code 6 to its own idx 6; only NULL/other falls to the
# Unknown idx 5. De Graaf has no multi-race anchor, so idx 6 carries the same weak
# s(race, year) fallback as Unknown - see priors.RACE_LEVELS / recording_anchor.py.)
RACE_MAP: dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 6}
RACE_UNKNOWN_IDX = 5

# ``meduc`` (2003 cert) -> edu_idx (see priors.EDU_LEVELS).
# 1 <=8th, 2 9-12 no diploma -> 0 (<HS)
# 3 HS/GED -> 1
# 4 Some college, 5 Associate -> 2 (Some college)
# 6 Bachelor's -> 3
# 7 Master's, 8 Doctorate -> 4 (Master's+)
# 9 or NULL -> 5 (Unknown)
EDU_MAP: dict[int, int] = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3, 7: 4, 8: 4}
EDU_UNKNOWN_IDX = 5

# ``pay_rec`` -> payer_idx (see priors.PAYER_LEVELS).
# 1 Medicaid -> 0, 2 Private -> 1, 3 Self-pay -> 2, 4 Other -> 2,
# 9 or NULL -> 3 (Unknown).
PAYER_MAP: dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 2}
PAYER_UNKNOWN_IDX = 3


# --------------------------------------------------------------------------- #
# Schema (column names in the DB). Override at the callsite if the DB schema  #
# drifts — a dict rather than hardcoded strings so the test fixture can       #
# swap names without editing SQL.                                             #
# --------------------------------------------------------------------------- #

DEFAULT_COLUMNS: dict[str, str] = {
    "year": "year",
    "mage_c": "mage_c",
    "mracehisp_c": "mracehisp_c",
    "meduc": "meduc",
    "pay_rec": "pay_rec",
    "gestrec10": "gestrec10",
    "ca_cchd": "ca_cchd",
    "ab_nicu": "ab_nicu",
    "ab_aven1": "ab_aven1",
    "down_ind": "down_ind",
}


def _build_sql(
    *,
    table: str,
    columns: dict[str, str],
    year_range: tuple[int, int],
    missing_flag_column: str | None = None,
    predictions_column: str | None = None,
) -> str:
    """Return SQL aggregating the raw table into selection-model cells.

    ``R_cell`` defaults to recorded DS (``SUM(down_ind)``). Two GB-corrected modes
    feed the variant-D comparative track (pass at most one):

    - ``missing_flag_column`` (a ``ds_pred_missing_*`` flag): ``R_cell`` counts the
      *union* ``down_ind = 1 OR flag = 1`` -- the project's R' (recorded plus
      predicted-missing). Coarse: the flag is a thresholded year-month quota.
    - ``predictions_column`` (a ``p_ds_lb_pred_*`` probability): ``R_cell`` is the
      *calibrated expected* DS count -- recorded births contribute 1, unrecorded
      births contribute the model's predicted DS probability -- summed per cell and
      rounded. Independent of any quota/multiplier choice.

    "Predicted" refers to the GB prediction, not a C+P training label (the model is
    trained confirmed-only).
    """
    if missing_flag_column and predictions_column:
        raise ValueError(
            "Pass at most one of missing_flag_column / predictions_column."
        )
    from_year, to_year = year_range
    c = columns
    pred_missing_expr = (
        f"COALESCE(CAST({missing_flag_column} AS INTEGER), 0)"
        if missing_flag_column
        else "0"
    )
    prob_expr = (
        f"COALESCE(CAST({predictions_column} AS DOUBLE), 0.0)"
        if predictions_column
        else "0.0"
    )
    if predictions_column:
        # Calibrated expected DS per cell (float): recorded contribute 1, unrecorded
        # contribute the model's predicted DS probability. Integerised by
        # largest-remainder rounding in prepare_cells (preserves the total); a per-cell
        # ROUND here would bias the total down ~9% via the many sub-0.5 cells.
        r_cell_expr = "SUM(CASE WHEN down_ind = 1 THEN 1.0 ELSE prob END)"
    elif missing_flag_column:
        r_cell_expr = (
            "SUM(CASE WHEN down_ind = 1 OR pred_missing = 1 THEN 1 ELSE 0 END)"
        )
    else:
        r_cell_expr = "SUM(down_ind)"
    # ``mage_c`` binning — last edge is the 45+ open bin.
    age_case = (
        f"CASE "
        f"WHEN {c['mage_c']} < 20 THEN 0 "
        f"WHEN {c['mage_c']} < 25 THEN 1 "
        f"WHEN {c['mage_c']} < 30 THEN 2 "
        f"WHEN {c['mage_c']} < 35 THEN 3 "
        f"WHEN {c['mage_c']} < 40 THEN 4 "
        f"WHEN {c['mage_c']} < 45 THEN 5 "
        f"ELSE 6 END"
    )
    race_case = (
        f"CASE {c['mracehisp_c']} "
        f"WHEN 1 THEN 0 WHEN 2 THEN 1 WHEN 3 THEN 2 "
        f"WHEN 4 THEN 3 WHEN 5 THEN 4 WHEN 6 THEN 6 "
        f"ELSE {RACE_UNKNOWN_IDX} END"
    )
    edu_case = (
        f"CASE {c['meduc']} "
        f"WHEN 1 THEN 0 WHEN 2 THEN 0 WHEN 3 THEN 1 "
        f"WHEN 4 THEN 2 WHEN 5 THEN 2 WHEN 6 THEN 3 "
        f"WHEN 7 THEN 4 WHEN 8 THEN 4 "
        f"ELSE {EDU_UNKNOWN_IDX} END"
    )
    payer_case = (
        f"CASE {c['pay_rec']} "
        f"WHEN 1 THEN 0 WHEN 2 THEN 1 WHEN 3 THEN 2 WHEN 4 THEN 2 "
        f"ELSE {PAYER_UNKNOWN_IDX} END"
    )
    # gestrec10 1-5 preterm, 6-10 term, 99 unknown (drop).
    preterm_case = (
        f"CASE WHEN {c['gestrec10']} BETWEEN 1 AND 5 THEN 1 "
        f"WHEN {c['gestrec10']} BETWEEN 6 AND 10 THEN 0 END"
    )
    cchd_case = (
        f"CASE UPPER({c['ca_cchd']}) WHEN 'Y' THEN 1 WHEN 'N' THEN 0 END"
    )
    nicu_case = (
        f"CASE UPPER({c['ab_nicu']}) WHEN 'Y' THEN 1 WHEN 'N' THEN 0 END"
    )
    aven_case = (
        f"CASE UPPER({c['ab_aven1']}) WHEN 'Y' THEN 1 WHEN 'N' THEN 0 END"
    )

    return f"""
        WITH coded AS (
            SELECT
                CAST({c['year']} AS INTEGER) - {from_year} AS year_idx,
                {age_case}    AS age_idx,
                {race_case}   AS race_idx,
                {edu_case}    AS edu_idx,
                {payer_case}  AS payer_idx,
                {preterm_case} AS preterm,
                {cchd_case}    AS cchd,
                {nicu_case}    AS nicu,
                {aven_case}    AS aven,
                CAST({c['down_ind']} AS INTEGER) AS down_ind,
                {pred_missing_expr} AS pred_missing,
                {prob_expr} AS prob
            FROM {table}
            WHERE {c['year']} BETWEEN {from_year} AND {to_year}
              AND {c['mage_c']} IS NOT NULL
              AND {c['down_ind']} IS NOT NULL
        )
        SELECT
            year_idx, age_idx, race_idx, edu_idx, payer_idx,
            preterm, cchd, nicu, aven,
            COUNT(*) AS N_cell,
            {r_cell_expr} AS R_cell
        FROM coded
        WHERE preterm IS NOT NULL
          AND cchd IS NOT NULL
          AND nicu IS NOT NULL
          AND aven IS NOT NULL
        GROUP BY year_idx, age_idx, race_idx, edu_idx, payer_idx,
                 preterm, cchd, nicu, aven
    """


def _largest_remainder_round(x: np.ndarray) -> np.ndarray:
    """Round non-negative reals to integers preserving the rounded grand total.

    Floor every value, then hand the leftover (``round(sum) - sum(floors)``) as +1 to
    the cells with the largest fractional parts. This avoids the systematic downward
    bias of independent per-cell rounding when many cells carry sub-0.5 expected
    counts (as in the variant-D probability track).
    """
    floor = np.floor(x)
    remainder = x - floor
    out = floor.astype(np.int64)
    deficit = int(round(float(x.sum()))) - int(floor.sum())
    if deficit > 0:
        top = np.argsort(remainder)[::-1][:deficit]
        out[top] += 1
    return out


def prepare_cells(
    con: duckdb.DuckDBPyConnection,
    *,
    year_range: tuple[int, int] = DEFAULT_YEAR_RANGE,
    table: str = "us_births",
    columns: dict[str, str] | None = None,
    missing_flag_column: str | None = None,
    predictions_column: str | None = None,
) -> pd.DataFrame:
    """Aggregate raw NCHS rows into selection-model cells.

    Args:
        con: Open DuckDB connection (read-only is fine).
        year_range: Inclusive ``(from_year, to_year)``.
        table: Name of the births table (default ``us_births``).
        columns: Optional override for column names (schema drift).
        missing_flag_column: Optional GB ``ds_pred_missing_*`` flag. When set,
            ``R_cell`` counts the R' union ``down_ind = 1 OR flag = 1`` (recorded
            plus predicted-missing, the variant-D flag track).
        predictions_column: Optional GB ``p_ds_lb_pred_*`` probability. When set,
            ``R_cell`` is the calibrated expected DS count (recorded contribute 1,
            unrecorded contribute their predicted probability, summed and rounded) --
            the variant-D probability track, independent of any quota/multiplier.
            Mutually exclusive with ``missing_flag_column``.

    Returns:
        A DataFrame with the integer index columns + ``N_cell`` / ``R_cell``,
        and ``attrs = {"n_year", "year_range", "N_total", "R_total"}``.
    """
    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    sql = _build_sql(
        table=table,
        columns=cols,
        year_range=year_range,
        missing_flag_column=missing_flag_column,
        predictions_column=predictions_column,
    )
    cells = con.execute(sql).df()

    # DuckDB returns SUM() as nullable; cast for clean downstream use.
    cells["N_cell"] = cells["N_cell"].astype("int64")
    if predictions_column is not None:
        # R_cell is the per-cell expected DS count (float); integerise preserving the
        # total (per-cell rounding would bias the total down).
        cells["R_cell"] = _largest_remainder_round(cells["R_cell"].to_numpy(float))
    cells["R_cell"] = cells["R_cell"].astype("int64")
    for col in (
        "year_idx",
        "age_idx",
        "race_idx",
        "edu_idx",
        "payer_idx",
        "preterm",
        "cchd",
        "nicu",
        "aven",
    ):
        cells[col] = cells[col].astype("int32")

    from_year, to_year = year_range
    n_year = to_year - from_year + 1
    cells.attrs.update(
        {
            "n_year": n_year,
            "year_range": year_range,
            "N_total": int(cells["N_cell"].sum()),
            "R_total": int(cells["R_cell"].sum()),
        }
    )
    _sanity_check(cells)
    return cells


def _sanity_check(cells: pd.DataFrame) -> None:
    """Fail loudly if the aggregated indices are out of vocabulary."""
    bounds = {
        "age_idx": N_AGE,
        "race_idx": N_RACE,
        "edu_idx": N_EDU,
        "payer_idx": N_PAYER,
    }
    for col, n in bounds.items():
        values = cells[col]
        if not values.between(0, n - 1).all():
            bad = values[~values.between(0, n - 1)].unique()
            raise ValueError(
                f"{col} values out of range [0, {n - 1}]: "
                f"found {sorted(bad.tolist())!r}"
            )
    for col in ("preterm", "cchd", "nicu", "aven"):
        uniq = set(cells[col].unique().tolist())
        if not uniq.issubset({0, 1}):
            raise ValueError(f"{col} must be 0/1, got {sorted(uniq)!r}")
    if (cells["R_cell"] > cells["N_cell"]).any():
        raise ValueError("R_cell > N_cell in at least one cell")


def summarise_cells(cells: pd.DataFrame) -> dict[str, Any]:
    """Return a small dict summary for CLI logging."""
    n_total = int(cells["N_cell"].sum())
    r_total = int(cells["R_cell"].sum())
    return {
        "n_cells": int(len(cells)),
        "n_total": n_total,
        "r_total": r_total,
        "recorded_rate": (r_total / n_total) if n_total else float("nan"),
        **{k: cells.attrs.get(k) for k in cells.attrs},
    }


def describe_age_bins() -> np.ndarray:
    """Return the half-open age-bin edges used by ``prepare_cells``."""
    return np.asarray(AGE_BIN_EDGES, dtype=int)

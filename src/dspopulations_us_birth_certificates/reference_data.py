"""Report external-input coverage and refuse incomplete expected-count analyses."""

from __future__ import annotations

import re

import duckdb


def require_complete_estimates(
    con: duckdb.DuckDBPyConnection, columns: list[str]
) -> None:
    """Check every birth with a known year before summing modelled estimates.

    SUM ignores NULL values. Missing years must not silently reduce an expected
    count or disappear from a reported recording rate.
    """
    failures = {}
    for column in columns:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", column):
            raise ValueError(f"Invalid estimate column: {column!r}")
        rows = con.execute(f"""
            SELECT year, count(*) FILTER (WHERE "{column}" IS NULL)
            FROM us_births WHERE year IS NOT NULL GROUP BY year
            HAVING count(*) FILTER (WHERE "{column}" IS NULL) > 0
            ORDER BY year
        """).fetchall()
        if rows:
            failures[column] = {year: count for year, count in rows}
    if failures:
        raise ValueError(
            "Missing model inputs by year (birth counts): "
            f"{failures}. Review reference coverage before estimating counts; "
            "do not treat missing inputs as zero or carry earlier years forward."
        )


REFERENCE_TABLES = (
    (
        "prevalence_year",
        "us-births-surveillance-prevalence-1989-2024.csv",
        ["p_ds_lb_wt"],
        "p_ds_lb_wt",
        "r.p_ds_lb_wt",
    ),
    (
        "us_births_est_prevalence_age",
        "us-births-estimated-prevalence-maternal-age-1989-2018.csv",
        ["p_ds_lb_wt_lt35_sv", "p_ds_lb_wt_gte35_sv"],
        "p_ds_lb_wt_mage",
        "CASE WHEN b.mage_c < 35 THEN r.p_ds_lb_wt_lt35_sv ELSE r.p_ds_lb_wt_gte35_sv END",
    ),
    (
        "reduction_rate_year",
        "us-births-reduction-rates-1989-2024.csv",
        ["reduction"],
        "p_ds_lb_nt_reduc",
        "b.p_ds_lb_nt * (1-r.reduction)",
    ),
    (
        "ds_case_weights",
        "us-births-ds-rec-weights.csv",
        ["nhw", "nhb", "ai_an", "as_pi", "his", "total"],
        "ds_case_weight",
        """CASE WHEN b.down_ind=1 THEN
            CASE b.mracehisp_c WHEN 1 THEN r.nhw WHEN 2 THEN r.nhb
                WHEN 3 THEN r.ai_an WHEN 4 THEN r.as_pi WHEN 5 THEN r.his
                ELSE r.total END ELSE 0 END""",
    ),
    (
        "us_births_est_prevalence_ethnicity",
        "us-births-estimated-prevalence-ethnicity-2000-2018.csv",
        ["prevalence"],
        None,
        None,
    ),
)


def reference_coverage(con: duckdb.DuckDBPyConnection, year: int) -> dict:
    """Audit all five stored references without filling gaps or changing data."""
    results = {}
    for table, filename, values, column, expression in REFERENCE_TABLES:
        keys = "year, mracehisp_c" if column is None else "year"
        duplicates = con.execute(
            f"SELECT count(*) FROM (SELECT {keys} FROM {table} "
            f"GROUP BY {keys} HAVING count(*)>1)"
        ).fetchone()[0]
        key_null = " OR ".join(f"{k.strip()} IS NULL" for k in keys.split(","))
        missing_keys = con.execute(
            f"SELECT count(*) FROM {table} WHERE {key_null}"
        ).fetchone()[0]
        if duplicates or missing_keys:
            raise ValueError(f"Missing or duplicate reference keys in {table}")
        years = [
            r[0]
            for r in con.execute(
                f"SELECT DISTINCT year FROM {table} ORDER BY year"
            ).fetchall()
        ]
        null_values = " OR ".join(f"{c} IS NULL" for c in values)
        rows, incomplete = con.execute(
            f"SELECT count(*), count(*) FILTER (WHERE {null_values}) FROM {table} WHERE year=?",
            [year],
        ).fetchone()
        join = "b.year=r.year" + (
            " AND b.mracehisp_c=r.mracehisp_c" if column is None else ""
        )
        unmatched = con.execute(
            f"SELECT count(*) FROM us_births b LEFT JOIN {table} r ON {join} WHERE b.year=? AND r.year IS NULL",
            [year],
        ).fetchone()[0]
        result = {
            "file": filename,
            "available_years": years,
            "rows_for_year": rows,
            "rows_with_missing_values_for_year": incomplete,
            "births_without_reference_key": unmatched,
            "derived_column": column,
            "complete_for_year": bool(rows and not incomplete and not unmatched),
        }
        if column:
            # An unmatched year leaves even non-case weights NULL. A zero would
            # conceal the absence of the year-specific recording assumptions.
            expected = f"CASE WHEN r.year IS NOT NULL THEN {expression} END"
            mismatch = con.execute(
                f"""
                SELECT count(*) FROM us_births b LEFT JOIN {table} r ON {join}
                WHERE b.year=? AND round(b.{column},12) IS DISTINCT FROM round(({expected}),12)
            """,
                [year],
            ).fetchone()[0]
            result["derived_value_mismatches"] = mismatch
            result["missing_derived_values"] = con.execute(
                f"SELECT count(*) FROM us_births WHERE year=? AND {column} IS NULL",
                [year],
            ).fetchone()[0]
        else:
            result["note"] = (
                "Loaded reference only; ethnicity probability columns remain unimplemented placeholders."
            )
        results[table] = result
    return {
        "year": year,
        "tables": results,
        "all_reference_inputs_available": all(
            r["complete_for_year"] for r in results.values()
        ),
        "derived_values_match_available_references": all(
            not r.get("derived_value_mismatches", 0) for r in results.values()
        ),
        "policy": "Missing inputs remain NULL. No extrapolation or carry-forward is applied.",
    }

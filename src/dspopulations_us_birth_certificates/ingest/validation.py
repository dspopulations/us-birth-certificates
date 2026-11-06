"""Read-only comparisons with the saved SAS-derived database."""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb

from dspopulations_us_birth_certificates.variables import IMPORTED_VARS

from .database import MemoryMonitor, protect_output
from .layouts import TYPES, Layout, identifier, sha256
from .sources import save_json

DERIVED = [
    "year",
    "mage_c",
    "mrace_c",
    "mhisp_c",
    "mracehisp_c",
    "down_ind",
    "ca_down_c",
    "p_ds_lb_nt",
    "p_ds_lb_wt",
    "p_ds_lb_wt_mage",
    "p_ds_lb_nt_mage",
    "p_ds_lb_wt_ethn",
    "p_ds_lb_nt_ethn",
    "p_ds_lb_nt_reduc",
    "ds_case_weight",
]
EXPECTED_DIFFERENCES = {
    "dob_tt": "Legacy unsigned eight-bit cast loses values above 255.",
    "no_risks": "Legacy range check converts source 0 to NULL.",
    "no_congen": "Legacy range check converts source 9 to NULL.",
    "bmi": "Legacy binary16 rounding; recover the source's one decimal place for comparison.",
    "imp_plural": "CDC IMP_PLUR is explicitly mapped; the legacy import omitted this alias.",
}


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _normalised(field: dict, *, legacy: bool) -> str:
    name = field["name"]
    expression = identifier(name)
    if name == "bmi" and legacy:
        expression = f"round({expression}::DOUBLE, 1)"
    if not legacy:
        if name == "dob_tt":
            expression = f"CASE WHEN {expression} <= 255 THEN {expression} END"
        elif name == "no_risks":
            expression = f"nullif({expression}, 0)"
        elif name == "no_congen":
            expression = f"nullif({expression}, 9)"
        elif name == "imp_plural":
            expression = "NULL"
    return f"CAST({expression} AS {TYPES[field['type']][1]}) AS {identifier(name)}"


def _difference_count(con, left: str, right: str) -> int:
    return con.execute(
        f"SELECT count(*) FROM (({left}) EXCEPT ALL ({right}))"
    ).fetchone()[0]


def compare_database(
    layout: Layout,
    candidate: Path,
    baseline: Path,
    *,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> dict:
    """Compare all imported fields and complete row multiplicities, by month."""
    protect_output(candidate, baseline)
    started = time.monotonic()
    before_stat = baseline.stat()
    before_hash = sha256(baseline)
    with duckdb.connect(str(candidate), read_only=True) as check:
        build = json.loads(
            check.execute("SELECT state FROM __natality_build WHERE id=1").fetchone()[0]
        )
        if (
            build["status"] != "source_validated"
            or build["identity"]["layout_sha256"] != layout.digest
        ):
            raise ValueError(
                "Candidate has not passed source validation with this layout"
            )
    fields = {f["name"]: f for f in layout.fields}
    report = {
        "year": layout.year,
        "comparison_source": "existing SAS-derived DuckDB; no fresh SAS import",
        "baseline_path": str(baseline.resolve()),
        "baseline_sha256_before": before_hash,
        "candidate_path": str(candidate.resolve()),
        "build_identity": build["identity"],
        "field_comparisons": {},
        "row_comparisons": [],
        "expected_differences": EXPECTED_DIFFERENCES,
        "float_comparison": "Derived floating-point columns rounded to 12 decimal places; BMI compared at its source precision.",
    }
    with MemoryMonitor() as monitor, duckdb.connect() as con:
        con.execute("SET memory_limit = ?", [memory_limit])
        con.execute("SET threads = ?", [threads])
        con.execute(
            "SET temp_directory = ?", [str(candidate.with_suffix(".validation.db.tmp"))]
        )
        con.execute(f"ATTACH {_literal(str(candidate.resolve()))} AS pilot (READ_ONLY)")
        con.execute(f"ATTACH {_literal(str(baseline.resolve()))} AS old (READ_ONLY)")
        baseline_schema = con.execute("DESCRIBE old.us_births").fetchall()
        report["baseline_schema"] = baseline_schema
        baseline_names = {row[0] for row in baseline_schema}
        if not set(IMPORTED_VARS + DERIVED).issubset(baseline_names):
            raise ValueError("Baseline is missing expected import or derived columns")
        for side, table, where in (
            ("candidate", "pilot", "dob_yy"),
            ("baseline", "old", "year"),
        ):
            report[f"{side}_count"] = con.execute(
                f"SELECT count(*) FROM {table}.us_births WHERE {where}=?", [layout.year]
            ).fetchone()[0]
        expected_count = layout.source["expected_counts"]["occurrence"]
        # Scan one column at a time. Outputs contain aggregates, never source rows.
        for name in IMPORTED_VARS:
            field = fields[name]
            cast = "DOUBLE" if name == "bmi" else TYPES[field["type"]][1]
            queries = [
                f"SELECT CAST({identifier(name)} AS {cast}) value, count(*) n FROM {table}.us_births WHERE {year_col}={layout.year} GROUP BY value"
                for table, year_col in (("pilot", "dob_yy"), ("old", "year"))
            ]
            changed = con.execute(f"""
                WITH a AS ({queries[0]}), b AS ({queries[1]})
                SELECT coalesce(sum(abs(coalesce(a.n,0)-coalesce(b.n,0))),0)::BIGINT
                FROM a FULL OUTER JOIN b ON a.value IS NOT DISTINCT FROM b.value
            """).fetchone()[0]
            report["field_comparisons"][name] = {
                "frequency_count_absolute_difference": changed,
                "expected_explanation": EXPECTED_DIFFERENCES.get(name)
                if changed
                else None,
            }
        projected = ", ".join(
            _normalised(fields[n], legacy=False) for n in IMPORTED_VARS
        )
        old_projected = ", ".join(
            _normalised(fields[n], legacy=True) for n in IMPORTED_VARS
        )
        derived = ", ".join(
            f"round({identifier(n)}::DOUBLE, 12) AS {identifier(n)}"
            if n.startswith("p_ds_") or n == "ds_case_weight"
            else identifier(n)
            for n in DERIVED
        )
        for month in range(1, 13):
            left = f"SELECT {projected}, {derived} FROM pilot.us_births WHERE dob_yy={layout.year} AND dob_mm={month}"
            right = f"SELECT {old_projected}, {derived} FROM old.us_births WHERE year={layout.year} AND dob_mm={month}"
            missing_old = _difference_count(con, left, right)
            missing_new = _difference_count(con, right, left)
            report["row_comparisons"].append(
                {
                    "month": month,
                    "candidate_only": missing_old,
                    "baseline_only": missing_new,
                }
            )
            print(
                f"Compared {layout.year}-{month:02d}: {missing_old:,} candidate-only, {missing_new:,} baseline-only rows",
                flush=True,
            )
        grouped = "year, mage_c, mracehisp_c, ca_down_c, down_ind"
        left = f"SELECT {grouped}, count(*) FROM pilot.us_births GROUP BY {grouped}"
        right = f"SELECT {grouped}, count(*) FROM old.us_births WHERE year={layout.year} GROUP BY {grouped}"
        report["analysis_cells"] = {
            "candidate_only": _difference_count(con, left, right),
            "baseline_only": _difference_count(con, right, left),
        }
    after_stat = baseline.stat()
    report.update(
        baseline_sha256_after=sha256(baseline),
        elapsed_seconds=time.monotonic() - started,
        peak_rss_bytes=monitor.peak,
        rss_includes_children=monitor.includes_children,
    )
    report["baseline_unchanged"] = (
        before_hash == report["baseline_sha256_after"]
        and before_stat.st_size == after_stat.st_size
        and before_stat.st_mtime_ns == after_stat.st_mtime_ns
    )
    report["passed"] = (
        report["baseline_unchanged"]
        and report["candidate_count"] == report["baseline_count"] == expected_count
        and all(
            not r["candidate_only"] and not r["baseline_only"]
            for r in report["row_comparisons"]
        )
        and not any(report["analysis_cells"].values())
        and all(
            not v["frequency_count_absolute_difference"] or v["expected_explanation"]
            for v in report["field_comparisons"].values()
        )
    )
    save_json(candidate.with_suffix(".validation.json"), report)
    return report

"""Read-only comparisons with the saved SAS-derived database."""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb

from dspopulations_us_birth_certificates.reference_data import reference_coverage
from dspopulations_us_birth_certificates.variables import IMPORTED_VARS

from .database import MemoryMonitor, protect_output
from .layouts import TYPES, Layout, identifier, sha256
from .sources import save_json
from .stored_validation import check_stored_source

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
    "mracehisp": "Legacy range check drops code 9 in the 2003-2013 coding scheme.",
    "mrace31": "Legacy range check converts the reviewed 2015 source code 99 to NULL.",
    "mrace15": "Legacy range check converts the reviewed 2015 source code 99 to NULL.",
    "dmar": "The legacy SAS conversion writes historical integer marital-status codes as strings ending in .0.",
}


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _normalised(field: dict, *, legacy: bool) -> str:
    name = field["name"]
    expression = identifier(name)
    if name == "bmi" and legacy:
        expression = f"round({expression}::DOUBLE, 1)"
    if name == "dmar" and legacy:
        expression = f"regexp_replace({expression}, '^([0-9]+)\\.0$', '\\1')"
    if not legacy:
        if name == "dob_tt":
            expression = f"CASE WHEN {expression} <= 255 THEN {expression} END"
        elif name == "no_risks":
            expression = f"nullif({expression}, 0)"
        elif name == "no_congen":
            expression = f"nullif({expression}, 9)"
        elif name == "imp_plural":
            expression = "NULL"
        elif name == "mracehisp":
            expression = f"nullif({expression}, 9)"
        elif name in ("mrace31", "mrace15"):
            expression = f"nullif({expression}, 99)"
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
    """Compare all imported fields and complete row multiplicities."""
    protect_output(candidate, baseline)
    started = time.monotonic()
    before_stat = baseline.stat()
    before_hash = sha256(baseline)
    candidate_before = sha256(candidate)
    with duckdb.connect(str(candidate), read_only=True) as check:
        check.execute("SET memory_limit = ?", [memory_limit])
        check.execute("SET threads = ?", [threads])
        build = json.loads(
            check.execute("SELECT state FROM __natality_build WHERE id=1").fetchone()[0]
        )
        if (
            build["status"] != "source_validated"
            or build["identity"]["layout_sha256"] != layout.digest
            or build["identity"]["source"] != layout.source
            or not build["identity"]["with_derivations"]
        ):
            raise ValueError(
                "Candidate has not passed source validation with this layout"
            )
        stored = check_stored_source(check, layout, candidate, build)
        coverage = reference_coverage(check, layout.year)
    missing = layout.source.get("legacy_missing_records", {})
    expected_extra = (
        missing.get("count", 0) if missing.get("baseline_sha256") == before_hash else 0
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
        "legacy_missing_records": missing if expected_extra else None,
        "expected_additional_records": expected_extra,
        "reference_coverage": coverage,
        **stored,
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
            ("candidate", "pilot", "year"),
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
                for table, year_col in (("pilot", "year"), ("old", "year"))
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
            if (
                changed
                and not report["field_comparisons"][name]["expected_explanation"]
                and expected_extra
            ):
                report["field_comparisons"][name]["expected_explanation"] = (
                    "Additional source records missing from the saved legacy year."
                    if changed <= expected_extra
                    else None
                )
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
        # Old project extracts do not include the birth month. Partition on a
        # shared age expression instead. Include NULL explicitly, so no record
        # can escape the complete-row comparison.
        partition = "coalesce(mage_c::INTEGER % 12, -1)"
        for bucket in range(-1, 12):
            left = f"SELECT {projected}, {derived} FROM pilot.us_births WHERE year={layout.year} AND {partition}={bucket}"
            right = f"SELECT {old_projected}, {derived} FROM old.us_births WHERE year={layout.year} AND {partition}={bucket}"
            missing_old = _difference_count(con, left, right)
            missing_new = _difference_count(con, right, left)
            report["row_comparisons"].append(
                {
                    "age_modulo_12": bucket,
                    "candidate_only": missing_old,
                    "baseline_only": missing_new,
                }
            )
            print(
                f"Compared {layout.year}, age partition {bucket}: {missing_old:,} candidate-only, {missing_new:,} baseline-only rows",
                flush=True,
            )
        grouped = "year, mage_c, mracehisp_c, ca_down_c, down_ind"
        left = f"SELECT {grouped}, count(*) AS births FROM pilot.us_births GROUP BY {grouped}"
        right = f"SELECT {grouped}, count(*) AS births FROM old.us_births WHERE year={layout.year} GROUP BY {grouped}"
        report["analysis_cells"] = {
            "candidate_only": _difference_count(con, left, right),
            "baseline_only": _difference_count(con, right, left),
        }
        report["analysis_cell_count_changes"] = con.execute(f"""
            WITH a AS ({left}), b AS ({right}), differences AS (
                SELECT coalesce(a.births, 0)-coalesce(b.births, 0) AS delta
                FROM a FULL JOIN b ON
                    {" AND ".join(f"a.{n} IS NOT DISTINCT FROM b.{n}" for n in grouped.split(", "))}
            )
            SELECT coalesce(sum(greatest(delta,0)),0)::BIGINT,
                   coalesce(sum(greatest(-delta,0)),0)::BIGINT FROM differences
        """).fetchone()
    after_stat = baseline.stat()
    report.update(
        baseline_sha256_after=sha256(baseline),
        candidate_sha256=sha256(candidate),
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
        and candidate_before == report["candidate_sha256"]
        and report["candidate_count"] == expected_count
        and report["baseline_count"] + expected_extra == expected_count
        and sum(r["candidate_only"] for r in report["row_comparisons"])
        == expected_extra
        and not any(r["baseline_only"] for r in report["row_comparisons"])
        and report["analysis_cell_count_changes"] == (expected_extra, 0)
        and report["record_identity_valid"]
        and all(report["stored_fields_match_source_frequencies"].values())
        and coverage["derived_values_match_available_references"]
        and all(
            not v["frequency_count_absolute_difference"] or v["expected_explanation"]
            for v in report["field_comparisons"].values()
        )
    )
    save_json(candidate.with_suffix(".validation.json"), report)
    return report

"""Validate 2025 against its published controls when no SAS baseline exists."""

from __future__ import annotations

import json
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import duckdb

from dspopulations_us_birth_certificates.chance import get_ds_lb_nt_probability
from dspopulations_us_birth_certificates.reference_data import reference_coverage

from .database import MemoryMonitor, protect_output
from .layouts import CONFIG, Layout, identifier, sha256
from .sources import save_json


def load_controls(layout: Layout) -> dict:
    path = (CONFIG / layout.source["controls_file"]).resolve()
    if (
        not path.is_relative_to(CONFIG.resolve())
        or sha256(path) != layout.source["controls_sha256"]
    ):
        raise ValueError("Published controls differ from the reviewed specification")
    controls = json.loads(path.read_text())
    if (
        controls["year"] != layout.year
        or controls["guide_sha256"] != layout.source["guide_sha256"]
        or not controls["controls"]
    ):
        raise ValueError("Published controls must match this year and guide")
    return controls


def _source_value(token: str, kind: str):
    if not token.strip():
        return None
    if kind == "string":
        return token
    if kind == "integer_string":
        return str(int(token))
    if kind == "decimal1":
        return Decimal(token)
    return int(token)


def validate_published(
    layout: Layout,
    candidate: Path,
    baseline: Path,
    *,
    memory_limit: str = "2GB",
    threads: int = 2,
) -> dict:
    """Check stored fields, derivations, reference coverage and published counts.

    This does not claim the complete-row comparison available for 2024. Source
    domains and framing are checked during import. Here, frequencies detect
    conversion or storage losses, and independently transcribed guide counts
    check selected joint distributions and recorded outcomes.
    """
    if layout.year != 2025:
        raise ValueError(
            "Published validation derivations have only been reviewed for 2025"
        )
    protect_output(candidate, baseline)
    controls = load_controls(layout)
    started = time.monotonic()
    before_stat = baseline.stat()
    before_hash = sha256(baseline)
    report = {
        "year": layout.year,
        "validation_mode": "published_controls",
        "baseline_path": str(baseline.resolve()),
        "baseline_sha256_before": before_hash,
        "candidate_path": str(candidate.resolve()),
        "controls_sha256": layout.source["controls_sha256"],
        "legacy_comparison": "not performed; current database has no 2025 source extract",
    }
    with (
        MemoryMonitor() as monitor,
        duckdb.connect(str(candidate), read_only=True) as con,
    ):
        con.execute("SET memory_limit = ?", [memory_limit])
        con.execute("SET threads = ?", [threads])
        build = json.loads(
            con.execute("SELECT state FROM __natality_build WHERE id=1").fetchone()[0]
        )
        if (
            build["status"] != "source_validated"
            or build["identity"]["layout_sha256"] != layout.digest
            or build["identity"]["source"] != layout.source
            or not build["identity"]["with_derivations"]
        ):
            raise ValueError(
                "Candidate has not passed source validation with this reviewed layout and source"
            )
        report["build_identity"] = build["identity"]
        frequencies_path = candidate.with_suffix(".frequencies.json")
        if sha256(frequencies_path) != build.get("frequencies_sha256"):
            raise ValueError("Source frequency report is missing or changed")
        frequencies = json.loads(frequencies_path.read_text())
        count, distinct, low, high, bad_ids, bad_year = con.execute("""
            SELECT count(*), count(DISTINCT source_record), min(source_record), max(source_record),
                count(*) FILTER (WHERE id IS DISTINCT FROM 2025000000000 + source_record),
                count(*) FILTER (WHERE dob_yy IS DISTINCT FROM 2025)
            FROM us_births
        """).fetchone()
        expected = layout.source["expected_counts"]["occurrence"]
        report["rows"] = count
        report["record_identity_valid"] = (
            count == distinct == high == expected
            and low == 1
            and not bad_ids
            and not bad_year
        )
        field_checks = {}
        for field in layout.fields:
            name = identifier(field["name"])
            actual = dict(
                con.execute(
                    f"SELECT {name},count(*) FROM us_births GROUP BY {name}"
                ).fetchall()
            )
            if field["status"] == "absent":
                desired = {None: count}
            else:
                desired = Counter()
                for token, n in frequencies[field["name"]].items():
                    desired[_source_value(token, field["type"])] += n
            field_checks[field["name"]] = (
                actual == desired and sum(desired.values()) == count
            )
        report["stored_fields_match_source_frequencies"] = field_checks
        report["published_controls"] = []
        for control in controls["controls"]:
            actual = con.execute(
                f"SELECT count(*) FROM us_births WHERE {control['where']}"
            ).fetchone()[0]
            report["published_controls"].append(
                {**control, "actual": actual, "passed": actual == control["expected"]}
            )
        expressions = {
            "year": "dob_yy",
            "mage_c": "mager",
            "ca_down_c": "ca_down",
            "down_ind": "CASE WHEN ca_down IN ('C','P') THEN 1 WHEN ca_down='N' THEN 0 END",
            "mrace_c": "CASE WHEN mrace6 IN (4,5) THEN 4 WHEN mrace6=6 THEN 5 ELSE mrace6 END",
            "mhisp_c": "CASE WHEN mhisp_r IN (4,5) THEN 4 WHEN mhisp_r=9 THEN 5 ELSE mhisp_r END",
            "mracehisp_c": "CASE WHEN mracehisp IN (4,5) THEN 4 WHEN mracehisp=7 THEN 5 WHEN mracehisp=8 THEN NULL ELSE mracehisp END",
        }
        checks = {
            name: con.execute(
                f"SELECT count(*) FROM us_births WHERE {identifier(name)} IS DISTINCT FROM ({expr})"
            ).fetchone()[0]
            for name, expr in expressions.items()
        }
        # Compare SQL-derived risks with the Python calculation by source age.
        age_risks = con.execute(
            "SELECT mager, min(p_ds_lb_nt), max(p_ds_lb_nt), count(*)-count(p_ds_lb_nt) FROM us_births GROUP BY mager"
        ).fetchall()
        checks["p_ds_lb_nt_age_groups"] = sum(
            lo is None
            or hi is None
            or nulls > 0
            or abs(lo - get_ds_lb_nt_probability(age)) > 1e-12
            or abs(hi - get_ds_lb_nt_probability(age)) > 1e-12
            for age, lo, hi, nulls in age_risks
        )
        for name in ("p_ds_lb_nt_mage", "p_ds_lb_nt_ethn", "p_ds_lb_wt_ethn"):
            checks[f"placeholder_{name}"] = con.execute(
                f"SELECT count({name}) FROM us_births"
            ).fetchone()[0]
        report["derived_value_mismatches"] = checks
        coverage = reference_coverage(con, layout.year)
        coverage["reference_sha256"] = build["identity"]["reference_sha256"]
        report["reference_coverage"] = coverage
        report["recorded_status_counts"] = [
            dict(zip(("status", "births"), row, strict=True))
            for row in con.execute(
                "SELECT ca_down,count(*) FROM us_births GROUP BY ca_down ORDER BY ca_down"
            ).fetchall()
        ]
        report["reporting_flag_counts"] = {
            f["name"]: dict(
                con.execute(
                    f"SELECT {identifier(f['name'])},count(*) FROM us_births GROUP BY {identifier(f['name'])}"
                ).fetchall()
            )
            for f in layout.fields
            if f["name"].startswith("f_")
        }
    after_stat = baseline.stat()
    report["baseline_sha256_after"] = sha256(baseline)
    report["baseline_unchanged"] = (
        before_hash == report["baseline_sha256_after"]
        and before_stat.st_size == after_stat.st_size
        and before_stat.st_mtime_ns == after_stat.st_mtime_ns
    )
    report["passed"] = bool(
        report["baseline_unchanged"]
        and report["record_identity_valid"]
        and all(field_checks.values())
        and all(c["passed"] for c in report["published_controls"])
        and not any(checks.values())
        and coverage["derived_values_match_available_references"]
    )
    report.update(
        elapsed_seconds=time.monotonic() - started,
        peak_rss_bytes=monitor.peak,
        rss_includes_children=monitor.includes_children,
    )
    save_json(candidate.with_suffix(".coverage.json"), coverage)
    save_json(candidate.with_suffix(".validation.json"), report)
    return report

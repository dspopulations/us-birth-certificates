"""Combine validated annual databases without touching the live database."""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb

from dspopulations_us_birth_certificates.reference_data import REFERENCE_TABLES

from .database import MemoryMonitor, protect_output
from .layouts import identifier, load_layout, sha256
from .sources import save_json


def _quote(value: Path) -> str:
    return "'" + str(value.resolve()).replace("'", "''") + "'"


def _managed_state(path: Path) -> dict:
    with duckdb.connect(str(path), read_only=True) as con:
        if "__natality_combined" not in {
            r[0] for r in con.execute("SHOW TABLES").fetchall()
        }:
            raise ValueError(
                f"Existing output is not a managed combined database: {path}"
            )
        return json.loads(
            con.execute("SELECT state FROM __natality_combined WHERE id=1").fetchone()[
                0
            ]
        )


def _state(con, state: dict) -> None:
    con.execute(
        "INSERT OR REPLACE INTO __natality_combined VALUES (1, ?)", [json.dumps(state)]
    )


def combine_years(
    annual: dict[int, Path],
    output: Path,
    baseline: Path,
    *,
    memory_limit: str = "2GB",
    threads: int = 2,
    restart: bool = False,
) -> dict:
    """Resume committed years in a separate build, then replace only its output.

    Each annual validation report must identify the exact database bytes read.
    A year and its provenance commit together. The combined copy is checked
    after reopening, including two aggregate row checksums for every year.
    """
    if not annual or threads < 1:
        raise ValueError("At least one year and one thread are required")
    protect_output(output, baseline)
    building = output.with_suffix(".building.db")
    protect_output(building, baseline)
    schema = {}
    inputs = []
    references = None
    for year, path in sorted(annual.items()):
        protect_output(output, path)
        protect_output(building, path)
        layout = load_layout(year)
        report_path = path.with_suffix(".validation.json")
        report = json.loads(report_path.read_text())
        fingerprint = sha256(path)
        with duckdb.connect(str(path), read_only=True) as con:
            build = json.loads(
                con.execute("SELECT state FROM __natality_build WHERE id=1").fetchone()[
                    0
                ]
            )
            identity = build["identity"]
            if (
                not report.get("passed")
                or report.get("candidate_sha256") != fingerprint
                or report.get("build_identity") != identity
                or build["status"] != "source_validated"
                or identity["layout_sha256"] != layout.digest
                or identity["source"] != layout.source
                or not identity["with_derivations"]
            ):
                raise ValueError(
                    f"Annual database has no matching successful validation: {year}"
                )
            if references is not None and references != identity["reference_sha256"]:
                raise ValueError("Annual builds use different reference files")
            references = identity["reference_sha256"]
            columns = {}
            for name, kind, *_ in con.execute("DESCRIBE us_births").fetchall():
                identifier(name)
                if name in schema and schema[name] != kind:
                    raise ValueError(f"Conflicting annual column types: {name}")
                columns[name] = kind
                schema[name] = kind
        inputs.append(
            {
                "year": year,
                "path": str(path.resolve()),
                "sha256": fingerprint,
                "validation_sha256": sha256(report_path),
                "identity": identity,
                "rows": layout.source["expected_counts"]["occurrence"],
                "columns": columns,
            }
        )
    identity = {"format_version": 1, "annual_inputs": inputs, "schema": schema}
    report_path = output.with_suffix(".build.json")
    if output.exists():
        existing = _managed_state(output)
        if existing["identity"] == identity and report_path.exists():
            saved = json.loads(report_path.read_text())
            if saved.get("passed") and saved.get("candidate_sha256") == sha256(output):
                return {**saved, "reused": True}
        if not restart:
            raise ValueError(
                "Combined output changed; use --restart to rebuild this separate candidate"
            )
    if building.exists():
        previous = _managed_state(building)
        if previous["identity"] != identity:
            if not restart:
                raise ValueError(
                    "Incomplete combined build has different inputs; use --restart"
                )
            building.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline_before = sha256(baseline)
    started = time.monotonic()
    names = ", ".join(identifier(n) for n in schema)
    row_hash = f"hash({names})"
    state = {"identity": identity, "status": "loading"}
    with MemoryMonitor() as monitor:
        with duckdb.connect(str(building)) as con:
            con.execute("SET memory_limit = ?", [memory_limit])
            con.execute("SET threads = ?", [threads])
            con.execute("SET preserve_insertion_order = false")
            con.execute(
                "SET temp_directory = ?", [str(building.with_suffix(".db.tmp"))]
            )
            if "__natality_combined" not in {
                r[0] for r in con.execute("SHOW TABLES").fetchall()
            }:
                con.execute("BEGIN")
                try:
                    con.execute(
                        "CREATE TABLE __natality_combined (id INTEGER PRIMARY KEY, state VARCHAR)"
                    )
                    con.execute(
                        "CREATE TABLE __natality_years (year INTEGER PRIMARY KEY, input VARCHAR, checksums VARCHAR)"
                    )
                    con.execute(
                        "CREATE TABLE us_births ("
                        + ", ".join(
                            f"{identifier(n)} {kind}" for n, kind in schema.items()
                        )
                        + ")"
                    )
                    _state(con, state)
                    con.execute("COMMIT")
                except BaseException:
                    con.execute("ROLLBACK")
                    raise
            done = {
                r[0]
                for r in con.execute("SELECT year FROM __natality_years").fetchall()
            }
            for item in inputs:
                year = item["year"]
                if year in done:
                    continue
                path = Path(item["path"])
                if sha256(path) != item["sha256"]:
                    raise ValueError(f"Annual database changed before copying: {year}")
                con.execute(f"ATTACH {_quote(path)} AS annual (READ_ONLY)")
                projection = ", ".join(
                    identifier(n)
                    if n in item["columns"]
                    else f"NULL::{kind} AS {identifier(n)}"
                    for n, kind in schema.items()
                )
                con.execute(
                    f"CREATE TEMP VIEW incoming AS SELECT {projection} FROM annual.us_births"
                )
                expected = con.execute(
                    f"SELECT count(*), bit_xor({row_hash}), sum({row_hash}::HUGEINT)::VARCHAR FROM incoming"
                ).fetchone()
                if expected[0] != item["rows"]:
                    raise ValueError(f"Annual row count changed: {year}")
                con.execute("BEGIN")
                try:
                    if not done:
                        for table, *_ in REFERENCE_TABLES:
                            con.execute(
                                f"CREATE TABLE {identifier(table)} AS SELECT * FROM annual.{identifier(table)}"
                            )
                    con.execute(
                        f"INSERT INTO us_births ({names}) SELECT {names} FROM incoming"
                    )
                    if sha256(path) != item["sha256"]:
                        raise ValueError(
                            f"Annual database changed while copying: {year}"
                        )
                    con.execute(
                        "INSERT INTO __natality_years VALUES (?, ?, ?)",
                        [year, json.dumps(item), json.dumps(expected)],
                    )
                    con.execute("COMMIT")
                except BaseException:
                    con.execute("ROLLBACK")
                    raise
                con.execute("DROP VIEW incoming")
                con.execute("DETACH annual")
                con.execute("CHECKPOINT")
                done.add(year)
                print(
                    f"Combined {year}: {item['rows']:,} validated records", flush=True
                )
            _state(con, {**state, "status": "checking"})
            con.execute("CHECKPOINT")
        # Reopen the completed file before accepting it. Every union column
        # contributes to both checksums, including columns absent in some years.
        with duckdb.connect(str(building)) as con:
            con.execute("SET memory_limit = ?", [memory_limit])
            con.execute("SET threads = ?", [threads])
            actual = {
                row[0]: list(row[1:])
                for row in con.execute(
                    f"SELECT year, count(*), bit_xor({row_hash}), sum({row_hash}::HUGEINT)::VARCHAR FROM us_births GROUP BY year"
                ).fetchall()
            }
            expected = {
                year: json.loads(value)
                for year, value in con.execute(
                    "SELECT year, checksums FROM __natality_years"
                ).fetchall()
            }
            if actual != expected or set(actual) != set(annual):
                raise ValueError(
                    "Combined copy differs from the validated annual inputs"
                )
            if sha256(baseline) != baseline_before:
                raise ValueError("Protected baseline changed during combination")
            state.update(
                status="validated",
                annual_checksums=actual,
                rows=sum(r[0] for r in actual.values()),
            )
            _state(con, state)
            con.execute("CHECKPOINT")
    report = {
        **state,
        "passed": True,
        "candidate_path": str(output.resolve()),
        "candidate_sha256": sha256(building),
        "baseline_sha256": baseline_before,
        "baseline_unchanged": True,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": monitor.peak,
        "rss_includes_children": monitor.includes_children,
        "database_bytes": building.stat().st_size,
        "prediction_policy": "No predictions copied from the old database; regenerate them for this build.",
    }
    # Persist validation before replacement. A failure leaves the old candidate
    # or the fully checked building file available for recovery.
    save_json(report_path, report)
    building.replace(output)
    return report

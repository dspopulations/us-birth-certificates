"""Build a separate, recoverable pilot database. Never promote it automatically."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import psutil
import pyarrow as pa

from dspopulations_us_birth_certificates.preparation import derive_columns

from .layouts import ROOT, TYPES, Layout, identifier, sha256
from .parser import iter_batches
from .sources import download, open_member, save_json


def protect_output(output: Path, baseline: Path | None = None) -> None:
    """Reject the live file, symlinks and hard links to any protected input."""
    if output.suffix != ".db":
        raise ValueError("Pilot database output must end in .db")
    protected = [ROOT / "data" / "us_births.db", ROOT / "data" / "us_births_temp.db"]
    if baseline is not None:
        protected.append(baseline)
    for path in protected:
        if output.resolve() == path.resolve() or (
            output.exists() and path.exists() and os.path.samefile(output, path)
        ):
            raise ValueError(f"Refusing to write to protected database: {path}")


def code_digest() -> str:
    files = (
        sorted(Path(__file__).parent.glob("*.py"))
        + [
            ROOT / "src/dspopulations_us_birth_certificates" / name
            for name in (
                "preparation.py",
                "variables.py",
                "chance.py",
                "reference_data.py",
            )
        ]
        + [ROOT / "uv.lock"]
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class MemoryMonitor:
    """Sample resident memory, including an external ZIP decompressor."""

    def __init__(self):
        self.peak = 0
        self.includes_children = True
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            try:
                total = process.memory_info().rss
                try:
                    children = process.children(recursive=True)
                except PermissionError, psutil.AccessDenied:
                    self.includes_children = False
                    children = []
                for child in children:
                    try:
                        total += child.memory_info().rss
                    except psutil.NoSuchProcess, psutil.AccessDenied:
                        self.includes_children = False
                self.peak = max(self.peak, total)
            except psutil.NoSuchProcess:
                pass
            self.stop.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def _state(con, report: dict) -> None:
    con.execute(
        "INSERT OR REPLACE INTO __natality_build VALUES (1, ?)", [json.dumps(report)]
    )


def import_year(
    layout: Layout,
    *,
    cache: Path,
    output: Path,
    baseline: Path | None = None,
    batch_size: int = 50_000,
    memory_limit: str = "2GB",
    threads: int = 2,
    restart: bool = False,
    reference_dir: Path = ROOT / "data",
    with_derivations: bool = True,
) -> dict:
    protect_output(output, baseline)
    if not 1 <= batch_size <= 1_000_000 or threads < 1:
        raise ValueError("Invalid batch size or thread count")
    source = layout.source
    archive = cache / Path(source["archive_url"]).name
    guide = cache / Path(source["guide_urls"][0]).name
    # Inputs must never be used as output or as its report paths.
    for path in (archive, guide):
        if output.resolve() == path.resolve():
            raise ValueError("Database output aliases a source file")
    archive_metadata = download(
        source["archive_url"], archive, source["archive_sha256"]
    )
    guide_metadata = download(source["guide_urls"][0], guide, source["guide_sha256"])
    reference_hashes = (
        {p.name: sha256(p) for p in sorted(reference_dir.glob("us-births-*.csv"))}
        if with_derivations
        else {}
    )
    identity = {
        "year": layout.year,
        "archive_sha256": source["archive_sha256"],
        "guide_sha256": source["guide_sha256"],
        "layout_sha256": layout.digest,
        "code_sha256": code_digest(),
        "reference_sha256": reference_hashes,
        "source": source,
        "with_derivations": with_derivations,
    }
    existed = output.exists()
    if existed:
        with duckdb.connect(str(output), read_only=True) as check:
            if "__natality_build" not in {
                r[0] for r in check.execute("SHOW TABLES").fetchall()
            }:
                raise ValueError(
                    "Existing output is not a managed natality pilot; refusing to modify it"
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(output))
    report = {
        "identity": identity,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "loading",
    }
    loading_started = False
    try:
        if existed:
            tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
            if "__natality_build" not in tables:
                raise ValueError(
                    "Existing output is not a managed natality pilot; refusing to modify it"
                )
            previous = json.loads(
                con.execute("SELECT state FROM __natality_build WHERE id=1").fetchone()[
                    0
                ]
            )
            if previous["identity"]["year"] != layout.year:
                raise ValueError("The pilot output must contain only the selected year")
            if (
                previous["identity"] == identity
                and previous["status"] == "source_validated"
            ):
                count = con.execute("SELECT count(*) FROM us_births").fetchone()[0]
                if count != source["expected_counts"]["occurrence"]:
                    raise ValueError("Completed pilot row count changed")
                return {**previous, "reused": True}
            if not restart:
                raise ValueError(
                    "Incomplete or changed build; use --restart with this separate pilot output"
                )
        else:
            con.execute(
                "CREATE TABLE __natality_build (id INTEGER PRIMARY KEY, state VARCHAR)"
            )
        con.execute("SET memory_limit = ?", [memory_limit])
        con.execute("SET threads = ?", [threads])
        con.execute("SET temp_directory = ?", [str(output.with_suffix(".db.tmp"))])
        con.execute("SET preserve_insertion_order = false")
        _state(con, report)
        loading_started = True
        con.execute("DROP TABLE IF EXISTS __incoming")
        columns = ", ".join(
            f"{identifier(f['name'])} {TYPES[f['type']][1]}" for f in layout.fields
        )
        con.execute(f"CREATE TABLE __incoming ({columns}, source_record BIGINT)")
        names = ", ".join(identifier(n) for n in layout.schema.names)
        started = time.monotonic()
        frequencies: dict = {}
        rows = 0
        with MemoryMonitor() as monitor:
            with open_member(archive, layout) as (reader, backend):
                for batch in iter_batches(reader, layout, batch_size, frequencies):
                    con.register("incoming_batch", pa.Table.from_batches([batch]))
                    try:
                        con.execute(
                            f"INSERT INTO __incoming ({names}) SELECT {names} FROM incoming_batch"
                        )
                    finally:
                        con.unregister("incoming_batch")
                    rows += batch.num_rows
                    if rows % (batch_size * 10) == 0:
                        print(
                            f"Imported {rows:,} records for {layout.year}", flush=True
                        )
            # The member context has checked EOF, byte count, CRC and child exit status.
            exceptions = []
            for field in layout.fields:
                for token, alias in field.get("lexical_aliases", {}).items():
                    observed = frequencies[field["name"]].get(token, 0)
                    if observed != alias["expected_count"]:
                        raise ValueError(
                            f"Lexical exception count changed for {field['name']}: {observed}"
                        )
                    exceptions.append(
                        {
                            "field": field["name"],
                            "source_token": token,
                            **alias,
                            "observed_count": observed,
                        }
                    )
            actual = con.execute("""
                SELECT count(*), count(*) FILTER (WHERE restatus IN (1,2,3)),
                    count(*) FILTER (WHERE restatus=4), min(dob_yy), max(dob_yy),
                    count(dob_yy), min(source_record), max(source_record)
                FROM __incoming
            """).fetchone()
            expected = source["expected_counts"]
            if actual != (
                expected["occurrence"],
                expected["resident"],
                expected["foreign_resident"],
                layout.year,
                layout.year,
                rows,
                1,
                rows,
            ):
                raise ValueError(f"Source controls failed: {actual}")
            report.update(
                rows=rows,
                counts=dict(
                    zip(
                        ("occurrence", "resident", "foreign_resident"),
                        actual[:3],
                        strict=True,
                    )
                ),
                source_member=layout.data["member"],
                decompressor=backend,
                uncompressed_bytes=reader.size,
                crc32=reader.crc,
                load_seconds=time.monotonic() - started,
                archive=archive_metadata,
                guide=guide_metadata,
                source_exceptions=exceptions,
            )
            # Save the frequency evidence before marking the database accepted.
            # A crash after the transaction must not leave an accepted build
            # without the evidence required by its validator.
            save_json(output.with_suffix(".frequencies.json"), frequencies)
            report["frequencies_sha256"] = sha256(
                output.with_suffix(".frequencies.json")
            )
            # Swapping the accepted table and deriving columns is all-or-nothing.
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute("DROP TABLE IF EXISTS us_births")
                con.execute("ALTER TABLE __incoming RENAME TO us_births")
                con.execute("ALTER TABLE us_births ADD COLUMN id BIGINT")
                con.execute(
                    "UPDATE us_births SET id = ? * 1000000000::BIGINT + source_record",
                    [layout.year],
                )
                derive_start = time.monotonic()
                if with_derivations:
                    for table in (
                        "prevalence_year",
                        "us_births_est_prevalence_age",
                        "reduction_rate_year",
                        "ds_case_weights",
                        "us_births_est_prevalence_ethnicity",
                    ):
                        con.execute(f"DROP TABLE IF EXISTS {identifier(table)}")
                    derive_columns(con, reference_dir)
                report.update(
                    status="source_validated",
                    derive_seconds=time.monotonic() - derive_start,
                    validated_at=datetime.now(UTC).isoformat(),
                )
                _state(con, report)
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
            con.execute("CHECKPOINT")
        report.update(
            peak_rss_bytes=monitor.peak,
            rss_includes_children=monitor.includes_children,
            database_bytes=output.stat().st_size,
            elapsed_seconds=time.monotonic() - started,
            duckdb_version=duckdb.__version__,
            pyarrow_version=pa.__version__,
            batch_size=batch_size,
            memory_limit=memory_limit,
            threads=threads,
        )
        _state(con, report)
        con.execute("CHECKPOINT")
        save_json(output.with_suffix(".import.json"), report)
        return report
    except BaseException as exc:
        if loading_started:
            # Do not rewrite a completed build on a refused restart or identity mismatch.
            current = con.execute(
                "SELECT state FROM __natality_build WHERE id=1"
            ).fetchone()
            if current and json.loads(current[0]).get("status") == "loading":
                report.update(status="failed", error=str(exc))
                _state(con, report)
                save_json(output.with_suffix(".import.json"), report)
        raise
    finally:
        con.close()

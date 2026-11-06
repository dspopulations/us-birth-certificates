"""Check stored annual data against the independently counted source tokens."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from .layouts import Layout, identifier, sha256


def source_value(token: str, kind: str):
    if not token.strip():
        return None
    if kind == "string":
        return token
    if kind == "integer_string":
        return str(int(token))
    if kind == "decimal1":
        return Decimal(token)
    return int(token)


def check_stored_source(con, layout: Layout, candidate: Path, build: dict) -> dict:
    """Use a connection whose main database contains this annual build."""
    frequencies_path = candidate.with_suffix(".frequencies.json")
    if sha256(frequencies_path) != build.get("frequencies_sha256"):
        raise ValueError("Source frequency report is missing or changed")
    frequencies = json.loads(frequencies_path.read_text())
    year_field = identifier(layout.data.get("year_field", "dob_yy"))
    count, distinct, low, high, bad_ids, bad_year = con.execute(f"""
        SELECT count(*), count(DISTINCT source_record), min(source_record), max(source_record),
            count(*) FILTER (WHERE id IS DISTINCT FROM {layout.year}000000000::BIGINT + source_record),
            count(*) FILTER (WHERE {year_field} IS DISTINCT FROM {layout.year})
        FROM us_births
    """).fetchone()
    expected = layout.source["expected_counts"]["occurrence"]
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
                canonical = {
                    **field.get("lexical_aliases", {}),
                    **field.get("source_code_exceptions", {}),
                }.get(token, {}).get("canonical", token)
                desired[source_value(canonical, field["type"])] += n
        field_checks[field["name"]] = (
            actual == desired and sum(desired.values()) == count
        )
    return {
        "rows": count,
        "record_identity_valid": count == distinct == high == expected
        and low == 1
        and not bad_ids
        and not bad_year,
        "stored_fields_match_source_frequencies": field_checks,
    }

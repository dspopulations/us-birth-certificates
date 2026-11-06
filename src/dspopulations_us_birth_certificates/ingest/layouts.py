"""Explicit annual layouts. Unreviewed years cannot be imported."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pyarrow as pa

from dspopulations_us_birth_certificates.variables import IMPORTED_VARS

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "natality"
TYPES = {
    "uint8": (pa.uint8(), "UTINYINT"),
    "uint16": (pa.uint16(), "USMALLINT"),
    "string": (pa.string(), "VARCHAR"),
    "integer_string": (pa.string(), "VARCHAR"),
    "decimal1": (pa.decimal128(3, 1), "DECIMAL(3,1)"),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identifier(value: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return f'"{value}"'


@dataclass(frozen=True)
class Layout:
    data: dict
    source: dict
    digest: str

    @property
    def year(self) -> int:
        return self.data["year"]

    @property
    def fields(self) -> list[dict]:
        return self.data["fields"]

    @property
    def schema(self) -> pa.Schema:
        return pa.schema(
            [(f["name"], TYPES[f["type"]][0]) for f in self.fields]
            + [("source_record", pa.int64())]
        )


def allowed_tokens(field: dict) -> list[str]:
    """Expand reviewed domains, preserving zero padding and decimal notation."""
    width = field["width"]
    values = set(field.get("values", []))
    if field["type"] == "decimal1":
        values = {str(Decimal(str(v)).quantize(Decimal("0.1"))) for v in values}
        for low, high in field.get("ranges", []):
            values.update(
                str(Decimal(n) / 10) if n % 10 else f"{n // 10}.0"
                for n in range(int(Decimal(low) * 10), int(Decimal(high) * 10) + 1)
            )
    elif field["type"] != "string":
        for low, high in field.get("ranges", []):
            values.update(range(low, high + 1))
        values = {str(v).zfill(width) for v in values}
    if field["blank"] != "forbidden":
        values.add(" " * width)
    values.update(field.get("lexical_aliases", {}))
    return sorted(values)


def validate_layout(data: dict, source: dict) -> None:
    if data["schema_version"] != 1 or data["year"] != source["year"]:
        raise ValueError("Layout version or year mismatch")
    if (
        data["population"] != "US occurrences"
        or source["population"] != data["population"]
    ):
        raise ValueError("Only US occurrence files are supported")
    if data["encoding"] != "ascii" or data["separator"] not in ("CRLF", "LF", "NONE"):
        raise ValueError("Unsupported record encoding or separator")
    names = [f["name"] for f in data["fields"]]
    if len(set(names)) != len(names) or not set(IMPORTED_VARS).issubset(names):
        raise ValueError(
            "Every imported column needs exactly one availability decision"
        )
    if "source_record" in names:
        raise ValueError("Reserved source_record column")
    used: set[int] = set()
    for field in data["fields"]:
        identifier(field["name"])
        if field["type"] not in TYPES:
            raise ValueError("Unsupported field type")
        if field["status"] == "absent":
            if not field.get("reason") or "start" in field:
                raise ValueError(
                    "Absent columns need evidence and cannot have positions"
                )
            continue
        if field["status"] != "present" or field["blank"] not in (
            "forbidden",
            "not_on_certificate",
            "not_imputed",
        ):
            raise ValueError("Unsupported field status or blank meaning")
        start, width = field["start"], field["width"]
        span = set(range(start, start + width))
        if (
            start < 1
            or width < 1
            or max(span) > data["documented_record_length"]
            or span & used
        ):
            raise ValueError(f"Invalid or overlapping span: {field['name']}")
        used.update(span)
        tokens = allowed_tokens(field)
        if not tokens or any(len(v) != width or not v.isascii() for v in tokens):
            raise ValueError(f"Invalid domain: {field['name']}")
        canonical = allowed_tokens(
            {k: v for k, v in field.items() if k != "lexical_aliases"}
        )
        for raw, alias in field.get("lexical_aliases", {}).items():
            if (
                field["type"] not in ("uint8", "uint16", "integer_string")
                or raw in canonical
                or alias["canonical"] not in canonical
                or not raw.strip().isdigit()
                or int(raw) != int(alias["canonical"])
                or not isinstance(alias["expected_count"], int)
                or alias["expected_count"] < 1
                or not alias.get("reason")
            ):
                raise ValueError(f"Invalid lexical exception: {field['name']}")
        if field.get("reporting_flag") not in (None, *names):
            raise ValueError("Reporting flag is missing from layout")
    padding = data["padding"]
    if (
        padding["start"] != data["documented_record_length"] + 1
        or padding["width"] != data["record_length"] - data["documented_record_length"]
        or padding["width"] < 0
        or padding["byte"] != " "
    ):
        raise ValueError("Invalid physical padding specification")


def load_layout(year: int, config: Path = CONFIG) -> Layout:
    catalogue = json.loads((config / "sources.json").read_text())
    matches = [s for s in catalogue["sources"] if s["year"] == year]
    if len(matches) != 1 or matches[0]["layout_status"] != "reviewed":
        raise ValueError(f"No reviewed layout for {year}; refusing to infer one")
    source = matches[0]
    path = (config / source["layout_file"]).resolve()
    if not path.is_relative_to(config.resolve()):
        raise ValueError("Layout must be inside the catalogue directory")
    data = json.loads(path.read_text())
    validate_layout(data, source)
    if data["guide_sha256"] != source["guide_sha256"]:
        raise ValueError("Source catalogue and layout refer to different guides")
    return Layout(data, source, sha256(path))

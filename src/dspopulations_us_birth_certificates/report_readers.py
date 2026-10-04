# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Report parsing with explicit defaults for absent files and loud invalid reads."""

from pathlib import Path
from typing import Any

from dse_research_utils.report.readers import FileRead, read_csv, read_json

_REQUIRED = object()


def _report_value(result: FileRead[Any], missing: Any, kind: str) -> Any:
    if result.status == "present":
        return result.value
    if result.status == "missing":
        if missing is _REQUIRED:
            raise FileNotFoundError(result.path)
        return missing
    raise ValueError(f"Cannot read required {kind} artefact {result.path}: {result.reason}")


def read_json_artefact(path: str | Path, *, missing: Any = _REQUIRED) -> Any:
    """Read JSON, with a caller-supplied default only for an absent file."""
    return _report_value(read_json(path), missing, "JSON")


def read_csv_artefact(
    path: str | Path, *, index_col: int | str | None = None, missing: Any = _REQUIRED
) -> Any:
    """Read CSV, preserving invalid-file failures and any selected index column."""
    return _report_value(read_csv(path, index_col=index_col), missing, "CSV")

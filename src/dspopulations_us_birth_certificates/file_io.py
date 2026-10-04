# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Atomic artefact writes that keep this project's file permissions.

``dse_research_utils.storage.files.atomic_write`` writes through a sibling
temporary file and one ``os.replace``, so a reader never sees a half-written
manifest, config or summary, and a failed write leaves the previous file
intact. It deliberately creates that temporary file with owner-only
permissions. Every artefact here has instead been produced by
``Path.write_text`` or ``DataFrame.to_csv``, which apply the process umask to
``0o666`` — 0o644 in the usual case, which report tooling and other readers
rely on. These wrappers restore that mode before the file is moved into place.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from dse_research_utils.storage.files import atomic_write
from dse_research_utils.storage.files import (
    default_file_mode as _shared_default_file_mode,
)

_FILE_MODE: int | None = None


def default_file_mode(directory: Path | str = ".") -> int:
    """Permission bits a plain ``write_text`` would give a newly created file.

    The shared probe reads ordinary creation permissions in ``directory``
    without changing the process umask. This project retains the first mode
    for the life of the process. The no-argument compatibility call probes
    the current directory.
    """
    global _FILE_MODE
    if _FILE_MODE is None:
        _FILE_MODE = _shared_default_file_mode(directory)
    return _FILE_MODE


def write_atomically(path: Path | str, writer: Callable[[Path], object]) -> Path:
    """Replace ``path`` with whatever ``writer`` puts at the temporary path."""
    destination = Path(path)

    def _write(temporary: Path) -> None:
        writer(temporary)
        temporary.chmod(default_file_mode(temporary.parent))

    atomic_write(destination, _write)
    return destination


def write_text_atomically(
    path: Path | str, text: str, *, encoding: str | None = None
) -> Path:
    """Replace ``path`` with ``text``, keeping the caller's encoding choice."""
    return write_atomically(
        path, lambda temporary: temporary.write_text(text, encoding=encoding)
    )


__all__ = ["default_file_mode", "write_atomically", "write_text_atomically"]

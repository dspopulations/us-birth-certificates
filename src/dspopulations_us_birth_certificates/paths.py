"""Project output locations shared by fitting and discovery commands."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_output_root(root: Path | str | None = None) -> Path:
    """Use an explicit root, then DSPOP_OUTPUT_ROOT, then ``output/``.

    An unset or empty environment variable keeps the local default. Expand
    ``~`` but keep relative paths relative to the working directory, as the
    existing commands do. Read the environment at call time and create no files.
    """
    if root is None:
        root = os.environ.get("DSPOP_OUTPUT_ROOT") or "output"
    return Path(root).expanduser()

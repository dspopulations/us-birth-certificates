"""AI-assisted documentation revision by Codex (GPT-6).

Copy and render Quarto templates in model run directories.

See ``docs/modelling-workflow.md`` for report inputs and rendering commands.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspopulations_us_birth_certificates.models.common import ModelFitContext

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE = Path("docs/models/usbc10/index.qmd")


def copy_template(
    context: ModelFitContext, template_path: Path = DEFAULT_TEMPLATE
) -> Path:
    """Copy a Quarto template into the run's output dir.

    Returns the path to the copied ``index.qmd``. The template is copied
    verbatim. The document loads artefacts from the run directory at render
    time. Rendering also requires the Python environment and Quarto CLI.
    """
    src = Path(template_path)
    if not src.is_file():
        raise FileNotFoundError(f"Quarto template not found: {src}")
    dst = Path(context.output_dir) / "index.qmd"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def render_quarto_report(qmd_path: Path, output_format: str = "html") -> Path | None:
    """Subprocess-invoke ``quarto render`` for ``qmd_path``.

    Returns the rendered file's path on success, or ``None`` when the
    render finished but the expected output file could not be located
    (some formats land in sibling directories with non-obvious suffixes,
    and callers should treat ``None`` as "rendered, but artefact path
    unknown"). Raises ``subprocess.CalledProcessError`` on non-zero exit
    and ``FileNotFoundError`` if the ``quarto`` binary isn't on PATH.
    """
    qmd_path = Path(qmd_path)
    if not qmd_path.is_file():
        raise FileNotFoundError(f"Quarto source not found: {qmd_path}")

    try:
        subprocess.run(
            ["quarto", "render", str(qmd_path), "--to", output_format],
            check=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "`quarto` binary not found on PATH. Install Quarto CLI or "
            "omit --render to skip the render step."
        ) from exc

    rendered = qmd_path.with_suffix(f".{output_format}")
    if rendered.is_file():
        return rendered
    logger.warning(
        "Could not locate rendered file for %s (format=%s); "
        "render completed but output path is unknown.",
        qmd_path,
        output_format,
    )
    return None

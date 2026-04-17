"""Quarto reporting helpers.

Generic enough to migrate upstream into ``dse_research_utils`` once stable.
See ``docs/refactor-plan.md`` step 10 for the upstreaming plan.

Implementation is populated in refactor step 7.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dspopulations_us_birth_certificates.models.common import ModelFitContext


def copy_template(context: ModelFitContext, template_path: Path) -> Path:
    """Copy a Quarto template into the run's output dir.

    Returns the path to the copied ``index.qmd``.
    """
    raise NotImplementedError("populated in refactor step 7")


def render_quarto_report(qmd_path: Path, output_format: str = "html") -> Path:
    """Subprocess-invoke ``quarto render`` for ``qmd_path``.

    Returns the path to the rendered report. Raises ``subprocess.CalledProcessError``
    on a non-zero exit.
    """
    raise NotImplementedError("populated in refactor step 7")

"""Artefact persistence for Bayesian fits."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dspopulations_us_birth_certificates.bayes.config import BayesFitContext

if TYPE_CHECKING:
    import arviz as az


DOCS_TEMPLATE_ROOT = Path("docs/models")


def save_artefacts(context: BayesFitContext, output_dir: Path) -> None:
    """Write InferenceData, configs, and the aggregated cell frame.

    Layout::

        output_dir/
            idata.nc
            cells.parquet
            config.json
            run_config.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if context.idata is not None:
        context.idata.to_netcdf(str(output_dir / "idata.nc"))

    if context.cells is not None:
        context.cells.to_parquet(output_dir / "cells.parquet", index=False)

    (output_dir / "config.json").write_text(
        json.dumps(context.config.to_dict(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(asdict(context.run_config), indent=2),
        encoding="utf-8",
    )


def save_summary(summary: Any, output_dir: Path, *, name: str = "summary.csv") -> None:
    """Write an ``az.summary`` DataFrame to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / name)


def save_prior_predictive_summary(
    summary: Any,
    output_dir: Path,
    *,
    name: str = "prior_predictive_summary.csv",
) -> None:
    """Write a prior-predictive summary DataFrame to CSV next to the fit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / name)


def load_idata(output_dir: Path) -> az.InferenceData:
    """Load a previously-saved InferenceData from an artefact directory."""
    import arviz as az

    return az.from_netcdf(str(output_dir / "idata.nc"))


def copy_docs_template(
    model_id: str,
    output_dir: Path,
    *,
    docs_root: Path = DOCS_TEMPLATE_ROOT,
) -> Path | None:
    """Copy ``docs/models/<model_id>/index.qmd`` next to the fit artefacts.

    Mirrors the sibling-repo pattern: the Quarto template lives in
    version control and renders against whichever run directory it is
    copied into. Returns the destination path, or ``None`` if no
    template exists for this ``model_id``.
    """
    src = docs_root / model_id / "index.qmd"
    if not src.exists():
        return None
    dst = output_dir / "index.qmd"
    shutil.copy(src, dst)
    return dst


def render_quarto(qmd_path: Path) -> None:
    """Invoke ``quarto render`` on a QMD file.

    Raises ``FileNotFoundError`` if Quarto isn't on PATH, or
    ``subprocess.CalledProcessError`` if the render fails.
    """
    subprocess.run(["quarto", "render", str(qmd_path)], check=True)

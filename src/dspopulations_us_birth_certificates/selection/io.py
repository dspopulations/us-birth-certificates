"""Artefact persistence for selection-model fits."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dspopulations_us_birth_certificates import cli_output
from dspopulations_us_birth_certificates.selection.config import FitContext

DOCS_TEMPLATE_ROOT = Path("docs/models")


def save_artefacts(context: FitContext, output_dir: Path) -> None:
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
    manifest = fit_manifest(context, output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_manifest(context: FitContext, output_dir: Path) -> dict:
    """Fingerprint actual likelihood inputs and code, without publishing records."""

    def git(*args):
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    versions = {}
    for package in (
        "pymc",
        "pytensor",
        "nutpie",
        "arviz",
        "numpy",
        "scipy",
        "dse-research-utils",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    files = {}
    for name in ("cells.parquet", "config.json", "run_config.json"):
        path = output_dir / name
        if path.is_file():
            files[name] = _sha256(path)
    source_files = {}
    for root in (Path("src"), Path("scripts")):
        if root.is_dir():
            source_files.update(
                {str(path): _sha256(path) for path in sorted(root.rglob("*.py"))}
            )
    config = context.config.to_dict()
    inputs = {}
    for candidate in (
        config.get("priors", {}).get("reduction_source"),
        (config.get("surveillance_anchor") or {}).get("source"),
        (config.get("anomaly_panel") or {}).get("conditions_source"),
    ):
        if candidate and Path(candidate).is_file():
            inputs[str(candidate)] = _sha256(Path(candidate))
    attrs = getattr(context.idata, "attrs", {})
    return {
        "schema_version": 1,
        "code_revision": git("rev-parse", "HEAD"),
        "worktree_changes": git("status", "--porcelain"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "actual_sampler": attrs.get("dsp_actual_sampler", "unavailable"),
        "prior_sampling": {
            k: v for k, v in attrs.items() if k.startswith("dsp_prior_")
        },
        "artefact_sha256": files,
        "source_sha256": source_files,
        "input_sha256": inputs,
        "lockfile_sha256": _sha256(Path("uv.lock"))
        if Path("uv.lock").is_file()
        else None,
        "validation_file": "validation.json",
        "validation_status": context.metrics.get("validation_status", "not_validated"),
    }


def save_summary(summary: Any, output_dir: Path, *, name: str = "summary.csv") -> None:
    """Write an ``az.summary`` DataFrame to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / name)


def latest_fit_dir(
    variant: str,
    *,
    spec: str = "full",
    root: Path | str = "output/selection",
    require_validated: bool = False,
) -> Path:
    """Return the most recently modified completed fit dir for ``variant``/``spec``.

    Looks under ``<root>/<variant>/<spec>/*`` for run directories written by
    ``scripts/fit_selection_model.py``; a run is "completed" if it has an
    ``idata.nc``. A saved validation status must be "passed". Legacy fits with
    no status remain discoverable unless ``require_validated`` is True; their
    absence of validation must not be interpreted as a pass.
    Centralises the run-layout convention that the analysis
    scripts under ``scripts/`` (year trends, ethnicity breakdowns, coefficient
    dumps, etc.) each need to locate the latest fit to read.

    Raises:
        FileNotFoundError: if no completed fit dir exists.
    """
    parent = Path(root) / variant / spec
    candidates = (
        [p for p in parent.iterdir() if p.is_dir() and (p / "idata.nc").is_file()]
        if parent.is_dir()
        else []
    )

    def eligible(path: Path) -> bool:
        validation = path / "validation.json"
        if not validation.is_file():
            return not require_validated
        try:
            return json.loads(validation.read_text()).get("status") == "passed"
        except OSError, ValueError, AttributeError:
            return False

    candidates = [path for path in candidates if eligible(path)]
    if not candidates:
        raise FileNotFoundError(f"No completed fit (idata.nc) found under {parent}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


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


def render_report(qmd_path: Path | None, *, do_render: bool) -> None:
    """Render ``qmd_path`` via Quarto when requested, reporting but not raising on failure.

    No-ops quietly if ``do_render`` is False or no template was copied
    (``qmd_path is None``). A missing ``quarto`` on PATH or a render failure
    is logged as a warning rather than raised, so a fit CLI still completes
    without a rendered HTML.
    """
    if not do_render or qmd_path is None:
        return
    cli_output.section("Render")
    try:
        render_quarto(qmd_path)
        cli_output.success(f"Rendered {qmd_path.with_suffix('.html')}")
    except FileNotFoundError:
        cli_output.warning(
            f"`quarto` not on PATH — render manually: quarto render {qmd_path}"
        )
    except Exception as exc:  # noqa: BLE001 — rendering is optional
        cli_output.warning(f"Quarto render raised {type(exc).__name__}: {exc}")

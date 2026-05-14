"""Reproducibility manifest for a model run.

Captures the provenance needed to reconstitute a result:

- git SHA, branch, and whether the working tree was dirty
- runtime environment (platform, Python version)
- relevant package versions (lightgbm, optuna, shap, sklearn, numpy,
  pandas, scipy, plus the project distribution itself)
- ``ModelConfig`` snapshot
- ``RunConfig`` snapshot
- input row count and positive count
- ``random_seed`` (shared by the model and the split; widen this if we
  decouple them in a follow-up)
- final validation metrics
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dspopulations_us_birth_certificates.models.common import ModelFitContext


# Packages whose versions are recorded in every manifest. Kept deliberately
# short: only the dependencies whose behaviour would change the numbers.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "dspopulations-us-birth-certificates",
    "lightgbm",
    "numpy",
    "optuna",
    "pandas",
    "scikit-learn",
    "scipy",
    "shap",
)


def _default_repo_root() -> Path:
    """Best-guess location of the repo root from this file's path.

    ``manifest.py`` lives at ``<repo>/src/<pkg>/manifest.py`` when installed
    editable, so two ``parents`` hops land at the repo root. When installed
    from a wheel this points inside site-packages, which is harmless —
    ``_git_info`` will just return ``None`` because there's no .git there.
    """
    return Path(__file__).resolve().parents[2]


def _git_info(repo_root: Path | None = None) -> dict[str, Any]:
    """Return git SHA and dirty flag. Silent fallback when git isn't available."""
    if repo_root is None:
        repo_root = _default_repo_root()
    cwd = str(repo_root)
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode()
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return {"sha": sha, "branch": branch, "dirty": bool(status.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "branch": None, "dirty": None}


def _package_versions() -> dict[str, str | None]:
    """Version for each tracked package, or None if not installed."""
    out: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }


def data_fingerprint(
    X_train: pd.DataFrame | None,
    X_valid: pd.DataFrame | None,
    y_train: pd.Series | None,
    y_valid: pd.Series | None,
) -> dict[str, Any]:
    """Row counts, positive counts, and a schema hash for the split data.

    The schema hash is a stable digest of the column names + dtypes; it
    changes iff the feature set or dtypes change. It is *not* a hash of the
    data values themselves, so two runs on the same cut of NVSS produce
    identical fingerprints even though the hash is cheap to compute.
    """
    fp: dict[str, Any] = {}
    if X_train is not None:
        fp["n_train"] = int(len(X_train))
        fp["schema"] = _schema_hash(X_train)
    if X_valid is not None:
        fp["n_valid"] = int(len(X_valid))
    if y_train is not None:
        fp["n_positive_train"] = int(np.asarray(y_train).sum())
    if y_valid is not None:
        fp["n_positive_valid"] = int(np.asarray(y_valid).sum())
    return fp


def _schema_hash(df: pd.DataFrame) -> str:
    """Hash of the column names + dtypes. Stable across runs on the same schema."""
    parts = [f"{c}:{str(dt)}" for c, dt in df.dtypes.items()]
    joined = "|".join(parts).encode()
    return hashlib.sha256(joined).hexdigest()[:16]


def write_manifest(
    context: ModelFitContext,
    output_dir: Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Serialise a run manifest to ``output_dir/manifest.json``.

    Returns the path written. Existing manifests are overwritten.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git": _git_info(),
        "environment": _environment_snapshot(),
        "config": context.config.to_dict(),
        "run_config": asdict(context.run_config),
        "data_fingerprint": data_fingerprint(
            context.X_train, context.X_valid, context.y_train, context.y_valid
        ),
        "metrics": context.metrics,
        "seeds": {
            "random_seed": context.run_config.random_seed,
        },
    }
    if extra:
        manifest["extra"] = extra

    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
